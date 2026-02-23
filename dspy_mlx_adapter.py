"""
dspy_mlx_adapter.py
===================
Custom DSPy language model adapter for MLX (Apple Silicon optimization).

DSPy by default uses `transformers` with PyTorch, but this project uses MLX
for efficient inference on Apple Silicon. This adapter implements the DSPy LM
interface while using mlx-lm under the hood.

Usage:
    from dspy_mlx_adapter import MLXLM
    import dspy

    # Initialize MLX-backed LM
    lm = MLXLM(model_id="mlx-community/Qwen2.5-7B-Instruct-4bit")

    # Configure DSPy to use this LM
    dspy.configure(lm=lm)

    # Use DSPy modules as normal
    signature = dspy.Signature("question -> answer")
    predictor = dspy.ChainOfThought(signature)
    result = predictor(question="What is 2+2?")

Integration with existing BaseLoadableModel pattern:
    class DSPyValidatorAgent(BaseLoadableModel):
        def _do_load(self):
            self._lm = MLXLM(self.model_id)
            dspy.configure(lm=self._lm)
            self._predictor = dspy.ChainOfThought(ValidatorSignature)
"""

from __future__ import annotations

import logging
from typing import Any

import dspy
from mlx_lm import generate, load

log = logging.getLogger(__name__)


class MLXLM(dspy.LM):
    """
    DSPy language model wrapper for MLX-based models.

    This adapter allows DSPy modules to use mlx-lm for inference on Apple
    Silicon, maintaining compatibility with the existing model loading and
    generation patterns used throughout the project.

    Args:
        model_id: Hugging Face model ID or local path (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")
        max_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.0 for deterministic output)
        **kwargs: Additional generation parameters passed to mlx_lm.generate()

    Attributes:
        model: The loaded MLX model
        tokenizer: The loaded MLX tokenizer
        model_id: The model identifier
        history: List of interaction history (for DSPy compatibility)
    """

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MLX language model adapter.

        The model is loaded immediately upon initialization. For integration
        with BaseLoadableModel pattern, wrap this initialization in _do_load().
        """
        super().__init__(model=model_id)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs

        log.info(f"Loading MLX model: {model_id}")
        self._model, self._tokenizer = load(model_id)
        log.info(f"✓ MLX model loaded: {model_id}")

        # DSPy expects a history attribute for tracking interactions
        self.history: list[dict[str, Any]] = []

    def __call__(
        self,
        prompt: str | list[dict[str, str]] | None = None,
        messages: str | list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate text completion for the given prompt.

        This is the main interface method required by DSPy. It accepts either
        a string prompt or a list of message dicts (OpenAI chat format).
        DSPy adapters call this with `messages` kwarg, but direct calls can use `prompt`.

        Args:
            prompt: Either a raw string or list of chat messages (positional/keyword)
            messages: Alternative parameter name used by DSPy adapters
            **kwargs: Generation parameters (max_tokens, temperature, etc.)

        Returns:
            List of generated completions (typically length 1 for greedy decoding)
        """
        # Check if model is loaded
        if self._model is None or self._tokenizer is None:
            log.error("Model not loaded. Call load() first or reinitialize MLXLM.")
            return [""]

        # DSPy adapters use 'messages' kwarg, but allow 'prompt' for direct calls
        input_prompt = messages if messages is not None else prompt
        if input_prompt is None:
            log.error("No prompt or messages provided to MLXLM.__call__")
            return [""]

        # Merge kwargs - only include parameters that mlx-lm supports
        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "verbose": False,
        }

        # Only add temperature if explicitly requested and non-zero
        # (mlx-lm may not support it in all versions/configurations)
        temp = kwargs.get("temperature", self.temperature)
        if temp != 0.0:
            gen_kwargs["temp"] = temp

        gen_kwargs.update(self.kwargs)
        gen_kwargs.update(kwargs)

        # Remove DSPy-specific kwargs that mlx-lm doesn't understand
        for key in ["n", "stop", "top_p", "frequency_penalty", "presence_penalty", "messages"]:
            gen_kwargs.pop(key, None)

        # Handle both string prompts and message lists
        if isinstance(input_prompt, list):
            # Chat format: apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                input_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            formatted_prompt = input_prompt

        # Generate using mlx-lm
        try:
            output = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                **gen_kwargs,
            )

            # Record interaction in history (for DSPy introspection)
            self.history.append(
                {
                    "prompt": input_prompt,
                    "output": output,
                    "kwargs": gen_kwargs,
                }
            )

            # DSPy expects a list of completions
            return [output]

        except Exception as e:
            log.error(f"MLX generation failed: {e}")
            # Return empty list on failure (DSPy will handle)
            return [""]

    def basic_request(
        self,
        prompt: str | list[dict[str, str]] | None = None,
        messages: str | list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Alternative interface for direct requests (used by some DSPy optimizers).

        Args:
            prompt: String or chat message list
            messages: Alternative parameter name used by DSPy adapters
            **kwargs: Generation parameters

        Returns:
            Dict with 'choices' key containing list of completion dicts
        """
        completions = self(prompt=prompt, messages=messages, **kwargs)

        # Format as OpenAI-style response for DSPy compatibility
        return {"choices": [{"text": completion, "message": {"content": completion}} for completion in completions]}

    def inspect_history(self, n: int = 1) -> list[dict[str, Any]]:
        """
        Retrieve the last n interactions from history.

        Useful for debugging and understanding DSPy's prompt optimization process.

        Args:
            n: Number of recent interactions to return

        Returns:
            List of interaction dictionaries
        """
        return self.history[-n:] if self.history else []

    def unload(self) -> None:
        """
        Explicitly unload the model from memory.

        This mirrors the BaseLoadableModel pattern used elsewhere in the project.
        After calling unload(), this LM instance should not be used for generation.
        """
        log.info(f"Unloading MLX model: {self.model_id}")
        self._model = None
        self._tokenizer = None
        self.history.clear()


# ═══════════════════════════════════════════════════════════
# Helper function for easy DSPy configuration
# ═══════════════════════════════════════════════════════════


def configure_mlx_lm(model_id: str, **kwargs: Any) -> MLXLM:
    """
    Convenience function to load an MLX model and configure DSPy in one call.

    Args:
        model_id: Model identifier
        **kwargs: Additional parameters for MLXLM initialization

    Returns:
        Configured MLXLM instance

    Example:
        lm = configure_mlx_lm("mlx-community/Qwen2.5-7B-Instruct-4bit")
        predictor = dspy.ChainOfThought("question -> answer")
        result = predictor(question="What is the capital of France?")
    """
    lm = MLXLM(model_id, **kwargs)
    dspy.configure(lm=lm)
    return lm
