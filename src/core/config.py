"""
Configuration management for model IDs and pipeline settings.

The ConfigLoader reads from settings.json to allow customization of:
- Model IDs for extraction, validation, and reasoning
- Model cache behavior
- Performance tuning parameters
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and manages pipeline configuration from settings.json.

    Provides fallback defaults if settings.json is missing or incomplete.
    """

    # Default model IDs
    _DEFAULTS = {
        "models": {
            "text_extraction": "mlx-community/Phi-3.5-mini-Instruct-4bit",
            "table_extraction": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "vision_extraction": "mlx-community/SmolVLM-256M-Instruct-4bit",
            "chunk_validator": "mlx-community/SmolVLM-256M-Instruct-4bit",
            "orchestrator": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
            "answer_validator": "mlx-community/Qwen3-8B-4bit",
            "dspy_lm": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "embedder": "intfloat/multilingual-e5-small",
        },
        "pipeline": {
            "max_context_chunks": 8,
            "embedder_batch_size": 32,
            "chunk_size": 800,
        },
        "cache": {
            "enable_hf_cache": True,
            "cache_dir": "./models",
        },
        "validation": {
            "confidence_threshold": 0.5,
            "enable_checkpoint_a": True,
            "enable_checkpoint_b": True,
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to settings.json. If None, looks for
                        settings.json in project root.
        """
        self.config_path = config_path or Path(__file__).parent.parent.parent / "settings.json"
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """
        Load settings.json, falling back to defaults if missing.

        Returns:
            Merged configuration dictionary
        """
        config = self._DEFAULTS.copy()

        if not self.config_path.exists():
            log.warning(
                f"settings.json not found at {self.config_path}. Using defaults. "
                "Copy settings.example.json to settings.json to customize."
            )
            return config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)

            # Deep merge user config with defaults
            config = self._deep_merge(config, user_config)
            log.info(f"Loaded configuration from {self.config_path}")

        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in {self.config_path}: {e}. Using defaults.")
        except Exception as e:
            log.error(f"Error loading {self.config_path}: {e}. Using defaults.")

        return config

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Recursively merge override dict into base dict.

        Args:
            base: Base configuration (defaults)
            override: User configuration (overrides)

        Returns:
            Merged configuration
        """
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_model(self, model_key: str) -> str:
        """
        Get model ID for a specific role.

        Args:
            model_key: Key like "text_extraction", "orchestrator", etc.

        Returns:
            Model ID (HuggingFace path or local path)
        """
        return self.config.get("models", {}).get(model_key, self._DEFAULTS["models"].get(model_key, ""))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot notation key.

        Example: config.get("pipeline.max_context_chunks")

        Args:
            key: Dot-separated config key
            default: Default value if key not found

        Returns:
            Config value or default
        """
        parts = key.split(".")
        value = self.config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
        return value if value is not None else default


# Global config instance
config = ConfigLoader()
