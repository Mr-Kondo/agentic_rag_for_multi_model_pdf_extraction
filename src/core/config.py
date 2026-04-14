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
            "text_extraction": "qwen3:8b",
            "table_extraction": "qwen2.5:3b",
            "vision_extraction": "qwen2.5vl:7b",
            "chunk_validator": "qwen2.5vl:7b",
            "orchestrator": "gemma4:latest",
            "answer_validator": "gemma4:latest",
            "dspy_lm": "gemma4:latest",
            "embedder": "kun432/cl-nagoya-ruri-large",
        },
        "embedder": {
            "backend": "ollama",  # "sentence_transformers" | "ollama"
            "query_prefix": None,  # None = use backend default
            "passage_prefix": None,  # None = use backend default
            "batch_size": 32,
            "max_input_chars": 800,
            "retry_trim_enabled": True,
            "retry_trim_min_chars": 128,
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
        "ollama": {
            "base_url": "http://localhost:11434",
        },
        "parser": {
            "enable_native_pdf_heuristic": True,
            "native_pdf_max_images": 2,
            "native_pdf_min_words": 50,
        },
        "ocr": {
            "engine": "easyocr",
            "default_lang": "jpn+eng",
            "japanese_lang": "jpn",
            "fallback_lang": "eng",
            "config": "--oem 3 --psm 6",
            "render_scale": 1.5,
            "tesseract_render_scale": 1.0,
            "prewarm_easyocr": True,
            "max_rescue_blocks_per_page": 6,
            "min_reocr_text_length": 6,
            "cache_tesseract_languages": True,
            "line_confidence_threshold": 0.55,
            "enable_line_reocr": True,
            "max_line_reocr_attempts": 2,
            "region_policies": {
                "text": {
                    "engine": "easyocr",
                    "line_confidence_threshold": 0.55,
                    "enable_reocr": True,
                    "max_reocr_attempts": 2,
                    "apply_post_correction": True,
                },
                "table": {
                    "engine": "easyocr",
                    "line_confidence_threshold": 0.65,
                    "enable_reocr": True,
                    "max_reocr_attempts": 1,
                    "apply_post_correction": False,
                },
                "figure": {
                    "engine": "tesseract",
                    "line_confidence_threshold": 0.5,
                    "enable_reocr": False,
                    "max_reocr_attempts": 0,
                    "apply_post_correction": False,
                },
            },
            "post_correction": {
                "enabled": False,
                "dictionary_paths": [],
                "apply_to_ocr_only": True,
            },
        },
        "audit": {
            "render_page_previews": True,
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

    def get_ollama_base_url(self) -> str:
        """
        Get the Ollama server base URL.

        Returns:
            URL string, e.g. "http://localhost:11434"
        """
        return self.config.get("ollama", {}).get("base_url", self._DEFAULTS["ollama"]["base_url"])

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
