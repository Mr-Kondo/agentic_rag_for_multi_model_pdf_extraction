"""
Model caching system for managing in-memory model instances.

Handles loading, caching, and cleanup of MLX text and vision models.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from mlx_lm import load
from mlx_vlm import load as vlm_load
from mlx_vlm.utils import load_config

log = logging.getLogger(__name__)

# Configure model cache directory
MODEL_CACHE_DIR = Path.home() / ".models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR.resolve())


class ModelCache:
    """
    Manages in-memory caching of loaded models and tracks their usage.

    Models are cached in memory while loaded to avoid redundant downloads.
    Cleanup removes models from .models/ that are no longer being used.

    Attributes:
        _text_models: Dictionary of cached text models {model_id: model}
        _vision_models: Dictionary of cached vision models {model_id: (model, processor, config)}
        _model_usage: Set of model IDs currently in use
    """

    def __init__(self):
        self._text_models: dict[str, Any] = {}
        self._vision_models: dict[str, Any] = {}
        self._model_usage: set[str] = set()
        self._lock = None  # Could be threading.Lock() for thread safety if needed

    def load_text_model(self, model_id: str) -> Any:
        """
        Load a text model from cache or download it.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Loaded MLX text model
        """
        if model_id in self._text_models:
            log.debug(f"📦 Returning cached text model: {model_id}")
            return self._text_models[model_id]

        log.info(f"🔄 Loading text model: {model_id}")
        model = load(model_id)
        self._text_models[model_id] = model
        self._model_usage.add(model_id)
        return model

    def load_vision_model(self, model_id: str) -> tuple[Any, Any, Any]:
        """
        Load a vision model from cache or download it.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Tuple of (model, processor, config)
        """
        if model_id in self._vision_models:
            log.debug(f"📦 Returning cached vision model: {model_id}")
            return self._vision_models[model_id]

        log.info(f"🔄 Loading vision model: {model_id}")
        model, processor = vlm_load(model_id)
        config = load_config(model_id)
        cached_model = (model, processor, config)
        self._vision_models[model_id] = cached_model
        self._model_usage.add(model_id)
        return cached_model

    def cleanup_unused_models(self):
        """
        Remove model directories from .models/ that are not currently loaded.

        This frees up disk space after models are no longer needed.
        """
        try:
            if not MODEL_CACHE_DIR.exists():
                return

            # Get list of cached model directories
            cached_models = set(d.name for d in MODEL_CACHE_DIR.glob("**/") if d.is_dir())

            # Find models not currently in use
            loaded_models = set(self._text_models.keys()) | set(self._vision_models.keys())
            unused = cached_models - self._model_usage

            if unused:
                log.info(f"🧹 Cleaning up {len(unused)} unused model(s)...")
                for model_name in unused:
                    model_path = MODEL_CACHE_DIR / model_name
                    if model_path.exists():
                        log.info(f"  Removing {model_name}...")
                        shutil.rmtree(model_path, ignore_errors=True)
        except Exception as e:
            log.warning(f"⚠️  Error during model cleanup: {e}")


# Global model cache instance
_model_cache = ModelCache()
