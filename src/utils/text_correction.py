"""Dictionary-based text post-correction utilities."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DictionaryCorrector:
    """Apply exact and regex replacement rules to OCR text."""

    exact_rules: dict[str, str] = field(default_factory=dict)
    regex_rules: list[tuple[re.Pattern[str], str]] = field(default_factory=list)

    @classmethod
    def from_paths(cls, paths: list[str]) -> "DictionaryCorrector":
        """Load and merge correction rules from JSON files.

        Supported file format:
        {
          "exact": {"old": "new"},
          "regex": [{"pattern": "...", "repl": "..."}]
        }
        """
        merged_exact: dict[str, str] = {}
        merged_regex: list[tuple[re.Pattern[str], str]] = []

        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                log.warning("Dictionary file not found: %s", path)
                continue

            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                log.warning("Failed to load dictionary file %s: %s", path, exc)
                continue

            exact = payload.get("exact", {})
            if isinstance(exact, dict):
                for key, value in exact.items():
                    if isinstance(key, str) and isinstance(value, str):
                        merged_exact[key] = value

            regex_entries = payload.get("regex", [])
            if isinstance(regex_entries, list):
                for entry in regex_entries:
                    if not isinstance(entry, dict):
                        continue
                    pattern = entry.get("pattern")
                    repl = entry.get("repl")
                    if not isinstance(pattern, str) or not isinstance(repl, str):
                        continue
                    try:
                        merged_regex.append((re.compile(pattern), repl))
                    except re.error as exc:
                        log.warning("Invalid regex pattern in %s (%s): %s", path, pattern, exc)

        return cls(exact_rules=merged_exact, regex_rules=merged_regex)

    def correct(self, text: str) -> tuple[str, int]:
        """Apply correction rules and return corrected text with replacement count."""
        if not text:
            return text, 0

        corrected = text
        replacements = 0

        for src, dst in self.exact_rules.items():
            occurrences = corrected.count(src)
            if occurrences:
                corrected = corrected.replace(src, dst)
                replacements += occurrences

        for pattern, repl in self.regex_rules:
            corrected, count = pattern.subn(repl, corrected)
            replacements += count

        return corrected, replacements
