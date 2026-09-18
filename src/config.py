"""Central configuration loader.

Loads ``config/config.yaml`` once and exposes it as a nested ``Config`` object
whose keys are reachable both as attributes (``cfg.data.test_size``) and as
dict items (``cfg["data"]["test_size"]``).  Every other module imports
``get_config()`` so that paths, seeds and hyper-parameters live in exactly one
place.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import yaml

# Repository root = two levels up from this file (src/config.py -> repo/)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


class Config(dict):
    """A dict that also allows attribute access and recurses into children."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def path(self, key: str) -> Path:
        """Resolve a path defined under the ``paths`` section against ROOT."""
        return (ROOT / self["paths"][key]).resolve()


@lru_cache(maxsize=1)
def get_config(path: Path | str = CONFIG_PATH) -> Config:
    """Load and cache the YAML configuration."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def resolve(relative: str) -> Path:
    """Turn a repo-relative string path into an absolute :class:`Path`."""
    return (ROOT / relative).resolve()


if __name__ == "__main__":  # quick sanity check
    cfg = get_config()
    print("Project:", cfg.project.name)
    print("Seed:", cfg.project.random_seed)
    print("Raw data:", cfg.path("raw_data"))
    print("Classifiers:", cfg.training.classifiers)
