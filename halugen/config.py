"""Load config.yaml. Paths in the config are resolved relative to the project root."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg["paths"].items():
        p = Path(val)
        cfg["paths"][key] = p if p.is_absolute() else PROJECT_ROOT / p
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    names = set(cfg["datasets"])
    assert set(cfg["splits"]["dedup_priority"]) == names, "dedup_priority must list every dataset exactly once"
    for name, d in cfg["datasets"].items():
        assert d["task"] in ("factual", "math"), name
        assert d["role"] in ("train_source", "test_set"), name
    assert cfg["generation"]["sampling"]["backend"] in ("auto", "vllm", "hf")


def splits_for(cfg: dict, dataset: str) -> list[str]:
    return ["train", "val"] if cfg["datasets"][dataset]["role"] == "train_source" else ["test"]


def all_dataset_splits(cfg: dict) -> list[tuple[str, str]]:
    return [(d, s) for d in cfg["datasets"] for s in splits_for(cfg, d)]
