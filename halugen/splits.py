"""Cross-dataset dedup, few-shot selection, and train/val/test sampling.

Procedure (all deterministic given config.seed and pinned dataset revisions):
1. Load every pool. Normalize questions: lowercase, drop punctuation, collapse whitespace.
2. Walk datasets in `dedup_priority` order; a question whose normalized form was
   already seen (in the same or an earlier dataset) is removed.
3. Shuffle each deduplicated pool with random.Random(f"{seed}:{dataset}").
4. Training sources: the first question in shuffled order that is short enough
   becomes that source's few-shot example (removed from all splits). The next
   n_train go to train and the following n_val to val.
   Test sets: the first n_test go to test.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from .config import splits_for
from .data import load_pool

_WS = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).lower()
    q = "".join(ch for ch in q if not unicodedata.category(ch).startswith("P"))
    return _WS.sub(" ", q).strip()


def dedup_pools(pools: dict[str, list[dict]], priority: list[str]):
    seen: dict[str, str] = {}  # normalized question -> id that kept it
    kept, log = {}, {}
    for name in priority:
        keep, within, cross = [], [], []
        for item in pools[name]:
            key = normalize_question(item["question"])
            if key in seen:
                other = seen[key]
                rec = {"removed": item["id"], "question": item["question"], "kept": other}
                (within if other.startswith(name + "-") else cross).append(rec)
            else:
                seen[key] = item["id"]
                keep.append(item)
        kept[name] = keep
        log[name] = {"pool_size": len(pools[name]), "kept": len(keep),
                     "removed_within_dataset": len(within), "removed_cross_dataset": len(cross),
                     "cross_dataset_examples": cross[:25], "within_dataset_examples": within[:10]}
    log["_total_removed"] = sum(v["removed_within_dataset"] + v["removed_cross_dataset"]
                                for k, v in log.items() if not k.startswith("_"))
    return kept, log


def _shuffled(items: list[dict], seed: int, name: str) -> list[dict]:
    items = list(items)
    random.Random(f"{seed}:{name}").shuffle(items)
    return items


def _fewshot_ok(item: dict, fs_cfg: dict) -> bool:
    return (item["id"] not in set(fs_cfg.get("exclude_ids") or [])
            and len(item["question"].split()) <= fs_cfg["max_question_words"]
            and 1 <= len(item["fewshot_answer"].split()) <= fs_cfg["max_answer_words"])


def make_splits(cfg: dict) -> dict:
    seed, sp, fs_cfg = cfg["seed"], cfg["splits"], cfg["fewshot"]
    pools, notes = {}, {}
    for name, spec in cfg["datasets"].items():
        pools[name], notes[name] = load_pool(name, spec)
    kept, dedup_log = dedup_pools(pools, sp["dedup_priority"])
    for name, n in notes.items():
        dedup_log[name]["source_notes"] = n

    splits: dict[tuple[str, str], list[dict]] = {}
    fewshot: list[dict] = []
    for name, spec in cfg["datasets"].items():
        order = _shuffled(kept[name], seed, name)
        if spec["role"] == "train_source":
            chosen = [x for x in order if _fewshot_ok(x, fs_cfg)][: fs_cfg["per_source"]]
            assert len(chosen) == fs_cfg["per_source"], f"no few-shot candidate in {name}"
            fewshot.extend(chosen)
            chosen_ids = {x["id"] for x in chosen}
            rest = [x for x in order if x["id"] not in chosen_ids]
            n_tr, n_va = sp["train_source"]["n_train"], sp["train_source"]["n_val"]
            assert len(rest) >= n_tr + n_va, f"{name}: pool too small ({len(rest)})"
            splits[(name, "train")] = rest[:n_tr]
            splits[(name, "val")] = rest[n_tr:n_tr + n_va]
        else:
            n = sp["test_set"]["n_test"]
            assert len(order) >= n, f"{name}: pool too small ({len(order)})"
            splits[(name, "test")] = order[:n]
    return {"splits": splits, "fewshot": fewshot, "dedup_log": dedup_log}


def _public(item: dict) -> dict:
    return {"id": item["id"], "question": item["question"], "gold_answers": item["gold_answers"]}


def write_splits(cfg: dict, result: dict) -> None:
    out_dir: Path = cfg["paths"]["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for (name, split), items in result["splits"].items():
        spec = cfg["datasets"][name]
        payload = {
            "dataset": name, "split": split, "n": len(items), "seed": cfg["seed"],
            "source": {k: spec[k] for k in ("hf_id", "hf_config", "hf_split", "revision")},
            "items": [_public(x) for x in items],
        }
        _dump(out_dir / f"{name}_{split}.json", payload)
    _dump(out_dir / "fewshot.json",
          [{"id": x["id"], "dataset": x["dataset"], "question": x["question"], "answer": x["fewshot_answer"]}
           for x in result["fewshot"]])
    _dump(out_dir / "dedup_log.json", result["dedup_log"])


def _dump(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def load_split(cfg: dict, dataset: str, split: str) -> list[dict]:
    if split not in splits_for(cfg, dataset):
        raise ValueError(f"{dataset} has no '{split}' split (has {splits_for(cfg, dataset)})")
    path = cfg["paths"]["splits_dir"] / f"{dataset}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/make_splits.py first")
    with open(path) as f:
        return json.load(f)["items"]


def load_fewshot(cfg: dict) -> list[dict]:
    with open(cfg["paths"]["splits_dir"] / "fewshot.json") as f:
        return json.load(f)
