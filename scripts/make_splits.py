"""Build data/splits/*.json (deterministic). Run once; commit the result so the whole team shares it.

    python scripts/make_splits.py [--config config.yaml] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halugen.config import load_config  # noqa: E402
from halugen.splits import make_splits, write_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = cfg["paths"]["splits_dir"]
    if any(out_dir.glob("*.json")) and not args.force:
        sys.exit(f"{out_dir} already has split files; refusing to overwrite (use --force).")

    result = make_splits(cfg)
    write_splits(cfg, result)

    log = result["dedup_log"]
    print("Dedup (normalized: lowercase, no punctuation, collapsed whitespace):")
    for name in cfg["splits"]["dedup_priority"]:
        d = log[name]
        print(f"  {name:14s} pool={d['pool_size']:6d} kept={d['kept']:6d} "
              f"removed_within={d['removed_within_dataset']:4d} removed_cross={d['removed_cross_dataset']:4d}")
    print(f"  total removed: {log['_total_removed']}")
    print("\nSplits:")
    for (name, split), items in result["splits"].items():
        print(f"  {name}_{split}: {len(items)}")
    print("\nFew-shot examples (must match prompts.FEWSHOT_EXAMPLES):")
    for x in result["fewshot"]:
        print("  " + json.dumps({"id": x["id"], "question": x["question"], "answer": x["fewshot_answer"]},
                                ensure_ascii=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
