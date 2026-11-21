"""Generate greedy + sampled answers. Resumable: re-run the same command after a preemption.

    # first check (stop after this and inspect the output):
    python scripts/generate.py --dataset triviaqa --split train --limit 50
    # full run:
    python scripts/generate.py --dataset all --split all

--phase all runs greedy for every target first (one HF model load), then sampling,
then merges into outputs/generations/{dataset}_{split}.jsonl and prints a summary.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halugen import pipeline as P  # noqa: E402
from halugen.chat import check_fewshot_matches_splits, load_tokenizer  # noqa: E402
from halugen.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dataset", required=True, help="dataset name or 'all'")
    ap.add_argument("--split", default="all", help="train | val | test | all")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions of each split")
    ap.add_argument("--phase", default="all", choices=["greedy", "sample", "merge", "all"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    check_fewshot_matches_splits(cfg)
    tok = load_tokenizer(cfg)
    targets = P.make_targets(cfg, args.dataset, args.split, args.limit)

    model = None
    if args.phase in ("greedy", "sample", "all"):
        from halugen import hf_backend as hf
        hf.setup_determinism(cfg["seed"])
        backend = P.resolve_sampling_backend(cfg) if args.phase != "greedy" else None
        if args.phase in ("greedy", "all") or backend == "hf":
            model = hf.load_model(cfg)
        if args.phase in ("greedy", "all"):
            P.run_greedy(cfg, tok, model, targets)
        if args.phase in ("sample", "all"):
            if backend == "vllm" and model is not None:
                model = None  # a 16 GB T4 cannot hold two copies of the model
                P.release_cuda_memory()
            P.run_sampling(cfg, tok, targets, backend, hf_model=model)

    if args.phase in ("merge", "all"):
        for t in targets:
            n = P.merge(cfg, tok, t)
            print(f"[merge] {P.final_path(cfg, t.dataset, t.split)}: {n} records")
        P.summarize(cfg, targets)


if __name__ == "__main__":
    main()
