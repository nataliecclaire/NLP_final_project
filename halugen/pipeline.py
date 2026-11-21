"""Generation phases: greedy -> sample -> merge. Each phase is resumable on its own.

Partial files (outputs/generations/_partial/{dataset}_{split}.{greedy|samples}.jsonl) are
append-only; on restart, completed IDs are skipped. Every partial record carries a
`run_key` (hash of the settings that affect its content); resuming with different
settings is refused rather than silently mixing outputs.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .chat import prompt_token_ids, prompts
from .config import all_dataset_splits
from .io import JsonlAppender, read_jsonl, write_jsonl_atomic
from .splits import load_split


@dataclass
class Target:
    dataset: str
    split: str
    task: str
    items: list[dict]                       # full split, in saved order
    limit: int | None = None
    _pids: dict = field(default_factory=dict)

    @property
    def todo_items(self) -> list[dict]:
        return self.items[: self.limit] if self.limit else self.items

    def prompt_ids(self, tok, item) -> list[int]:
        if item["id"] not in self._pids:
            self._pids[item["id"]] = prompt_token_ids(tok, self.task, item["question"])
        return self._pids[item["id"]]


def make_targets(cfg, dataset: str, split: str, limit: int | None) -> list[Target]:
    pairs = [(d, s) for d, s in all_dataset_splits(cfg)
             if dataset in ("all", d) and split in ("all", s)]
    if not pairs:
        raise SystemExit(f"no such dataset/split: {dataset}/{split}")
    return [Target(d, s, cfg["datasets"][d]["task"], load_split(cfg, d, s), limit) for d, s in pairs]


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def sample_seed(cfg, qid: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{qid}".encode()).digest()[:4], "big")


def max_new_tokens(cfg, task: str) -> int:
    return cfg["generation"]["max_new_tokens"][task]


def greedy_run_key(cfg, task) -> str:
    m = cfg["model"]
    return _hash({"model": m["name"], "rev": m["revision"], "dtype": m["dtype"], "prompts": prompts.prompt_fingerprint(),
                  "max_new_tokens": max_new_tokens(cfg, task), "mode": "greedy"})


def sample_run_key(cfg, task) -> str:
    m, s = cfg["model"], cfg["generation"]["sampling"]
    return _hash({"model": m["name"], "rev": m["revision"], "dtype": m["dtype"], "prompts": prompts.prompt_fingerprint(),
                  "max_new_tokens": max_new_tokens(cfg, task), "seed": cfg["seed"],
                  "n": s["n"], "temperature": s["temperature"], "top_p": s["top_p"], "top_k": s["top_k"]})


def partial_path(cfg, t: Target, phase: str) -> Path:
    return cfg["paths"]["partial_dir"] / f"{t.dataset}_{t.split}.{phase}.jsonl"


def final_path(cfg, dataset: str, split: str) -> Path:
    return cfg["paths"]["generations_dir"] / f"{dataset}_{split}.jsonl"


def _done_ids(path: Path, run_key: str) -> set[str]:
    recs = read_jsonl(path)
    bad = {r["run_key"] for r in recs if r["run_key"] != run_key}
    if bad:
        raise RuntimeError(f"{path} was produced with different settings (run_key {bad} != {run_key}). "
                           "Settings changed since the last run; move the file aside deliberately to regenerate.")
    return {r["id"] for r in recs}


# --------------------------------------------------------------------------- greedy
def run_greedy(cfg, tok, model, targets: list[Target]) -> None:
    from . import hf_backend as hf
    env = hf.versions()
    for t in targets:
        path, key = partial_path(cfg, t, "greedy"), greedy_run_key(cfg, t.task)
        done = _done_ids(path, key)
        todo = [x for x in t.todo_items if x["id"] not in done]
        print(f"[greedy] {t.dataset}/{t.split}: {len(todo)} to do ({len(done)} already done)")
        with JsonlAppender(path) as w:
            for item in tqdm(todo, desc=f"greedy {t.dataset}/{t.split}"):
                pids = t.prompt_ids(tok, item)
                g = hf.greedy(model, tok, pids, max_new_tokens(cfg, t.task))
                w.write({"id": item["id"], "run_key": key, "prompt_token_ids": pids, **g,
                         "backend": "hf", "env": env})


# --------------------------------------------------------------------------- sampling
def resolve_sampling_backend(cfg) -> str:
    from . import vllm_backend
    b = cfg["generation"]["sampling"]["backend"]
    if b == "auto":
        b = "vllm" if vllm_backend.available() else "hf"
        if b == "hf":
            print("[sample] vLLM not installed -> using the HF transformers fallback")
    return b


def run_sampling(cfg, tok, targets: list[Target], backend: str, hf_model=None) -> None:
    s = cfg["generation"]["sampling"]
    if backend == "vllm":
        from . import vllm_backend
        sampler, env = vllm_backend.VllmSampler(cfg), vllm_backend.versions()
    else:
        from . import hf_backend as hf
        assert hf_model is not None
        env = hf.versions()
    for t in targets:
        path, key = partial_path(cfg, t, "samples"), sample_run_key(cfg, t.task)
        done = _done_ids(path, key)
        todo = [x for x in t.todo_items if x["id"] not in done]
        mnt = max_new_tokens(cfg, t.task)
        print(f"[sample/{backend}] {t.dataset}/{t.split}: {len(todo)} to do ({len(done)} already done)")
        with JsonlAppender(path) as w:
            if backend == "vllm":
                bs = cfg["generation"]["vllm"]["batch_questions"]
                for i in tqdm(range(0, len(todo), bs), desc=f"sample {t.dataset}/{t.split}"):
                    batch = todo[i:i + bs]
                    seeds = [sample_seed(cfg, x["id"]) for x in batch]
                    res = sampler.sample([t.prompt_ids(tok, x) for x in batch], mnt, seeds)
                    for x, sd, samples in zip(batch, seeds, res):
                        w.write({"id": x["id"], "run_key": key, "samples": samples, "sample_seed": sd,
                                 "backend": "vllm", "env": env})
            else:
                for x in tqdm(todo, desc=f"sample {t.dataset}/{t.split}"):
                    sd = sample_seed(cfg, x["id"])
                    samples = hf.sample(hf_model, tok, t.prompt_ids(tok, x), mnt, s["n"], s["temperature"],
                                        s["top_p"], s["top_k"], sd, s["hf_chunk_size"])
                    w.write({"id": x["id"], "run_key": key, "samples": samples, "sample_seed": sd,
                             "backend": "hf", "env": env})


def release_cuda_memory() -> None:
    """Call after dropping every reference to the HF model (the caller must `model = None` first)."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- merge
def merge(cfg, tok, t: Target) -> int:
    """Join greedy + samples into the final contract file (only IDs complete in both phases)."""
    greedy = {r["id"]: r for r in read_jsonl(partial_path(cfg, t, "greedy"))}
    samples = {r["id"]: r for r in read_jsonl(partial_path(cfg, t, "samples"))}
    gkey, skey = greedy_run_key(cfg, t.task), sample_run_key(cfg, t.task)
    s = cfg["generation"]["sampling"]
    out = []
    for item in t.items:
        g, sm = greedy.get(item["id"]), samples.get(item["id"])
        if g is None or sm is None:
            continue
        assert g["run_key"] == gkey and sm["run_key"] == skey, f"stale partial record for {item['id']}"
        assert g["prompt_token_ids"] == t.prompt_ids(tok, item), f"prompt tokens changed for {item['id']}"
        assert len(g["greedy_token_ids"]) == len(g["greedy_token_logprobs"])
        assert len(sm["samples"]) == s["n"]
        out.append({
            "id": item["id"], "dataset": t.dataset, "split": t.split,
            "question": item["question"], "gold_answers": item["gold_answers"],
            "prompt_token_ids": g["prompt_token_ids"],
            "greedy_response": g["greedy_response"],
            "greedy_token_ids": g["greedy_token_ids"],
            "greedy_token_logprobs": g["greedy_token_logprobs"],
            "samples": sm["samples"],
            "model": cfg["model"]["name"],
            "gen_config": {
                "seed": cfg["seed"], "model_revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
                "task": t.task, "prompt_fingerprint": prompts.prompt_fingerprint(),
                "max_new_tokens": max_new_tokens(cfg, t.task), "stop": "eos",
                "greedy": {"do_sample": False, "backend": g["backend"], **g["env"]},
                "sampling": {"do_sample": True, "n": s["n"], "temperature": s["temperature"], "top_p": s["top_p"],
                             "top_k": s["top_k"], "seed": sm["sample_seed"], "backend": sm["backend"], **sm["env"]},
            },
        })
    write_jsonl_atomic(final_path(cfg, t.dataset, t.split), out)
    return len(out)


# --------------------------------------------------------------------------- summary
def summarize(cfg, targets: list[Target], n_examples: int = 3) -> None:
    rng = random.Random(cfg["seed"])
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    by_ds: dict[str, list[dict]] = {}
    print(f"{'file':28s} {'done':>6s} {'split':>6s} {'avg tok':>8s} {'avg words':>10s} {'hit max':>8s}")
    for t in targets:
        recs = read_jsonl(final_path(cfg, t.dataset, t.split))
        by_ds.setdefault(t.dataset, []).extend(recs)
        if not recs:
            print(f"{t.dataset + '_' + t.split:28s} {0:6d} {len(t.items):6d}")
            continue
        n_tok = [len(r["greedy_token_ids"]) for r in recs]
        n_words = [len(r["greedy_response"].split()) for r in recs]
        mnt = max_new_tokens(cfg, t.task)
        hit = sum(n >= mnt and r["greedy_token_ids"][-1] != 2 for n, r in zip(n_tok, recs))
        print(f"{t.dataset + '_' + t.split:28s} {len(recs):6d} {len(t.items):6d} {sum(n_tok)/len(recs):8.1f} "
              f"{sum(n_words)/len(recs):10.1f} {hit:8d}")
    for ds, recs in by_ds.items():
        if not recs:
            continue
        print(f"\n--- {ds}: {min(n_examples, len(recs))} random examples ---")
        for r in rng.sample(recs, min(n_examples, len(recs))):
            resp = r["greedy_response"].strip().replace("\n", " | ")
            print(f"[{r['id']}] Q: {r['question']}")
            print(f"    gold: {r['gold_answers'][:4]}")
            print(f"    greedy: {resp[:300]}")
            print(f"    sample[0]: {r['samples'][0]['text'].strip().replace(chr(10), ' | ')[:200]}")
