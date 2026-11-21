"""Model-dependent checks. Run on the GPU machine:  pytest tests/test_gpu.py -s

What the hidden-state stage relies on is that feeding the saved IDs
(prompt_token_ids + greedy_token_ids) back into the model reproduces the run.
Re-tokenizing *decoded text* is not guaranteed to give the same IDs with
Mistral's SentencePiece tokenizer, so that is reported, not asserted.
"""
import json

import pytest

from conftest import requires_gpu
from halugen.chat import prompt_token_ids
from halugen.io import read_jsonl

# fp16 incremental decoding (KV cache) vs. one full forward pass differ by rounding.
MAX_ABS_LOGPROB_DIFF = 0.15
MEAN_ABS_LOGPROB_DIFF = 0.02

pytestmark = requires_gpu


def _examples(cfg, n_factual=3, n_math=2):
    def first(name, k):
        return json.load(open(cfg["paths"]["splits_dir"] / f"{name}_{'test' if name == 'gsm8k' else 'train'}.json"))["items"][:k]
    return [("factual", x) for x in first("triviaqa", n_factual)] + [("math", x) for x in first("gsm8k", n_math)]


def _mnt(cfg, task):
    return cfg["generation"]["max_new_tokens"][task]


def _check_teacher_forcing(model, pids, rids, saved_lps, label):
    from halugen import hf_backend as hf
    lps, argmax = hf.teacher_forced_logprobs(model, pids, rids)
    diffs = [abs(a - b) for a, b in zip(lps, saved_lps)]
    agree = sum(a == b for a, b in zip(argmax, rids)) / len(rids)
    print(f"{label}: n_tok={len(rids)} max|d|={max(diffs):.4f} mean|d|={sum(diffs)/len(diffs):.5f} argmax_agree={agree:.3f}")
    assert max(diffs) < MAX_ABS_LOGPROB_DIFF and sum(diffs) / len(diffs) < MEAN_ABS_LOGPROB_DIFF, label


def test_greedy_deterministic(cfg, tok, model):
    from halugen import hf_backend as hf
    for task, x in _examples(cfg):
        pids = prompt_token_ids(tok, task, x["question"])
        a = hf.greedy(model, tok, pids, _mnt(cfg, task))
        b = hf.greedy(model, tok, pids, _mnt(cfg, task))
        assert a == b, x["id"]


def test_ids_reproduce_logprobs_fresh(cfg, tok, model):
    from halugen import hf_backend as hf
    for task, x in _examples(cfg):
        pids = prompt_token_ids(tok, task, x["question"])
        g = hf.greedy(model, tok, pids, _mnt(cfg, task))
        _check_teacher_forcing(model, pids, g["greedy_token_ids"], g["greedy_token_logprobs"], x["id"])


def test_ids_reproduce_logprobs_saved_outputs(cfg, model):
    recs = read_jsonl(cfg["paths"]["generations_dir"] / "triviaqa_train.jsonl")[:5]
    if not recs:
        pytest.skip("no saved outputs yet")
    for r in recs:
        _check_teacher_forcing(model, r["prompt_token_ids"], r["greedy_token_ids"], r["greedy_token_logprobs"], r["id"])


def test_text_roundtrip_rate_report(cfg, tok, model):
    """Informational: how often re-encoding the decoded text reproduces the exact IDs."""
    from halugen import hf_backend as hf
    hits, n = 0, 0
    for task, x in _examples(cfg):
        pids = prompt_token_ids(tok, task, x["question"])
        g = hf.greedy(model, tok, pids, _mnt(cfg, task))
        full = tok.decode(pids + g["greedy_token_ids"])
        hits += tok(full, add_special_tokens=False).input_ids == pids + g["greedy_token_ids"]
        n += 1
    print(f"text round-trip exact match: {hits}/{n} (not required; use the saved IDs)")
