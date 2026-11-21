"""Contract checks on whatever final output files exist (skips if none yet)."""
import pytest

from halugen.chat import prompt_token_ids
from halugen.config import all_dataset_splits
from halugen.io import read_jsonl

FIELDS = ["id", "dataset", "split", "question", "gold_answers", "prompt_token_ids", "greedy_response",
          "greedy_token_ids", "greedy_token_logprobs", "samples", "model", "gen_config"]
FORBIDDEN = {"answer_span", "correct", "semantic_entropy", "se_label"}


def _records(cfg, per_file=None):
    out = []
    for d, s in all_dataset_splits(cfg):
        recs = read_jsonl(cfg["paths"]["generations_dir"] / f"{d}_{s}.jsonl")
        out += [(d, s, r) for r in (recs[:per_file] if per_file else recs)]
    if not out:
        pytest.skip("no output files yet")
    return out


def test_schema(cfg):
    n = cfg["generation"]["sampling"]["n"]
    for d, s, r in _records(cfg):
        assert list(r) == FIELDS, r["id"]
        assert not FORBIDDEN & set(r)
        assert (r["dataset"], r["split"]) == (d, s)
        assert r["model"] == cfg["model"]["name"]
        assert len(r["greedy_token_ids"]) == len(r["greedy_token_logprobs"]) > 0
        assert all(lp <= 0 for lp in r["greedy_token_logprobs"])
        assert len(r["samples"]) == n
        assert all(set(x) == {"text", "total_logprob"} and x["total_logprob"] <= 0 for x in r["samples"])


def test_saved_prompt_ids_reproduce(cfg, tok):
    for d, s, r in _records(cfg, per_file=5):
        assert r["prompt_token_ids"] == prompt_token_ids(tok, cfg["datasets"][d]["task"], r["question"]), r["id"]


def test_greedy_response_is_decoded_ids(cfg, tok):
    for _, _, r in _records(cfg, per_file=5):
        assert r["greedy_response"] == tok.decode(r["greedy_token_ids"], skip_special_tokens=True)
