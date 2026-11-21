import json

import pytest

from halugen.config import all_dataset_splits
from halugen.splits import load_fewshot, make_splits, normalize_question


def _all_items(cfg):
    return {(d, s): json.load(open(cfg["paths"]["splits_dir"] / f"{d}_{s}.json"))["items"]
            for d, s in all_dataset_splits(cfg)}


def test_split_sizes(cfg):
    for (d, s), items in _all_items(cfg).items():
        role = cfg["datasets"][d]["role"]
        expected = (cfg["splits"]["test_set"]["n_test"] if role == "test_set"
                    else cfg["splits"]["train_source"]["n_train" if s == "train" else "n_val"])
        assert len(items) == expected, (d, s)


def test_split_files_match_config(cfg):
    for d, s in all_dataset_splits(cfg):
        meta = json.load(open(cfg["paths"]["splits_dir"] / f"{d}_{s}.json"))
        spec = cfg["datasets"][d]
        assert meta["seed"] == cfg["seed"]
        assert meta["source"] == {k: spec[k] for k in ("hf_id", "hf_config", "hf_split", "revision")}


def test_no_id_overlap(cfg):
    seen = {}
    for key, items in _all_items(cfg).items():
        for x in items:
            assert x["id"] not in seen, f"{x['id']} in both {seen.get(x['id'])} and {key}"
            seen[x["id"]] = key
    for fs in load_fewshot(cfg):
        assert fs["id"] not in seen, f"few-shot example {fs['id']} also appears in {seen[fs['id']]}"


def test_no_normalized_question_overlap(cfg):
    seen = {}
    for key, items in _all_items(cfg).items():
        for x in items:
            q = normalize_question(x["question"])
            assert q not in seen, f"duplicate question {x['question']!r} in {seen.get(q)} and {key}"
            seen[q] = key
    for fs in load_fewshot(cfg):
        assert normalize_question(fs["question"]) not in seen, fs


def test_gold_answers_nonempty(cfg):
    for key, items in _all_items(cfg).items():
        for x in items:
            assert x["gold_answers"] and all(isinstance(a, str) and a for a in x["gold_answers"]), (key, x["id"])


def test_normalize_question():
    assert normalize_question("What is the capital of Venezuela") == normalize_question("what is the capital of venezuela?")
    assert normalize_question("Who's  there?!") == "whos there"


@pytest.mark.slow
def test_splits_reproducible(cfg):
    """Rebuilding from the pinned dataset revisions gives exactly the committed split files."""
    result = make_splits(cfg)
    saved = _all_items(cfg)
    for key, items in result["splits"].items():
        assert [x["id"] for x in items] == [x["id"] for x in saved[key]], key
    assert [x["id"] for x in result["fewshot"]] == [x["id"] for x in load_fewshot(cfg)]
