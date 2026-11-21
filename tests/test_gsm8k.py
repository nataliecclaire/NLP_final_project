import json

import pytest

from halugen.data import parse_gsm8k_answer


@pytest.mark.parametrize("sol,expected", [
    ("blah\n#### 18", "18"),
    ("x\n#### 1,234", "1234"),
    ("x\n####  -5 ", "-5"),
    ("x\n#### 2.5", "2.5"),
])
def test_parse(sol, expected):
    assert parse_gsm8k_answer(sol) == expected


def test_parse_rejects_missing_marker():
    with pytest.raises(ValueError):
        parse_gsm8k_answer("the answer is 4")


def test_gsm8k_split_golds_are_numbers(cfg):
    items = json.load(open(cfg["paths"]["splits_dir"] / "gsm8k_test.json"))["items"]
    assert len(items) == cfg["splits"]["test_set"]["n_test"]
    for x in items:
        assert len(x["gold_answers"]) == 1, x["id"]
        float(x["gold_answers"][0])  # raises if not a number
