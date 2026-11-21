"""Prompt construction and prompt tokenization (CPU; needs access to the Mistral tokenizer)."""
import json

import prompts
from halugen.chat import check_fewshot_matches_splits, prompt_token_ids
from halugen.config import all_dataset_splits


def _sample_items(cfg, k=3):
    for d, s in all_dataset_splits(cfg):
        items = json.load(open(cfg["paths"]["splits_dir"] / f"{d}_{s}.json"))["items"]
        for x in items[:k]:
            yield cfg["datasets"][d]["task"], x


def test_fewshot_matches_split_file(cfg):
    check_fewshot_matches_splits(cfg)


def test_prompt_ids_match_fresh_chat_template(cfg, tok):
    """Saved prompt IDs == a direct apply_chat_template call, with exactly one leading BOS."""
    for task, x in _sample_items(cfg):
        ids = prompt_token_ids(tok, task, x["question"])
        enc = tok.apply_chat_template(prompts.build_messages(task, x["question"]), tokenize=True, return_dict=True)
        assert ids == list(enc["input_ids"])
        assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
        assert tok.decode(ids).rstrip().endswith("[/INST]")


def test_factual_prompt_identical_across_datasets(cfg, tok):
    """Everything before the target question is the same token sequence for every factual dataset."""
    prefix = None
    for task, x in _sample_items(cfg, k=2):
        if task != "factual":
            continue
        msgs = prompts.build_messages(task, x["question"])
        assert msgs[:-1] == prompts.build_messages(task, "PLACEHOLDER")[:-1]
        rendered = tok.apply_chat_template(msgs, tokenize=False)
        p = rendered[: rendered.rindex("[INST]")]
        assert prefix is None or p == prefix
        prefix = p


def test_math_prompt_asks_for_final_answer_format():
    m = prompts.build_messages("math", "What is 2+2?")
    assert len(m) == 1 and 'The answer is <number>.' in m[0]["content"] and "step by step" in m[0]["content"]
