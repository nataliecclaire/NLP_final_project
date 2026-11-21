"""The ONLY tokenization path for prompts: question -> chat messages -> token IDs."""
from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

from .config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import prompts  # noqa: E402  (project-root module, kept there for easy review)


def load_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])


def prompt_token_ids(tokenizer, task: str, question: str) -> list[int]:
    """Exact model input IDs, including the leading <s> (BOS).

    apply_chat_template with tokenize=True adds BOS exactly once. Never re-tokenize the
    rendered string (that would add a second BOS). return_dict=True normalizes the return
    type across transformers 4.x (list) and 5.x (BatchEncoding).
    """
    messages = prompts.build_messages(task, question)
    enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
    ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        assert len(ids) == 1
        ids = ids[0]
    return [int(i) for i in ids]


def check_fewshot_matches_splits(cfg: dict) -> None:
    from .splits import load_fewshot
    saved = [{k: x[k] for k in ("id", "question", "answer")} for x in load_fewshot(cfg)]
    if saved != prompts.FEWSHOT_EXAMPLES:
        raise RuntimeError("prompts.FEWSHOT_EXAMPLES does not match data/splits/fewshot.json")


__all__ = ["load_tokenizer", "prompt_token_ids", "check_fewshot_matches_splits", "prompts", "Path"]
