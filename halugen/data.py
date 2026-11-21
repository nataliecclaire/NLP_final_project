"""Dataset loaders. Each returns the full pool as a list of question dicts:
{"id": str, "dataset": str, "question": str, "gold_answers": [str], "fewshot_answer": str}

`fewshot_answer` is the single short answer used if the question is chosen as a
few-shot example; it is not written to the generation outputs.
"""
from __future__ import annotations

import json
import re

from datasets import load_dataset


def _dedupe_keep_order(xs):
    seen, out = set(), []
    for x in xs:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


_GSM8K_RE = re.compile(r"####\s*(.+?)\s*$", re.S)


def parse_gsm8k_answer(solution: str) -> str:
    """Return the number after '####', with thousands separators removed."""
    m = _GSM8K_RE.search(solution)
    if not m:
        raise ValueError(f"no '####' answer in: {solution[-80:]!r}")
    ans = m.group(1).replace(",", "").replace("$", "").strip()
    float(ans)  # raises if not numeric
    return ans


def _load_hf(spec: dict):
    return load_dataset(spec["hf_id"], spec["hf_config"], split=spec["hf_split"], revision=spec["revision"])


def load_pool(name: str, spec: dict) -> tuple[list[dict], dict]:
    """Return (items, notes). `notes` records source-level cleanup for the dedup log."""
    ds = _load_hf(spec)
    if name == "triviaqa":
        ds = ds.select_columns(["question_id", "question", "answer"])  # skip the empty evidence columns
    hf_split = spec["hf_split"]
    out = []
    for i, r in enumerate(ds):
        if name == "triviaqa":
            a = r["answer"]
            item = dict(id=f"triviaqa-{r['question_id']}",
                        gold_answers=_dedupe_keep_order(a["aliases"] + a["normalized_aliases"]),
                        fewshot_answer=a["value"])
        elif name == "nq_open":
            item = dict(id=f"nq_open-{hf_split}-{i}", gold_answers=_dedupe_keep_order(r["answer"]),
                        fewshot_answer=r["answer"][0])
        elif name == "web_questions":
            item = dict(id=f"web_questions-{hf_split}-{i}", gold_answers=_dedupe_keep_order(r["answers"]),
                        fewshot_answer=r["answers"][0])
        elif name == "popqa":
            golds = _dedupe_keep_order(json.loads(r["possible_answers"]))
            item = dict(id=f"popqa-{r['id']}", gold_answers=golds, fewshot_answer=golds[0])
        elif name == "gsm8k":
            num = parse_gsm8k_answer(r["answer"])
            item = dict(id=f"gsm8k-{hf_split}-{i}", gold_answers=[num], fewshot_answer=num)
        else:
            raise ValueError(f"unknown dataset {name}")
        item["dataset"] = name
        item["question"] = r["question"].strip()
        out.append(item)
    notes = {"source_rows": len(out)}
    if name == "triviaqa":
        # rc.nocontext/validation is the union of the Wikipedia and web subsets, so most
        # question_ids appear twice. Collapse by id, keeping first-occurrence order and
        # merging gold aliases. At the pinned revision only 2 ids have non-identical rows
        # (bb_956: "June" vs "November" in the question; qz_4576: different alias lists);
        # the first row's question text is kept.
        merged: dict[str, dict] = {}
        conflicts = []
        for item in out:
            if item["id"] in merged:
                prev = merged[item["id"]]
                if prev["question"] != item["question"] or prev["gold_answers"] != item["gold_answers"]:
                    conflicts.append(item["id"])
                prev["gold_answers"] = _dedupe_keep_order(prev["gold_answers"] + item["gold_answers"])
            else:
                merged[item["id"]] = item
        out = list(merged.values())
        notes["collapsed_repeated_question_id_rows"] = notes["source_rows"] - len(out)
        notes["ids_with_differing_rows_merged"] = conflicts
    ids = [x["id"] for x in out]
    assert len(ids) == len(set(ids)), f"non-unique ids in {name}"
    return out, notes
