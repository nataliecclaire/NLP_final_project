# Generation stage: Mistral-7B-Instruct-v0.2 answers for hallucination probing

Loads the datasets, makes fixed splits, and generates greedy and sampled answers.
The output format is documented in [SCHEMA.md](SCHEMA.md). All settings are in [config.yaml](config.yaml),
and all prompts are in [prompts.py](prompts.py).

```
config.yaml            all settings (seed, pinned model/dataset revisions, sizes, decoding)
prompts.py             prompt templates + the 3 few-shot examples
halugen/               library code (data, splits, chat/tokenization, HF + vLLM backends, pipeline)
scripts/make_splits.py builds data/splits/ (already done and committed - do not re-run)
scripts/generate.py    greedy -> sample -> merge, resumable
data/splits/           {dataset}_{split}.json, fewshot.json, dedup_log.json
outputs/generations/   final {dataset}_{split}.jsonl (+ _partial/ per-phase files)
tests/                 pytest (GPU tests skip automatically without CUDA)
```

## Running on Google Colab (T4)

1. (Optional) Create a Read token at huggingface.co/settings/tokens. The model is not gated; logging in just makes the ~15 GB download less likely to be rate-limited.
2. Put this folder in Google Drive, so outputs survive a disconnect. Then in a T4 notebook:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/coms4705-hallucination-gen
!pip install -q -r requirements.txt
from huggingface_hub import login; login()        # optional: paste a Read token
```

3. First check (50 TriviaQA train questions), then inspect `outputs/generations/triviaqa_train.jsonl`:

```python
!python scripts/generate.py --dataset triviaqa --split train --limit 50
!python -m pytest -s
```

4. Full run (re-run the same command after any disconnect; finished IDs are skipped):

```python
!python scripts/generate.py --dataset all --split all
```

Useful flags: `--phase greedy|sample|merge|all`, `--dataset <name>|all`, `--split train|val|test|all`, `--limit N`.

## Notes

- **fp16**: the T4 has no bfloat16. Keep `float16` on any GPU so all outputs are numerically consistent.
- **Sampling backend**: `auto` uses vLLM if it is installed and otherwise falls back to HF transformers.
  Colab does not ship vLLM. On a T4, the 7B fp16 weights leave very little memory for vLLM's KV cache,
  so the HF fallback is the dependable option there.
- **Resumability**: each phase appends one fsynced line per question. A line cut off by a preemption
  is dropped on restart. Each question's sampling seed comes from (seed, id), so resumed samples
  match those from an uninterrupted run. If settings change between runs, resuming is refused.
- **Dedup**: questions are normalized (lowercase, no punctuation, collapsed whitespace) and deduplicated
  within and across datasets before sampling. Held-out sets keep shared questions. See
  `data/splits/dedup_log.json`.
