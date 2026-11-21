# Generation output schema

One JSONL file per dataset and split: `outputs/generations/{dataset}_{split}.jsonl`, one JSON object per line.
Records are in split order (the order in `data/splits/{dataset}_{split}.json`). A file only contains
questions that have both greedy and sampled outputs.

| file | split | n |
|---|---|---|
| `triviaqa_train`, `nq_open_train`, `web_questions_train` | train | 1,200 each |
| `triviaqa_val`, `nq_open_val`, `web_questions_val` | val | 300 each |
| `popqa_test`, `gsm8k_test` | test (held out, never used for training) | 500 each |

## Fields

| field | type | description |
|---|---|---|
| `id` | str | Stable question ID, unique across all datasets. Format: `triviaqa-<question_id>`, `popqa-<id>`, or `<dataset>-<hf_split>-<row index>` for `nq_open`, `web_questions`, `gsm8k` (which have no native IDs). |
| `dataset` | str | `triviaqa` \| `nq_open` \| `web_questions` \| `popqa` \| `gsm8k` |
| `split` | str | `train` \| `val` \| `test` |
| `question` | str | Question text exactly as inserted into the prompt (source text, whitespace-stripped). |
| `gold_answers` | [str] | Acceptable answers. TriviaQA: `answer.aliases` + `answer.normalized_aliases` (deduplicated, order kept). NQ-open: `answer`. WebQuestions: `answers`. PopQA: `possible_answers`. GSM8K: a single string holding the number after `####`, commas removed (e.g. `"1234"`, parses as a float). |
| `prompt_token_ids` | [int] | **Exact** model input: output of `tokenizer.apply_chat_template(messages, tokenize=True)` for the messages built by `prompts.py`. Starts with BOS (`<s>` = 1), exactly once. |
| `greedy_response` | str | `tokenizer.decode(greedy_token_ids, skip_special_tokens=True)`, not stripped. For display and judging only. |
| `greedy_token_ids` | [int] | **Exact** generated tokens from greedy decoding (`do_sample=False`). Includes the final `</s>` (EOS = 2) if the model stopped by itself. It has no `</s>` if it hit `max_new_tokens`. |
| `greedy_token_logprobs` | [float] | One per entry of `greedy_token_ids` (same length, including `</s>`): `log_softmax` of the model's raw next-token logits, at the chosen token. |
| `samples` | [{`text`: str, `total_logprob`: float}] | Exactly 10 sampled answers (temperature 1.0, top_p 1.0, no top-k). `text` is decoded without special tokens. `total_logprob` is the sum of log-probabilities of the sampled tokens under the model's raw distribution, **including** the `</s>` if emitted. Sample token IDs are not stored. |
| `model` | str | `mistralai/Mistral-7B-Instruct-v0.2` |
| `gen_config` | object | Settings that produced this record (below). |

### `gen_config`

| key | description |
|---|---|
| `seed` | global seed (42) |
| `model_revision` | Hugging Face commit hash of the model (pinned in `config.yaml`) |
| `dtype` | `float16` |
| `task` | `factual` (few-shot short-answer prompt) or `math` (step-by-step prompt) |
| `prompt_fingerprint` | hash of all templates and few-shot examples in `prompts.py`; records with equal fingerprints used identical prompts |
| `max_new_tokens` | 48 (factual) or 384 (math) |
| `stop` | `"eos"`: generation stops only at `</s>` or `max_new_tokens` |
| `greedy` | `{do_sample: false, backend: "hf", torch, transformers, gpu}` |
| `sampling` | `{do_sample: true, n, temperature, top_p, top_k (null = disabled), seed, backend: "vllm" or "hf", library versions}`. `seed` is this question's own sampling seed, derived from (42, id). |

Fields intentionally **absent** (added by later stages): `answer_span`, `correct`, `semantic_entropy`, `se_label`.

## Notes for downstream stages

- **Hidden-state extraction:** run the model on `prompt_token_ids + greedy_token_ids` directly. Do **not**
  re-tokenize `question` or `greedy_response`. Mistral's tokenizer does not reliably round-trip
  decoded text back to the same IDs (leading spaces get merged, and the model sometimes emits token splits
  the tokenizer would not produce). Use the same model revision and fp16. A forward pass over those IDs reproduces
  `greedy_token_logprobs` within fp16 rounding (checked by `tests/test_gpu.py`).
- Token `i` of the response sits at absolute position `len(prompt_token_ids) + i`. Its logprob comes from
  the logits at position `len(prompt_token_ids) + i - 1`.
- Splits, few-shot examples and the dedup log are in `data/splits/`. Every split file has
  `items: [{id, question, gold_answers}]` plus source metadata (HF ID, config, split, pinned revision).
