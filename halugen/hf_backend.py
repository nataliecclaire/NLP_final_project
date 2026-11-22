"""Hugging Face transformers backend: model loading, greedy decoding, and the sampling fallback.

Log-probabilities are always log_softmax of the model's RAW next-token logits
(`output_logits=True`), i.e. under the untempered, untruncated model distribution.
That is exactly what a teacher-forced forward pass over prompt+response reproduces.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS; set before CUDA init

import torch  # noqa: E402
import transformers  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoModelForCausalLM, GenerationConfig  # noqa: E402


def setup_determinism(seed: int) -> None:
    transformers.set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(cfg: dict, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg["model"]["dtype"])
    kw = dict(revision=cfg["model"]["revision"], attn_implementation=cfg["model"]["attn_implementation"])
    # transformers renamed torch_dtype -> dtype in 4.56; older versions silently ignore `dtype`
    # (which would load fp32 and OOM a T4), so choose by version and verify below.
    if Version(transformers.__version__) >= Version("4.56"):
        kw["dtype"] = dtype
    else:
        kw["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], device_map=device, **kw)
    model.eval()
    assert model.dtype == dtype, f"model loaded as {model.dtype}, expected {dtype}"
    return model


def _gen_config(tokenizer, max_new_tokens: int, **kw) -> GenerationConfig:
    # return_dict_in_generate/output_logits live in the config: passing them alongside a
    # generation_config is deprecated in recent transformers.
    return GenerationConfig(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                            return_dict_in_generate=True, output_logits=True, **kw)


@torch.inference_mode()
def greedy(model, tokenizer, prompt_ids: list[int], max_new_tokens: int) -> dict:
    """Batch size 1, no padding. Response IDs include the final </s> if the model emitted it."""
    inp = torch.tensor([prompt_ids], device=model.device)
    out = model.generate(inp, attention_mask=torch.ones_like(inp),
                         generation_config=_gen_config(tokenizer, max_new_tokens, do_sample=False))
    gen = out.sequences[0, len(prompt_ids):]
    logits = torch.stack(out.logits, 0)[:, 0, :].float()  # (T, V)
    assert logits.shape[0] == gen.shape[0]
    lp = torch.log_softmax(logits, -1).gather(1, gen[:, None])[:, 0]
    ids = gen.tolist()
    return {
        "greedy_response": tokenizer.decode(ids, skip_special_tokens=True),
        "greedy_token_ids": ids,
        "greedy_token_logprobs": [float(x) for x in lp.tolist()],
    }


@torch.inference_mode()
def teacher_forced_logprobs(model, prompt_ids: list[int], response_ids: list[int]) -> tuple[list[float], list[int]]:
    """Single forward pass over prompt+response. Returns (logprob of each response token, argmax at each step)."""
    ids = torch.tensor([prompt_ids + response_ids], device=model.device)
    logits = model(ids).logits[0, len(prompt_ids) - 1:-1].float()
    lp = torch.log_softmax(logits, -1)
    tgt = torch.tensor(response_ids, device=model.device)
    return lp.gather(1, tgt[:, None])[:, 0].tolist(), logits.argmax(-1).tolist()


@torch.inference_mode()
def sample(model, tokenizer, prompt_ids: list[int], max_new_tokens: int, n: int, temperature: float,
           top_p: float, top_k: int | None, seed: int, chunk_size: int) -> list[dict]:
    """n samples; RNG seeded per question so a resumed run reproduces an uninterrupted one."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    eos = tokenizer.eos_token_id
    inp = torch.tensor([prompt_ids], device=model.device)
    results = []
    while len(results) < n:
        k = min(chunk_size, n - len(results))
        gc = _gen_config(tokenizer, max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p,
                         top_k=0 if top_k is None else top_k,  # 0 disables HF's default top_k=50
                         num_return_sequences=k)
        out = model.generate(inp, attention_mask=torch.ones_like(inp), generation_config=gc)
        seqs = out.sequences[:, len(prompt_ids):]
        total = torch.zeros(k, device=seqs.device, dtype=torch.float64)
        alive = torch.ones(k, dtype=torch.bool, device=seqs.device)
        for t, step_logits in enumerate(out.logits):
            tok_t = seqs[:, t]
            lp_t = torch.log_softmax(step_logits.float(), -1).gather(1, tok_t[:, None])[:, 0]
            total += torch.where(alive, lp_t.double(), torch.zeros_like(total))
            alive &= tok_t != eos  # the </s> itself is counted; padding after it is not
        for row, tot in zip(seqs.tolist(), total.tolist()):
            if eos in row:
                row = row[:row.index(eos)]
            results.append({"text": tokenizer.decode(row, skip_special_tokens=True), "total_logprob": float(tot)})
        del out
    return results


def versions() -> dict:
    v = {"torch": torch.__version__, "transformers": transformers.__version__}
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name(0)
    return v
