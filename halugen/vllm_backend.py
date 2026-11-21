"""vLLM backend for the 10 sampled answers per question."""
from __future__ import annotations


def available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


class VllmSampler:
    def __init__(self, cfg: dict):
        from vllm import LLM
        v = cfg["generation"]["vllm"]
        self.cfg = cfg
        self.llm = LLM(model=cfg["model"]["name"], revision=cfg["model"]["revision"],
                       tokenizer_revision=cfg["model"]["revision"], dtype=cfg["model"]["dtype"],
                       seed=cfg["seed"], gpu_memory_utilization=v["gpu_memory_utilization"],
                       max_model_len=v["max_model_len"], enforce_eager=v["enforce_eager"])

    def sample(self, prompt_ids_list: list[list[int]], max_new_tokens: int, seeds: list[int]) -> list[list[dict]]:
        from vllm import SamplingParams
        s = self.cfg["generation"]["sampling"]
        params = []
        for seed in seeds:
            kw = dict(n=s["n"], temperature=s["temperature"], top_p=s["top_p"], max_tokens=max_new_tokens,
                      seed=seed, logprobs=0, skip_special_tokens=True)
            if s["top_k"] is not None:
                kw["top_k"] = s["top_k"]  # omitted -> vLLM default, which is "disabled"
            p = SamplingParams(**kw)
            assert s["top_k"] is not None or p.top_k in (-1, 0), f"unexpected vLLM top_k default {p.top_k}"
            params.append(p)
        outs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids_list], params, use_tqdm=False)
        results = []
        for o in outs:
            samples = []
            for c in o.outputs:
                if c.logprobs is not None and len(c.logprobs) == len(c.token_ids):
                    # Sum of the sampled tokens' logprobs (includes </s> when emitted). With
                    # temperature 1.0 and no top-k/top-p this is the raw model distribution.
                    total = sum(step[tid].logprob for step, tid in zip(c.logprobs, c.token_ids))
                else:
                    total = c.cumulative_logprob
                samples.append({"text": c.text, "total_logprob": float(total)})
            results.append(samples)
        return results


def versions() -> dict:
    import vllm
    return {"vllm": vllm.__version__}
