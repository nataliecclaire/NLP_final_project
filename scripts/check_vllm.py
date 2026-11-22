"""Sanity-check the vLLM sampling backend before the full sampling run (GPU, vLLM installed).

    python scripts/check_vllm.py

1. vLLM greedy (temperature 0) on 5 saved HF-greedy prompts: same token IDs, and the summed
   logprobs (incl. </s>) match the saved HF logprobs -> prompt IDs are passed through intact and
   total_logprob uses the same convention as greedy_token_logprobs.
2. Sampling: 10 samples per prompt, all total_logprob <= 0, and the same seed gives the same samples.
Run it in its own process (not inside pytest with the HF model loaded): two copies of the model
do not fit on one GPU.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halugen.config import load_config  # noqa: E402
from halugen.pipeline import sample_seed  # noqa: E402
from halugen.vllm_backend import VllmSampler  # noqa: E402

LOGPROB_SUM_TOL = 0.1   # fp16 kernels differ between HF and vLLM


def main():
    cfg = load_config()
    recs = [json.loads(line) for line in open(cfg["paths"]["generations_dir"] / "triviaqa_train.jsonl")][:5]
    sampler = VllmSampler(cfg)
    from vllm import SamplingParams

    ok = True
    params = SamplingParams(n=1, temperature=0.0, max_tokens=cfg["generation"]["max_new_tokens"]["factual"], logprobs=0)
    outs = sampler.llm.generate([{"prompt_token_ids": r["prompt_token_ids"]} for r in recs], params, use_tqdm=False)
    n_same = 0
    for r, o in zip(recs, outs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        total = sum(step[t].logprob for step, t in zip(c.logprobs, ids))
        same = ids == r["greedy_token_ids"]
        n_same += same
        d = abs(total - sum(r["greedy_token_logprobs"]))
        print(f"{r['id']}: ids_match={same} vllm_sum={total:.4f} hf_sum={sum(r['greedy_token_logprobs']):.4f} |d|={d:.4f}")
        if same and d > LOGPROB_SUM_TOL:
            ok = False
            print("   FAIL: logprob sums disagree although tokens match")
    if n_same < 4:
        ok = False
        print(f"FAIL: vLLM greedy matched HF greedy on only {n_same}/5 prompts")

    prompts = [r["prompt_token_ids"] for r in recs[:3]]
    seeds = [sample_seed(cfg, r["id"]) for r in recs[:3]]
    a = sampler.sample(prompts, cfg["generation"]["max_new_tokens"]["factual"], seeds)
    b = sampler.sample(prompts, cfg["generation"]["max_new_tokens"]["factual"], seeds)
    for r, s in zip(recs, a):
        print(f"{r['id']}: {[x['text'].strip()[:20] for x in s[:4]]} ...")
    if not all(len(s) == cfg["generation"]["sampling"]["n"] for s in a):
        ok = False
        print("FAIL: wrong number of samples")
    if not all(x["total_logprob"] <= 0 for s in a for x in s):
        ok = False
        print("FAIL: positive total_logprob")
    if [[x["text"] for x in s] for s in a] != [[x["text"] for x in s] for s in b]:
        ok = False
        print("FAIL: same seeds gave different samples")
    print("\nvLLM CHECK PASSED" if ok else "\nvLLM CHECK FAILED - do not run the sampling phase with vLLM; send this output")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
