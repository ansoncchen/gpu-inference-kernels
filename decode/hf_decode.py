"""FP16 greedy decode speed with Hugging Face transformers on the GPU (SDPA attention).
hf_decode.py <model dir> <prompt ids> <new tokens> <repeats> <out.jsonl>"""
import json, statistics as st, sys, time, torch
from transformers import AutoModelForCausalLM
mdir, idf, n, reps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
m = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
ids = torch.tensor([[int(t) for t in open(idf).read().split()]]).cuda()
us = []
with torch.inference_mode():
    for r in range(reps + 1):
        o = m(ids, use_cache=True)
        cache, tok = o.past_key_values, o.logits[:, -1:].argmax(-1)
        tok.item()
        t0 = time.perf_counter()
        for _ in range(n):
            o = m(tok, past_key_values=cache, use_cache=True)
            tok = o.logits[:, -1:].argmax(-1)
            tok.item()
        if r:
            us.append((time.perf_counter() - t0) * 1e6 / n)
med = st.median(us)
rec = dict(run="decode", engine=f"Hugging Face transformers, PyTorch {torch.__version__}", mode="fp16", prompt_tokens=ids.shape[1],
           new_tokens=n, tok_per_s=1e6 / med, median_us=med, samples_us=us)
open(out, "a").write(json.dumps(rec) + "\n")
print(rec)
