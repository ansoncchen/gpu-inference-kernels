"""Token files and reference logits for the decode engine (needs torch + transformers).
prepare.py <model dir> <wikitext-2 raw test file> <out dir>"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mdir, wiki, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(mdir)
prompt = ("The history of computing hardware covers the developments from early simple devices to aid calculation "
          "to modern day computers. Memory bandwidth became the main limit for many programs because")
ids = tok(prompt).input_ids
open(f"{out}/prompt.ids", "w").write(" ".join(map(str, ids)))
w = tok(open(wiki).read()).input_ids
open(f"{out}/wiki.ids", "w").write(" ".join(map(str, w)))
print(len(ids), len(w))
# FP32 CPU reference logits for the prompt, from Hugging Face.
m = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).eval()
with torch.no_grad():
    l = m(torch.tensor([ids])).logits[0]
l.numpy().astype("float32").tofile(f"{out}/ref_logits.bin")
