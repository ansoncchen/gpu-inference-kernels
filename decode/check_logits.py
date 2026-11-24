"""Compares engine logits with the Hugging Face FP32 reference."""
import sys, numpy as np
ref = np.fromfile(sys.argv[1], "float32")
for p in sys.argv[2:]:
    a = np.fromfile(p, "float32").reshape(-1, 151936)
    r = ref.reshape(-1, 151936)
    print(p.split("/")[-1], a.shape, "max|diff| %.4f" % np.abs(a - r).max(), "mean|diff| %.4f" % np.abs(a - r).mean(),
          "top-1 agreement %d/%d" % ((a.argmax(1) == r.argmax(1)).sum(), len(a)))
