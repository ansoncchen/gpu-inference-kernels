"""Turns results/raw/*.jsonl into results/tables.md, results/summary.json and
docs/img/*.png. Every number in the README comes from this script's output."""
import collections, json, os, statistics as st

RAW, IMG = "results/raw", "docs/img"
os.makedirs(IMG, exist_ok=True)

def load(name):
    p = f"{RAW}/{name}"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

main, tune, l2 = load("gpu_main.jsonl"), load("gpu_tune.jsonl"), load("gpu_l2.jsonl")
ggml, ggml_ng, cpu, dec = load("ggml.jsonl"), load("ggml_nograph.jsonl"), load("cpu.jsonl"), load("decode.jsonl")
ppl = load("ppl.jsonl")
SHAPES = [(1536, 8960), (8960, 1536), (4096, 4096), (11008, 4096), (8192, 8192), (32768, 8192)]
STEPS = ["s0_naive", "s1_coalesced", "s2_vec16", "s3_mantissa", "s4_xregs", "s5_tiled", "s6_block", "s7_loop", "s8_octet"]

def agg(rows):
    """(variant, shape) -> median of the per-process medians, with their min and max."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["variant"], r["rows"], r["cols"])].append(r)
    return {k: dict(us=st.median(x["median_us"] for x in v), lo=min(x["median_us"] for x in v),
                    hi=max(x["median_us"] for x in v), bytes=v[0]["bytes"], n=len(v),
                    err=max(x["max_rel_err"] for x in v),
                    lat=st.median(x["latency_median_us"] for x in v) if v[0].get("latency_median_us") else None)
            for k, v in g.items()}

M, G, GN = agg(main), agg(ggml), agg(ggml_ng)
peak = max(M[(f"read{b}", 0, 0)]["bytes"] / M[(f"read{b}", 0, 0)]["us"] * 1e-3 for b in (4, 8, 16))
out, summ = [], {"peak_read_gbps": peak, "theoretical_gbps": 616}
P = out.append

def q4bytes(s):  # bytes of the INT4 matrix itself, before any layout padding
    return s[0] * s[1] * 0.5625

def best_fp16(s):
    c = {k[0]: v for k, v in M.items() if k[0].startswith("fp16") and k[1:] == s}
    n = min(c, key=lambda k: c[k]["us"])
    return n, c[n]

P("## Machine limits\n")
P("| probe | result |\n|---|---|")
P("| theoretical memory bandwidth | 616 GB/s |")
for b in (1, 4, 8, 16):
    v = M[(f"read{b}", 0, 0)]
    P(f"| measured read, {b}-byte loads, 4 GiB buffer | {v['bytes']/v['us']*1e-3:.1f} GB/s |")
P(f"| **measured peak read (best of the above)** | **{peak:.1f} GB/s** ({peak/616*100:.1f}% of theoretical) |")
P(f"| empty kernel, back to back | {M[('empty',0,0)]['us']:.2f} us per launch |")
fma = st.median(r["fma_per_s"] for r in main if r["variant"] == "fma")
P(f"| FP32 FMA rate | {fma/1e12:.2f} T/s |")
P("")

P("## Steps, microseconds per GEMV\n")
P("Median over 3 separate processes of each process's median (at least 30 timed passes each).\n")
P("| variant | " + " | ".join(f"{r}x{c}" for r, c in SHAPES) + " | % of peak read, 8192x8192 |")
P("|---|" + "---:|" * (len(SHAPES) + 1))
for v in STEPS + ["a3_xcontig", "a7_convert", "a8_convert"]:
    cells = [f"{M[(v,)+s]['us']:.1f}" for s in SHAPES]
    s = (8192, 8192)
    pct = q4bytes(s) / M[(v,) + s]["us"] * 1e-3 / peak * 100
    P(f"| {v} | " + " | ".join(cells) + f" | {pct:.0f}% |")
P("| read-only bound (weights, no scales) | " + " | ".join(f"{M[('readbound',)+s]['us']:.1f}" for s in SHAPES) + " | |")
P("")

P("## Against baselines, microseconds per GEMV\n")
P("| shape | FP32 cuBLAS (best) | FP16 cuBLAS (best of 6 routes) | which | ggml Q4_0 | ggml Q4_0, no CUDA graphs | s8_octet | s8 vs FP16 | s8 vs ggml | s8 GB/s | % of peak read |")
P("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
summ["shapes"] = {}
for s in SHAPES:
    f32 = min(M[(f"fp32_sgemv_{x}",) + s]["us"] for x in "TN")
    n16, f16 = best_fp16(s)
    g, gn, m = G.get(("ggml_q4_0",) + s), GN.get(("ggml_q4_0",) + s), M[("s8_octet",) + s]
    bw = q4bytes(s) / m["us"] * 1e-3
    P(f"| {s[0]}x{s[1]} | {f32:.1f} | {f16['us']:.1f} | {n16} | {g['us']:.1f} | {gn['us']:.1f} | **{m['us']:.1f}** | "
      f"**{f16['us']/m['us']:.2f}x** | {g['us']/m['us']:.2f}x | {bw:.0f} | {bw/peak*100:.0f}% |")
    summ["shapes"][f"{s[0]}x{s[1]}"] = dict(fp32_us=f32, fp16_us=f16["us"], fp16_route=n16, ggml_us=g["us"], ggml_nograph_us=gn["us"],
        s8_us=m["us"], s8_lo=m["lo"], s8_hi=m["hi"], speedup_fp16=f16["us"] / m["us"], speedup_ggml=g["us"] / m["us"],
        speedup_naive=M[("s0_naive",) + s]["us"] / m["us"],
        gbps=bw, pct_peak=bw / peak * 100, err_s8=m["err"], err_ggml=g["err"], err_fp16=f16["err"])
geo = lambda xs: st.geometric_mean(list(xs))
summ["geomean_vs_fp16"] = geo(x["speedup_fp16"] for x in summ["shapes"].values())
summ["geomean_vs_ggml"] = geo(x["speedup_ggml"] for x in summ["shapes"].values())
summ["geomean_vs_naive"] = geo(x["speedup_naive"] for x in summ["shapes"].values())
P(f"\nGeometric mean: {summ['geomean_vs_fp16']:.2f}x vs FP16 cuBLAS, {summ['geomean_vs_ggml']:.2f}x vs ggml, "
  f"{summ['geomean_vs_naive']:.1f}x vs s0_naive. The FP16 to INT4 size ratio is 3.56x, so that is the ceiling for a "
  "memory-bound kernel.\n")
P("Largest error relative to the row bound (see tests): "
  f"s8 {max(x['err_s8'] for x in summ['shapes'].values()):.1e}, ggml {max(x['err_ggml'] for x in summ['shapes'].values()):.1e} "
  f"(int8 x), FP16 cuBLAS {max(x['err_fp16'] for x in summ['shapes'].values()):.1e}.\n")
P("Single call, launch + sync on the host clock, 4096x4096: "
  + ", ".join(f"{v} {M[(v,4096,4096)]['lat']:.1f} us" for v in ("s8_octet", "s7_loop", best_fp16((4096, 4096))[0])) + ".\n")

P("## Spread between the 3 processes (s8_octet)\n")
P("| shape | median | lowest | highest |\n|---|---:|---:|---:|")
for s in SHAPES:
    m = M[("s8_octet",) + s]
    P(f"| {s[0]}x{s[1]} | {m['us']:.2f} | {m['lo']:.2f} | {m['hi']:.2f} |")
P("")
nv = [r["nvml"] for r in main if r["variant"] == "s8_octet"]
P(f"Clocks while s8 ran: SM {min(x['sm_mhz_min'] for x in nv):.0f}-{max(x['sm_mhz_max'] for x in nv):.0f} MHz, memory "
  f"{min(x['mem_mhz_mean'] for x in nv):.0f}-{max(x['mem_mhz_mean'] for x in nv):.0f} MHz, up to {max(x['temp_c_max'] for x in nv):.0f} C, "
  f"mean power {min(x['power_w_mean'] for x in nv):.0f}-{max(x['power_w_mean'] for x in nv):.0f} W (limit 300 W).\n")

if tune:
    T = agg(tune)
    names = sorted({k[0] for k in T})
    best = {s: min(T[(n,) + s]["us"] for n in names) for s in SHAPES}
    score = {n: geo(T[(n,) + s]["us"] / best[s] for s in SHAPES) for n in names}
    for pre, title in (("t_loop", "s7: b = rows per iteration, h = rows per block"), ("t_oct", "s8: p = loads in flight, c = 128-column chunks per warp")):
        P(f"## Tuning {title}\n")
        P("Slowdown against the best setting found for each shape, geometric mean over the 6 shapes.\n")
        P("| setting | slowdown | " + " | ".join(f"{r}x{c}" for r, c in SHAPES) + " |\n|---|---:|" + "---:|" * len(SHAPES))
        for n in sorted((n for n in names if n.startswith(pre)), key=score.get):
            P(f"| {n[2:]} | {score[n]:.3f} | " + " | ".join(f"{T[(n,)+s]['us']:.1f}" for s in SHAPES) + " |")
        P("")
if l2:
    P("## Matrix small enough to sit in the L2 (2048x4096, one copy, 4.7 MB)\n")
    P("| variant | us |\n|---|---:|")
    for k, v in agg(l2).items():
        P(f"| {k[0]} | {v['us']:.1f} |")
    P("")

if cpu:
    C = collections.defaultdict(dict)
    for r in cpu:
        C[(r["variant"], r["rows"], r["cols"])][r["threads"]] = r
    cread = max(r["gbps_median"] for r in C[("read", 0, 0)].values())
    summ["cpu_peak_read_gbps"] = cread
    P("## CPU (Threadripper 1950X), best thread count per entry\n")
    P(f"Measured peak read bandwidth: {cread:.1f} GB/s.\n")
    P("| variant | " + " | ".join(f"{r}x{c} us (threads)" for r, c in ((4096, 4096), (11008, 4096), (8192, 8192))) + " | GB/s at 8192x8192 | % of CPU peak read |")
    P("|---|---:|---:|---:|---:|---:|")
    for v in ("openblas_fp32", "fp32_avx2", "c0_scalar", "c1_avx2", "c2_avx2_int8x"):
        cells = []
        for s in ((4096, 4096), (11008, 4096), (8192, 8192)):
            b = min(C[(v,) + s].values(), key=lambda r: r["median_us"])
            cells.append(f"{b['median_us']:.0f} ({b['threads']})")
            summ[f"cpu_{v}_{s[0]}x{s[1]}_us"] = b["median_us"]
        P(f"| {v} | " + " | ".join(cells) + f" | {b['gbps_median']:.1f} | {b['gbps_median']/cread*100:.0f}% |")
    s = (8192, 8192)
    f32best = {x: min(summ[f"cpu_openblas_fp32_{x}_us"], summ[f"cpu_fp32_avx2_{x}_us"]) for x in ("4096x4096", "11008x4096", "8192x8192")}
    for x in f32best:
        summ[f"cpu_int4_vs_best_fp32_{x}"] = f32best[x] / summ[f"cpu_c1_avx2_{x}_us"]
        summ[f"cpu_int4_vs_openblas_{x}"] = summ[f"cpu_openblas_fp32_{x}_us"] / summ[f"cpu_c1_avx2_{x}_us"]
        summ[f"cpu_int4_vs_scalar_{x}"] = summ[f"cpu_c0_scalar_{x}_us"] / summ[f"cpu_c1_avx2_{x}_us"]
    summ["cpu_int4_vs_fp32_8192"] = summ["cpu_int4_vs_best_fp32_8192x8192"]
    summ["gpu_vs_cpu_8192"] = summ["cpu_c1_avx2_8192x8192_us"] / M[("s8_octet",) + s]["us"]
    P(f"\n8192x8192: AVX2 INT4 is {summ['cpu_int4_vs_fp32_8192']:.1f}x faster than the better FP32 baseline ({summ['cpu_int4_vs_openblas_8192x8192']:.1f}x vs OpenBLAS, {summ['cpu_int4_vs_scalar_8192x8192']:.1f}x vs plain C++ INT4); the GPU s8 kernel is "
      f"{summ['gpu_vs_cpu_8192']:.1f}x faster than AVX2 INT4, and the measured bandwidths differ by {peak/cread:.1f}x.\n")

import glob
if dec:
    P("## Decode: Qwen2.5-1.5B, greedy, one sequence, 30-token prompt + 256 new tokens\n")
    P("| engine | weights | GB read per token | tok/s |\n|---|---|---:|---:|")
    D = {}
    for r in dec:
        eng = r.get("engine", "this repo")
        wb = f"{r['weight_bytes']/1e9:.3f}" if r.get("weight_bytes") else ""
        P(f"| {eng} | {r['mode']} | {wb} | {r['tok_per_s']:.1f} |")
        D[r["mode"] if eng == "this repo" else "hf_fp16"] = r["tok_per_s"]
    for m in ("f16", "q4_0"):
        rs = [json.load(open(p))[0] for p in glob.glob(f"{RAW}/llama_bench_{m}_fa*.json")]
        if rs:
            b = max(rs, key=lambda r: r["avg_ts"])
            P(f"| llama.cpp llama-bench (best of flash-attn on/off; no prompt) | {m} | | {b['avg_ts']:.1f} |")
            D["llama_" + m] = b["avg_ts"]
    summ["decode"] = D
    if "fp16" in D and "int4s" in D:
        summ["decode_speedup"] = D["int4s"] / D["fp16"]
        if "llama_q4_0" in D:
            summ["decode_vs_llama_q4_0"] = D["int4s"] / D["llama_q4_0"]
        P(f"\nint4s / fp16 in this repo: {D['int4s']/D['fp16']:.2f}x.\n")
if ppl:
    P("## Perplexity, WikiText-2 test, 40 windows of 512, second half scored\n")
    P("| engine | weights | ppl | change vs FP16 |\n|---|---|---:|---:|")
    base = next(r["ppl"] for r in ppl if r["mode"] == "fp16")
    for r in ppl:
        P(f"| this repo | {r['mode']} | {r['ppl']:.3f} | {100*(r['ppl']/base-1):+.1f}% |")
        summ["ppl_" + r["mode"]] = r["ppl"]
    import re
    lp = {}
    for m in ("f16", "q4_0"):
        f = f"{RAW}/llama_ppl_{m}.txt"
        if os.path.exists(f):
            lp[m] = float(re.search(r"PPL = ([\d.]+)", open(f).read()).group(1))
            P(f"| llama.cpp | {m} | {lp[m]:.3f} | {100*(lp[m]/lp['f16']-1):+.1f}% |")
            summ["ppl_llama_" + m] = lp[m]
    P("")
open("results/tables.md", "w").write("<!-- generated by scripts/report.py from results/raw -->\n\n" + "\n".join(out))
json.dump(summ, open("results/summary.json", "w"), indent=1)

# ---------------------------------------------------------------- charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTE, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "text.color": INK})

def clean(ax, grid="x"):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

LABEL = {"s0_naive": "s0  thread per row, byte loads", "s1_coalesced": "s1  warp per row, coalesced bytes",
         "s2_vec16": "s2  16-byte loads", "s3_mantissa": "s3  + nibble-in-mantissa", "s4_xregs": "s4  x in registers, shuffles",
         "s5_tiled": "s5  tiled, fully unrolled", "s6_block": "s6  block reduction, unrolled", "s7_loop": "s7  short hot loop",
         "s8_octet": "s8  8 rows x 4 cols per load"}
s = (8192, 8192)
fig, ax = plt.subplots(figsize=(8.4, 4.4))
vals = [q4bytes(s) / M[(v,) + s]["us"] * 1e-3 for v in STEPS]
ax.barh(range(len(STEPS)), vals, height=0.62, color=[BLUE] * len(STEPS))
for i, v in enumerate(vals):
    inside = v > 480
    ax.text(v - 8 if inside else v + 8, i, f"{v:.0f}", va="center", ha="right" if inside else "left", color="white" if inside else INK, fontsize=9)
ax.axvline(peak, color=INK, lw=1.2)
ax.axvline(616, color=MUTE, lw=1.0, ls=(0, (4, 3)))
ax.text(peak - 4, -0.85, f"measured peak read {peak:.0f}", ha="right", fontsize=8.5, color=INK)
ax.text(620, -0.85, "spec 616", ha="left", fontsize=8.5, color=MUTE)
ax.set_yticks(range(len(STEPS)), [LABEL[v] for v in STEPS])
ax.invert_yaxis()
ax.set_xlim(0, 680)
ax.set_xlabel("weight bytes consumed per second, GB/s (8192 x 8192 INT4)")
ax.set_title("Two steps made the kernel slower before it reached the memory limit", loc="left", fontsize=11, pad=22)
clean(ax)
fig.tight_layout()
fig.savefig(f"{IMG}/steps.png", dpi=160)

fig, ax = plt.subplots(figsize=(8.4, 4.0))
names = [f"{r}x{c}" for r, c in SHAPES]
w = 0.36
for j, (key, col, lab) in enumerate((("ggml", ORANGE, "ggml Q4_0 (int8 x)"), ("s8", BLUE, "this repo, s8 (FP32 x)"))):
    v = [summ["shapes"][n]["fp16_us"] / summ["shapes"][n]["ggml_us" if key == "ggml" else "s8_us"] for n in names]
    xs = [i + (j - 0.5) * (w + 0.03) for i in range(len(names))]
    ax.bar(xs, v, width=w, color=col, label=lab)
    for x, y in zip(xs, v):
        ax.text(x, y - 0.06, f"{y:.2f}", ha="center", va="top", fontsize=8.5, color="white" if key == "s8" else INK)
ax.axhline(3.5556, color=MUTE, lw=1.0, ls=(0, (4, 3)))
ax.text(-0.48, 3.6, "3.56x = FP16 / INT4 size ratio", fontsize=8.5, color=MUTE)
ax.set_ylim(0, 4.2)
ax.set_xticks(range(len(names)), names)
ax.set_ylabel("speedup over best FP16 cuBLAS GEMV")
ax.set_title("Speedup over FP16 cuBLAS, RTX 2080 Ti", loc="left", fontsize=11)
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
clean(ax, "y")
fig.tight_layout()
fig.savefig(f"{IMG}/speedup.png", dpi=160)

if cpu:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for v, col, lab in (("read", INK, "read-only probe"), ("c1_avx2", BLUE, "INT4, AVX2 + FMA"), ("c2_avx2_int8x", ORANGE, "INT4, AVX2, int8 x"), ("c0_scalar", AQUA, "INT4, plain C++")):
        d = C[(v, 0, 0)] if v == "read" else C[(v, 8192, 8192)]
        t = sorted(d)
        y = [d[k]["gbps_median"] for k in t]
        ax.plot(t, y, "-o", color=col, lw=2, ms=6, label=lab, mec=SURF, mew=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32], ["1", "4", "8", "16", "32"])
    ax.set_xlabel("threads (16 cores, 32 hardware threads)")
    ax.set_ylabel("GB/s of matrix data")
    ax.set_ylim(0, 60)
    ax.set_title("CPU INT4 GEMV approaches the DRAM limit at 16 threads (8192 x 8192)", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    clean(ax, "y")
    fig.tight_layout()
    fig.savefig(f"{IMG}/cpu.png", dpi=160)
if dec and "llama_q4_0" in summ["decode"]:
    D = summ["decode"]
    rows = [("PyTorch / HF transformers, FP16", D["hf_fp16"], 0), ("llama.cpp, FP16", D["llama_f16"], 0), ("this repo, FP16 (cuBLAS)", D["fp16"], 0),
            ("llama.cpp, Q4_0 (Q6_K head)", D["llama_q4_0"], 1), ("this repo, INT4, FP16 head", D["int4sh"], 1), ("this repo, INT4", D["int4s"], 1)]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.barh(range(len(rows)), [r[1] for r in rows], height=0.62, color=[BLUE if r[2] else "#9ec5f0" for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[1] + 5, i, f"{r[1]:.0f}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("tokens per second, Qwen2.5-1.5B, RTX 2080 Ti")
    ax.set_title("Single-stream decode speed (light = FP16, dark = INT4)", loc="left", fontsize=11)
    ax.set_xlim(0, 470)
    clean(ax)
    fig.tight_layout()
    fig.savefig(f"{IMG}/decode.png", dpi=160)
print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, (dict, list))}, indent=1))
