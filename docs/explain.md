# Explaining every step

Written for an interview. Each section says what the code does, why, and what a good interviewer will ask next.
All numbers are for an RTX 2080 Ti and can be found in `results/tables.md`.

## 0. The problem in three lines

Generating one token with a large language model reads every weight once. For a 1.5B model in FP16 that is 3.1 GB per
token, and the 2080 Ti reads about 587 GB/s, so one token cannot take less than about 5.3 ms. The only ways to go faster are to read fewer bytes
(quantize) or to have more memory bandwidth. INT4 reads 0.87 GB, about 1.5 ms. Arithmetic is not the problem: the card does 8.4 T FMAs per second and
the model only needs about 1.5 G per token.

**Roofline version.** Operational intensity for FP16 GEMV is 2 flops per 2 bytes = 1 flop/byte. At 587 GB/s that is 0.59 Tflop/s,
far below the measured 8.4 T FMA/s (about 17 Tflop/s). So the kernel sits on the sloped (memory-bound) part of the roofline.

## 1. The INT4 format

Each 32 weights share a scale. `w = scale * q`, q in -8..7. Store `u = q + 8` (0..15) as four bits, so unpacking needs no sign handling.
Scale is FP16. Bytes per weight: 16/32 + 2/32 = 0.5625. Why not one scale per row? A single outlier weight would force a coarse step for
thousands of weights. Why 32? It is the usual trade between scale overhead (12.5% of the bytes) and accuracy.

*Q: What error does rounding add?* Half a step, scale/2 per weight at most; a weight clipped at +7 can be off by up to a full step
because the range is -8..7 and the extreme is mapped to -8.

*Q: How do you choose the scale?* The baseline maps the largest-magnitude weight to -8 (same rule as ggml Q4_0). `int4s` tries 25 scales with both
signs and keeps the one with least squared error. Same format and same speed. Perplexity goes from 9.78 to 9.64.

## 2. s0: naive

One thread per row; thread r walks along row r a byte at a time. Threads in a warp read addresses one row apart (kilobytes apart), so every
thread pulls in its own 32-byte sector and uses one byte of it. 8% of peak.

*Q: What is coalescing?* A warp's 32 loads are merged if they fall in the same cache lines. Memory is fetched in 32-byte sectors grouped in 128-byte lines.

## 3. s1: one warp per row

The lanes of a warp read adjacent bytes: a 32-byte load with no waste. 53% of peak. Each warp reduces its 32 partial sums through shared memory.

## 4. s2: 16-byte loads (slower)

`uint4` loads fetch 16 bytes per lane, so each warp-wide load is 512 contiguous bytes. Fewer load instructions for the weights, but each lane now handles
32 *consecutive* columns, so it needs 32 consecutive x values, and the warp's 32 lanes start 128 bytes apart. One warp-wide x load touches 32 cache lines and takes
16 bytes from each. 24% of peak.

*Q: How do you know it was x?* `a3_xcontig` rearranges x first (one extra kernel) and changes nothing else in the loop. 275.8 us goes to 71.6 us.

*Q: Why does touching many lines cost so much?* The L1 works by looking up tags, and each distinct line in one warp instruction needs its own lookup, so the instruction
is replayed over many cycles. I did not measure the tag rate with counters; the claim rests on the ablation.

## 5. s3: building floats from nibbles

Idea: avoid the int-to-float conversion instruction, which I expected to have lower throughput than simple logic ops. Put the nibble `u` into the top 4 of the 23 mantissa bits of 1.0f:

    bits = 0x3F800000 | (u << 19)    which is exactly 1 + u/16

Why it works: a float is 1.mantissa x 2^e. The top mantissa bit is worth 1/2, then 1/4, 1/8, 1/16. So a 4-bit value written there adds u/16.
Extracting one nibble is a shift plus one LOP3 (and + or in a single instruction). We do not want 1 + u/16, we want u - 8:

    sum (u - 8) x = 16 * ( sum (1 + u/16) x  -  1.5 * sum x )

`sum x` over a group is the same for every row, so it is computed once and the subtraction costs almost nothing.

*Q: Did it help?* No. At these sizes the kernels wait for memory, so saving arithmetic does not show (under 1% difference on large shapes, up to 3% on the smallest). It helps only when the matrix is in the L2
(8.6 vs 10.0 us). It also loses some accuracy through cancellation: error 3e-8 of the row bound against 6e-9 with real conversion.
I kept it because it shows the hot loop, and I report that it did not pay.

*Q: Where could the accuracy get worse?* When x has a large mean, `1.5 * sum x` is large compared with the real answer. The tests include all-positive large x and the error stays
under 2e-7 of the bound.

## 6. s4: x in registers, warp-shuffle reduction

A warp owns 32 rows x 1024 columns. Lane g handles columns 32g..32g+31 and keeps those 32 x values in registers for all 32 rows, so x is read once per tile and not once per row.
At the end of each row the 32 lanes' partial sums are combined with a butterfly of 5 shuffles (`__shfl_xor_sync` with distances 16, 8, 4, 2, 1): each lane adds the value of the lane whose index differs in one bit.
86% of peak. Rows are written with atomics, because several warps add to the same row.

*Q: Why shuffles and not shared memory?* They need no memory traffic and no barrier; they are register moves between lanes.

## 7. s5, s6: tiling, reduce-scatter and the unrolling mistake

Layout: 32x1024 tiles, weights of a tile contiguous. Scales are stored so one 16-byte load gives 8 rows' scales for a group.

**Reduce-scatter.** Doing a 5-step reduction for 32 rows costs 160 shuffles. Instead: each lane holds 32 partial sums. Step 1: exchange 16 of them with the lane 16 away and add; keep 16.
Step 2: 8 with the lane 8 away. After 5 steps each lane holds one finished row: 16+8+4+2+1 = 31 shuffles. This is the same idea as MPI's reduce-scatter.

s6 also does the final add across column tiles in shared memory, so no memset and no atomics.

Both were slow (58%, 60%) because everything was unrolled into one long body. s7 does the same work in a short loop.

*Q: How much code?* About 3,500 instructions at 16 bytes each, over 50 KB (an estimate), against instruction caches that are much smaller near the SM. s7's loop body is 8 rows.
I did not read cache-miss counters; the evidence is that s6 and s7 differ in this and 32768x8192 goes from 453 to 267 us.

## 8. s7: short hot loop, tuned

B = 8 rows per loop iteration (8 loads of 512 bytes in flight per warp, one scale load), H = 16 rows per block. Reduction for a batch of 8: 3 reduce-scatter steps, then 2 all-reduce steps (4+2+1 shuffles
for the first part, 2 for the second, 9 in total). 90% of peak.

*Q: Why not always big blocks?* With H = 128 a block is one long unit of work. Small matrices then have few blocks and few loads in flight. Little's law: to sustain bandwidth B with latency L you need B x L bytes in
flight: 587 GB/s at roughly 0.5 to 1 us is 300 to 600 KB. (The latency is my assumption, not a measurement.) 4096x4096 with H = 128 has 32 blocks for 68 SMs: most of the GPU is idle.

## 9. s8: the octet layout

This is the version I would show first. One 16-byte load holds 8 rows x 4 columns (2 bytes per row). Lane l takes columns 4l..4l+3 of a 128-column chunk. So in one step:

- the warp reads 512 contiguous bytes of weights (8 rows x 128 columns), and
- 512 contiguous bytes of x (128 floats, 16 per lane as one float4).

Each lane keeps 8 running sums (one per row). A group of 32 columns is spread over 8 lanes, which all load the same scale block (it is a broadcast, cheap), and each multiplies by its own part. After all columns, a 9-shuffle reduction gives
8 row results. With one warp per 1024-column piece, pieces are added in shared memory. No atomics, no memset, 64 registers.

*Q: Why is reloading x fine now?* Contiguous loads from L1 are cheap (4 lines per instruction rather than 32). The trouble was never the number of x bytes.

*Q: Why not only one warp per row-octet?* A very wide matrix then has too few warps. Splitting each row into 1024-column pieces (c8 in the tuning table) gave the best average. Prefetch 2 (two steps loaded before computing) was within 0.2% of prefetch 1, so loads-in-flight per warp is not the limit any more.

*Q: Can it go faster?* At 574 of 587 GB/s on large matrices, no. Smaller matrices still lose about 2 us to the launch gap and to load balance.

## 10. Baselines and fairness

- **cuBLAS FP16** has several routes: strided-batched HSH and HSS GEMV, and `gemmEx` with n = 1, each with and without transpose. I measured all six on every shape and compare with the best one. The best route reads 88 to 99% of the peak depending on the shape, so they are memory-bound too, which is why the ceiling is the 3.56x size ratio.
- **cuBLAS FP32**: twice the bytes again.
- **ggml (llama.cpp) Q4_0**: same weights bit for bit. It quantizes x to int8 and uses integer dot products, so its error is about 5e-4 of the row bound against 3e-8 for mine. It can also capture launches into a CUDA graph; I measured both ways. Mine is equal or up to 4% faster.
- FP16 x in the FP16 baseline against FP32 x in mine: not identical, and I say so. The FP16 baseline's error is 4e-5.

*Q: Isn't your advantage over FP16 just the smaller data?* Yes, up to 3.56x. The kernel's job is to not lose that factor. It keeps 3.5x at the largest size.

## 11. CPU version

Unpack 16 bytes of nibbles with AND and a shift, zero-extend to 32-bit, convert to float, and FMA with x. The same `- 8 * sum x` correction. Threads split the rows. AVX2 gives 8 floats per instruction; Zen 1 executes a 256-bit
instruction as two 128-bit halves, so the gain over scalar is about 3x to 4x, not 8x.

*Q: Why does int8 x look much better on one thread and the same on 16?* One thread is limited by arithmetic, and `maddubs` does 32 products at once. 16 threads are limited by DRAM, so better arithmetic buys nothing.

*Q: Why is the GPU 12x faster?* Memory bandwidth: 587 vs 51 GB/s is 11.4x. The arithmetic peak ratio would be much bigger.

*Q: Why does read bandwidth fall past 8 threads?* More threads than the memory controllers can keep busy means extra queueing. I did not investigate it further; it is a measurement, not an explanation I can prove.

## 12. Decode engine

Layer by layer: RMSNorm, GEMV for q/k/v stacked in one matrix, attention kernel (RoPE on q and k, dot with the KV cache, softmax, weighted sum), GEMV for the output, RMSNorm, GEMV for gate and up stacked, SiLU times up, GEMV down.
Stacking means 4 GEMVs per layer and not 7, for both modes.

*Q: 2.6x is less than 3.56x. Where does the rest go?* INT4 reads 0.868 GB per token. At 587 GB/s that is 1.5 ms; a token takes 2.4 ms. The other ~0.9 ms is launch gaps (8 kernels per layer, 28 layers, around 2 us each), attention, norms, copying the token back, and small matrices that run below full bandwidth. FP16 has the same overhead but it is a smaller share of 6.3 ms.

*Q: Is the output right?* FP16 logits agree with Hugging Face FP32 on CPU to 0.03, 30 of 30 top-1. FP16 perplexity is 8.726 against llama.cpp's 8.736. INT4 perplexity is 9.643, which is 10.5% worse; with the head in FP16 it is 6.3% worse. The quality loss is from rounding without calibration.

*Q: What would you do next?* A quantizer that uses calibration data (GPTQ/AWQ style), batching (a GEMM), CUDA graphs to remove launch gaps, and Nsight Compute on a machine where counters are allowed to check the cache and instruction-cache explanations.

## 13. Things to be honest about

- I did not have hardware counters. Explanations of *why* something was slow are backed by one-change experiments, not counters.
- Clocks were not pinned and the card was power-limited (300 W).
- One GPU model only.
- The project was built with Claude Code. Be ready to rewrite the dot product (section 5), the reduction (6 and 7) and the octet loop (9) on a whiteboard.
