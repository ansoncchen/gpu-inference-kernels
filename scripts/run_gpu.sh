#!/bin/bash
# Full GPU measurement. Run from the repo root after building into build/.
# Uses one GPU; set CUDA_VISIBLE_DEVICES first. Takes about 12 minutes.
set -euo pipefail
RAW=results/raw; mkdir -p $RAW; rm -f $RAW/gpu_*.jsonl $RAW/ggml*.jsonl
python3 scripts/env_info.py $RAW/env.json
./build/test_gpu > $RAW/test_gpu.txt; tail -1 $RAW/test_gpu.txt
SHAPES=1536x8960,8960x1536,4096x4096,11008x4096,8192x8192,32768x8192
STEPS='s*,a*,readbound,fp32*,fp16*'
for i in 1 2 3; do
  ./build/bench_gpu --label main$i --out $RAW/gpu_main.jsonl --shapes $SHAPES --variants "$STEPS,empty,fma,read*" > $RAW/gpu_main$i.txt
done
./build/bench_gpu --label tune --out $RAW/gpu_tune.jsonl --shapes $SHAPES --variants 't_loop*,t_oct*' > $RAW/gpu_tune.txt
# 4.7 MB matrix, one copy: lives in the L2, so DRAM is not the limit
./build/bench_gpu --label l2 --footprint 1 --out $RAW/gpu_l2.jsonl --shapes 2048x4096 --variants 's*,a*' > $RAW/gpu_l2.txt
if [ -x build/bench_ggml ]; then
  for i in 1 2 3; do ./build/bench_ggml $RAW/ggml.jsonl "${GPU_BUS_ID:-}" > $RAW/ggml$i.txt 2>/dev/null; done
  # same thing with ggml's CUDA graph capture off (plain launches, like ours)
  for i in 1 2 3; do
    GGML_CUDA_DISABLE_GRAPHS=1 ./build/bench_ggml $RAW/ggml_nograph.jsonl "${GPU_BUS_ID:-}" > $RAW/ggml_nograph$i.txt 2>/dev/null
  done
fi
