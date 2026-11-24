#!/bin/bash
# CPU measurement. Run from the repo root on an otherwise idle machine.
set -euo pipefail
RAW=results/raw; mkdir -p $RAW; rm -f $RAW/cpu.jsonl
./build/test_cpu > $RAW/test_cpu.txt; tail -1 $RAW/test_cpu.txt
OMP_PLACES=cores OMP_PROC_BIND=spread ./build/bench_cpu $RAW/cpu.jsonl 1,4,8,16,32 q4 > $RAW/cpu_q4.txt
OMP_WAIT_POLICY=passive ./build/bench_cpu $RAW/cpu.jsonl 8,16,32 blas > $RAW/cpu_blas.txt
