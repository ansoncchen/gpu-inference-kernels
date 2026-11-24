#!/bin/bash
# Real-model decode. Needs models/qwen2.5-1.5b, models/{prompt,wiki}.ids (decode/prepare.py),
# models/{f16,q4_0}.gguf and a built llama.cpp in third_party/. PY = python with torch + transformers.
set -euo pipefail
RAW=results/raw; M=models/qwen2.5-1.5b; rm -f $RAW/decode.jsonl $RAW/ppl.jsonl
for m in fp16 int4s int4sh; do ./build/decode $M $m logits models/prompt.ids models/l_$m.bin; done
$PY decode/check_logits.py models/ref_logits.bin models/l_{fp16,int4s,int4sh}.bin > $RAW/logits_check.txt
for m in fp16 int4s int4sh; do ./build/decode $M $m gen models/prompt.ids 256 10 $RAW/decode.jsonl > $RAW/decode_$m.txt; done
$PY decode/hf_decode.py $M models/prompt.ids 256 5 $RAW/decode.jsonl > /dev/null 2>&1
B=third_party/llama.cpp/build/bin
for m in f16 q4_0; do
  for fa in 0 1; do $B/llama-bench -m models/$m.gguf -ngl 99 -p 0 -n 256 -r 10 -fa $fa -o json 2>/dev/null > $RAW/llama_bench_${m}_fa$fa.json; done
  $B/llama-perplexity -m models/$m.gguf -ngl 99 -f models/wt/wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 2>&1 | grep -i "final estimate" > $RAW/llama_ppl_$m.txt
done
for m in fp16 int4 int4s int4sh; do ./build/decode $M $m ppl models/wiki.ids 512 40 $RAW/ppl.jsonl; done
