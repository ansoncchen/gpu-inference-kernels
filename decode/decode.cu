// Token-by-token decode of a Qwen2-architecture model (Qwen2.5-0.5B / 1.5B),
// with the linear layers done either in FP16 by cuBLAS or in INT4 by the s8
// kernel. Everything else (norms, RoPE, attention, activation) is shared, so the
// difference between the two modes is the GEMV.
//
// int4 = largest weight of each group maps to -8 (ggml Q4_0 rule); int4s = scale
// searched per group for the smallest squared error. Same format, same speed.
// int4sh = int4s, but the output head stays FP16.
// decode <model dir> <fp16|int4|int4s|int4sh> gen <ids file> <n new> <repeats> [out.jsonl]
// decode <model dir> <fp16|int4> ppl <ids file> <window> <n windows> [out.jsonl]
// decode <model dir> <fp16|int4> logits <ids file> <out.bin>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include "../bench/stats.h"
#include "gpu.h"
#include "safetensors.h"

using namespace q4;
using Clock = std::chrono::steady_clock;

namespace {

struct Config {
  int hidden, layers, heads, kv_heads, ffn, vocab;
  int head_dim() const { return hidden / heads; }
};

// --------------------------------------------------------------- kernels
__global__ void k_embed(const __half* table, int token, int n, float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __half2float(table[(size_t)token * n + i]);
}

// x += add (if given), then out = RMSNorm(x). Writes FP32 and FP16 copies. One block.
__global__ void k_add_norm(float* x, const float* add, const float* w, int n, float eps, float* out, __half* outh) {
  __shared__ float part[16];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float acc = 0;
  for (int i = threadIdx.x; i < n; i += 512) {
    float v = x[i];
    if (add) x[i] = v = v + add[i];
    acc += v * v;
  }
  for (int d = 16; d; d >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
  if (lane == 0) part[warp] = acc;
  __syncthreads();
  float s = 0;
  for (int i = 0; i < 16; i++) s += part[i];
  float inv = rsqrtf(s / n + eps);
  for (int i = threadIdx.x; i < n; i += 512) {
    float v = x[i] * inv * w[i];
    out[i] = v;
    outh[i] = __float2half(v);
  }
}

// Attention for the newest token, one block per head. qkv holds q, then k, then
// v for this token (before bias). The cache is [kv head][pos][dim] FP32. A warp
// handles one position at a time with 4 dims per lane.
constexpr int kMaxCtx = 2048;
template <int HD>
__global__ void k_attn(const float* qkv, const float* bias, float* cache_k, float* cache_v, int hidden, int kv_heads,
                       int heads, int pos, float* out, __half* outh) {
  constexpr int L4 = HD / 4;  // lanes used (HD 128 -> 32, HD 64 -> 16)
  __shared__ float score[kMaxCtx];
  __shared__ float4 wsum[16 * 32];
  __shared__ float red[16];
  __shared__ float qs[HD], ks[HD], vs[HD];
  int head = blockIdx.x, kvh = head / (heads / kv_heads);
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int kbase = hidden + kvh * HD, vbase = hidden + kv_heads * HD + kvh * HD;
  if (threadIdx.x < HD) {
    int i = threadIdx.x, half = HD / 2, j = i % half;
    float th = pos * powf(1.0e6f, -(float)j / half), c = cosf(th), s = sinf(th);
    auto rot = [&](int base) {
      float a = qkv[base + j] + bias[base + j], b = qkv[base + j + half] + bias[base + j + half];
      return i < half ? a * c - b * s : b * c + a * s;
    };
    qs[i] = rot(head * HD);
    ks[i] = rot(kbase);
    vs[i] = qkv[vbase + i] + bias[vbase + i];
    if (head % (heads / kv_heads) == 0) {
      cache_k[((size_t)kvh * kMaxCtx + pos) * HD + i] = ks[i];
      cache_v[((size_t)kvh * kMaxCtx + pos) * HD + i] = vs[i];
    }
  }
  __syncthreads();
  bool on = lane < L4;
  float4 q = on ? ((const float4*)qs)[lane] : make_float4(0, 0, 0, 0);
  const float4* K = (const float4*)(cache_k + (size_t)kvh * kMaxCtx * HD);
  const float4* V = (const float4*)(cache_v + (size_t)kvh * kMaxCtx * HD);
  float scale = 1.0f / sqrtf((float)HD), mx = -1e30f;
  for (int p = warp; p <= pos; p += 16) {
    float d = 0;
    if (on) {
      float4 k = p == pos ? ((const float4*)ks)[lane] : K[p * L4 + lane];
      d = k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w;
    }
    for (int s = 16; s; s >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, s);
    d *= scale;
    if (lane == 0) score[p] = d;
    mx = fmaxf(mx, d);
  }
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  for (int i = 0; i < 16; i++) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float total = 0;
  float4 acc = make_float4(0, 0, 0, 0);
  for (int p = warp; p <= pos; p += 16) {
    float e = expf(score[p] - mx);
    total += e;
    if (on) {
      float4 v = p == pos ? ((const float4*)vs)[lane] : V[p * L4 + lane];
      acc.x += e * v.x, acc.y += e * v.y, acc.z += e * v.z, acc.w += e * v.w;
    }
  }
  if (lane == 0) red[warp] = total;
  if (on) wsum[warp * 32 + lane] = acc;
  __syncthreads();
  if (threadIdx.x < L4) {
    float t = 0;
    float4 s = make_float4(0, 0, 0, 0);
    for (int i = 0; i < 16; i++) {
      t += red[i];
      float4 v = wsum[i * 32 + lane];
      s.x += v.x, s.y += v.y, s.z += v.z, s.w += v.w;
    }
    t = 1.0f / t;
    float o[4] = {s.x * t, s.y * t, s.z * t, s.w * t};
    for (int j = 0; j < 4; j++) {
      out[head * HD + lane * 4 + j] = o[j];
      outh[head * HD + lane * 4 + j] = __float2half(o[j]);
    }
  }
}

// out = silu(gate) * up, with gate and up stacked in gu.
__global__ void k_act(const float* gu, int n, float* out, __half* outh) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gu[i], v = g / (1.0f + expf(-g)) * gu[n + i];
  out[i] = v;
  outh[i] = __float2half(v);
}

// Best logit of each 1024-entry slice; the host picks the winner of ~150 slices.
struct Best {
  float v;
  int i;
};
__global__ void k_argmax(const float* logits, int n, Best* best) {
  __shared__ Best part[16];
  int lo = blockIdx.x * 1024, lane = threadIdx.x % 32;
  Best b = {-1e30f, 0};
  for (int i = lo + threadIdx.x; i < lo + 1024 && i < n; i += 512)
    if (logits[i] > b.v) b = {logits[i], i};
  for (int d = 16; d; d >>= 1) {
    float v = __shfl_xor_sync(0xFFFFFFFFu, b.v, d);
    int j = __shfl_xor_sync(0xFFFFFFFFu, b.i, d);
    if (v > b.v || (v == b.v && j < b.i)) b = {v, j};
  }
  if (lane == 0) part[threadIdx.x / 32] = b;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < 16; i++)
      if (part[i].v > b.v || (part[i].v == b.v && part[i].i < b.i)) b = part[i];
    best[blockIdx.x] = b;
  }
}

// ---------------------------------------------------------------- engine
struct Linear {
  int rows = 0, cols = 0;
  GpuMat q4m;
  __half* fp16 = nullptr;
};

struct Engine {
  Config c;
  bool int4, search, fp16_head;
  cudaStream_t st;
  cublasHandle_t h;
  void (*gemv_q4)(const GpuMat&, const float*, float*, float*, cudaStream_t) = nullptr;
  size_t weight_bytes = 0;  // linear-layer weights read per token

  struct Layer {
    Linear qkv, o, gu, down;
    float *bias, *n1, *n2, *ck, *cv;
  };
  std::vector<Layer> layers;
  Linear head;
  __half* embed;
  float *fnorm, *x, *nx, *qkv_out, *attn_out, *o_out, *gu_out, *act, *down_out, *logits;
  __half *nxh, *attn_h, *act_h;
  Best *best, *best_d;

  template <class T>
  static T* alloc(size_t n) {
    T* p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
  }
  template <class T>
  static T* up(const std::vector<T>& v) {
    T* p = alloc<T>(v.size());
    CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }

  Linear make(const std::vector<float>& w, int cols, bool q4 = true) {
    Linear l;
    l.cols = cols, l.rows = (int)(w.size() / cols);
    if (int4 && q4) {
      Matrix m = quantize(w.data(), l.rows, cols, search);
      Octet o = to_octet(m);
      l.q4m.rows = l.rows, l.q4m.cols = cols, l.q4m.octets = o.octets, l.q4m.chunks = o.chunks;
      l.q4m.ow = up(o.w), l.q4m.oscale = up(o.scale);
      weight_bytes += m.bytes();
    } else {
      std::vector<uint16_t> hw(w.size());
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < w.size(); i++) hw[i] = float_to_half(w[i]);
      l.fp16 = (__half*)up(hw);
      weight_bytes += w.size() * 2;
    }
    return l;
  }

  Engine(const std::string& dir, bool int4_, bool search_, bool head_) : int4(int4_), search(search_), fp16_head(head_) {
    std::ifstream f(dir + "/config.json");
    std::string j((std::istreambuf_iterator<char>(f)), {});
    auto num = [&](const char* k) { return atoi(j.c_str() + j.find(':', j.find(std::string("\"") + k + "\"")) + 1); };
    c = {num("hidden_size"), num("num_hidden_layers"), num("num_attention_heads"), num("num_key_value_heads"),
         num("intermediate_size"), num("vocab_size")};
    for (const Variant& v : variants())
      if (v.name == "s8_octet") gemv_q4 = v.launch;
    CUDA_CHECK(cudaStreamCreate(&st));
    cublasCreate(&h);
    cublasSetStream(h, st);
    SafeTensors s(dir + "/model.safetensors");
    auto cat = [](std::vector<float> a, const std::vector<float>& b) {
      a.insert(a.end(), b.begin(), b.end());
      return a;
    };
    for (int l = 0; l < c.layers; l++) {
      std::string p = "model.layers." + std::to_string(l) + ".";
      auto g = [&](const char* n) { return s.get(p + n); };
      Layer y;
      // q, k and v (and gate and up) share their input, so they are stacked into
      // one matrix: one GEMV instead of three (two), in both modes.
      y.qkv = make(cat(cat(g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight")), g("self_attn.v_proj.weight")), c.hidden);
      y.bias = up(cat(cat(g("self_attn.q_proj.bias"), g("self_attn.k_proj.bias")), g("self_attn.v_proj.bias")));
      y.o = make(g("self_attn.o_proj.weight"), c.hidden);
      y.gu = make(cat(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")), c.hidden);
      y.down = make(g("mlp.down_proj.weight"), c.ffn);
      y.n1 = up(g("input_layernorm.weight"));
      y.n2 = up(g("post_attention_layernorm.weight"));
      y.ck = alloc<float>((size_t)c.kv_heads * kMaxCtx * c.head_dim());
      y.cv = alloc<float>((size_t)c.kv_heads * kMaxCtx * c.head_dim());
      layers.push_back(y);
    }
    fnorm = up(s.get("model.norm.weight"));
    auto e = s.get("model.embed_tokens.weight");
    std::vector<uint16_t> eh(e.size());
    for (size_t i = 0; i < e.size(); i++) eh[i] = float_to_half(e[i]);
    embed = (__half*)up(eh);
    head = make(e, c.hidden, !fp16_head);  // tied embeddings
    x = alloc<float>(c.hidden), nx = alloc<float>(c.hidden), nxh = alloc<__half>(c.hidden);
    qkv_out = alloc<float>(c.hidden * 3), attn_out = alloc<float>(c.hidden), attn_h = alloc<__half>(c.hidden);
    o_out = alloc<float>(c.hidden), gu_out = alloc<float>(c.ffn * 2);
    act = alloc<float>(c.ffn), act_h = alloc<__half>(c.ffn), down_out = alloc<float>(c.hidden);
    logits = alloc<float>(c.vocab + 32);
    best_d = alloc<Best>(c.vocab / 1024 + 1);
    CUDA_CHECK(cudaMallocHost(&best, sizeof(Best) * (c.vocab / 1024 + 1)));
  }

  void apply(const Linear& l, const float* in, const __half* inh, float* out) {
    if (!l.fp16) {
      gemv_q4(l.q4m, in, out, nullptr, st);
    } else {
      const float one = 1, zero = 0;
      auto r = cublasHSSgemvStridedBatched(h, CUBLAS_OP_T, l.cols, l.rows, &one, l.fp16, l.cols, (long long)l.cols * l.rows,
                                           inh, 1, l.cols, &zero, out, 1, l.rows, 1);
      if (r != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas %d\n", (int)r); exit(1); }
    }
  }

  // Queues one decode step; afterwards `logits` holds the next-token scores.
  void step(int token, int pos) {
    const float eps = 1e-6f;
    k_embed<<<(c.hidden + 255) / 256, 256, 0, st>>>(embed, token, c.hidden, x);
    const float* add = nullptr;
    for (Layer& l : layers) {
      k_add_norm<<<1, 512, 0, st>>>(x, add, l.n1, c.hidden, eps, nx, nxh);
      apply(l.qkv, nx, nxh, qkv_out);
      if (c.head_dim() == 128)
        k_attn<128><<<c.heads, 512, 0, st>>>(qkv_out, l.bias, l.ck, l.cv, c.hidden, c.kv_heads, c.heads, pos, attn_out, attn_h);
      else
        k_attn<64><<<c.heads, 512, 0, st>>>(qkv_out, l.bias, l.ck, l.cv, c.hidden, c.kv_heads, c.heads, pos, attn_out, attn_h);
      apply(l.o, attn_out, attn_h, o_out);
      k_add_norm<<<1, 512, 0, st>>>(x, o_out, l.n2, c.hidden, eps, nx, nxh);
      apply(l.gu, nx, nxh, gu_out);
      k_act<<<(c.ffn + 255) / 256, 256, 0, st>>>(gu_out, c.ffn, act, act_h);
      apply(l.down, act, act_h, down_out);
      add = down_out;
    }
    k_add_norm<<<1, 512, 0, st>>>(x, add, fnorm, c.hidden, eps, nx, nxh);
    apply(head, nx, nxh, logits);
  }

  int argmax() {
    int slices = (c.vocab + 1023) / 1024;
    k_argmax<<<slices, 512, 0, st>>>(logits, c.vocab, best_d);
    CUDA_CHECK(cudaMemcpyAsync(best, best_d, sizeof(Best) * slices, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    Best b = best[0];
    for (int i = 1; i < slices; i++)
      if (best[i].v > b.v) b = best[i];
    return b.i;
  }
  std::vector<float> host_logits() {
    std::vector<float> v(c.vocab);
    CUDA_CHECK(cudaStreamSynchronize(st));
    CUDA_CHECK(cudaMemcpy(v.data(), logits, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
  }
};

std::vector<int> read_ids(const char* path) {
  std::ifstream f(path);
  std::vector<int> v;
  for (int t; f >> t;) v.push_back(t);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) return 1;
  std::string mode = argv[2], cmd = argv[3];
  Engine e(argv[1], mode.rfind("int4", 0) == 0, mode.find('s') != std::string::npos, mode.find('h') != std::string::npos);
  auto ids = read_ids(argv[4]);
  if (cmd == "gen") {
    int n_new = atoi(argv[5]), reps = atoi(argv[6]);
    std::vector<double> us;
    std::vector<int> gen;
    for (int r = 0; r < reps + 1; r++) {  // first run is warm-up
      gen.clear();
      for (size_t i = 0; i < ids.size(); i++) e.step(ids[i], (int)i);
      int tok = e.argmax(), pos = (int)ids.size();
      auto t0 = Clock::now();
      for (int i = 0; i < n_new; i++) {
        gen.push_back(tok);
        e.step(tok, pos++);
        tok = e.argmax();
      }
      double s = std::chrono::duration<double>(Clock::now() - t0).count();
      if (r) us.push_back(s * 1e6 / n_new);
    }
    CUDA_CHECK(cudaGetLastError());
    double med = quantile(us, 0.5);
    printf("%s: %.1f tok/s (%.1f us/token, min %.1f max %.1f), weights read per token %.3f GB\n", mode.c_str(), 1e6 / med, med,
           quantile(us, 0), quantile(us, 1), e.weight_bytes / 1e9);
    printf("ids:");
    for (int t : gen) printf(" %d", t);
    printf("\n");
    if (argc > 7) {
      std::ofstream f(argv[7], std::ios::app);
      f << "{\"run\":\"decode\",\"mode\":\"" << mode << "\",\"hidden\":" << e.c.hidden << ",\"layers\":" << e.c.layers
        << ",\"prompt_tokens\":" << ids.size() << ",\"new_tokens\":" << n_new << ",\"weight_bytes\":" << e.weight_bytes
        << ",\"tok_per_s\":" << 1e6 / med << "," << stats_json(us) << "}\n";
    }
  } else if (cmd == "ppl") {
    int win = atoi(argv[5]), n = atoi(argv[6]);
    double nll = 0;
    long count = 0;
    for (int w = 0; w < n && (size_t)(w + 1) * win <= ids.size(); w++) {
      for (int i = 0; i + 1 < win; i++) {
        e.step(ids[w * win + i], i);
        // Same protocol as llama-perplexity: only the second half of each window counts.
        if (i < win / 2) continue;
        auto l = e.host_logits();
        double mx = -1e30, sum = 0;
        for (float v : l) mx = std::max(mx, (double)v);
        for (float v : l) sum += std::exp(v - mx);
        nll += -(l[ids[w * win + i + 1]] - mx - std::log(sum));
        count++;
      }
    }
    printf("%s: ppl %.4f over %ld tokens\n", mode.c_str(), std::exp(nll / count), count);
    if (argc > 7) {
      std::ofstream f(argv[7], std::ios::app);
      f << "{\"run\":\"ppl\",\"mode\":\"" << mode << "\",\"window\":" << win << ",\"tokens\":" << count
        << ",\"ppl\":" << std::exp(nll / count) << "}\n";
    }
  } else if (cmd == "logits") {
    std::ofstream f(argv[5], std::ios::binary);
    for (size_t i = 0; i < ids.size(); i++) {
      e.step(ids[i], (int)i);
      auto l = e.host_logits();
      f.write((const char*)l.data(), l.size() * 4);
    }
  }
}
