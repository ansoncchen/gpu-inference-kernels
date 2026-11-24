#pragma once
#include <string>
#include <vector>

#include "q4.h"

namespace q4 {
struct CpuVariant {
  std::string name;
  bool int8_x;  // quantizes x to int8 per group (extra error)
  void (*run)(const Matrix&, const float* x, float* y);
  const char* what;
};
const std::vector<CpuVariant>& cpu_variants();
// FP32 AVX2+FMA GEMV with the same threading, as an unquantized baseline.
void cpu_gemv_fp32(const float* a, int rows, int cols, const float* x, float* y);
// Sums a buffer with every OpenMP thread; returns a value to keep it alive.
double cpu_read(const float* p, size_t n);
}  // namespace q4
