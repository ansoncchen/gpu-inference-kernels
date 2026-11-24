#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

inline double quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double at = q * (v.size() - 1);
  size_t i = (size_t)at;
  return i + 1 < v.size() ? v[i] + (at - i) * (v[i + 1] - v[i]) : v[i];
}
inline std::string stats_json(const std::vector<double>& v) {
  char b[256];
  snprintf(b, sizeof b, "\"reps\":%zu,\"median_us\":%.4f,\"min_us\":%.4f,\"p5_us\":%.4f,\"p95_us\":%.4f,\"max_us\":%.4f",
           v.size(), quantile(v, 0.5), quantile(v, 0), quantile(v, 0.05), quantile(v, 0.95), quantile(v, 1));
  std::string s = b;
  s += ",\"samples_us\":[";
  for (size_t i = 0; i < v.size(); i++) {
    snprintf(b, sizeof b, "%s%.4f", i ? "," : "", v[i]);
    s += b;
  }
  return s + "]";
}
