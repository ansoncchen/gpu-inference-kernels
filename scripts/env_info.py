"""Records the hardware and software used, without host or user names."""
import json, re, subprocess, sys

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

cpu = re.search(r"model name\s*:\s*(.*)", open("/proc/cpuinfo").read()).group(1)
info = {
    "gpu": sh("nvidia-smi -i 0 --query-gpu=name --format=csv,noheader"),
    "driver": sh("nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader"),
    "power_limit_w": sh("nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader"),
    "max_sm_clock": sh("nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,noheader"),
    "max_mem_clock": sh("nvidia-smi -i 0 --query-gpu=clocks.max.mem --format=csv,noheader"),
    "nvcc": sh("nvcc --version | grep release"),
    "gxx": sh("/usr/bin/g++ --version | head -1"),
    "cpu": cpu,
    "cpu_threads": sh("nproc"),
    "kernel": sh("uname -sr"),
    "llama_cpp_commit": sh("git -C third_party/llama.cpp rev-parse HEAD") or None,
    "openblas": "0.3.34 (conda-forge, pthreads)",
    "theoretical_gpu_gbps": 616,
}
json.dump(info, open(sys.argv[1], "w"), indent=1)
