#!/usr/bin/env python3
"""
Benchmark CPU embedding models for ROCm-RAG.

Compares throughput, latency, memory, and quality of three candidate models
on CPU (i9-12th gen) to pick the best embedding model for the pipeline.

Usage:
    python tests/benchmark_embeddings.py
"""

import gc
import os
import time
import traceback

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "BAAI/bge-large-en-v1.5": {
        "dims": 1024,
        "notes": "335M params, fp32 baseline",
    },
    "Intel/bge-large-en-v1.5-rag-int8-static": {
        "dims": 1024,
        "notes": "335M base, int8 quantized + IPEX optimized",
        "ipex": True,
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "dims": 768,
        "notes": "137M params, Matryoshka dims",
        "trust_remote_code": True,
        "prompt_name": "document",
        "query_prompt": "query",
    },
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "dims": 1024,
        "notes": "303M params, high MTEB scores",
        "prompt_name": "document",
        "query_prompt": "query",
    },
}

# Force CPU even if ROCm GPU is visible
DEVICE = "cpu"

# Number of warmup + timed runs
WARMUP_RUNS = 2
TIMED_RUNS = 5

# ---------------------------------------------------------------------------
# Test corpus — ROCm documentation snippets at varying lengths
# ---------------------------------------------------------------------------

SHORT_TEXTS = [
    "ROCm is AMD's open-source GPU computing platform.",
    "HIP allows developers to write portable GPU code.",
    "The rocBLAS library provides BLAS operations on AMD GPUs.",
    "AMD Instinct MI300X accelerators support FP8 training.",
    "rocSPARSE provides sparse matrix operations for ROCm.",
]

MEDIUM_TEXTS = [
    (
        "ROCm (Radeon Open Compute) is AMD's open-source software platform for "
        "GPU computing. It includes compilers, libraries, and runtime components "
        "that enable developers to build and run HPC and AI workloads on AMD GPUs. "
        "ROCm supports multiple programming models including HIP, OpenCL, and "
        "OpenMP offloading. The platform is designed to be compatible with CUDA "
        "through the HIP portability layer."
    ),
    (
        "The HIP programming model allows developers to write GPU kernels that "
        "can run on both AMD and NVIDIA hardware. HIP provides a C++ runtime API "
        "and kernel language that is similar to CUDA. The hipify tools can "
        "automatically convert CUDA source code to HIP, making it easier to port "
        "existing CUDA applications to AMD GPUs. HIP supports features like "
        "shared memory, warp-level primitives, and cooperative groups."
    ),
    (
        "AMD Instinct MI300X is a data center GPU accelerator designed for AI "
        "and HPC workloads. It features 192GB of HBM3 memory with 5.3 TB/s "
        "bandwidth. The MI300X supports FP8, FP16, BF16, and FP32 data types "
        "for deep learning training and inference. Multiple MI300X GPUs can be "
        "connected via Infinity Fabric for scale-out training across nodes."
    ),
]

LONG_TEXTS = [
    (
        "ROCm Installation Guide for Linux\n\n"
        "Prerequisites: ROCm supports Ubuntu 22.04, Ubuntu 24.04, RHEL 9, and "
        "SLES 15 SP5. Your system must have a supported AMD GPU (Instinct MI series "
        "or Radeon Pro/RX 7000 series). Ensure your kernel version is compatible "
        "and that you have the latest firmware installed.\n\n"
        "Step 1: Add the ROCm repository. Download and install the amdgpu-install "
        "package from the AMD ROCm repository. This package manages the repository "
        "configuration and GPG keys.\n\n"
        "Step 2: Install the ROCm meta-packages. Use 'amdgpu-install --usecase=rocm' "
        "to install the complete ROCm stack including the kernel driver, runtime, "
        "compilers, and math libraries. For development, add '--usecase=rocm,hip'.\n\n"
        "Step 3: Configure user permissions. Add your user to the 'render' and "
        "'video' groups to allow GPU access without root privileges.\n\n"
        "Step 4: Verify the installation. Run 'rocminfo' to list detected GPUs "
        "and 'rocm-smi' to check GPU status and temperatures. Run the 'hipcc' "
        "compiler on a simple test program to verify the toolchain works correctly.\n\n"
        "Troubleshooting: If rocminfo shows no agents, check that the amdgpu "
        "kernel driver is loaded with 'lsmod | grep amdgpu'. Ensure IOMMU is "
        "enabled in BIOS for proper PCIe passthrough support. Check dmesg for "
        "any firmware loading errors."
    ),
    (
        "rocBLAS Performance Tuning Guide\n\n"
        "rocBLAS is AMD's GPU-accelerated BLAS library optimized for AMD Instinct "
        "and Radeon GPUs. It provides Level 1, 2, and 3 BLAS routines including "
        "GEMM, GEMV, and AXPY operations.\n\n"
        "Tuning GEMM Performance:\n"
        "Matrix multiplication (GEMM) is the most critical operation for deep "
        "learning. rocBLAS uses a library of pre-tuned kernels selected based on "
        "matrix dimensions, data types, and GPU architecture. For optimal performance:\n"
        "- Pad matrices to multiples of 256 for MI200/MI300 series\n"
        "- Use FP16 or BF16 for training workloads (2x throughput vs FP32)\n"
        "- Enable FP8 for inference on MI300X (4x throughput vs FP16)\n"
        "- Set ROCBLAS_LAYER=2 to enable performance logging\n\n"
        "Batched Operations:\n"
        "For small matrix sizes common in transformer attention layers, use "
        "batched GEMM (rocblas_gemm_strided_batched) to amortize kernel launch "
        "overhead. Group multiple small GEMMs into a single batched call.\n\n"
        "Memory Layout:\n"
        "Column-major layout is the default and generally optimal for rocBLAS. "
        "For row-major data (common in PyTorch), use the transpose flags rather "
        "than explicitly transposing the data. This avoids unnecessary memory copies."
    ),
]

# Quality check pairs: (text_a, text_b, expected_similarity)
# "high" = should be very similar, "low" = should be dissimilar
QUALITY_PAIRS = [
    (
        "ROCm is AMD's GPU computing platform for HPC and AI.",
        "AMD ROCm provides an open-source software stack for GPU-accelerated computing.",
        "high",
    ),
    (
        "HIP allows writing portable GPU code across AMD and NVIDIA hardware.",
        "The HIP programming model enables cross-platform GPU development.",
        "high",
    ),
    (
        "ROCm installation requires adding the AMD repository and installing drivers.",
        "The rocBLAS library provides optimized matrix multiplication on AMD GPUs.",
        "low",
    ),
    (
        "AMD Instinct MI300X has 192GB HBM3 memory for AI training.",
        "Python virtual environments isolate package dependencies per project.",
        "low",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_memory_mb():
    """Current RSS of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def benchmark_model(name: str, cfg: dict) -> dict | None:
    """Load a model, run all benchmarks, return results dict."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {cfg['notes']}")
    print(f"{'='*60}")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_before = get_memory_mb()

    # --- Load model ---
    print("  Loading model...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        if cfg.get("ipex"):
            # Intel int8 TorchScript model — load via optimum-intel or jit
            if not HAS_IPEX:
                print("SKIPPED (intel-extension-for-pytorch not installed)")
                return None
            # Load the base model, then apply IPEX optimization
            base_name = "BAAI/bge-large-en-v1.5"
            model = SentenceTransformer(base_name, device=DEVICE)
            # Apply IPEX optimization to the transformer module
            model[0].auto_model = ipex.optimize(model[0].auto_model, dtype=torch.float32)
        else:
            st_kwargs = {"device": DEVICE}
            if cfg.get("trust_remote_code"):
                st_kwargs["trust_remote_code"] = True
            model = SentenceTransformer(name, **st_kwargs)
    except Exception:
        print(f"FAILED\n{traceback.format_exc()}")
        return None
    load_time = time.perf_counter() - t0
    mem_after_load = get_memory_mb()
    print(f"done in {load_time:.1f}s  (RAM: {mem_after_load:.0f} MB, delta: {mem_after_load - mem_before:.0f} MB)")

    results = {
        "load_time_s": load_time,
        "ram_delta_mb": mem_after_load - mem_before,
        "ram_peak_mb": mem_after_load,
    }

    # --- Helper to encode with the right prompts ---
    def encode_docs(texts):
        kwargs = {}
        if "prompt_name" in cfg:
            kwargs["prompt_name"] = cfg["prompt_name"]
        return model.encode(texts, **kwargs, convert_to_numpy=True, show_progress_bar=False)

    def encode_queries(texts):
        kwargs = {}
        if "query_prompt" in cfg:
            kwargs["prompt_name"] = cfg["query_prompt"]
        return model.encode(texts, **kwargs, convert_to_numpy=True, show_progress_bar=False)

    all_docs = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS
    n_docs = len(all_docs)

    try:
        # --- Warmup ---
        print("  Warming up...", end=" ", flush=True)
        for _ in range(WARMUP_RUNS):
            encode_docs(all_docs)
        print("done")

        # --- Throughput (batch) ---
        print("  Measuring throughput...", end=" ", flush=True)
        times = []
        for _ in range(TIMED_RUNS):
            t0 = time.perf_counter()
            encode_docs(all_docs)
            times.append(time.perf_counter() - t0)
        avg_batch = np.mean(times)
        results["throughput_docs_per_sec"] = n_docs / avg_batch
        results["batch_time_s"] = avg_batch
        print(f"{results['throughput_docs_per_sec']:.1f} docs/s  ({avg_batch:.3f}s for {n_docs} docs)")

        # --- Single-query latency ---
        print("  Measuring single-query latency...", end=" ", flush=True)
        query = "How do I install ROCm on Ubuntu?"
        # warmup
        for _ in range(WARMUP_RUNS):
            encode_queries([query])
        latencies = []
        for _ in range(TIMED_RUNS * 3):
            t0 = time.perf_counter()
            encode_queries([query])
            latencies.append(time.perf_counter() - t0)
        results["query_latency_ms"] = np.mean(latencies) * 1000
        results["query_latency_p95_ms"] = np.percentile(latencies, 95) * 1000
        print(f"mean={results['query_latency_ms']:.1f}ms  p95={results['query_latency_p95_ms']:.1f}ms")

        # --- Peak memory under load ---
        encode_docs(all_docs * 5)  # larger batch
        results["ram_peak_mb"] = get_memory_mb()
        print(f"  Peak RAM (under load): {results['ram_peak_mb']:.0f} MB")

        # --- Quality: cosine similarity on known pairs ---
        print("  Evaluating quality (cosine sim on known pairs)...")
        quality_scores = []
        for text_a, text_b, expected in QUALITY_PAIRS:
            emb_a = encode_queries([text_a])[0]
            emb_b = encode_queries([text_b])[0]
            sim = cosine_sim(emb_a, emb_b)
            label = "HIGH" if expected == "high" else " low"
            ok = (expected == "high" and sim > 0.7) or (expected == "low" and sim < 0.7)
            marker = "OK" if ok else "!!"
            quality_scores.append((expected, sim, ok))
            print(f"    [{marker}] {label} expected | sim={sim:.4f} | {text_a[:50]}...")
        results["quality_pairs"] = quality_scores
    except Exception:
        print(f"FAILED during benchmark\n{traceback.format_exc()}")
        del model
        gc.collect()
        return None

    # cleanup
    del model
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CPU Embedding Model Benchmark")
    print(f"  Device: {DEVICE}")
    print(f"  CPU: {psutil.cpu_count(logical=True)} logical cores")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Torch threads: {torch.get_num_threads()}")
    print(f"  IPEX available: {HAS_IPEX}")
    print("=" * 60)

    all_results = {}
    for name, cfg in MODELS.items():
        result = benchmark_model(name, cfg)
        if result is not None:
            all_results[name] = result

    # --- Summary table ---
    print("\n\n")
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    # Header
    col_w = 42
    print(f"{'Metric':<30}", end="")
    short_names = []
    for name in all_results:
        short = name.split("/")[-1][:col_w]
        short_names.append(short)
        print(f"  {short:>{col_w}}", end="")
    print()
    print("-" * (30 + (col_w + 2) * len(all_results)))

    def row(label, key, fmt=".1f", suffix=""):
        print(f"{label:<30}", end="")
        for r in all_results.values():
            val = r.get(key, float("nan"))
            print(f"  {f'{val:{fmt}}{suffix}':>{col_w}}", end="")
        print()

    row("Load time", "load_time_s", ".1f", "s")
    row("RAM delta (model)", "ram_delta_mb", ".0f", " MB")
    row("RAM peak (under load)", "ram_peak_mb", ".0f", " MB")
    row("Throughput", "throughput_docs_per_sec", ".1f", " docs/s")
    row("Batch time (10 docs)", "batch_time_s", ".3f", "s")
    row("Query latency (mean)", "query_latency_ms", ".1f", "ms")
    row("Query latency (p95)", "query_latency_p95_ms", ".1f", "ms")

    # Quality summary
    print(f"{'Quality (correct/total)':<30}", end="")
    for r in all_results.values():
        pairs = r.get("quality_pairs", [])
        correct = sum(1 for _, _, ok in pairs if ok)
        total = len(pairs)
        print(f"  {f'{correct}/{total}':>{col_w}}", end="")
    print()

    print("-" * (30 + (col_w + 2) * len(all_results)))
    print()


if __name__ == "__main__":
    main()
