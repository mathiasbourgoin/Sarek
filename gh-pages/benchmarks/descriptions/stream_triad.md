# STREAM Triad

**Category**: Memory Bandwidth  
**Optimization Level**: Industry Standard Baseline  
**Demonstrates**: Memory subsystem stress test, sustainable bandwidth

## Overview

**STREAM Triad** is the most demanding operation in the industry-standard STREAM benchmark suite. It has been used since 1991 to measure memory bandwidth on HPC systems worldwide.

The kernel performs: `A[i] = B[i] + C[i] * scalar`

This combines:
- 2 memory reads (B and C)
- 1 memory write (A)
- 1 FMA (fused multiply-add) operation

STREAM Triad is the **gold standard** for memory bandwidth benchmarking.

## STREAM Benchmark Suite

The complete STREAM benchmark has 4 operations:

| Operation | Formula | Bytes/Element | FLOPs/Element |
|-----------|---------|---------------|---------------|
| Copy | `C[i] = A[i]` | 8 | 0 |
| Scale | `B[i] = scalar * C[i]` | 8 | 1 |
| Add | `C[i] = A[i] + B[i]` | 12 | 1 |
| **Triad** | `A[i] = B[i] + C[i] * scalar` | **12** | **2** |

**Triad is the hardest** - it stresses both memory bandwidth and compute units.

## STREAM Triad Kernel

```ocaml
let stream_triad_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (scalar : float32)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then 
        a.(tid) <- b.(tid) +. (c.(tid) *. scalar)]
```

## Memory Traffic Analysis

For N elements (float32):

```
Reads:  2 vectors (B, C) = 2 × N × 4 bytes = 8N bytes
Writes: 1 vector (A)     = 1 × N × 4 bytes = 4N bytes
Total:                                        12N bytes

Bandwidth (GB/s) = 12N / (Time × 10⁹)
```

**Arithmetic Intensity**: 2 FLOPs / 12 bytes = 0.167 FLOPs/byte

This is **memory-bound** on all modern architectures (compute is negligible).

## Why STREAM Triad Matters

### 1. Industry Standard
- Used in Top500 supercomputer rankings
- Cited in thousands of academic papers
- Every major HPC vendor publishes STREAM results
- Enables apples-to-apples hardware comparisons

### 2. Sustainable Bandwidth
Unlike burst benchmarks, STREAM measures **sustained** memory bandwidth:
- Large arrays (don't fit in cache)
- Multiple iterations (averages over time)
- Real-world memory access patterns

### 3. System Bottleneck Identification
```
if (STREAM BW < 50% of theoretical peak):
    "Memory subsystem has issues"
    "Check: memory clocks, channel config, NUMA"
else if (STREAM BW ≈ 80-90% of peak):
    "Excellent - memory is well-utilized"
```

### 4. Power Efficiency Metric
```
Energy Efficiency = Bandwidth (GB/s) / Power (Watts)
                  = GB/s/W
```

Modern systems target **5-10 GB/s/W** for good efficiency.

## Performance Expectations

### High-End GPUs
- **NVIDIA A100** (1.6 TB/s HBM2): ~1400 GB/s (87% efficiency)
- **AMD MI250X** (3.2 TB/s HBM2e): ~2800 GB/s (87% efficiency)
- **NVIDIA H100** (3.35 TB/s HBM3): ~2900 GB/s (86% efficiency)

### Consumer GPUs
- **RTX 4090** (1008 GB/s GDDR6X): ~850 GB/s (84%)
- **RX 7900 XTX** (960 GB/s GDDR6): ~800 GB/s (83%)
- **Intel Arc A770** (560 GB/s GDDR6): ~450 GB/s (80%)

### CPUs
- **Dual AMD EPYC 9654** (12-channel DDR5): ~450 GB/s
- **Intel Xeon 8480+** (8-channel DDR5): ~320 GB/s
- **Apple M2 Ultra** (800 GB/s unified): ~650 GB/s

## Comparison with Vector Operations

| Benchmark | Reads | Writes | FLOPs | Bytes | BW Ratio |
|-----------|-------|--------|-------|-------|----------|
| Vector Copy | 1 | 1 | 0 | 8N | 1.00× |
| Vector Add | 2 | 1 | 1 | 12N | 0.83× |
| Vector Scale | 1 | 1 | 1 | 8N | 1.00× |
| **STREAM Triad** | **2** | **1** | **2** | **12N** | **0.83×** |

Triad and Add have identical memory traffic, but Triad has 2× FLOPs.

## Running STREAM Properly

### Array Size Requirements
```
Minimum Size = 4 × L3 Cache Size
```

For accurate results:
- **GPUs**: Use > 100M elements (400 MB)
- **CPUs**: Use > 4× largest cache (e.g., 256 MB L3 → 1 GB arrays)

This ensures data doesn't fit in cache.

### Iteration Count
- Minimum 10 iterations (20 recommended)
- Report **best** result (not average) per STREAM rules
- Modern practice: report median for stability

### Verification
Always verify results:
```
A[i] = B[i] + C[i] × scalar

With B[i] = 2.0, C[i] = 1.0, scalar = 3.0:
Expected A[i] = 2.0 + (1.0 × 3.0) = 5.0
```

## Interpreting Results

### Good Performance (>80% of peak)
Your memory subsystem is well-optimized:
- Coalesced memory accesses
- Full bus utilization
- Proper memory controller configuration

### Moderate Performance (50-80% of peak)
Room for improvement:
- Check memory clocks
- Optimize access patterns
- Consider memory affinity (NUMA)

### Poor Performance (<50% of peak)
Major issues:
- Non-coalesced accesses
- Bank conflicts
- Insufficient parallelism
- Hardware configuration problems

## STREAM vs Roofline Model

STREAM Triad sits at the **memory-bound region** of the Roofline:
- Arithmetic Intensity: 0.167 FLOPs/byte
- Performance: Limited by memory bandwidth, not compute

On a Roofline plot:
```
          │    /  Compute Bound
 GFLOPS   │   /   (flat roof)
          │  /
          │ /
          │/_____ Memory Bound (slope = BW)
          │        ← STREAM Triad here
          └────────────────────
               AI (FLOPs/byte)
```

---

<!-- GENERATED_CODE_TABS: stream_triad -->

## References

- McCalpin, John D., "STREAM: Sustainable Memory Bandwidth in High Performance Computers" (1995-present)
- https://www.cs.virginia.edu/stream/
- Standard Performance Evaluation Corporation (SPEC), STREAM documentation
- Top500.org - Uses STREAM for memory benchmarking
