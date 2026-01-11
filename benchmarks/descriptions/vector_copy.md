# Vector Copy

**Category**: Memory Bandwidth  
**Optimization Level**: Baseline  
**Demonstrates**: Pure memory bandwidth measurement, minimal compute overhead

## Overview

Vector Copy is the **simplest possible GPU kernel** - it just copies data from one array to another. This benchmark serves as a **baseline for memory bandwidth** measurement, with virtually no computational overhead.

## Why Measure Vector Copy?

Understanding pure memory bandwidth is crucial for:
- **Identifying memory-bound kernels** - Compare against peak bandwidth
- **Baseline for optimization** - How much is spent on memory vs compute?
- **Hardware comparison** - PCIe, memory clock speeds, bus widths
- **Validation** - Sanity check for more complex benchmarks

## Algorithm

The kernel is trivial:

```
for each element i in parallel:
    B[i] = A[i]
```

**Memory Operations**:
- 1 read from global memory (array A)
- 1 write to global memory (array B)
- **Total**: 2 memory accesses per element

**Arithmetic Operations**: 0  
**Arithmetic Intensity**: 0 FLOPs/byte (pure memory-bound)

## Vector Copy Kernel

```ocaml
let vector_copy_kernel =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then 
        b.(tid) <- a.(tid)]
```

That's it! One line of code.

## Performance Characteristics

**Memory Pattern**: Coalesced sequential access (optimal)  
**Parallelism**: O(N) threads  
**Bottleneck**: **Memory bandwidth** (100%)  
**Compute**: Negligible

**Expected Performance**:
- **High-end GPU**: 80-90% of peak memory bandwidth
- **Mid-range GPU**: 70-85% of peak
- **CPU**: Limited by memory bus (40-60 GB/s typical)

### Why Not 100% Efficiency?

Real-world factors limiting bandwidth:
- Kernel launch overhead
- Cache warming effects
- Bus arbitration
- Minor compute for address calculation

Still, 80-90% is excellent for production code.

## Bandwidth Calculation

For a vector copy of N elements (float32):

```
Bytes Transferred = 2 × N × 4 bytes
                  = 8N bytes
                  (1 read + 1 write, 4 bytes per float)

Bandwidth (GB/s) = Bytes Transferred / Time(seconds)
                 = 8N / (Time × 10⁹)
```

## Comparison with Vector Add

| Benchmark | Memory Ops | Compute Ops | AI (FLOPs/byte) |
|-----------|------------|-------------|-----------------|
| **Vector Copy** | 2 | 0 | 0.0 |
| **Vector Add** | 3 | 1 | 0.125 |

Vector Add has 50% more memory traffic plus one floating-point addition. Comparing the two shows the cost of that single ADD operation.

**Typical Results**:
- Vector Copy: ~150 GB/s
- Vector Add: ~140 GB/s (slight overhead from ADD)

This shows that Vector Add is also memory-bound, not compute-bound.

## Use Cases

### 1. Bandwidth Measurement
```
Peak Bandwidth = median(Vector Copy measurements)
Efficiency = (Actual BW) / (Theoretical Peak BW) × 100%
```

### 2. Identifying Bottlenecks
```
if (Kernel BW ≈ Vector Copy BW):
    "Your kernel is memory-bound"
else if (Kernel BW < 50% of Vector Copy BW):
    "Look for memory access issues (non-coalesced, bank conflicts)"
```

### 3. Cache Analysis
Run with increasing sizes:
- Small sizes: Higher BW (L1/L2 cache hits)
- Large sizes: Lower BW (main memory bound)
- Plateau size: Total cache size

### 4. Transfer Overhead
Compare:
- GPU → GPU copy (this benchmark)
- CPU → GPU copy (PCIe limited, ~12 GB/s for PCIe 3.0 x16)

## Example Results

**NVIDIA RTX 3090** (936 GB/s theoretical):
- Vector Copy: 850 GB/s (91% efficiency)
- Interpretation: Excellent - near-optimal memory access

**AMD Radeon RX 6800 XT** (512 GB/s theoretical):
- Vector Copy: 450 GB/s (88% efficiency)
- Interpretation: Excellent - hardware limitations respected

**Intel Arc A770** (512 GB/s theoretical):
- Vector Copy: 380 GB/s (74% efficiency)
- Interpretation: Good - room for optimization in drivers

## Roofline Model

On a Roofline plot, Vector Copy sits at:
- **X-axis (Arithmetic Intensity)**: 0 FLOPs/byte
- **Y-axis (Performance)**: Peak memory bandwidth

It defines the **left edge of the Roofline** - any kernel with AI=0 hits this ceiling.

## Comparison with Industry Standards

STREAM Benchmark's **COPY** operation is equivalent to Vector Copy.

**STREAM COPY** formula:
```c
for i = 0 to N:
    C[i] = A[i]  // Same as Vector Copy
```

Compare your GPU's Vector Copy bandwidth against:
- Published STREAM results for CPUs
- GPU vendor specifications
- Peer benchmarks on similar hardware

---

<!-- GENERATED_CODE_TABS: vector_copy -->

## References

- McCalpin, John D., "Memory Bandwidth and Machine Balance in Current High Performance Computers" (1995)
- NVIDIA, "GPU Performance Analysis and Optimization"
- Roofline Model: Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)
