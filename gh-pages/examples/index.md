---
layout: page
title: Examples
---

# Sarek Examples

Learn Sarek through practical examples that demonstrate different GPU computing patterns and optimizations.

## Memory & Bandwidth

<div style="border-left: 4px solid var(--link-color); padding-left: 20px; margin: 20px 0;">

### [Vector Addition](vector_add.html)
The classic "Hello World" of GPU computing. Demonstrates basic kernel structure, memory operations, and how to achieve peak memory bandwidth.

**Key concepts:** Thread indexing, memory coalescing, bandwidth optimization

</div>

<div style="border-left: 4px solid var(--link-color); padding-left: 20px; margin: 20px 0;">

### [Matrix Transpose](transpose.html)
Shows the impact of memory access patterns on performance. Compares naive vs tiled implementations.

**Key concepts:** Memory access patterns, shared memory, bank conflicts, tiling optimization

</div>

## Compute-Bound Operations

<div style="border-left: 4px solid var(--link-color); padding-left: 20px; margin: 20px 0;">

### [Matrix Multiplication](matrix_mul.html)
A fundamental compute-intensive operation. Demonstrates how to maximize arithmetic throughput.

**Key concepts:** FLOPS optimization, cache utilization, algorithmic complexity

</div>

<div style="border-left: 4px solid var(--link-color); padding-left: 20px; margin: 20px 0;">

### [Mandelbrot Set](mandelbrot.html)
Classic fractal generation with heavy arithmetic per pixel. Shows embarrassingly parallel computation.

**Key concepts:** Complex arithmetic, iteration, 2D thread grids

</div>

## Parallel Patterns

<div style="border-left: 4px solid var(--link-color); padding-left: 20px; margin: 20px 0;">

### [Parallel Reduction](reduction.html)
Efficiently compute aggregate operations (sum, max, min) on large arrays using tree-based reduction.

**Key concepts:** Tree reduction, synchronization, warp-level primitives

</div>

---

## Performance Data

For detailed performance comparisons across different GPUs and backends, see the [Benchmarks](../benchmarks/) section.

## Next Steps

- [Getting Started Guide](../docs/getting_started.html) - Set up your environment
- [Concepts](../docs/concepts.html) - Understanding Sarek's design
- [API Documentation](../spoc_docs/index.html) - Complete API reference
