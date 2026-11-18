# Matrix Multiplication

Dense matrix multiplication: **C = A × B** where A, B, C are N×N matrices.

## Description

This benchmark computes the product of two square matrices using a naive implementation. Each thread computes one element of the output matrix by performing a dot product of a row from matrix A and a column from matrix B.

## Why It Matters

Matrix multiplication is fundamental to:
- **Machine Learning**: Neural network training and inference
- **Scientific Computing**: Linear algebra operations, simulations
- **Graphics**: Transformations, projections
- **Signal Processing**: Convolutions, filtering

This is a **compute-bound** operation that tests the raw arithmetic throughput of the GPU. The ratio of floating-point operations to memory accesses is high (O(N) operations per element loaded), making it ideal for measuring peak GFLOPS.

## Sarek Kernel

```ocaml
[%kernel
  fun (a : float32 vector)
      (b : float32 vector)
      (c : float32 vector)
      (m : int32) (n : int32) (k : int32) ->
    let open Std in
    let tid = global_thread_id in
    let row = tid / n in
    let col = tid mod n in
    if row < m && col < n then begin
      let sum = mut 0.0 in
      for i = 0 to k - 1l do
        sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done;
      c.((row * n) + col) <- sum
    end]
```

<!-- GENERATED_CODE_TABS: matrix_mul -->

## Key Features

- **Mutable accumulator**: Uses `mut` for the running sum
- **2D thread mapping**: Maps 1D thread ID to 2D matrix coordinates
- **Bounds checking**: Ensures threads don't access out-of-bounds memory
- **Inner loop**: Each thread performs N multiply-add operations

## Performance Characteristics

- **Arithmetic Intensity**: 2N operations per N memory accesses = O(N)
- **Memory Pattern**: Non-coalesced reads from matrix B (column access)
- **Optimization Potential**: Tiled/blocked implementations can achieve 10-100× speedup
- **Typical Performance**: 100-1000 GFLOPS depending on GPU and size
