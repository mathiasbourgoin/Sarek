# 2D Convolution (3×3 Box Blur)

## Overview

The 2D Convolution benchmark implements image filtering using a 3×3 convolution kernel. Convolution is a fundamental operation in image processing, computer vision, and convolutional neural networks (CNNs).

This benchmark tests both **memory access patterns** and **arithmetic throughput**, with a balance between the two. It's classified as a **stencil pattern** where each output depends on a fixed neighborhood of inputs.

## Algorithm

2D convolution with a 3×3 kernel:

1. For each pixel (x, y) in the interior (excluding 1-pixel border):
   - Load 9 neighboring input pixels
   - Load 9 kernel weights
   - Compute weighted sum: `output[y][x] = Σᵢⱼ input[y+i-1][x+j-1] × kernel[i][j]`
   - Write result

Border pixels are set to zero (simple boundary handling).

For a **box blur**, all 9 kernel weights = 1/9, giving equal weight to all neighbors.

## Sarek Kernel Implementation

```ocaml
[%kernel
  fun (input : float32 vector)
      (output : float32 vector)
      (width : int32)
      (height : int32) ->
    let tid = thread_idx_x + (block_idx_x * block_dim_x) in
    let x = tid % width in
    let y = tid / width in
    
    if tid < width * height then
      if x = 0 || x = width - 1 || y = 0 || y = height - 1 then
        output.(tid) <- 0.0  (* Border pixels *)
      else
        let sum = mut 0.0 in
        for ky = 0 to 2 do
          for kx = 0 to 2 do
            let px = x + kx - 1 in
            let py = y + ky - 1 in
            let pidx = (py * width) + px in
            sum := sum +. input.(pidx)
          done
        done ;
        output.(tid) <- sum /. 9.0]
```

**Key features:**
- **1D thread indexing** with 2D coordinate conversion
- **Boundary handling**: Explicit check for border pixels
- **Nested loops**: 3×3 neighborhood iteration
- **Box blur kernel**: Uniform weights (1/9 each)
- **Arithmetic**: 9 reads, 9 adds, 1 divide, 1 write per pixel

## Performance Characteristics

### Complexity
- **Time**: O(N) for N pixels (assuming constant kernel size)
- **Space**: O(N) input + O(N) output
- **Parallelism**: O(N) - one thread per pixel

### Memory Access
- **Input**: Each thread reads 9 neighboring pixels (potentially irregular)
- **Output**: Each thread writes 1 pixel (coalesced)
- **Reuse**: Neighboring threads access overlapping input regions (4-way reuse for 3×3)

```text
Memory traffic per pixel:
- Worst case (no cache): 9 reads + 1 write = 10 floats = 40 bytes
- Best case (perfect cache): 2.25 reads + 1 write = 3.25 floats = 13 bytes
```

### Arithmetic Intensity
```text
FLOPs per pixel = 9 multiplies + 8 adds + 1 divide = 18 ops (box blur: 8 adds + 1 div = 9 ops)
Memory traffic = 40 bytes (worst case)
AI = 9 / 40 = 0.225 FLOPs/byte (box blur, worst case)
```

This is a **memory-bound** operation without optimization.

### Performance Metrics
- **M pixels/s**: Million pixels processed per second
- **GB/s**: Memory bandwidth (N × 40 bytes / time)
- **GFLOPS**: (N × 9) / time_seconds / 1e9 (for box blur)

## Typical Results

| Size (W×H) | Pixels  | Time (ms) | M pixels/s | GB/s  |
|------------|---------|-----------|------------|-------|
| 256×256    | 65K     | 0.02      | 3,277      | 131   |
| 512×512    | 262K    | 0.07      | 3,743      | 150   |
| 1024×1024  | 1.05M   | 0.28      | 3,750      | 150   |
| 2048×2048  | 4.19M   | 1.12      | 3,741      | 150   |

#### Intel Arc A770 GPU (OpenCL backend)

Performance is consistent across sizes, indicating good scaling and memory system utilization.

## Optimization Opportunities

The naive implementation can be significantly optimized:

### 1. Shared Memory Tiling
Load tiles of input image into shared memory, reducing global memory traffic:

```text
For each block:
  Load (TILE_SIZE + 2)² input pixels into shared memory (halo included)
  Synchronize threads
  Compute TILE_SIZE² outputs from shared memory
```

**Benefit:** Reduces global memory reads from 9N to ~2.25N (with proper tile sizes)

### 2. Separable Kernels
Many kernels (Gaussian, Sobel) are separable into row × column passes:

```
output = row_conv(col_conv(input))
```

**Benefit:** Reduces 9 reads to 6 reads (3 per pass), improves cache locality

### 3. Constant Memory for Kernel
Store convolution weights in constant memory (read-only cache):

```ocaml
let kernel_weights = constant_array [1./9.; 1./9.; ...; 1./9.]
```

**Benefit:** Faster access, reduced register pressure

### 4. Specialized Border Handling
Use separate kernels for border vs. interior, or use texture memory with border modes.

## Stencil Pattern Classification

| Stencil Type     | Radius | Points | Memory Reads | Arithmetic |
|------------------|--------|--------|--------------|------------|
| 2D - 5-point     | 1      | 5      | 5            | Low        |
| **2D - 9-point** | **1**  | **9**  | **9**        | **Medium** |
| 3D - 7-point     | 1      | 7      | 7            | Low        |
| 3D - 27-point    | 1      | 27     | 27           | High       |

The 3×3 convolution is a 9-point stencil, the most common in 2D image processing.

## Applications

2D convolution is used in:

- **Image processing**: Blur, sharpen, edge detection
- **Computer vision**: Feature extraction, filtering
- **CNNs**: Fundamental building block (though modern CNNs use im2col + GEMM)
- **Scientific computing**: Heat diffusion, wave propagation
- **Signal processing**: 2D filtering

## Verification

CPU reference computes the same convolution in double precision:

```ocaml
let cpu_conv2d input output width height kernel =
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      if x = 0 || x = width - 1 || y = 0 || y = height - 1 then
        output.(idx) <- 0.0
      else
        let sum = ref 0.0 in
        for ky = 0 to 2 do
          for kx = 0 to 2 do
            let px = x + kx - 1 in
            let py = y + ky - 1 in
            let pidx = (py * width) + px in
            sum := !sum +. (input.(pidx) *. kernel.(ky * 3 + kx))
          done
        done ;
        output.(idx) <- !sum
    done
  done
```

Verification checks interior pixels only (excluding 1-pixel border) with tolerance 0.001.
