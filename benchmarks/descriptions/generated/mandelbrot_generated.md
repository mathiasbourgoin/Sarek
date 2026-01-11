# Generated Backend Code: mandelbrot

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 01:54:44

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(int* __restrict__ output, int sarek_output_length, int width, int height, int max_iter) {
  int px = (threadIdx.x + blockIdx.x * blockDim.x);
  int py = (threadIdx.y + blockIdx.y * blockDim.y);
  if (((px < width) && (py < height))) {
    float x0 = ((4.0f * ((float)(px) / (float)(width))) - 2.5f);
    float y0 = ((3.0f * ((float)(py) / (float)(height))) - 1.5f);
    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;
    while (((((x * x) + (y * y)) <= 4.0f) && (iter < max_iter))) {
      float xtemp = (((x * x) - (y * y)) + x0);
      y = (((2.0f * x) * y) + y0);
      x = xtemp;
      iter = (iter + 1);
    }
    output[((py * width) + px)] = iter;
  }
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global int* restrict output, int sarek_output_length, int width, int height, int max_iter) {
  int px = get_global_id(0);
  int py = get_global_id(1);
  if (((px < width) && (py < height))) {
    float x0 = ((4.0f * ((float)(px) / (float)(width))) - 2.5f);
    float y0 = ((3.0f * ((float)(py) / (float)(height))) - 1.5f);
    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;
    while (((((x * x) + (y * y)) <= 4.0f) && (iter < max_iter))) {
      float xtemp = (((x * x) - (y * y)) + x0);
      y = (((2.0f * x) * y) + y0);
      x = xtemp;
      iter = (iter + 1);
    }
    output[((py * width) + px)] = iter;
  }
}
```

## Vulkan GLSL

```glsl
#version 450

// Sarek-generated compute shader: sarek_kern
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) buffer Buffer_outputv {
  int outputv[];
};
layout(push_constant) uniform PushConstants {
  int outputv_len;
  int width;
  int height;
  int max_iter;
} pc;

#define outputv_len pc.outputv_len
#define width pc.width
#define height pc.height
#define max_iter pc.max_iter

void main() {
  int px = int(gl_GlobalInvocationID.x);
  int py = int(gl_GlobalInvocationID.y);
  if (((px < width) && (py < height))) {
    float x0 = ((4.0 * (float(px) / float(width))) - 2.5);
    float y0 = ((3.0 * (float(py) / float(height))) - 1.5);
    float x = 0.0;
    float y = 0.0;
    int iter = 0;
    while (((((x * x) + (y * y)) <= 4.0) && (iter < max_iter))) {
      float xtemp = (((x * x) - (y * y)) + x0);
      y = (((2.0 * x) * y) + y0);
      x = xtemp;
      iter = (iter + 1);
    }
    outputv[((py * width) + px)] = iter;
  }
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device int* output [[buffer(0)]], constant int &sarek_output_length [[buffer(1)]], constant int &width [[buffer(2)]], constant int &height [[buffer(3)]], constant int &max_iter [[buffer(4)]], 
  uint3 __metal_gid [[thread_position_in_grid]],
  uint3 __metal_tid [[thread_position_in_threadgroup]],
  uint3 __metal_bid [[threadgroup_position_in_grid]],
  uint3 __metal_tpg [[threads_per_threadgroup]],
  uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  int px = __metal_gid.x;
  int py = __metal_gid.y;
  if (((px < width) && (py < height))) {
    float x0 = ((4.0f * ((float)(px) / (float)(width))) - 2.5f);
    float y0 = ((3.0f * ((float)(py) / (float)(height))) - 1.5f);
    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;
    while (((((x * x) + (y * y)) <= 4.0f) && (iter < max_iter))) {
      float xtemp = (((x * x) - (y * y)) + x0);
      y = (((2.0f * x) * y) + y0);
      x = xtemp;
      iter = (iter + 1);
    }
    output[((py * width) + px)] = iter;
  }
}
```

