# Generated Backend Code: transpose_naive

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 02:00:43

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(float* __restrict__ input, int sarek_input_length, float* __restrict__ output, int sarek_output_length, int width, int height) {
  int tid = (threadIdx.x + blockIdx.x * blockDim.x);
  int n = (width * height);
  if ((tid < n)) {
    int col = (tid % width);
    int row = (tid / width);
    int in_idx = ((row * width) + col);
    int out_idx = ((col * height) + row);
    output[out_idx] = input[in_idx];
  }
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global float* restrict input, int sarek_input_length, __global float* restrict output, int sarek_output_length, int width, int height) {
  int tid = get_global_id(0);
  int n = (width * height);
  if ((tid < n)) {
    int col = (tid % width);
    int row = (tid / width);
    int in_idx = ((row * width) + col);
    int out_idx = ((col * height) + row);
    output[out_idx] = input[in_idx];
  }
}
```

## Vulkan GLSL

```glsl
#version 450

// Sarek-generated compute shader: sarek_kern
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) buffer Buffer_inputv {
  float inputv[];
};
layout(std430, set=0, binding = 1) buffer Buffer_outputv {
  float outputv[];
};
layout(push_constant) uniform PushConstants {
  int inputv_len;
  int outputv_len;
  int width;
  int height;
} pc;

#define inputv_len pc.inputv_len
#define outputv_len pc.outputv_len
#define width pc.width
#define height pc.height

void main() {
  int tid = int(gl_GlobalInvocationID.x);
  int n = (width * height);
  if ((tid < n)) {
    int col = (tid % width);
    int row = (tid / width);
    int in_idx = ((row * width) + col);
    int out_idx = ((col * height) + row);
    outputv[out_idx] = inputv[in_idx];
  }
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device float* input [[buffer(0)]], constant int &sarek_input_length [[buffer(1)]], device float* output [[buffer(2)]], constant int &sarek_output_length [[buffer(3)]], constant int &width [[buffer(4)]], constant int &height [[buffer(5)]], 
  uint3 __metal_gid [[thread_position_in_grid]],
  uint3 __metal_tid [[thread_position_in_threadgroup]],
  uint3 __metal_bid [[threadgroup_position_in_grid]],
  uint3 __metal_tpg [[threads_per_threadgroup]],
  uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  int tid = __metal_gid.x;
  int n = (width * height);
  if ((tid < n)) {
    int col = (tid % width);
    int row = (tid / width);
    int in_idx = ((row * width) + col);
    int out_idx = ((col * height) + row);
    output[out_idx] = input[in_idx];
  }
}
```

