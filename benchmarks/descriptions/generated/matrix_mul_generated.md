# Generated Backend Code: matrix_mul

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 01:56:48

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(float* __restrict__ a, int sarek_a_length, float* __restrict__ b, int sarek_b_length, float* __restrict__ c, int sarek_c_length, int m, int n, int k) {
  int tid = (threadIdx.x + blockIdx.x * blockDim.x);
  int row = (tid / n);
  int col = (tid % n);
  if (((row < m) && (col < n))) {
    float sum = 0.0f;
    for (int i = 0; i <= (k - 1); i++) {
      sum = (sum + (a[((row * k) + i)] * b[((i * n) + col)]));
    }
    c[((row * n) + col)] = sum;
  }
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global float* restrict a, int sarek_a_length, __global float* restrict b, int sarek_b_length, __global float* restrict c, int sarek_c_length, int m, int n, int k) {
  int tid = get_global_id(0);
  int row = (tid / n);
  int col = (tid % n);
  if (((row < m) && (col < n))) {
    float sum = 0.0f;
    for (int i = 0; i <= (k - 1); i++) {
      sum = (sum + (a[((row * k) + i)] * b[((i * n) + col)]));
    }
    c[((row * n) + col)] = sum;
  }
}
```

## Vulkan GLSL

```glsl
#version 450

// Sarek-generated compute shader: sarek_kern
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) buffer Buffer_a {
  float a[];
};
layout(std430, set=0, binding = 1) buffer Buffer_b {
  float b[];
};
layout(std430, set=0, binding = 2) buffer Buffer_c {
  float c[];
};
layout(push_constant) uniform PushConstants {
  int a_len;
  int b_len;
  int c_len;
  int m;
  int n;
  int k;
} pc;

#define a_len pc.a_len
#define b_len pc.b_len
#define c_len pc.c_len
#define m pc.m
#define n pc.n
#define k pc.k

void main() {
  int tid = int(gl_GlobalInvocationID.x);
  int row = (tid / n);
  int col = (tid % n);
  if (((row < m) && (col < n))) {
    float sum = 0.0;
    for (int i = 0; i <= (k - 1); i++) {
      sum = (sum + (a[((row * k) + i)] * b[((i * n) + col)]));
    }
    c[((row * n) + col)] = sum;
  }
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device float* a [[buffer(0)]], constant int &sarek_a_length [[buffer(1)]], device float* b [[buffer(2)]], constant int &sarek_b_length [[buffer(3)]], device float* c [[buffer(4)]], constant int &sarek_c_length [[buffer(5)]], constant int &m [[buffer(6)]], constant int &n [[buffer(7)]], constant int &k [[buffer(8)]], 
  uint3 __metal_gid [[thread_position_in_grid]],
  uint3 __metal_tid [[thread_position_in_threadgroup]],
  uint3 __metal_bid [[threadgroup_position_in_grid]],
  uint3 __metal_tpg [[threads_per_threadgroup]],
  uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  int tid = __metal_gid.x;
  int row = (tid / n);
  int col = (tid % n);
  if (((row < m) && (col < n))) {
    float sum = 0.0f;
    for (int i = 0; i <= (k - 1); i++) {
      sum = (sum + (a[((row * k) + i)] * b[((i * n) + col)]));
    }
    c[((row * n) + col)] = sum;
  }
}
```

