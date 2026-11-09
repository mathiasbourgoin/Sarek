# Generated Backend Code: matrix_mul_tiled

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 01:54:44

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(float* __restrict__ a, int sarek_a_length, float* __restrict__ b, int sarek_b_length, float* __restrict__ c, int sarek_c_length, int m, int n, int k) {
  __shared__ float tile_a[256];
  __shared__ float tile_b[256];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = (ty + (blockDim.y * blockIdx.y));
  int col = (tx + (blockDim.x * blockIdx.x));
  int tile_size = 16;
  int num_tiles = (((k + tile_size) - 1) / tile_size);
  float sum = 0.0f;
  for (int t = 0; t <= (num_tiles - 1); t++) {
    {
      int a_col = ((t * tile_size) + tx);
      if (((row < m) && (a_col < k))) {
        tile_a[((ty * tile_size) + tx)] = a[((row * k) + a_col)];
      } else {
        tile_a[((ty * tile_size) + tx)] = 0.0f;
      }
    }
    __syncthreads();
    {
      int b_row = ((t * tile_size) + ty);
      if (((b_row < k) && (col < n))) {
        tile_b[((ty * tile_size) + tx)] = b[((b_row * n) + col)];
      } else {
        tile_b[((ty * tile_size) + tx)] = 0.0f;
      }
    }
    __syncthreads();
    {
      for (int i = 0; i <= (tile_size - 1); i++) {
        sum = (sum + (tile_a[((ty * tile_size) + i)] * tile_b[((i * tile_size) + tx)]));
      }
    }
    __syncthreads();
  }
  if (((row < m) && (col < n))) {
    c[((row * n) + col)] = sum;
  }
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global float* restrict a, int sarek_a_length, __global float* restrict b, int sarek_b_length, __global float* restrict c, int sarek_c_length, int m, int n, int k) {
  __local float tile_a[256];
  __local float tile_b[256];
  int tx = get_local_id(0);
  int ty = get_local_id(1);
  int row = (ty + (get_local_size(1) * get_group_id(1)));
  int col = (tx + (get_local_size(0) * get_group_id(0)));
  int tile_size = 16;
  int num_tiles = (((k + tile_size) - 1) / tile_size);
  float sum = 0.0f;
  for (int t = 0; t <= (num_tiles - 1); t++) {
    {
      int a_col = ((t * tile_size) + tx);
      if (((row < m) && (a_col < k))) {
        tile_a[((ty * tile_size) + tx)] = a[((row * k) + a_col)];
      } else {
        tile_a[((ty * tile_size) + tx)] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      int b_row = ((t * tile_size) + ty);
      if (((b_row < k) && (col < n))) {
        tile_b[((ty * tile_size) + tx)] = b[((b_row * n) + col)];
      } else {
        tile_b[((ty * tile_size) + tx)] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      for (int i = 0; i <= (tile_size - 1); i++) {
        sum = (sum + (tile_a[((ty * tile_size) + i)] * tile_b[((i * tile_size) + tx)]));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (((row < m) && (col < n))) {
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

// Shared memory
shared float tile_a[256];
shared float tile_b[256];

void main() {
  int tx = int(gl_LocalInvocationID.x);
  int ty = int(gl_LocalInvocationID.y);
  int row = (ty + (int(gl_WorkGroupSize.y) * int(gl_WorkGroupID.y)));
  int col = (tx + (int(gl_WorkGroupSize.x) * int(gl_WorkGroupID.x)));
  int tile_size = 16;
  int num_tiles = (((k + tile_size) - 1) / tile_size);
  float sum = 0.0;
  for (int t = 0; t <= (num_tiles - 1); t++) {
    {
      int a_col = ((t * tile_size) + tx);
      if (((row < m) && (a_col < k))) {
        tile_a[((ty * tile_size) + tx)] = a[((row * k) + a_col)];
      } else {
        tile_a[((ty * tile_size) + tx)] = 0.0;
      }
    }
    barrier();
    {
      int b_row = ((t * tile_size) + ty);
      if (((b_row < k) && (col < n))) {
        tile_b[((ty * tile_size) + tx)] = b[((b_row * n) + col)];
      } else {
        tile_b[((ty * tile_size) + tx)] = 0.0;
      }
    }
    barrier();
    {
      for (int i = 0; i <= (tile_size - 1); i++) {
        sum = (sum + (tile_a[((ty * tile_size) + i)] * tile_b[((i * tile_size) + tx)]));
      }
    }
    barrier();
  }
  if (((row < m) && (col < n))) {
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
  threadgroup float tile_a[256];
  threadgroup float tile_b[256];
  int tx = __metal_tid.x;
  int ty = __metal_tid.y;
  int row = (ty + (__metal_tpg.y * __metal_bid.y));
  int col = (tx + (__metal_tpg.x * __metal_bid.x));
  int tile_size = 16;
  int num_tiles = (((k + tile_size) - 1) / tile_size);
  float sum = 0.0f;
  for (int t = 0; t <= (num_tiles - 1); t++) {
    {
      int a_col = ((t * tile_size) + tx);
      if (((row < m) && (a_col < k))) {
        tile_a[((ty * tile_size) + tx)] = a[((row * k) + a_col)];
      } else {
        tile_a[((ty * tile_size) + tx)] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int b_row = ((t * tile_size) + ty);
      if (((b_row < k) && (col < n))) {
        tile_b[((ty * tile_size) + tx)] = b[((b_row * n) + col)];
      } else {
        tile_b[((ty * tile_size) + tx)] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      for (int i = 0; i <= (tile_size - 1); i++) {
        sum = (sum + (tile_a[((ty * tile_size) + i)] * tile_b[((i * tile_size) + tx)]));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (((row < m) && (col < n))) {
    c[((row * n) + col)] = sum;
  }
}
```

