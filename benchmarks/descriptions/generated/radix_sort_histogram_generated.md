# Generated Backend Code: radix_sort_histogram

This file is auto-generated. Do not edit manually.

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(int* __restrict__ input, int sarek_input_length, int* __restrict__ histogram, int sarek_histogram_length, int n, int shift, int mask) {
  __shared__ int local_hist[256];
  int tid = threadIdx.x;
  int gid = (threadIdx.x + blockIdx.x * blockDim.x);
  int num_bins = 256;
  {
    if ((tid < num_bins)) {
      local_hist[tid] = 0;
    }
  }
  __syncthreads();
  {
    if ((gid < n)) {
      int value = input[gid];
      int digit = ((value >> shift) & mask);
      int _old = atomicAdd(&local_hist[digit], 1);
    }
  }
  __syncthreads();
  {
    if ((tid < num_bins)) {
      int _old = atomicAdd(&histogram[tid], local_hist[tid]);
    }
  }
  __syncthreads();
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global int* restrict input, int sarek_input_length, __global int* restrict histogram, int sarek_histogram_length, int n, int shift, int mask) {
  __local int local_hist[256];
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int num_bins = 256;
  {
    if ((tid < num_bins)) {
      local_hist[tid] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((gid < n)) {
      int value = input[gid];
      int digit = ((value >> shift) & mask);
      int _old = atomic_add(&local_hist[digit], 1);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < num_bins)) {
      int _old = atomic_add(&histogram[tid], local_hist[tid]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
```

## Vulkan GLSL

```glsl
#version 450

// Sarek-generated compute shader: sarek_kern
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) buffer Buffer_inputv {
  int inputv[];
};
layout(std430, set=0, binding = 1) buffer Buffer_histogram {
  int histogram[];
};
layout(push_constant) uniform PushConstants {
  int inputv_len;
  int histogram_len;
  int n;
  int shift;
  int mask;
} pc;

#define inputv_len pc.inputv_len
#define histogram_len pc.histogram_len
#define n pc.n
#define shift pc.shift
#define mask pc.mask

// Shared memory
shared int local_hist[256];

void main() {
  int tid = int(gl_LocalInvocationID.x);
  int gid = int(gl_GlobalInvocationID.x);
  int num_bins = 256;
  {
    if ((tid < num_bins)) {
      local_hist[tid] = 0;
    }
  }
  barrier();
  {
    if ((gid < n)) {
      int value = inputv[gid];
      int digit = ((value >> shift) & mask);
      int _old = atomicAdd(local_hist[digit], 1);
    }
  }
  barrier();
  {
    if ((tid < num_bins)) {
      int _old = atomicAdd(histogram[tid], local_hist[tid]);
    }
  }
  barrier();
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device int* input [[buffer(0)]], constant int &sarek_input_length [[buffer(1)]], device atomic_int* histogram [[buffer(2)]], constant int &sarek_histogram_length [[buffer(3)]], constant int &n [[buffer(4)]], constant int &shift [[buffer(5)]], constant int &mask [[buffer(6)]],
uint3 __metal_gid [[thread_position_in_grid]],
uint3 __metal_tid [[thread_position_in_threadgroup]],
uint3 __metal_bid [[threadgroup_position_in_grid]],
uint3 __metal_tpg [[threads_per_threadgroup]],
uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  threadgroup int local_hist[256];
  int tid = __metal_tid.x;
  int gid = __metal_gid.x;
  int num_bins = 256;
  {
    if ((tid < num_bins)) {
      local_hist[tid] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((gid < n)) {
      int value = input[gid];
      int digit = ((value >> shift) & mask);
      int _old = atomic_fetch_add_explicit((volatile threadgroup atomic_int*)&local_hist[digit], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < num_bins)) {
      int _old = atomic_fetch_add_explicit((volatile device atomic_int*)&histogram[tid], local_hist[tid], memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

```

