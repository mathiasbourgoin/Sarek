# Generated Backend Code: radix_sort_scatter

This file is auto-generated. Do not edit manually.

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(int* __restrict__ input, int sarek_input_length, int* __restrict__ output, int sarek_output_length, int* __restrict__ counters, int sarek_counters_length, int n, int shift, int mask) {
  int gid = (threadIdx.x + blockIdx.x * blockDim.x);
  if ((gid < n)) {
    int value = input[gid];
    int digit = ((value >> shift) & mask);
    int pos = atomicAdd(&counters[digit], 1);
    output[pos] = value;
  }
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global int* restrict input, int sarek_input_length, __global int* restrict output, int sarek_output_length, __global int* restrict counters, int sarek_counters_length, int n, int shift, int mask) {
  int gid = get_global_id(0);
  if ((gid < n)) {
    int value = input[gid];
    int digit = ((value >> shift) & mask);
    int pos = atomic_add(&counters[digit], 1);
    output[pos] = value;
  }
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
layout(std430, set=0, binding = 1) buffer Buffer_outputv {
  int outputv[];
};
layout(std430, set=0, binding = 2) buffer Buffer_counters {
  int counters[];
};
layout(push_constant) uniform PushConstants {
  int inputv_len;
  int outputv_len;
  int counters_len;
  int n;
  int shift;
  int mask;
} pc;

#define inputv_len pc.inputv_len
#define outputv_len pc.outputv_len
#define counters_len pc.counters_len
#define n pc.n
#define shift pc.shift
#define mask pc.mask

void main() {
  int gid = int(gl_GlobalInvocationID.x);
  if ((gid < n)) {
    int value = inputv[gid];
    int digit = ((value >> shift) & mask);
    int pos = atomicAdd(counters[digit], 1);
    outputv[pos] = value;
  }
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device int* input [[buffer(0)]], constant int &sarek_input_length [[buffer(1)]], device int* output [[buffer(2)]], constant int &sarek_output_length [[buffer(3)]], device atomic_int* counters [[buffer(4)]], constant int &sarek_counters_length [[buffer(5)]], constant int &n [[buffer(6)]], constant int &shift [[buffer(7)]], constant int &mask [[buffer(8)]],
uint3 __metal_gid [[thread_position_in_grid]],
uint3 __metal_tid [[thread_position_in_threadgroup]],
uint3 __metal_bid [[threadgroup_position_in_grid]],
uint3 __metal_tpg [[threads_per_threadgroup]],
uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  int gid = __metal_gid.x;
  if ((gid < n)) {
    int value = input[gid];
    int digit = ((value >> shift) & mask);
    int pos = atomic_fetch_add_explicit((volatile device atomic_int*)&counters[digit], 1, memory_order_relaxed);
    output[pos] = value;
  }
}

```

