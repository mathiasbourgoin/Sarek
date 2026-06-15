# Generated Backend Code: gather

This file is auto-generated. Do not edit manually.

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(int* __restrict__ input, int sarek_input_length, int* __restrict__ indices, int sarek_indices_length, int* __restrict__ output, int sarek_output_length, int n) {
  int i = (threadIdx.x + blockIdx.x * blockDim.x);
  if ((i < n)) {
    int idx = indices[i];
    output[i] = input[idx];
  }
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global int* restrict input, int sarek_input_length, __global int* restrict indices, int sarek_indices_length, __global int* restrict output, int sarek_output_length, int n) {
  int i = get_global_id(0);
  if ((i < n)) {
    int idx = indices[i];
    output[i] = input[idx];
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
layout(std430, set=0, binding = 1) buffer Buffer_indices {
  int indices[];
};
layout(std430, set=0, binding = 2) buffer Buffer_outputv {
  int outputv[];
};
layout(push_constant) uniform PushConstants {
  int inputv_len;
  int indices_len;
  int outputv_len;
  int n;
} pc;

#define inputv_len pc.inputv_len
#define indices_len pc.indices_len
#define outputv_len pc.outputv_len
#define n pc.n

void main() {
  int i = int(gl_GlobalInvocationID.x);
  if ((i < n)) {
    int idx = indices[i];
    outputv[i] = inputv[idx];
  }
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device int* input [[buffer(0)]], constant int &sarek_input_length [[buffer(1)]], device int* indices [[buffer(2)]], constant int &sarek_indices_length [[buffer(3)]], device int* output [[buffer(4)]], constant int &sarek_output_length [[buffer(5)]], constant int &n [[buffer(6)]],
uint3 __metal_gid [[thread_position_in_grid]],
uint3 __metal_tid [[thread_position_in_threadgroup]],
uint3 __metal_bid [[threadgroup_position_in_grid]],
uint3 __metal_tpg [[threads_per_threadgroup]],
uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  int i = __metal_gid.x;
  if ((i < n)) {
    int idx = indices[i];
    output[i] = input[idx];
  }
}

```

