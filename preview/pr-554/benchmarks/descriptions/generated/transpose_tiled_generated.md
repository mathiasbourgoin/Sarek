# Generated Backend Code: transpose_tiled

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 00:21:52

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(float* __restrict__ input, int sarek_input_length, float* __restrict__ output, int sarek_output_length, int width, int height) {
  __shared__ float tile[272];
  int tile_size = 16;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_col = (blockIdx.x * tile_size);
  int block_row = (blockIdx.y * tile_size);
  int read_col = (block_col + tx);
  int read_row = (block_row + ty);
  {
    if (((read_col < width) && (read_row < height))) {
      int read_idx = ((read_row * width) + read_col);
      int tile_idx = ((ty * 17) + tx);
      tile[tile_idx] = input[read_idx];
    }
  }
  __syncthreads();
  int write_col = (block_row + tx);
  int write_row = (block_col + ty);
  {
    if (((write_col < height) && (write_row < width))) {
      int tile_idx = ((tx * 17) + ty);
      int write_idx = ((write_row * height) + write_col);
      output[write_idx] = tile[tile_idx];
    }
  }
  __syncthreads();
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global float* restrict input, int sarek_input_length, __global float* restrict output, int sarek_output_length, int width, int height) {
  __local float tile[272];
  int tile_size = 16;
  int tx = get_local_id(0);
  int ty = get_local_id(1);
  int block_col = (get_group_id(0) * tile_size);
  int block_row = (get_group_id(1) * tile_size);
  int read_col = (block_col + tx);
  int read_row = (block_row + ty);
  {
    if (((read_col < width) && (read_row < height))) {
      int read_idx = ((read_row * width) + read_col);
      int tile_idx = ((ty * 17) + tx);
      tile[tile_idx] = input[read_idx];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int write_col = (block_row + tx);
  int write_row = (block_col + ty);
  {
    if (((write_col < height) && (write_row < width))) {
      int tile_idx = ((tx * 17) + ty);
      int write_idx = ((write_row * height) + write_col);
      output[write_idx] = tile[tile_idx];
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

// Shared memory
shared float tile[272];

void main() {
  int tile_size = 16;
  int tx = int(gl_LocalInvocationID.x);
  int ty = int(gl_LocalInvocationID.y);
  int block_col = (int(gl_WorkGroupID.x) * tile_size);
  int block_row = (int(gl_WorkGroupID.y) * tile_size);
  int read_col = (block_col + tx);
  int read_row = (block_row + ty);
  {
    if (((read_col < width) && (read_row < height))) {
      int read_idx = ((read_row * width) + read_col);
      int tile_idx = ((ty * 17) + tx);
      tile[tile_idx] = inputv[read_idx];
    }
  }
  barrier();
  int write_col = (block_row + tx);
  int write_row = (block_col + ty);
  {
    if (((write_col < height) && (write_row < width))) {
      int tile_idx = ((tx * 17) + ty);
      int write_idx = ((write_row * height) + write_col);
      outputv[write_idx] = tile[tile_idx];
    }
  }
  barrier();
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
  threadgroup float tile[272];
  int tile_size = 16;
  int tx = __metal_tid.x;
  int ty = __metal_tid.y;
  int block_col = (__metal_bid.x * tile_size);
  int block_row = (__metal_bid.y * tile_size);
  int read_col = (block_col + tx);
  int read_row = (block_row + ty);
  {
    if (((read_col < width) && (read_row < height))) {
      int read_idx = ((read_row * width) + read_col);
      int tile_idx = ((ty * 17) + tx);
      tile[tile_idx] = input[read_idx];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int write_col = (block_row + tx);
  int write_row = (block_col + ty);
  {
    if (((write_col < height) && (write_row < width))) {
      int tile_idx = ((tx * 17) + ty);
      int write_idx = ((write_row * height) + write_col);
      output[write_idx] = tile[tile_idx];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

