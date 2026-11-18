# Generated Backend Code: reduction_max

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 02:06:44

## CUDA C

```cuda

extern "C" {
__global__ void sarek_kern(float* __restrict__ input, int sarek_input_length, float* __restrict__ output, int sarek_output_length, int n) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int gid = (threadIdx.x + (blockDim.x * blockIdx.x));
  {
    if ((gid < n)) {
      sdata[tid] = input[gid];
    } else {
      sdata[tid] = -1000000.0f;
    }
  }
  __syncthreads();
  {
    if ((tid < 128)) {
      float a = sdata[tid];
      float b = sdata[(tid + 128)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 64)) {
      float a = sdata[tid];
      float b = sdata[(tid + 64)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 32)) {
      float a = sdata[tid];
      float b = sdata[(tid + 32)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 16)) {
      float a = sdata[tid];
      float b = sdata[(tid + 16)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 8)) {
      float a = sdata[tid];
      float b = sdata[(tid + 8)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 4)) {
      float a = sdata[tid];
      float b = sdata[(tid + 4)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 2)) {
      float a = sdata[tid];
      float b = sdata[(tid + 2)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid < 1)) {
      float a = sdata[tid];
      float b = sdata[(tid + 1)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  __syncthreads();
  {
    if ((tid == 0)) {
      output[blockIdx.x] = sdata[0];
    }
  }
  __syncthreads();
}
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(__global float* restrict input, int sarek_input_length, __global float* restrict output, int sarek_output_length, int n) {
  __local float sdata[256];
  int tid = get_local_id(0);
  int gid = (get_local_id(0) + (get_local_size(0) * get_group_id(0)));
  {
    if ((gid < n)) {
      sdata[tid] = input[gid];
    } else {
      sdata[tid] = -1000000.0f;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 128)) {
      float a = sdata[tid];
      float b = sdata[(tid + 128)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 64)) {
      float a = sdata[tid];
      float b = sdata[(tid + 64)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 32)) {
      float a = sdata[tid];
      float b = sdata[(tid + 32)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 16)) {
      float a = sdata[tid];
      float b = sdata[(tid + 16)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 8)) {
      float a = sdata[tid];
      float b = sdata[(tid + 8)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 4)) {
      float a = sdata[tid];
      float b = sdata[(tid + 4)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 2)) {
      float a = sdata[tid];
      float b = sdata[(tid + 2)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid < 1)) {
      float a = sdata[tid];
      float b = sdata[(tid + 1)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    if ((tid == 0)) {
      output[get_group_id(0)] = sdata[0];
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
  int n;
} pc;

#define inputv_len pc.inputv_len
#define outputv_len pc.outputv_len
#define n pc.n

// Shared memory
shared float sdata[256];

void main() {
  int tid = int(gl_LocalInvocationID.x);
  int gid = (int(gl_LocalInvocationID.x) + (int(gl_WorkGroupSize.x) * int(gl_WorkGroupID.x)));
  {
    if ((gid < n)) {
      sdata[tid] = inputv[gid];
    } else {
      sdata[tid] = -1000000.0;
    }
  }
  barrier();
  {
    if ((tid < 128)) {
      float a = sdata[tid];
      float b = sdata[(tid + 128)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 64)) {
      float a = sdata[tid];
      float b = sdata[(tid + 64)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 32)) {
      float a = sdata[tid];
      float b = sdata[(tid + 32)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 16)) {
      float a = sdata[tid];
      float b = sdata[(tid + 16)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 8)) {
      float a = sdata[tid];
      float b = sdata[(tid + 8)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 4)) {
      float a = sdata[tid];
      float b = sdata[(tid + 4)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 2)) {
      float a = sdata[tid];
      float b = sdata[(tid + 2)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid < 1)) {
      float a = sdata[tid];
      float b = sdata[(tid + 1)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  barrier();
  {
    if ((tid == 0)) {
      outputv[int(gl_WorkGroupID.x)] = sdata[0];
    }
  }
  barrier();
}
```

## Metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void sarek_kern(device float* input [[buffer(0)]], constant int &sarek_input_length [[buffer(1)]], device float* output [[buffer(2)]], constant int &sarek_output_length [[buffer(3)]], constant int &n [[buffer(4)]], 
  uint3 __metal_gid [[thread_position_in_grid]],
  uint3 __metal_tid [[thread_position_in_threadgroup]],
  uint3 __metal_bid [[threadgroup_position_in_grid]],
  uint3 __metal_tpg [[threads_per_threadgroup]],
  uint3 __metal_num_groups [[threadgroups_per_grid]]) {
  threadgroup float sdata[256];
  int tid = __metal_tid.x;
  int gid = (__metal_tid.x + (__metal_tpg.x * __metal_bid.x));
  {
    if ((gid < n)) {
      sdata[tid] = input[gid];
    } else {
      sdata[tid] = -1000000.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 128)) {
      float a = sdata[tid];
      float b = sdata[(tid + 128)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 64)) {
      float a = sdata[tid];
      float b = sdata[(tid + 64)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 32)) {
      float a = sdata[tid];
      float b = sdata[(tid + 32)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 16)) {
      float a = sdata[tid];
      float b = sdata[(tid + 16)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 8)) {
      float a = sdata[tid];
      float b = sdata[(tid + 8)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 4)) {
      float a = sdata[tid];
      float b = sdata[(tid + 4)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 2)) {
      float a = sdata[tid];
      float b = sdata[(tid + 2)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid < 1)) {
      float a = sdata[tid];
      float b = sdata[(tid + 1)];
      if ((b > a)) {
        sdata[tid] = b;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    if ((tid == 0)) {
      output[__metal_bid.x] = sdata[0];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

