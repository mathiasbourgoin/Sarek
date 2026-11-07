---
name: roster-spec
type: spec
status: live
feature: Launch bounds hint (sarek.max_threads)
brief: briefs/launch-bounds-hint-intake.md (VALIDATED by the owner in chat, 2026-10-17)
date: 2026-10-17
version: 1.0.0
---

# Spec - Launch bounds hint

Driver: a downstream LLM inference engine's fused expert GEMM tiles are limited by the 192-VGPR cap hipcc applies for a
1024-thread block. A declared bound lifts it (hipcc, gfx1100, 2026-10-16: 192 VGPR with 216 bytes
of scratch spill, versus 249 VGPR and no spill with `__launch_bounds__(256)`).

## Requirements

- FR-001. `pragma ["sarek.max_threads N"] body` (also `["sarek.max_threads"; "N"]`) around the whole
  kernel body sets `kern_launch_bounds = Some {max_threads = N}` and is removed from the IR body.
- FR-002. N is an integer in 1..1024; otherwise a located PPX error.
- FR-003. The pragma elsewhere than around the whole body, or twice on one kernel, is a located PPX error.
- FR-004. A kernel without the pragma generates byte-identical source on every backend.
- FR-005. HIP, CUDA-C: `__global__ __launch_bounds__(N) void`.
- FR-006. CUDA PTX: `.maxntid bx, by, bz` from the launch block when bx*by*bz <= N; nothing otherwise.
- FR-007. OpenCL: `__attribute__((reqd_work_group_size(bx, by, bz)))` under the same condition.
- FR-008. Metal: `[[max_total_threads_per_threadgroup(N)]]` before `kernel void`.
- FR-009. Vulkan, WebGPU, Native, Interpreter: no source change.
- FR-010. On every backend, a launch whose block has more than N threads raises
  `Execute_error.Invalid_dimensions` before launch, naming kernel, bound and block.
- FR-011. A fused kernel (horizontal, epilogue, vector fusion) takes the minimum of its members' bounds.
- FR-012. A matrix-pattern rewrite drops the bound (it chooses its own geometry).

## Scenarios

- Given a kernel with N=256, when launched with a 16x16 or 256x1x1 block, then it runs; with 32x16, then
  Invalid_dimensions (GWT-1).
- Given N=256 and the OpenCL backend with block (16,8,1), then the source contains
  `reqd_work_group_size(16, 8, 1)`; with block (32,32,1) it does not (GWT-2).
- Given `["sarek.max_threads abc"]`, `2048`, an inner placement, or two declarations, then compilation fails (GWT-3).
- Given members with bounds 256, 64 and none, the fused kernel has 64 (GWT-4).

## Out of scope

Minimum-blocks, waves-per-EU, maxnreg; shared-memory tuning; automatic derivation; WebGPU.
