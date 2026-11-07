(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The CUDA/HIP cooperative-matrix helpers, emitted before [extern "C"] when
    the kernel has a cooperative-matrix statement (Sarek_ir_cuda_coopmat).

    The AMD helpers have C++ linkage and one pointer type each, so a buffer of
    the wrong element type is a compile error rather than a reinterpretation.

    Each operand helper names an access pattern, not an operand: [load_a_*]
    reads 16 contiguous elements per lane starting at [p + i + l16 * s], which
    is an A fragment row-major and a B fragment column-major; [load_b_*] reads
    16 elements [s] apart starting at [p + i + l16], a B fragment row-major and
    an A fragment column-major. The contiguous form is the one the compiler
    turns into vector loads.

    The stride is [long long], so every offset product ([l16 * s], [e * s],
    [2 * j * s]) is computed in 64 bits: an [int] product overflows once the
    stride passes [2^31 / 15] elements, although every element the fragment
    touches is addressable. *)

let text =
  {|#if defined(__HIP_PLATFORM_AMD__)
/* Sarek cooperative matrix, HIP: RDNA3 (gfx11) WMMA, wave32
   (specs/coopmat-hip-wmma.md). For lane l, with l16 = l % 16 and h = l / 16:
   an A fragment holds row l16 (k = 0..15), a B fragment holds column l16, and
   accumulator element j holds row 2*j + h, column l16.
   load_a_* reads 16 contiguous elements at p + i + l16*s (A row-major, B
   column-major); load_b_* reads 16 elements s apart at p + i + l16 (B
   row-major, A column-major); *_acc_cm_* are the column-major accumulator. */
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX11__)
#error "Sarek cooperative matrix: the HIP lowering targets gfx11 (RDNA3) WMMA only"
#endif
typedef _Float16 sarek_cm_h16 __attribute__((ext_vector_type(16)));
typedef short sarek_cm_s16 __attribute__((ext_vector_type(16)));
typedef float sarek_cm_f8 __attribute__((ext_vector_type(8)));
typedef int sarek_cm_i4 __attribute__((ext_vector_type(4)));
typedef int sarek_cm_i8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int sarek_cm_l16() { return (int)(__lane_id() & 15u); }
__device__ __forceinline__ int sarek_cm_h() { return (int)((__lane_id() >> 4) & 1u); }
__device__ __forceinline__ unsigned sarek_cm_pack4(unsigned char b0, unsigned char b1,
                                                   unsigned char b2, unsigned char b3) {
  return (unsigned)b0 | ((unsigned)b1 << 8) | ((unsigned)b2 << 16) | ((unsigned)b3 << 24);
}
__device__ __forceinline__ void sarek_cm_load_a_f16(sarek_cm_h16 &f, const __half *p, int i, long long s) {
  const _Float16 *q = (const _Float16 *)p + i + sarek_cm_l16() * s;
  for (int e = 0; e < 16; ++e) f[e] = q[e];
}
__device__ __forceinline__ void sarek_cm_load_b_f16(sarek_cm_h16 &f, const __half *p, int i, long long s) {
  const _Float16 *q = (const _Float16 *)p + i + sarek_cm_l16();
  for (int e = 0; e < 16; ++e) f[e] = q[e * s];
}
/* bf16 operands are loaded from a float32 buffer and rounded to bfloat16
   (nearest even), specs/matrix-pattern-rewrite.md. */
__device__ __forceinline__ short sarek_cm_bf16(float x) {
  unsigned u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (short)((u >> 16) | 0x40u);
  return (short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ void sarek_cm_load_a_bf16(sarek_cm_s16 &f, const float *p, int i, long long s) {
  const float *q = p + i + sarek_cm_l16() * s;
  for (int e = 0; e < 16; ++e) f[e] = sarek_cm_bf16(q[e]);
}
__device__ __forceinline__ void sarek_cm_load_b_bf16(sarek_cm_s16 &f, const float *p, int i, long long s) {
  const float *q = p + i + sarek_cm_l16();
  for (int e = 0; e < 16; ++e) f[e] = sarek_cm_bf16(q[e * s]);
}
__device__ __forceinline__ void sarek_cm_load_a_i8(sarek_cm_i4 &f, const unsigned char *p, int i, long long s) {
  const unsigned char *q = p + i + sarek_cm_l16() * s;
  for (int w = 0; w < 4; ++w)
    f[w] = (int)sarek_cm_pack4(q[4 * w], q[4 * w + 1], q[4 * w + 2], q[4 * w + 3]);
}
__device__ __forceinline__ void sarek_cm_load_b_i8(sarek_cm_i4 &f, const unsigned char *p, int i, long long s) {
  const unsigned char *q = p + i + sarek_cm_l16();
  for (int w = 0; w < 4; ++w)
    f[w] = (int)sarek_cm_pack4(q[(4 * w) * s], q[(4 * w + 1) * s],
                               q[(4 * w + 2) * s], q[(4 * w + 3) * s]);
}
__device__ __forceinline__ void sarek_cm_load_acc_f32(sarek_cm_f8 &f, const float *p, int i, long long s) {
  const float *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) f[j] = q[2 * j * s];
}
__device__ __forceinline__ void sarek_cm_store_acc_f32(const sarek_cm_f8 &f, float *p, int i, long long s) {
  float *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) q[2 * j * s] = f[j];
}
__device__ __forceinline__ void sarek_cm_load_acc_i32(sarek_cm_i8 &f, const int *p, int i, long long s) {
  const int *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) f[j] = q[2 * j * s];
}
__device__ __forceinline__ void sarek_cm_store_acc_i32(const sarek_cm_i8 &f, int *p, int i, long long s) {
  int *q = p + i + sarek_cm_h() * s + sarek_cm_l16();
  for (int j = 0; j < 8; ++j) q[2 * j * s] = f[j];
}
__device__ __forceinline__ void sarek_cm_load_acc_cm_f32(sarek_cm_f8 &f, const float *p, int i, long long s) {
  const float *q = p + i + sarek_cm_l16() * s + sarek_cm_h();
  for (int j = 0; j < 8; ++j) f[j] = q[2 * j];
}
__device__ __forceinline__ void sarek_cm_store_acc_cm_f32(const sarek_cm_f8 &f, float *p, int i, long long s) {
  float *q = p + i + sarek_cm_l16() * s + sarek_cm_h();
  for (int j = 0; j < 8; ++j) q[2 * j] = f[j];
}
__device__ __forceinline__ void sarek_cm_load_acc_cm_i32(sarek_cm_i8 &f, const int *p, int i, long long s) {
  const int *q = p + i + sarek_cm_l16() * s + sarek_cm_h();
  for (int j = 0; j < 8; ++j) f[j] = q[2 * j];
}
__device__ __forceinline__ void sarek_cm_store_acc_cm_i32(const sarek_cm_i8 &f, int *p, int i, long long s) {
  int *q = p + i + sarek_cm_l16() * s + sarek_cm_h();
  for (int j = 0; j < 8; ++j) q[2 * j] = f[j];
}
#else
/* Sarek cooperative matrix, CUDA: nvcuda::wmma m16n16k16. COMPILE-CHECKED
   ONLY (no NVIDIA GPU was available; specs/coopmat-hip-wmma.md). wmma's
   load/store need a 32-byte aligned pointer and a row stride that is a
   multiple of 16 bytes; Sarek's CM_load promises any index and stride, so a
   launch that violates the precondition traps instead of computing on a
   misaligned fragment. */
#include <mma.h>
__device__ __forceinline__ bool sarek_cm_misaligned(const void *p, long long s, int bytes) {
  return (((unsigned long long)p) & 31ull) != 0ull || ((s * bytes) & 15) != 0;
}
__device__ __forceinline__ void sarek_cm_require(const void *p, long long s, int bytes) {
  if (sarek_cm_misaligned(p, s, bytes)) __trap();
}
/* A misaligned access (not proven aligned at code generation, block known) copies
   the 16x16 tile element by element through a 32-byte aligned per-warp buffer
   in shared memory (sarek_cm_tiles, declared by the generator) and runs wmma on
   the buffer with a leading dimension of 16: element t of the buffer is
   index + (t / 16) * stride + (t % 16), in either layout. Lane l copies
   t = l + 32 j, j < 8. */
template <typename T>
__device__ __forceinline__ void sarek_cm_stage_in(T *tile, const T *src, long long stride) {
  unsigned lane = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) & 31u;
  for (int j = 0; j < 8; ++j) {
    unsigned t = lane + 32u * j;
    tile[t] = src[(long long)(t >> 4) * stride + (t & 15u)];
  }
}
template <typename T>
__device__ __forceinline__ void sarek_cm_stage_out(T *dst, const T *tile, long long stride) {
  unsigned lane = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) & 31u;
  for (int j = 0; j < 8; ++j) {
    unsigned t = lane + 32u * j;
    dst[(long long)(t >> 4) * stride + (t & 15u)] = tile[t];
  }
}
#endif
|}
