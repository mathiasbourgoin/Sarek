(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The HIP cooperative-matrix probe (specs/coopmat-hip-wmma.md).

    HIP has no query that lists matrix-instruction configurations the way
    [vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR] does, so the probe is a
    rule over two facts the runtime does report: the gfx family, read from
    [hipDeviceComputeCapability] (gfx1100 reports 11.0, the Raphael iGPU gfx1036
    reports 10.3), and the wavefront size. The configurations offered are the
    ones Sarek's HIP lowering emits ([Sarek_ir_cuda_coopmat]) and that were
    executed on gfx1100: the gfx11 WMMA builtins are wave32-only.

    Every other device is reported as probed with NO configuration — [Some] with
    an empty list, not [None] — so a refusal names the rule instead of "not
    probed". That includes devices that do have matrix units this lowering does
    not emit: gfx12 (WMMA with different builtins and layouts) and CDNA (MFMA).
    They are refused until they are lowered and measured. *)

module C = Sarek_coopmat

let shape = {C.m = 16; n = 16; k = 16}

let config a b acc =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = b;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

(** [__builtin_amdgcn_wmma_f32_16x16x16_f16_w32], its bf16 twin
    [__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32]
    (specs/matrix-pattern-rewrite.md) and
    [__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32] (both operand signednesses are
    immediates of the latter; a 32-bit wrapping sum has the same bits whether it
    is read as s32 or u32). *)
let gfx11_configs =
  config C.Float16 C.Float16 C.Float32
  :: config C.BFloat16 C.BFloat16 C.Float32
  :: List.concat_map
       (fun (a, b) -> [config a b C.Sint32; config a b C.Uint32])
       [
         (C.Uint8, C.Uint8);
         (C.Uint8, C.Sint8);
         (C.Sint8, C.Uint8);
         (C.Sint8, C.Sint8);
       ]

(** [support ~compute_capability ~warp_size] is the rule, separated from the
    device record so a test can drive it with any device's facts. *)
let support ~compute_capability:(major, minor) ~warp_size =
  let lowered = major = 11 && warp_size = 32 in
  (* CM_scale_add is lowered for the whole gfx11 family but has been executed
     only on gfx1100 (compute capability 11.0), so only 11.0 offers it.
     docs/design/coopmat-scaled-accumulate.md §6. *)
  let scale_add = lowered && minor = 0 in
  let configs = if lowered then gfx11_configs else [] in
  {
    C.ds_configs = configs;
    ds_robust_buffer_access = false;
    ds_subgroup_size = warp_size;
    ds_advertised_count = List.length configs;
    ds_scale_add = scale_add;
    ds_evidence =
      (if lowered then
         Printf.sprintf
           "HIP: gfx%d family (compute capability %d.%d), wave%d: RDNA3 WMMA%s"
           major
           major
           minor
           warp_size
           (if scale_add then ""
            else
              "; CM_scale_add is lowered for gfx11 but executed only on \
               gfx1100 (compute capability 11.0), so it is not offered here")
       else
         Printf.sprintf
           "HIP: the WMMA lowering covers gfx11 (RDNA3) with 32-wide \
            wavefronts; this device reports compute capability %d.%d and \
            wavefront size %d"
           major
           minor
           warp_size);
  }

let probe (d : Hip_api.Device.t) =
  support
    ~compute_capability:d.Hip_api.Device.compute_capability
    ~warp_size:d.Hip_api.Device.warp_size
