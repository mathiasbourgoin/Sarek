(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The Metal cooperative-matrix capability rule
    (specs/coopmat-metal-simdgroup.md).

    Metal has no query that lists matrix configurations, and MSL offers exactly
    one lowered here: [simdgroup_half8x8 x half8x8 + float8x8 -> float8x8]
    (docs/design/f16-relaxed-accuracy.md §2). The probe is therefore a compile
    check on the device rather than a family table: a kernel using that
    configuration is compiled on the device (compile evidence, not execution),
    and its pipeline reports the SIMD-group width. The pure rule below turns
    those two facts into a {!Sarek_coopmat.device_support}, so it can be driven
    without a device. A device whose compiler rejects the probe is reported as
    probed with NO configuration — [Some] with an empty list — so a refusal says
    why instead of "not probed"; that pre-Apple7 GPUs and Intel/AMD Macs are in
    this class is inferred, not observed. The launch check and
    [Sarek_gemm.Coopmat] assume whole 32-wide SIMD-groups, so a device reporting
    another width is likewise offered no configuration. *)

module C = Sarek_coopmat

let shape = {C.m = 8; n = 8; k = 8}

(** [simdgroup_half8x8 x half8x8 -> float8x8], 8x8x8. *)
let f16_config =
  {
    C.cfg_shape = shape;
    cfg_a = C.Float16;
    cfg_b = C.Float16;
    cfg_c = C.Float32;
    cfg_result = C.Float32;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let probe_kernel_name = "sarek_coopmat_probe"

(** Exercises the lowered configuration with the same statements the generator
    emits: declare, load, multiply-accumulate, store. *)
let probe_source =
  "#include <metal_stdlib>\n\
   #include <metal_simdgroup_matrix>\n\
   using namespace metal;\n\
   kernel void sarek_coopmat_probe(device half* a [[buffer(0)]], device float* \
   d [[buffer(1)]]) {\n\
  \  simdgroup_half8x8 fa = make_filled_simdgroup_matrix<half, 8, 8>(0);\n\
  \  simdgroup_float8x8 fc = make_filled_simdgroup_matrix<float, 8, 8>(0);\n\
  \  simdgroup_load(fa, (a + (0)), (ulong)(8), ulong2(0, 0), false);\n\
  \  simdgroup_multiply_accumulate(fc, fa, fa, fc);\n\
  \  simdgroup_store(fc, (d + (0)), (ulong)(8), ulong2(0, 0), false);\n\
   }\n"

(** The SIMD-group width the launch check and the GEMM builder assume. *)
let required_simd_width = 32

(** [support ~simd_width ~probe] is the device's support given the SIMD-group
    width its pipeline reported and the outcome of compiling {!probe_source}. *)
let support ~simd_width ~(probe : (unit, string) result) : C.device_support =
  let configs, evidence =
    match probe with
    | Ok () when simd_width <> required_simd_width ->
        ( [],
          Printf.sprintf
            "Metal: the simdgroup_matrix probe compiled, but the SIMD-group \
             width is %d and the lowering needs %d"
            simd_width
            required_simd_width )
    | Ok () ->
        ( [f16_config],
          Printf.sprintf
            "Metal: a simdgroup_half8x8 x half8x8 -> float8x8 kernel compiled \
             on this device (8x8 only; there is no integer simdgroup_matrix); \
             SIMD-group width %d read from its pipeline"
            simd_width )
    | Error why ->
        ( [],
          Printf.sprintf
            "Metal: the simdgroup_matrix probe kernel did not compile on this \
             device: %s"
            why )
  in
  {
    C.ds_configs = configs;
    ds_robust_buffer_access = false;
    ds_subgroup_size = simd_width;
    ds_advertised_count = List.length configs;
    ds_scale_add = false;
    ds_evidence = evidence;
  }
