(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The CUDA cooperative-matrix probe (specs/coopmat-hip-wmma.md,
    specs/matrix-pattern-rewrite.md), shared by the CUDA/C and CUDA/PTX
    backends.

    The CUDA driver has no query that lists matrix-instruction configurations,
    so the probe is a rule over two facts it does report: the compute capability
    and the warp size. The configurations offered are the ones both lowerings
    emit ([Sarek_ir_cuda_coopmat]'s [nvcuda::wmma] arm and
    [Sarek_ir_ptx_coopmat]'s [wmma.*] instructions): 16x16x16 on 32-wide warps,
    f16 operands with an f32 accumulator from sm_70, and 8-bit operands of the
    SAME signedness with a 32-bit integer accumulator from sm_72 (the mixed-sign
    arms and bf16 loaded from float32 are refused by both, so they are not
    offered). [lowering] only labels the evidence.

    Executed in this repository only through ZLUDA on gfx1100 (which reports
    compute capability 8.8 and rejects [wmma.*]), never on NVIDIA hardware. *)

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

let f16_configs = [config C.Float16 C.Float16 C.Float32]

(** Which CUDA backend the probe answers for. *)
type lowering = Cuda_c | Ptx

let lowering_name = function Cuda_c -> "CUDA/C" | Ptx -> "CUDA/PTX"

(* Both accumulator signednesses: a 32-bit wrapping sum has the same bits
   either way, and the lowering accepts both. *)
let int8_configs =
  List.concat_map
    (fun operand ->
      [config operand operand C.Sint32; config operand operand C.Uint32])
    [C.Uint8; C.Sint8]

(** ZLUDA (a CUDA driver API over ROCm) rejects every [wmma.*] instruction and
    8-bit [mma] with unsigned operands: the module fails to load
    ([cuModuleGetFunction] -> CUDA_ERROR_NOT_FOUND), measured on gfx1100
    (specs/zluda-mma-nvrtc13.md, F-1;
    docs/measurements/zluda-mma-probe-2026-10-11). Its devices carry "ZLUDA" in
    Its devices carry "ZLUDA" in their name and report compute capability 8.8,
    which no NVIDIA part does. *)
let is_zluda ~device_name ~compute_capability =
  let affix = "ZLUDA" in
  let n = String.length device_name and m = String.length affix in
  let rec at i =
    i + m <= n && (String.sub device_name i m = affix || at (i + 1))
  in
  at 0 || compute_capability = (8, 8)

(** [support_for ~lowering ~device_name ~compute_capability ~warp_size] is the
    rule, separated from the device record so a test can drive it with any
    device's facts. *)
let support_for ~lowering ~device_name ~compute_capability:(major, minor)
    ~warp_size =
  let at_least (m, n) = (major, minor) >= (m, n) in
  let zluda = is_zluda ~device_name ~compute_capability:(major, minor) in
  let who = lowering_name lowering in
  let int8 = at_least (7, 2) in
  let configs =
    if warp_size <> 32 || zluda then []
    else
      (if at_least (7, 0) then f16_configs else [])
      @ if int8 then int8_configs else []
  in
  {
    C.ds_configs = configs;
    ds_robust_buffer_access = false;
    ds_subgroup_size = warp_size;
    ds_advertised_count = List.length configs;
    ds_evidence =
      (if configs <> [] then
         Printf.sprintf
           "%s: compute capability %d.%d, %d-wide warps: %s 16x16x16"
           who
           major
           minor
           warp_size
           (match lowering with
           | Cuda_c -> "nvcuda::wmma"
           | Ptx -> "wmma.* instructions")
       else if zluda then
         Printf.sprintf
           "%s: ZLUDA rejects wmma.* and unsigned 8-bit mma (the module does \
            not load); the rewrite is not offered there"
           who
       else
         Printf.sprintf
           "%s: wmma needs compute capability 7.0 (f16)%s and 32-wide warps; \
            this device reports %d.%d and warp size %d"
           who
           " or 7.2 (8-bit)"
           major
           minor
           warp_size);
  }

(** The CUDA/C rule. *)
let support = support_for ~lowering:Cuda_c
