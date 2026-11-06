(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix ([SCoopmat]) for the CUDA-C generator, which HIP shares.

    One generated source serves both toolchains, so every construct here is
    emitted twice under the platform guard the generator already uses for its
    f32 barrier: the [__HIP_PLATFORM_AMD__] arm is RDNA3 (gfx11) WMMA through
    the clang builtins, the other arm is [nvcuda::wmma]. See
    specs/coopmat-hip-wmma.md for the configurations, the per-lane layout and
    the reasons for both choices.

    {b What is validated where.} The AMD arm is executed on an RX 7900 XTX
    (gfx1100) against the interpreter: bit-identical for the integer
    configurations, within the derived bound for f16. The NVIDIA arm is
    compile-checked only (NVRTC, and ptxas/nvdisasm when present): no NVIDIA GPU
    was available to run it. The CUDA/C backend does probe [coopmat]
    ([Cuda_coopmat.support]: f16 from sm_70, 8-bit from sm_72, never on ZLUDA),
    so on an NVIDIA device the launch gate admits what this arm emits. The
    CUDA/PTX backend does not use this module: it lowers to [wmma.*]
    instructions itself ({!Sarek_ir_ptx_coopmat}), for the same configurations.
*)

open Sarek_ir_types
module C = Sarek_coopmat_types

(* The CUDA generator's error module, so a refusal here is the same exception,
   tagged the same way, as every other CUDA/HIP codegen refusal. *)
module Codegen_error = Sarek_backend_error.Backend_error.Make (struct
  let name = "CUDA"
end)

let unsupported what why =
  Codegen_error.raise_error (Codegen_error.unsupported_construct what why)

(** {1 What the lowering accepts} *)

let shape_16 = {C.m = 16; n = 16; k = 16}

let is_8bit = function C.Uint8 | C.Sint8 -> true | _ -> false

let is_int32 = function C.Uint32 | C.Sint32 -> true | _ -> false

(** Refuse, at code generation, a fragment this lowering cannot represent. Total
    over the fragment type: every case either names its representation or raises
    with the reason. *)
let check_fragment (f : C.fragment) =
  if f.C.frag_shape <> shape_16 then
    unsupported
      "cooperative-matrix shape"
      (Printf.sprintf
         "CUDA/HIP: only 16x16x16 is lowered (RDNA3 WMMA and nvcuda::wmma \
          m16n16k16); got %s"
         (C.shape_name f.C.frag_shape)) ;
  if f.C.frag_scope <> C.Subgroup then
    unsupported
      "cooperative-matrix scope"
      (Printf.sprintf
         "CUDA/HIP: only subgroup (wavefront / warp) scope exists; got %s"
         (C.scope_name f.C.frag_scope)) ;
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), (C.Float16 | C.BFloat16 | C.Uint8 | C.Sint8) ->
      ()
  | C.Accumulator, (C.Float32 | C.Uint32 | C.Sint32) -> ()
  | use, comp ->
      unsupported
        "cooperative-matrix fragment"
        (Printf.sprintf
           "CUDA/HIP: no %s fragment of %s (operands are f16, bf16, u8 or s8; \
            accumulators are f32, s32 or u32)"
           (C.use_name use)
           (C.component_name comp))

(** The configurations the lowering emits a multiply-add for. *)
let check_config (cfg : C.config) =
  let a, b, c, r = C.fragments_of_config cfg in
  List.iter check_fragment [a; b; c; r] ;
  if cfg.C.cfg_saturating then
    unsupported
      "saturating cooperative-matrix accumulation"
      "CUDA/HIP: the saturating variant computes a different function and the \
       interpreter, the oracle it would be checked against, refuses it" ;
  let f16 =
    (cfg.C.cfg_a = C.Float16 || cfg.C.cfg_a = C.BFloat16)
    && cfg.C.cfg_b = cfg.C.cfg_a
  in
  let int8 = is_8bit cfg.C.cfg_a && is_8bit cfg.C.cfg_b in
  let ok =
    (f16 && cfg.C.cfg_c = C.Float32 && cfg.C.cfg_result = C.Float32)
    || (int8 && is_int32 cfg.C.cfg_c && cfg.C.cfg_c = cfg.C.cfg_result)
  in
  if not ok then
    unsupported
      "cooperative-matrix configuration"
      (Printf.sprintf
         "CUDA/HIP: %s is not lowered (f16/bf16*same+f32->f32 and \
          8-bit*8-bit+i32->i32 with the same 32-bit accumulator type are)"
         (C.config_name cfg))

(** The kernel buffer element type a fragment may be loaded from or stored to.
    Integer accumulators of either signedness live in an [int32] buffer, and
    both 8-bit operand types in a [uint8] buffer (an s8 fragment also accepts an
    [int8] one, see {!accepts_buffer}): the signedness is the fragment's,
    applied by the instruction, exactly as the interpreter narrows the loaded
    byte. *)
let buffer_elttype (f : C.fragment) =
  match f.C.frag_component with
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

let elttype_name = function
  | TFloat16 -> "float16"
  | TFloat32 -> "float32"
  | TUint8 -> "uint8"
  | TInt8 -> "int8"
  | TInt32 -> "int32"
  | TInt64 -> "int64"
  | TFloat64 -> "float64"
  | _ -> "a non-numeric type"

(** An s8 fragment also loads from an [int8] buffer: the bytes are the same, and
    it is the natural home of int8-quantised data. [uint8] stays accepted. *)
let accepts_buffer (f : C.fragment) elt =
  elt = buffer_elttype f || (f.C.frag_component = C.Sint8 && elt = TInt8)

(** [buffers] maps a parameter or shared-array name to its element type, when
    the generator knows it. An unknown name is left to the C++ compiler, which
    rejects a mismatch too: each helper below takes exactly one pointer type. *)
let check_buffer ~buffers ~buf (f : C.fragment) =
  match List.assoc_opt buf buffers with
  | None -> ()
  | Some elt ->
      let want = buffer_elttype f in
      if not (accepts_buffer f elt) then
        unsupported
          "cooperative-matrix buffer"
          (Printf.sprintf
             "a %s %s fragment needs a %s buffer, but %s holds %s"
             (C.component_name f.C.frag_component)
             (C.use_name f.C.frag_use)
             (elttype_name want)
             buf
             (elttype_name elt))

(** Element types of the kernel's array parameters and shared arrays. A shared
    array is declared either in [kern_locals] ([DShared]) or in the body, as
    [SLet (v, EArrayCreate (elt, _, Shared), _)] (how the matrix rewrite
    declares its staging arrays); private arrays are not buffers. *)
let kernel_buffers (k : kernel) =
  let of_param = function
    | DParam (v, Some {arr_elttype; _}) -> Some (v.var_name, arr_elttype)
    | DParam ({var_type = TVec elt; var_name; _}, None) -> Some (var_name, elt)
    | DParam _ | DLocal _ -> None
    | DShared (name, elt, _) -> Some (name, elt)
  in
  let declared =
    Sarek_ir_analysis.kernel_fold
      {
        fe = (fun acc _ -> acc);
        ft = (fun acc _ -> acc);
        fs =
          (fun acc -> function
            | SLet (v, EArrayCreate (elt, _, Shared), _) ->
                (v.var_name, elt) :: acc
            | _ -> acc);
        fnative = (fun acc -> acc);
        visit_lvalue = false;
      }
      []
      k
  in
  List.filter_map of_param (k.kern_params @ k.kern_locals) @ List.rev declared

(** The layout every [CM_load] of an operand fragment uses, by fragment name.
    [nvcuda::wmma] fixes an operand's layout in its fragment TYPE, so the CUDA
    arm of the declaration needs it; a name loaded in both layouts has no CUDA
    spelling. Keyed by name across the whole kernel, scopes ignored: two
    fragments that share a name in different blocks and are loaded in different
    layouts get the [#error] too (conservative, never wrong code). *)
type operand_layout = Loaded of C.layout | Both_layouts

type context = {
  buffers : (string * elttype) list;
      (** element types of the kernel's array parameters and shared arrays *)
  operand_layouts : (string * operand_layout) list;
  shared : string list;  (** the kernel's [.shared] arrays *)
  globals : string list;
      (** the kernel's array parameters: the only operands whose base is known
          to be 32-byte aligned (device allocations) *)
  mutable proven : coopmat_op list;
      (** the loads and stores proven aligned ({!Sarek_ir_coopmat_align}), set
          by the generator for the body it emits; physical identity *)
  warps : int option;
      (** warps per block when the launch block is known: the size of the
          staging buffer. [None]: a misaligned access traps. *)
  mutable tiles_used : bool;  (** a staging path was emitted *)
}

let operand_layouts (k : kernel) =
  let note acc name layout =
    match List.assoc_opt name acc with
    | None -> (name, Loaded layout) :: acc
    | Some (Loaded l) when l = layout -> acc
    | Some _ -> (name, Both_layouts) :: List.remove_assoc name acc
  in
  List.fold_left
    (fun acc -> function
      | CM_load {dst; frag; layout; _} when frag.C.frag_use <> C.Accumulator ->
          note acc dst layout
      | CM_decl _ | CM_load _ | CM_store _ | CM_muladd _ | CM_scale_add _ -> acc)
    []
    (Sarek_ir_analysis.kernel_coopmat_ops k)

(** The kernel's [.shared] arrays, declared in [kern_locals] or in the body. *)
let shared_arrays (k : kernel) =
  let declared =
    Sarek_ir_analysis.kernel_fold
      {
        fe = (fun acc _ -> acc);
        ft = (fun acc _ -> acc);
        fs =
          (fun acc -> function
            | SLet (v, EArrayCreate (_, _, Shared), _) -> v.var_name :: acc
            | _ -> acc);
        fnative = (fun acc -> acc);
        visit_lvalue = false;
      }
      []
      k
  in
  List.filter_map
    (function DShared (name, _, _) -> Some name | _ -> None)
    k.kern_locals
  @ declared

(** The kernel's array parameters. *)
let global_arrays (k : kernel) =
  List.filter_map
    (function
      | DParam (v, Some _) -> Some v.var_name
      | DParam ({var_type = TVec _; var_name; _}, None) -> Some var_name
      | DParam _ | DLocal _ | DShared _ -> None)
    k.kern_params

(** What the [SCoopmat] arm needs from the kernel it is emitting. [block] is the
    launch block when it is known. *)
let kernel_context ?block (k : kernel) =
  {
    buffers = kernel_buffers k;
    operand_layouts = operand_layouts k;
    shared = shared_arrays k;
    globals = global_arrays k;
    proven = [];
    warps = Option.map (fun (bx, by, bz) -> bx * by * bz / 32) block;
    tiles_used = false;
  }

(** Bytes of staging shared memory per warp: one 16x16 tile of the widest
    element (4 bytes). *)
let tile_bytes_per_warp = 1024

(** File-scope declarations of the per-warp staging buffer, when a staging path
    was emitted; empty otherwise. Not compiled for HIP (the AMD arm has no
    alignment precondition). *)
let tiles_declaration (ctx : context) =
  match (ctx.tiles_used, ctx.warps) with
  | true, Some warps ->
      Printf.sprintf
        "#if !defined(__HIP_PLATFORM_AMD__)\n\
         __shared__ __align__(32) unsigned char sarek_cm_tiles[%d];\n\
         __device__ __forceinline__ unsigned char *sarek_cm_tile() {\n\
        \  unsigned w = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z \
         * blockDim.x * blockDim.y) >> 5;\n\
        \  if (w >= %d) __trap();\n\
        \  return sarek_cm_tiles + w * %d;\n\
         }\n\
         #endif\n"
        (warps * tile_bytes_per_warp)
        warps
        tile_bytes_per_warp
  | _ -> ""

(** {1 Preamble} *)

(** The helper definitions, emitted before [extern "C"] when the kernel has a
    cooperative-matrix statement: {!Sarek_ir_cuda_coopmat_preamble}, with the
    [CM_scale_add] helper only when the kernel has that statement. *)
let preamble (k : kernel) =
  Sarek_ir_cuda_coopmat_preamble.text
    ~scale_add:
      (List.exists
         (function
           | CM_scale_add _ -> true
           | CM_decl _ | CM_load _ | CM_store _ | CM_muladd _ -> false)
         (Sarek_ir_analysis.kernel_coopmat_ops k))

(** The header comment §6.1 of docs/design/f16-relaxed-accuracy.md requires on a
    generated source whose results are bounded rather than exact. *)
let relaxed_comment (k : kernel) =
  let relaxed =
    List.filter
      (fun cfg -> C.regime cfg = C.Relaxed_bounded)
      (Sarek_ir_analysis.kernel_coopmat_configs k)
  in
  match relaxed with
  | [] -> ""
  | cfgs ->
      Printf.sprintf
        "/* Sarek: %s accumulates in floating point in an order the hardware \
         chooses. Results are within gamma_n * (sum|A*B| + |C|) of the exact \
         value (u = 2^-24), NOT bit-identical to the interpreter: Regime B of \
         docs/design/f16-relaxed-accuracy.md; launching it requires \
         Sarek_coopmat.accept_relaxed_accumulation. */\n"
        (String.concat ", " (List.map C.config_name cfgs))

(** {1 Statements} *)

let amd_type (f : C.fragment) =
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), C.Float16 -> "sarek_cm_h16"
  | (C.Matrix_a | C.Matrix_b), C.BFloat16 -> "sarek_cm_s16"
  | (C.Matrix_a | C.Matrix_b), (C.Uint8 | C.Sint8) -> "sarek_cm_i4"
  | C.Accumulator, C.Float32 -> "sarek_cm_f8"
  | C.Accumulator, (C.Uint32 | C.Sint32) -> "sarek_cm_i8"
  | use, comp ->
      unsupported
        "cooperative-matrix fragment"
        (Printf.sprintf
           "HIP: no %s fragment of %s"
           (C.use_name use)
           (C.component_name comp))

(** Suffix of the AMD helper for a fragment in [layout]: [a_f16], [b_i8],
    [acc_f32], [acc_cm_f32]... An operand helper is named by its access pattern
    (see the preamble), so a column-major A takes the [b] helper and a
    column-major B the [a] one. *)
let amd_helper_suffix (f : C.fragment) layout =
  let role =
    match (f.C.frag_use, layout) with
    | C.Matrix_a, C.Row_major | C.Matrix_b, C.Col_major -> "a"
    | C.Matrix_a, C.Col_major | C.Matrix_b, C.Row_major -> "b"
    | C.Accumulator, C.Row_major -> "acc"
    | C.Accumulator, C.Col_major -> "acc_cm"
  in
  let ty =
    match f.C.frag_component with
    | C.Float16 -> "f16"
    | C.BFloat16 -> "bf16"
    | C.Float32 -> "f32"
    | C.Uint8 | C.Sint8 -> "i8"
    | C.Uint32 | C.Sint32 -> "i32"
  in
  role ^ "_" ^ ty

let cuda_elt = function
  | C.Float16 -> "__half"
  | C.BFloat16 -> "__nv_bfloat16"
  | C.Float32 -> "float"
  | C.Uint8 -> "unsigned char"
  | C.Sint8 -> "signed char"
  | C.Uint32 | C.Sint32 -> "int"

let cuda_elt_bytes = function
  | C.Float16 | C.BFloat16 -> 2
  | C.Float32 | C.Uint32 | C.Sint32 -> 4
  | C.Uint8 | C.Sint8 -> 1

let cuda_layout = function
  | C.Row_major -> "row_major"
  | C.Col_major -> "col_major"

(* An operand's layout is part of its nvcuda::wmma fragment type: [layout] is
   the one every load of it uses (row-major when it is never loaded). *)
let cuda_type (f : C.fragment) layout =
  let w = "nvcuda::wmma::" in
  let elt = cuda_elt f.C.frag_component in
  let operand role =
    Printf.sprintf
      "%sfragment<%s%s, 16, 16, 16, %s, %s%s>"
      w
      w
      role
      elt
      w
      (cuda_layout layout)
  in
  match f.C.frag_use with
  | C.Matrix_a -> operand "matrix_a"
  | C.Matrix_b -> operand "matrix_b"
  | C.Accumulator ->
      Printf.sprintf "%sfragment<%saccumulator, 16, 16, 16, %s>" w w elt

(** The two arms of one statement, under the platform guard. Preprocessor
    directives start at column 0. *)
let both_arms ~indent ~amd ~cuda =
  Printf.sprintf
    "#if defined(__HIP_PLATFORM_AMD__)\n%s%s\n#else\n%s%s\n#endif\n"
    indent
    amd
    indent
    cuda

(* The CUDA arm of a bf16 statement: nvcuda::wmma's bf16 fragments read a
   bf16 buffer, not a float32 buffer rounded on load, so there is no faithful
   CUDA spelling of this IR. A named #error, as for mixed-sign 8-bit (the CUDA
   launch gate refuses first: the CUDA backend probes no configuration). *)
let cuda_bf16_unsupported =
  "\n\
   #error \"Sarek cooperative matrix: bf16 operands loaded from float32 are \
   lowered on HIP gfx11 only (specs/matrix-pattern-rewrite.md)\"\n"

let cuda_arm (frag : C.fragment) cuda =
  if frag.C.frag_component = C.BFloat16 then cuda_bf16_unsupported else cuda

(* The CUDA arm of an operand loaded in both layouts: valid on HIP, whose
   fragment has no layout, so a named #error rather than a refusal of the
   shared source. *)
let cuda_both_layouts name =
  Printf.sprintf
    "\n\
     #error \"Sarek cooperative matrix: nvcuda::wmma fixes a fragment's layout \
     in its type, and %s is loaded both row- and column-major\"\n"
    name

let gen_decl ~indent ~ctx name (frag : C.fragment) =
  check_fragment frag ;
  let declare layout =
    Printf.sprintf
      "%s %s; nvcuda::wmma::fill_fragment(%s, 0);"
      (cuda_type frag layout)
      name
      name
  in
  let cuda =
    match List.assoc_opt name ctx.operand_layouts with
    | Some Both_layouts -> cuda_both_layouts name
    | Some (Loaded layout) -> declare layout
    | None -> declare C.Row_major
  in
  both_arms
    ~indent
    ~amd:(Printf.sprintf "%s %s = {};" (amd_type frag) name)
    ~cuda:(cuda_arm frag cuda)

(* The CUDA pointer for a fragment's buffer: [src + index], cast to the wmma
   element type (an s8 fragment reads a uint8 or int8 buffer as signed char). *)
let cuda_ptr ~const (frag : C.fragment) ~buf ~index =
  Printf.sprintf
    "((%s%s *)(%s) + (%s))"
    (if const then "const " else "")
    (cuda_elt frag.C.frag_component)
    buf
    index

(* nvcuda::wmma's memory-layout argument of an accumulator load or store. *)
let cuda_mem_layout = function
  | C.Row_major -> "nvcuda::wmma::mem_row_major"
  | C.Col_major -> "nvcuda::wmma::mem_col_major"

(** The CUDA arm of a load or store. Proven aligned (and not from a [.shared]
    array, whose alignment this generator does not declare): the bare [wmma]
    call. Otherwise, with a known block: a warp-uniform test, the bare call when
    aligned and the staged one when not. Block unknown: [sarek_cm_require],
    which traps. *)
let guarded ~(ctx : context) ~op ~buf ~(frag : C.fragment) ~ptr ~stride ~direct
    ~staged =
  let proven =
    List.mem buf ctx.globals && Sarek_ir_coopmat_align.is_proven ctx.proven op
  in
  let bytes = cuda_elt_bytes frag.C.frag_component in
  if proven then direct
  else
    match ctx.warps with
    | None ->
        Printf.sprintf
          "sarek_cm_require(%s, (%s), %d); %s"
          ptr
          stride
          bytes
          direct
    | Some _ ->
        ctx.tiles_used <- true ;
        let t = cuda_elt frag.C.frag_component in
        Printf.sprintf
          "if (sarek_cm_misaligned(%s, (%s), %d)) { %s *sarek_cm_t = (%s \
           *)sarek_cm_tile(); %s } else { %s }"
          ptr
          stride
          bytes
          t
          t
          (staged "sarek_cm_t")
          direct

let gen_load ~ctx ~op ~indent ~buffers ~expr_str ~dst ~(frag : C.fragment) ~src
    ~index ~stride ~layout =
  check_fragment frag ;
  check_buffer ~buffers ~buf:src frag ;
  let index = expr_str index and stride = expr_str stride in
  let ptr = cuda_ptr ~const:true frag ~buf:src ~index in
  let mem_layout =
    match frag.C.frag_use with
    | C.Accumulator -> ", " ^ cuda_mem_layout layout
    | C.Matrix_a | C.Matrix_b -> ""
  in
  (* The AMD load helpers take [const unsigned char *] for both 8-bit
     signednesses; an int8 buffer is [signed char *], which C++ does not
     convert implicitly. A uint8 buffer is passed as before. *)
  let amd_src =
    match List.assoc_opt src buffers with
    | Some TInt8 -> Printf.sprintf "(const unsigned char *)(%s)" src
    | _ -> src
  in
  both_arms
    ~indent
    ~amd:
      (Printf.sprintf
         "sarek_cm_load_%s(%s, %s, (%s), (%s));"
         (amd_helper_suffix frag layout)
         dst
         amd_src
         index
         stride)
    ~cuda:
      (cuda_arm
         frag
         (guarded
            ~ctx
            ~op
            ~buf:src
            ~frag
            ~ptr
            ~stride
            ~direct:
              (Printf.sprintf
                 "nvcuda::wmma::load_matrix_sync(%s, %s, (%s)%s);"
                 dst
                 ptr
                 stride
                 mem_layout)
            ~staged:(fun tile ->
              Printf.sprintf
                "sarek_cm_stage_in(%s, %s, (%s)); __syncwarp(); \
                 nvcuda::wmma::load_matrix_sync(%s, %s, 16%s); __syncwarp();"
                tile
                ptr
                stride
                dst
                tile
                mem_layout)))

let gen_store ~ctx ~op ~indent ~buffers ~expr_str ~src ~(frag : C.fragment) ~dst
    ~index ~stride ~layout =
  check_fragment frag ;
  if frag.C.frag_use <> C.Accumulator then
    unsupported
      "cooperative-matrix store"
      "CUDA/HIP: only an accumulator fragment can be stored (nvcuda::wmma has \
       no store for matrix_a / matrix_b)" ;
  check_buffer ~buffers ~buf:dst frag ;
  let index = expr_str index and stride = expr_str stride in
  let ptr = cuda_ptr ~const:false frag ~buf:dst ~index in
  both_arms
    ~indent
    ~amd:
      (Printf.sprintf
         "sarek_cm_store_%s(%s, %s, (%s), (%s));"
         (amd_helper_suffix frag layout)
         src
         dst
         index
         stride)
    ~cuda:
      (guarded
         ~ctx
         ~op
         ~buf:dst
         ~frag
         ~ptr
         ~stride
         ~direct:
           (Printf.sprintf
              "nvcuda::wmma::store_matrix_sync(%s, %s, (%s), %s);"
              ptr
              src
              stride
              (cuda_mem_layout layout))
         ~staged:(fun tile ->
           Printf.sprintf
             "nvcuda::wmma::store_matrix_sync(%s, %s, 16, %s); __syncwarp(); \
              sarek_cm_stage_out(%s, %s, (%s)); __syncwarp();"
             tile
             src
             (cuda_mem_layout layout)
             ptr
             tile
             stride))

let gen_muladd ~indent ~dst ~a ~b ~c (cfg : C.config) =
  check_config cfg ;
  let amd =
    if cfg.C.cfg_a = C.Float16 || cfg.C.cfg_a = C.BFloat16 then
      Printf.sprintf
        "%s = __builtin_amdgcn_wmma_f32_16x16x16_%s_w32(%s, %s, %s);"
        dst
        (if cfg.C.cfg_a = C.Float16 then "f16" else "bf16")
        a
        b
        c
    else
      let signed comp = if comp = C.Sint8 then "true" else "false" in
      Printf.sprintf
        "%s = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(%s, %s, %s, %s, %s, \
         false);"
        dst
        (signed cfg.C.cfg_a)
        a
        (signed cfg.C.cfg_b)
        b
        c
  in
  let cuda =
    if cfg.C.cfg_a = C.BFloat16 then cuda_bf16_unsupported
    else if cfg.C.cfg_a = C.Float16 then
      Printf.sprintf "nvcuda::wmma::mma_sync(%s, %s, %s, %s);" dst a b c
    else if cfg.C.cfg_a <> cfg.C.cfg_b then
      (* nvcuda::wmma's integer mma_sync takes two signed or two unsigned
         operands, never one of each. The AMD arm is valid, so this cannot be
         a codegen-time refusal of the shared source; the CUDA toolchain gets
         a named #error instead (and the CUDA launch gate refuses first). *)
      Printf.sprintf
        "\n\
         #error \"Sarek cooperative matrix: nvcuda::wmma has no mixed-sign \
         8-bit multiply (%s)\"\n"
        (C.config_name cfg)
    else
      Printf.sprintf "nvcuda::wmma::mma_sync(%s, %s, %s, %s, false);" dst a b c
  in
  both_arms ~indent ~amd ~cuda

(** [gen_op ~indent ~ctx ~expr_str op] is the source for one cooperative matrix
    statement, both arms; [ctx] is {!kernel_context} of the kernel. *)
let gen_op ~indent ~ctx ~expr_str (op : coopmat_op) =
  match op with
  | CM_decl {name; frag} -> gen_decl ~indent ~ctx name frag
  | CM_load {dst; frag; src; index; stride; layout} ->
      let buffers = ctx.buffers in
      gen_load
        ~ctx
        ~op
        ~indent
        ~buffers
        ~expr_str
        ~dst
        ~frag
        ~src
        ~index
        ~stride
        ~layout
  | CM_store {src; frag; dst; index; stride; layout} ->
      let buffers = ctx.buffers in
      gen_store
        ~ctx
        ~op
        ~indent
        ~buffers
        ~expr_str
        ~src
        ~frag
        ~dst
        ~index
        ~stride
        ~layout
  | CM_muladd {dst; a; b; c; cfg} -> gen_muladd ~indent ~dst ~a ~b ~c cfg
  | CM_scale_add {dst; dst_frag; src; src_frag; row; col; offset; accumulate} ->
      check_fragment dst_frag ;
      check_fragment src_frag ;
      let amd =
        Sarek_ir_cuda_coopmat_scale.amd_call
          ~buffers:ctx.buffers
          ~expr_str
          ~dst
          ~dst_frag
          ~src
          ~src_frag
          ~row
          ~col
          ~offset
          ~accumulate
      in
      both_arms ~indent ~amd ~cuda:Sarek_ir_cuda_coopmat_scale.cuda_error

(** {1 Launch shape} *)

(** The lowering runs the matrix instruction on whole wavefronts (wave32 on
    gfx11) and whole warps; a block whose thread count is not a multiple of 32
    has a partial wave whose inactive lanes would feed the instruction undefined
    operands. Refused before any source is compiled. *)
let check_block (k : kernel) (block : int * int * int) =
  let bx, by, bz = block in
  let threads = bx * by * bz in
  if threads mod 32 <> 0 && Sarek_ir_analysis.kernel_has_coopmat_op k then
    unsupported
      "cooperative-matrix launch shape"
      (Printf.sprintf
         "CUDA/HIP: a cooperative-matrix kernel must run whole 32-wide \
          wavefronts/warps, but the block has %d threads (%dx%dx%d)"
         threads
         bx
         by
         bz)
