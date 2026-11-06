(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix ([SCoopmat]) for the PTX generator: the [wmma.*]
    instructions of the PTX ISA (tensor cores from sm_70, PTX ISA 6.0).

    The lowering accepts exactly what the [nvcuda::wmma] arm of
    {!Sarek_ir_cuda_coopmat} accepts: 16x16x16 on subgroup (warp) scope, any
    layout of A (row/col), B (row/col) and of the accumulator in memory, and two
    families of configuration.
    - f16 operands with an f32 accumulator (sm_70);
    - 8-bit operands of the SAME signedness (s8*s8 or u8*u8) with a 32-bit
      integer accumulator of either signedness (sm_72),
      [wmma.mma ... .s32.s8.s8.s32] or [.s32.u8.u8.s32]: a 32-bit wrapping sum
      has the same bits either way. Mixed-sign operands have no [wmma] form and
      are refused (a named [#error] on CUDA/C). The non-saturating form only.
      bf16 is refused by both: [nvcuda::wmma]'s bf16 fragment reads a bf16
      buffer, not a float32 buffer rounded on load.

    [wmma] and not [mma.sync]: [mma.sync.m16n8k16] needs sm_80, and the Turing
    form [m16n8k8] is not translated by ZLUDA (specs/zluda-mma-nvrtc13.md).
    [wmma] is the one form sm_70 and later share, and CUDA/C already lowers to
    it.

    {b Fragments.} PTX fixes the per-lane register group of a [wmma] fragment
    (m16n16k16): an f16 operand is 8 [.b32] registers each holding an f16x2
    pair, an 8-bit operand 2 [.b32] registers of four bytes, an f32 accumulator
    8 [.f32] registers and an integer accumulator 8 [.b32] registers. The
    emitter never indexes into a group; it passes the whole group to [wmma.load]
    / [wmma.mma] / [wmma.store], which agree on the distribution across lanes.

    {b Precondition, checked at run time.} Exactly the CUDA/C one
    ([sarek_cm_require]): the element address is 32-byte aligned and
    [stride * element bytes] is a multiple of 16, otherwise the kernel executes
    [trap]. [wmma.load]/[wmma.store] are undefined on a misaligned address; a
    trap is preferable to wrong results. A [.shared] array a cooperative-matrix
    statement accesses is declared 32-byte aligned, so only a misaligned offset
    into it traps.

    {b What is validated where.} Text and refusals: unit tests, any machine. The
    instructions are checked by [ptxas] where it is installed. Execution needs
    an NVIDIA GPU of compute capability 7.0 or later; ZLUDA rejects [wmma.*]. *)

open Sarek_ir_types
open Sarek_ir_ptx_types
open Sarek_ir_ptx_mem
open Sarek_ir_ptx_expr
module C = Sarek_coopmat_types

(** First SM with [wmma], and with 8-bit [wmma]. *)
let min_sm = 70

let min_sm_int8 = 72

let shape_16 = {C.m = 16; n = 16; k = 16}

(** {1 What the lowering accepts} *)

let refuse what why =
  unsupported (Printf.sprintf "cooperative matrix: %s: %s" what why)

(** Refuse a fragment this lowering cannot represent. Total over the fragment
    type: every case is accepted or raises with the reason. *)
let check_fragment (f : C.fragment) =
  if f.C.frag_shape <> shape_16 then
    refuse
      "shape"
      (Printf.sprintf
         "CUDA/PTX: only 16x16x16 is lowered (wmma m16n16k16); got %s"
         (C.shape_name f.C.frag_shape)) ;
  if f.C.frag_scope <> C.Subgroup then
    refuse
      "scope"
      (Printf.sprintf
         "CUDA/PTX: only subgroup (warp) scope exists; got %s"
         (C.scope_name f.C.frag_scope)) ;
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), (C.Float16 | C.Uint8 | C.Sint8)
  | C.Accumulator, (C.Float32 | C.Uint32 | C.Sint32) ->
      ()
  | (C.Matrix_a | C.Matrix_b), C.BFloat16 ->
      refuse
        "bf16 operand"
        "no faithful lowering: wmma's bf16 fragment reads a bf16 buffer, not a \
         float32 buffer rounded on load (CUDA/C refuses it too)"
  | use, comp ->
      refuse
        "fragment"
        (Printf.sprintf
           "CUDA/PTX: no %s fragment of %s (operands are f16, s8 or u8; \
            accumulators are f32, s32 or u32)"
           (C.use_name use)
           (C.component_name comp))

let is_8bit = function C.Uint8 | C.Sint8 -> true | _ -> false

let is_int32 = function C.Uint32 | C.Sint32 -> true | _ -> false

(** The configurations a multiply-add is emitted for: f16*f16+f32->f32, and
    8-bit operands of one signedness with the same 32-bit integer accumulator
    type as addend and result (as the CUDA/C arm). The four fragments are
    checked first. *)
let check_config (cfg : C.config) =
  List.iter
    check_fragment
    (let a, b, c, r = C.fragments_of_config cfg in
     [a; b; c; r]) ;
  if cfg.C.cfg_saturating then
    refuse
      "saturating accumulation"
      "CUDA/PTX: the saturating variant computes a different function, and the \
       interpreter, the oracle it is checked against, refuses it" ;
  let int8 = is_8bit cfg.C.cfg_a && is_8bit cfg.C.cfg_b in
  if int8 && cfg.C.cfg_a <> cfg.C.cfg_b then
    refuse
      "mixed-sign 8-bit multiply"
      (Printf.sprintf
         "wmma has no form with one signed and one unsigned operand (%s)"
         (C.config_name cfg)) ;
  let f16 =
    cfg.C.cfg_a = C.Float16 && cfg.C.cfg_b = C.Float16
    && cfg.C.cfg_c = C.Float32
    && cfg.C.cfg_result = C.Float32
  in
  let int8_ok =
    int8 && is_int32 cfg.C.cfg_c && cfg.C.cfg_c = cfg.C.cfg_result
  in
  if not (f16 || int8_ok) then
    refuse
      "configuration"
      (Printf.sprintf
         "CUDA/PTX: %s is not lowered (f16*f16+f32->f32 and \
          8-bit*8-bit+i32->i32 with the same operand signedness and the same \
          32-bit accumulator type are)"
         (C.config_name cfg))

(** {1 Layouts} *)

type operand_layout = Sarek_ir_cuda_coopmat.operand_layout

(** The layout of every operand fragment of the kernel, by name; refuses a name
    loaded in both layouts. [wmma.mma] names the layouts of A and B and expects
    the fragments to have been loaded in them, so a fragment has exactly one
    (the CUDA/C arm fixes the same thing in the fragment's type). *)
let operand_layouts (k : kernel) : (string * C.layout) list =
  List.map
    (fun (name, (l : operand_layout)) ->
      match l with
      | Sarek_ir_cuda_coopmat.Loaded layout -> (name, layout)
      | Sarek_ir_cuda_coopmat.Both_layouts ->
          refuse
            "operand layout"
            (Printf.sprintf
               "wmma.mma fixes the layout of an operand fragment, and %s is \
                loaded both row- and column-major"
               name))
    (Sarek_ir_cuda_coopmat.operand_layouts k)

(** The arrays a load or store of the kernel accesses. A [.shared] one among
    them is declared 32-byte aligned ({!Sarek_ir_ptx_types.shared_align}), so a
    tile at index 0 passes the address check whatever its neighbours are. *)
let accessed_buffers (k : kernel) : string list =
  List.filter_map
    (function
      | CM_load {src; _} -> Some src
      | CM_store {dst; _} -> Some dst
      | CM_decl _ | CM_muladd _ -> None)
    (Sarek_ir_analysis.kernel_coopmat_ops k)
  |> List.sort_uniq String.compare

(** {1 Kernel-level gate} *)

(** [sm_number "sm_86"] is [86]; a suffix ([sm_90a]) is ignored. *)
let sm_number target =
  let digits =
    String.to_seq target
    |> Seq.filter (fun c -> c >= '0' && c <= '9')
    |> String.of_seq
  in
  int_of_string_opt digits

(** The whole-kernel refusal, run before any instruction is emitted (a per-node
    refusal would name whichever node the emitter reached first, and a [TUint8]
    buffer that is only stored to reaches none). Also the launch shape: the
    instructions run on whole warps, and a partial warp has inactive lanes that
    would feed them undefined operands. *)
let check_kernel ?block ~sm_target (k : kernel) : unit =
  let ops = Sarek_ir_analysis.kernel_coopmat_ops k in
  if ops = [] then begin
    if Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Coopmat k then
      refuse
        "uint8 buffer"
        "a uint8 operand buffer without a cooperative-matrix statement (uint8 \
         is the element type of 8-bit operand buffers only)"
  end
  else begin
    (match sm_number sm_target with
    | Some sm when sm >= min_sm -> ()
    | _ ->
        refuse
          "target"
          (Printf.sprintf
             "wmma needs .target sm_%d or later (PTX ISA 6.0; 8-bit: sm_72); \
              this kernel is generated for %s"
             min_sm
             sm_target)) ;
    List.iter
      (function
        | CM_decl {frag; _} | CM_load {frag; _} | CM_store {frag; _} ->
            check_fragment frag
        | CM_muladd {cfg; _} -> check_config cfg)
      ops ;
    if
      List.exists
        (function
          | CM_decl {frag; _} | CM_load {frag; _} | CM_store {frag; _} ->
              is_8bit frag.C.frag_component
          | CM_muladd {cfg; _} -> is_8bit cfg.C.cfg_a)
        ops
      &&
      match sm_number sm_target with
      | Some sm -> sm < min_sm_int8
      | None -> true
    then
      refuse
        "target"
        (Printf.sprintf
           "8-bit wmma needs .target sm_%d or later; this kernel is generated \
            for %s"
           min_sm_int8
           sm_target) ;
    ignore (operand_layouts k) ;
    Option.iter
      (fun (bx, by, bz) ->
        let threads = bx * by * bz in
        if threads mod 32 <> 0 then
          refuse
            "launch shape"
            (Printf.sprintf
               "CUDA/PTX: a cooperative-matrix kernel must run whole 32-wide \
                warps, but the block has %d threads (%dx%dx%d)"
               threads
               bx
               by
               bz))
      block
  end

(** {1 Statements} *)

let regs_text regs = "{" ^ String.concat ", " regs ^ "}"

let layout_suffix = function C.Row_major -> "row" | C.Col_major -> "col"

(** Buffer element type and byte size a fragment is loaded from or stored to (an
    integer accumulator of either signedness lives in an [int32] buffer, an
    8-bit operand of either signedness in a [uint8] one). *)
let elt_of (f : C.fragment) =
  match f.C.frag_component with
  | C.Float16 -> (TFloat16, 2)
  | C.BFloat16 | C.Float32 -> (TFloat32, 4)
  | C.Uint8 | C.Sint8 -> (TUint8, 1)
  | C.Uint32 | C.Sint32 -> (TInt32, 4)

let elt_name = function
  | TFloat16 -> "float16"
  | TFloat32 -> "float32"
  | TUint8 -> "uint8"
  | TInt32 -> "int32"
  | _ -> "a non-numeric type"

(** The PTX type letters of a fragment component in [wmma.load]/[wmma.store]
    ([f16], [s8], [u8], [f32], [s32]). *)
let ptx_type = function
  | C.Float16 | C.BFloat16 -> "f16"
  | C.Uint8 -> "u8"
  | C.Sint8 -> "s8"
  | C.Float32 -> "f32"
  | C.Uint32 | C.Sint32 -> "s32"

(** Registers per lane of a fragment at m16n16k16. *)
let frag_reg_count (f : C.fragment) =
  match (f.C.frag_use, f.C.frag_component) with
  | (C.Matrix_a | C.Matrix_b), (C.Uint8 | C.Sint8) -> 2
  | _ -> 8

let declared alloc name =
  match Hashtbl.find_opt alloc.coop_frags name with
  | Some cf -> cf
  | None ->
      fail
        (Printf.sprintf
           "PTX codegen: cooperative-matrix fragment '%s' used before its \
            declaration"
           name)

let operand_layout alloc name =
  Option.value (List.assoc_opt name alloc.coop_layouts) ~default:C.Row_major

let emit_decl buf alloc name (frag : C.fragment) =
  check_fragment frag ;
  (* Zero-initialised, as every backend does (Sarek_ir_types.CM_decl). *)
  let regs, zero =
    match (frag.C.frag_use, frag.C.frag_component) with
    | C.Accumulator, C.Float32 ->
        ( List.init 8 (fun _ -> new_f32 alloc),
          fun r -> emit buf "mov.f32 %s, 0F00000000;" r )
    | _ ->
        ( List.init (frag_reg_count frag) (fun _ -> new_frag_b32 alloc),
          fun r -> emit buf "mov.b32 %s, 0;" r )
  in
  List.iter zero regs ;
  Hashtbl.replace alloc.coop_frags name {cf_frag = frag; cf_regs = regs}

(** The element address of a fragment access, after the checks that make the
    access meaningful: the buffer holds the fragment's element type, and is not
    per-thread [.local] memory (a warp-collective access to it has no meaning).
    Returns the state-space word, the address register and the stride register.
*)
let emit_access buf alloc env ~what ~(frag : C.fragment) ~buf_name ~index
    ~stride =
  let elt, elt_bytes = elt_of frag in
  let held = infer_elt_type alloc buf_name in
  if held <> elt then
    refuse
      what
      (Printf.sprintf
         "a %s %s fragment needs a %s buffer, but %s holds %s"
         (C.component_name frag.C.frag_component)
         (C.use_name frag.C.frag_use)
         (elt_name elt)
         buf_name
         (elt_name held)) ;
  let space = arr_space_of alloc buf_name in
  if space = Some SpaceLocal then
    refuse
      what
      (Printf.sprintf
         "%s is per-thread .local memory; wmma reads global or shared memory"
         buf_name) ;
  let r_index = emit_expr buf alloc env index in
  let r_stride = emit_expr buf alloc env stride in
  check_index_reg (what ^ " index") r_index ;
  check_index_reg (what ^ " stride") r_stride ;
  let r_addr =
    emit_elt_addr buf alloc (env_lookup env buf_name) r_index elt ~space
  in
  (* sarek_cm_require of the CUDA/C arm: 32-byte aligned address, stride in
     whole 16-byte units; [mul.lo.u32] keeps the low bits, the only ones tested. *)
  let bad = new_pred alloc and bad_stride = new_pred alloc in
  (match space with
  | Some SpaceShared ->
      let low = new_u32 alloc in
      emit buf "and.b32 %s, %s, 31;" low r_addr ;
      emit buf "setp.ne.u32 %s, %s, 0;" bad low
  | Some SpaceLocal | None ->
      let low = new_u64 alloc in
      emit buf "and.b64 %s, %s, 31;" low r_addr ;
      emit buf "setp.ne.u64 %s, %s, 0;" bad low) ;
  let bytes = new_u32 alloc and low = new_u32 alloc in
  emit buf "mul.lo.u32 %s, %s, %d;" bytes r_stride elt_bytes ;
  emit buf "and.b32 %s, %s, 15;" low bytes ;
  emit buf "setp.ne.u32 %s, %s, 0;" bad_stride low ;
  emit buf "or.pred %s, %s, %s;" bad bad bad_stride ;
  emit buf "@%s trap;" bad ;
  (space_qualifier space, r_addr, r_stride)

let emit_load buf alloc env ~dst ~(frag : C.fragment) ~src ~index ~stride
    ~layout =
  check_fragment frag ;
  let cf = declared alloc dst in
  if cf.cf_frag <> frag then
    refuse
      "load"
      (Printf.sprintf
         "%s was declared as another fragment than it is loaded as"
         dst) ;
  let sp, addr, r_stride =
    emit_access buf alloc env ~what:"load" ~frag ~buf_name:src ~index ~stride
  in
  let role =
    match frag.C.frag_use with
    | C.Matrix_a -> "a"
    | C.Matrix_b -> "b"
    | C.Accumulator -> "c"
  in
  emit
    buf
    "wmma.load.%s.sync.aligned.%s.m16n16k16.%s.%s %s, [%s], %s;"
    role
    (layout_suffix layout)
    sp
    (ptx_type frag.C.frag_component)
    (regs_text cf.cf_regs)
    addr
    r_stride

let emit_store buf alloc env ~src ~(frag : C.fragment) ~dst ~index ~stride
    ~layout =
  check_fragment frag ;
  if frag.C.frag_use <> C.Accumulator then
    refuse
      "store"
      "only an accumulator fragment can be stored (wmma has no store for A or \
       B)" ;
  let cf = declared alloc src in
  let sp, addr, r_stride =
    emit_access buf alloc env ~what:"store" ~frag ~buf_name:dst ~index ~stride
  in
  emit
    buf
    "wmma.store.d.sync.aligned.%s.m16n16k16.%s.%s [%s], %s, %s;"
    (layout_suffix layout)
    sp
    (ptx_type frag.C.frag_component)
    addr
    (regs_text cf.cf_regs)
    r_stride

let emit_muladd buf alloc ~dst ~a ~b ~c (cfg : C.config) =
  check_config cfg ;
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let regs_of name expected =
    let cf = declared alloc name in
    if cf.cf_frag <> expected then
      refuse
        "multiply-add"
        (Printf.sprintf
           "%s is not a %s fragment of %s"
           name
           (C.use_name expected.C.frag_use)
           (C.component_name expected.C.frag_component)) ;
    regs_text cf.cf_regs
  in
  let ra = regs_of a fa and rb = regs_of b fb in
  let rc = regs_of c fc and rd = regs_of dst fd in
  let types =
    match cfg.C.cfg_a with
    | C.Float16 -> "f32.f32"
    | op -> Printf.sprintf "s32.%s.%s.s32" (ptx_type op) (ptx_type op)
  in
  emit
    buf
    "wmma.mma.sync.aligned.%s.%s.m16n16k16.%s %s, %s, %s, %s;"
    (layout_suffix (operand_layout alloc a))
    (layout_suffix (operand_layout alloc b))
    types
    rd
    ra
    rb
    rc

(** [emit_op buf alloc env op] emits one cooperative-matrix statement. *)
let emit_op buf alloc env = function
  | CM_decl {name; frag} -> emit_decl buf alloc name frag
  | CM_load {dst; frag; src; index; stride; layout} ->
      emit_load buf alloc env ~dst ~frag ~src ~index ~stride ~layout
  | CM_store {src; frag; dst; index; stride; layout} ->
      emit_store buf alloc env ~src ~frag ~dst ~index ~stride ~layout
  | CM_muladd {dst; a; b; c; cfg} -> emit_muladd buf alloc ~dst ~a ~b ~c cfg
