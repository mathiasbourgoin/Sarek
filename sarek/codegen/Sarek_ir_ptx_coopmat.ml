(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix ([SCoopmat]) for the PTX generator: the [wmma.*]
    instructions of the PTX ISA (tensor cores from sm_70, PTX ISA 6.0).

    The lowering accepts what the [nvcuda::wmma] arm of {!Sarek_ir_cuda_coopmat}
    accepts, minus the integer family: 16x16x16 on subgroup (warp) scope, f16
    operands with an f32 accumulator, any layout of A (row/col), B (row/col) and
    of the accumulator in memory. The CUDA/C arm also emits 8-bit operands with
    a 32-bit integer accumulator (sm_72); that is a separate piece of work here
    and is refused by name below, never lowered silently. bf16 is refused by
    both: [nvcuda::wmma]'s bf16 fragment reads a bf16 buffer, not a float32
    buffer rounded on load.

    [wmma] and not [mma.sync]: [mma.sync.m16n8k16] needs sm_80, and the Turing
    form [m16n8k8] is not translated by ZLUDA (specs/zluda-mma-nvrtc13.md).
    [wmma] is the one form sm_70 and later share, and CUDA/C already lowers to
    it.

    {b Fragments.} PTX fixes the per-lane register group of a [wmma] fragment
    (m16n16k16): an f16 operand is 8 [.b32] registers each holding an f16x2
    pair, an f32 accumulator is 8 [.f32] registers. The emitter never indexes
    into a group; it passes the whole group to [wmma.load] / [wmma.mma] /
    [wmma.store], which agree on the distribution across lanes.

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

(** First SM with [wmma]. *)
let min_sm = 70

let shape_16 = {C.m = 16; n = 16; k = 16}

(** {1 What the lowering accepts} *)

let refuse what why =
  unsupported (Printf.sprintf "cooperative matrix: %s: %s" what why)

let phase_two =
  "the PTX backend lowers f16 operands with an f32 accumulator only; 8-bit \
   operands with a 32-bit integer accumulator (wmma.*.s8/u8, sm_72) are not \
   lowered yet, although the CUDA/C backend lowers them"

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
  | (C.Matrix_a | C.Matrix_b), C.Float16 | C.Accumulator, C.Float32 -> ()
  | (C.Matrix_a | C.Matrix_b), C.BFloat16 ->
      refuse
        "bf16 operand"
        "no faithful lowering: wmma's bf16 fragment reads a bf16 buffer, not a \
         float32 buffer rounded on load (CUDA/C refuses it too)"
  | (C.Matrix_a | C.Matrix_b), (C.Uint8 | C.Sint8)
  | C.Accumulator, (C.Uint32 | C.Sint32) ->
      refuse "integer fragment" phase_two
  | use, comp ->
      refuse
        "fragment"
        (Printf.sprintf
           "CUDA/PTX: no %s fragment of %s (operands are f16; accumulators are \
            f32)"
           (C.use_name use)
           (C.component_name comp))

(** The configurations a multiply-add is emitted for. The four fragments of a
    configuration are checked first, which already pins both operands to f16 and
    the addend and result to f32. *)
let check_config (cfg : C.config) =
  List.iter
    check_fragment
    (let a, b, c, r = C.fragments_of_config cfg in
     [a; b; c; r]) ;
  if cfg.C.cfg_saturating then
    refuse
      "saturating accumulation"
      "CUDA/PTX: the saturating variant computes a different function, and the \
       interpreter, the oracle it is checked against, refuses it"

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
        ("a uint8 operand buffer without a cooperative-matrix statement; "
       ^ phase_two)
  end
  else begin
    (match sm_number sm_target with
    | Some sm when sm >= min_sm -> ()
    | _ ->
        refuse
          "target"
          (Printf.sprintf
             "wmma needs .target sm_%d or later (PTX ISA 6.0); this kernel is \
              generated for %s"
             min_sm
             sm_target)) ;
    List.iter
      (function
        | CM_decl {frag; _} | CM_load {frag; _} | CM_store {frag; _} ->
            check_fragment frag
        | CM_muladd {cfg; _} -> check_config cfg)
      ops ;
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

let elt_of (f : C.fragment) =
  match f.C.frag_component with C.Float16 -> (TFloat16, 2) | _ -> (TFloat32, 4)

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
    match frag.C.frag_use with
    | C.Accumulator ->
        ( List.init 8 (fun _ -> new_f32 alloc),
          fun r -> emit buf "mov.f32 %s, 0F00000000;" r )
    | C.Matrix_a | C.Matrix_b ->
        ( List.init 8 (fun _ -> new_frag_b32 alloc),
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
         (ptx_reg_type_of elt)
         buf_name
         (ptx_reg_type_of held)) ;
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
    (if frag.C.frag_use = C.Accumulator then "f32" else "f16")
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
    "wmma.store.d.sync.aligned.%s.m16n16k16.%s.f32 [%s], %s, %s;"
    (layout_suffix layout)
    sp
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
  emit
    buf
    "wmma.mma.sync.aligned.%s.%s.m16n16k16.f32.f32 %s, %s, %s, %s;"
    (layout_suffix (operand_layout alloc a))
    (layout_suffix (operand_layout alloc b))
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
