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

    {b Alignment.} [wmma.load]/[wmma.store] are undefined unless the address is
    32-byte aligned and [stride * element bytes] is a multiple of 16, while
    [CM_load]/[CM_store] promise any index and stride. An access that
    {!Sarek_ir_coopmat_align} proves aligned is the bare instruction. Any other
    access tests alignment at run time (the CUDA/C [sarek_cm_require] test) and,
    when the launch block is known, branches (warp-uniform) to an out-of-line
    slow path that stages the 16x16 tile element by element through a 32-byte
    aligned per-warp [.shared] buffer ([sarek_cm_tiles], 1 KiB per warp,
    declared only when an access is unproven): results are the IR's for any
    index and stride, in either layout; a store writes only its tile. With no
    block known no buffer can be sized and a misaligned access executes [trap].
    A [.shared] array a cooperative-matrix statement accesses is declared
    32-byte aligned.

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
      | CM_decl _ | CM_muladd _ | CM_scale_add _ -> None)
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
    refusal would name whichever node the emitter reached first). A uint8 buffer
    without a cooperative-matrix statement is ordinary 8-bit storage
    (Sarek_ir_int8) and is not this function's business. Also the launch shape:
    the instructions run on whole warps, and a partial warp has inactive lanes
    that would feed them undefined operands. *)
let check_kernel ?block ~sm_target (k : kernel) : unit =
  let ops = Sarek_ir_analysis.kernel_coopmat_ops k in
  if ops <> [] then begin
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
        | CM_muladd {cfg; _} -> check_config cfg
        | CM_scale_add _ ->
            refuse "CM_scale_add" Sarek_ir_cuda_coopmat_scale.cuda_reason)
      ops ;
    if
      List.exists
        (function
          | CM_decl {frag; _} | CM_load {frag; _} | CM_store {frag; _} ->
              is_8bit frag.C.frag_component
          | CM_muladd {cfg; _} -> is_8bit cfg.C.cfg_a
          | CM_scale_add _ -> false)
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
  | TInt8 -> "int8"
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

(** One fragment access, after the checks that make it meaningful: the buffer
    holds the fragment's element type, and is not per-thread [.local] memory (a
    warp-collective access to it has no meaning). *)
type access = {
  frag : C.fragment;
  space : arr_space option;
  base : string;  (** the array's base register *)
  index : string;  (** int32 register *)
  stride : string;  (** int32 register *)
  addr : string;  (** the element address, 32-bit for [.shared] else 64-bit *)
  bytes : int;  (** element bytes *)
}

let ss a = space_qualifier a.space

let emit_access buf alloc env ~what ~(frag : C.fragment) ~buf_name ~index
    ~stride =
  let elt, bytes = elt_of frag in
  let held = infer_elt_type alloc buf_name in
  (* An s8 fragment also loads from an int8 buffer (same bytes); the wmma load
     reads the address, whatever the declared element type. *)
  let accepted =
    held = elt || (frag.C.frag_component = C.Sint8 && held = TInt8)
  in
  if not accepted then
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
  let base = env_lookup env buf_name in
  let addr = emit_elt_addr buf alloc base r_index elt ~space in
  {frag; space; base; index = r_index; stride = r_stride; addr; bytes}

(** The predicate "this access is misaligned" ([sarek_cm_require] of the CUDA/C
    arm): the address is not 32-byte aligned, or the stride is not a whole
    number of 16-byte units. [mul.lo.u32] keeps the low bits, the only ones
    tested. *)
let emit_misaligned buf alloc a =
  let bad = new_pred alloc and bad_stride = new_pred alloc in
  (match a.space with
  | Some SpaceShared ->
      let low = new_u32 alloc in
      emit buf "and.b32 %s, %s, 31;" low a.addr ;
      emit buf "setp.ne.u32 %s, %s, 0;" bad low
  | Some SpaceLocal | None ->
      let low = new_u64 alloc in
      emit buf "and.b64 %s, %s, 31;" low a.addr ;
      emit buf "setp.ne.u64 %s, %s, 0;" bad low) ;
  let bytes = new_u32 alloc and low = new_u32 alloc in
  emit buf "mul.lo.u32 %s, %s, %d;" bytes a.stride a.bytes ;
  emit buf "and.b32 %s, %s, 15;" low bytes ;
  emit buf "setp.ne.u32 %s, %s, 0;" bad_stride low ;
  emit buf "or.pred %s, %s, %s;" bad bad bad_stride ;
  bad

(** The access must leave the bare instruction: misaligned, or a STORE whose
    stride is below the tile edge (its rows overlap in memory, and the answer is
    the last write of the interpreter's loop, which a matrix-unit store does not
    promise: such a store is written serially, see {!emit_slow_path}). *)
let emit_unsafe buf alloc a ~load =
  let bad = emit_misaligned buf alloc a in
  if not load then begin
    let magnitude = new_u32 alloc and overlap = new_pred alloc in
    emit buf "abs.s32 %s, %s;" magnitude a.stride ;
    emit buf "setp.lt.s32 %s, %s, 16;" overlap magnitude ;
    emit buf "or.pred %s, %s, %s;" bad bad overlap
  end ;
  bad

(** {1 Staging through shared memory}

    A misaligned access cannot use [wmma.load] / [wmma.store] on its own
    address. The slow path copies the tile element by element, with the IR's own
    formula, through a 32-byte aligned per-warp buffer in [.shared]: element [t]
    (0..255) of the buffer is the memory element at
    [index + (t / 16) * stride + (t mod 16)], in either layout (row-major:
    [r * stride + c]; column-major: [c * stride + r]); [wmma] then loads from /
    stores to the buffer with a leading dimension of 16, which is aligned for
    every element size. Lane [l] copies [t = l + 32 j], [j < 8]. Results are
    those of the IR whatever the address and stride. *)

let tile_name = "sarek_cm_tiles"

let tile_bytes_per_warp = 1024

(** The buffer, declared at its first use, [warps] tiles of 1 KiB. *)
let ensure_tiles alloc ~warps =
  if not alloc.coop_tiles then begin
    alloc.coop_tiles <- true ;
    Buffer.add_string
      alloc.shared_decls
      (Printf.sprintf
         "    .shared .align 32 .b8 %s[%d];\n"
         tile_name
         (warps * tile_bytes_per_warp))
  end

let move_width = function 1 -> "u8" | 2 -> "b16" | _ -> "b32"

(** One element register of the staging copy. *)
let copy_register alloc bytes =
  if bytes = 2 then new_f16 alloc else new_u32 alloc

(** The slow path of one access, appended out of line. [wmma] receives the
    address register of the buffer (a [.shared] window offset) and the register
    holding 16. [load]: copy in, then [wmma.load]; else [wmma.store], then copy
    out. *)
let emit_slow_path alloc ~label ~done_label (a : access) ~layout ~load ~wmma =
  let b = alloc.coop_slow in
  let warps = Option.get alloc.coop_warps in
  ensure_tiles alloc ~warps ;
  emit b "%s:" label ;
  (* this lane's buffer: warp = linear thread id / 32 *)
  let tx = new_u32 alloc in
  let ty = new_u32 alloc and tz = new_u32 alloc in
  let nx = new_u32 alloc and ny = new_u32 alloc in
  emit b "mov.u32 %s, %%tid.x;" tx ;
  emit b "mov.u32 %s, %%tid.y;" ty ;
  emit b "mov.u32 %s, %%tid.z;" tz ;
  emit b "mov.u32 %s, %%ntid.x;" nx ;
  emit b "mov.u32 %s, %%ntid.y;" ny ;
  let t1 = new_u32 alloc and lin = new_u32 alloc in
  emit b "mad.lo.u32 %s, %s, %s, %s;" t1 tz ny ty ;
  emit b "mad.lo.u32 %s, %s, %s, %s;" lin t1 nx tx ;
  let warp = new_u32 alloc and lane = new_u32 alloc in
  emit b "shr.u32 %s, %s, 5;" warp lin ;
  emit b "and.b32 %s, %s, 31;" lane lin ;
  (* the buffer was sized for the block this source was generated for *)
  let beyond = new_pred alloc in
  emit b "setp.ge.u32 %s, %s, %d;" beyond warp warps ;
  emit b "@%s trap;" beyond ;
  let tile = new_u32 alloc and tile_base = new_u32 alloc in
  emit b "mov.u32 %s, %s;" tile_base tile_name ;
  emit b "mad.lo.u32 %s, %s, %d, %s;" tile warp tile_bytes_per_warp tile_base ;
  let sixteen = new_u32 alloc in
  emit b "mov.u32 %s, 16;" sixteen ;
  (* this lane's first element: row lane/16, column lane mod 16 *)
  let o0 = new_u32 alloc and i0 = new_u32 alloc in
  emit b "shr.u32 %s, %s, 4;" o0 lane ;
  emit b "and.b32 %s, %s, 15;" i0 lane ;
  let src, step =
    match a.space with
    | Some SpaceShared ->
        let off = new_u32 alloc in
        emit b "mad.lo.s32 %s, %s, %s, %s;" off o0 a.stride a.index ;
        emit b "add.s32 %s, %s, %s;" off off i0 ;
        let offb = new_u32 alloc and src = new_u32 alloc in
        let step = new_u32 alloc in
        emit b "mul.lo.u32 %s, %s, %d;" offb off a.bytes ;
        emit b "add.u32 %s, %s, %s;" src a.base offb ;
        emit b "mul.lo.u32 %s, %s, %d;" step a.stride (2 * a.bytes) ;
        (src, step)
    | Some SpaceLocal | None ->
        (* signed 64-bit: a negative stride walks backwards inside the buffer,
           and an offset past 2^31 elements does not wrap *)
        let idx64 = new_u64 alloc and i64 = new_u64 alloc in
        let off64 = new_u64 alloc and offb = new_u64 alloc in
        let src = new_u64 alloc and step = new_u64 alloc in
        let stride64 = new_u64 alloc in
        emit b "cvt.s64.s32 %s, %s;" idx64 a.index ;
        emit b "mad.wide.s32 %s, %s, %s, %s;" off64 o0 a.stride idx64 ;
        emit b "cvt.u64.u32 %s, %s;" i64 i0 ;
        emit b "add.s64 %s, %s, %s;" off64 off64 i64 ;
        emit b "mul.lo.s64 %s, %s, %d;" offb off64 a.bytes ;
        emit b "add.u64 %s, %s, %s;" src a.base offb ;
        emit b "cvt.s64.s32 %s, %s;" stride64 a.stride ;
        emit b "mul.lo.s64 %s, %s, %d;" step stride64 (2 * a.bytes) ;
        (src, step)
  in
  let lane_slot = new_u32 alloc in
  emit b "mad.lo.u32 %s, %s, %d, %s;" lane_slot lane a.bytes tile ;
  let w = move_width a.bytes in
  let sp = ss a in
  let add_step = if a.space = Some SpaceShared then "add.u32" else "add.u64" in
  let copy j =
    let v = copy_register alloc a.bytes in
    let slot = Printf.sprintf "[%s+%d]" lane_slot (32 * j * a.bytes) in
    if load then begin
      emit b "ld.%s.%s %s, [%s];" sp w v src ;
      emit b "st.shared.%s %s, %s;" w slot v
    end
    else begin
      emit b "ld.shared.%s %s, %s;" w v slot ;
      emit b "st.%s.%s [%s], %s;" sp w src v
    end ;
    emit b "%s %s, %s, %s;" add_step src src step
  in
  let copy_all () = List.iter copy (List.init 8 Fun.id) in
  if load then begin
    copy_all () ;
    emit b "bar.warp.sync 0xffffffff;" ;
    wmma b tile sixteen
  end
  else begin
    wmma b tile sixteen ;
    emit b "bar.warp.sync 0xffffffff;" ;
    (* Rows (columns) closer than 16 elements overlap in memory, and then the
       interpreter's answer is the LAST write of its loop (rows outer, columns
       inner). Parallel lanes would race on the shared cells, so that case is
       written serially, by lane 0, in the interpreter's order. *)
    let overlap = new_pred alloc and magnitude = new_u32 alloc in
    let serial = new_label alloc and joined = new_label alloc in
    emit b "abs.s32 %s, %s;" magnitude a.stride ;
    emit b "setp.lt.s32 %s, %s, 16;" overlap magnitude ;
    emit b "@%s bra %s;" overlap serial ;
    copy_all () ;
    emit b "bra %s;" joined ;
    emit b "%s:" serial ;
    let not_leader = new_pred alloc in
    emit b "setp.ne.u32 %s, %s, 0;" not_leader lane ;
    emit b "@%s bra %s;" not_leader joined ;
    let r = new_u32 alloc and c = new_u32 alloc in
    let row_loop = new_label alloc and col_loop = new_label alloc in
    emit b "mov.u32 %s, 0;" r ;
    emit b "%s:" row_loop ;
    emit b "mov.u32 %s, 0;" c ;
    emit b "%s:" col_loop ;
    (* element (r, c): in the buffer at outer * 16 + inner, in memory at
       index + outer * stride + inner, outer / inner being (r, c) row-major and
       (c, r) column-major *)
    let outer, inner = if layout = C.Col_major then (c, r) else (r, c) in
    let toff = new_u32 alloc and taddr = new_u32 alloc in
    emit b "mad.lo.u32 %s, %s, 16, %s;" toff outer inner ;
    emit b "mad.lo.u32 %s, %s, %d, %s;" taddr toff a.bytes tile ;
    let v = copy_register alloc a.bytes in
    emit b "ld.shared.%s %s, [%s];" w v taddr ;
    (match a.space with
    | Some SpaceShared ->
        let off = new_u32 alloc and offb = new_u32 alloc in
        let addr = new_u32 alloc in
        emit b "mad.lo.s32 %s, %s, %s, %s;" off outer a.stride a.index ;
        emit b "add.s32 %s, %s, %s;" off off inner ;
        emit b "mul.lo.u32 %s, %s, %d;" offb off a.bytes ;
        emit b "add.u32 %s, %s, %s;" addr a.base offb ;
        emit b "st.shared.%s [%s], %s;" w addr v
    | Some SpaceLocal | None ->
        let idx64 = new_u64 alloc and in64 = new_u64 alloc in
        let off64 = new_u64 alloc and offb = new_u64 alloc in
        let addr = new_u64 alloc in
        emit b "cvt.s64.s32 %s, %s;" idx64 a.index ;
        emit b "mad.wide.s32 %s, %s, %s, %s;" off64 outer a.stride idx64 ;
        emit b "cvt.u64.u32 %s, %s;" in64 inner ;
        emit b "add.s64 %s, %s, %s;" off64 off64 in64 ;
        emit b "mul.lo.s64 %s, %s, %d;" offb off64 a.bytes ;
        emit b "add.u64 %s, %s, %s;" addr a.base offb ;
        emit b "st.global.%s [%s], %s;" w addr v) ;
    let more_cols = new_pred alloc and more_rows = new_pred alloc in
    emit b "add.u32 %s, %s, 1;" c c ;
    emit b "setp.lt.u32 %s, %s, 16;" more_cols c ;
    emit b "@%s bra %s;" more_cols col_loop ;
    emit b "add.u32 %s, %s, 1;" r r ;
    emit b "setp.lt.u32 %s, %s, 16;" more_rows r ;
    emit b "@%s bra %s;" more_rows row_loop ;
    emit b "%s:" joined
  end ;
  (* the buffer is reused by the next staged access *)
  emit b "bar.warp.sync 0xffffffff;" ;
  emit b "bra %s;" done_label

(** The shape shared by loads and stores: [direct buf addr stride] is the [wmma]
    instruction on [addr]/[stride]; [staged buf tile sixteen] the same on the
    staging buffer. Proven aligned: the bare instruction. Unproven with a known
    block: a warp-uniform branch to the out-of-line slow path. Unproven without
    a block (no buffer can be sized): the trap. *)
let emit_guarded buf alloc ~op (a : access) ~layout ~load ~direct ~staged =
  if Sarek_ir_coopmat_align.is_proven alloc.coop_proven op then
    direct buf a.addr a.stride
  else if alloc.coop_warps = None then begin
    let bad = emit_unsafe buf alloc a ~load in
    emit buf "@%s trap;" bad ;
    direct buf a.addr a.stride
  end
  else begin
    let bad = emit_unsafe buf alloc a ~load in
    let slow = new_label alloc and done_ = new_label alloc in
    emit buf "@%s bra %s;" bad slow ;
    direct buf a.addr a.stride ;
    emit buf "%s:" done_ ;
    emit_slow_path
      alloc
      ~label:slow
      ~done_label:done_
      a
      ~layout
      ~load
      ~wmma:staged
  end

let emit_load buf alloc env ~op ~dst ~(frag : C.fragment) ~src ~index ~stride
    ~layout =
  check_fragment frag ;
  let cf = declared alloc dst in
  if cf.cf_frag <> frag then
    refuse
      "load"
      (Printf.sprintf
         "%s was declared as another fragment than it is loaded as"
         dst) ;
  let a =
    emit_access buf alloc env ~what:"load" ~frag ~buf_name:src ~index ~stride
  in
  let role =
    match frag.C.frag_use with
    | C.Matrix_a -> "a"
    | C.Matrix_b -> "b"
    | C.Accumulator -> "c"
  in
  let instr sp b addr stride =
    emit
      b
      "wmma.load.%s.sync.aligned.%s.m16n16k16.%s.%s %s, [%s], %s;"
      role
      (layout_suffix layout)
      sp
      (ptx_type frag.C.frag_component)
      (regs_text cf.cf_regs)
      addr
      stride
  in
  emit_guarded
    buf
    alloc
    ~op
    a
    ~layout
    ~load:true
    ~direct:(instr (ss a))
    ~staged:(instr "shared")

let emit_store buf alloc env ~op ~src ~(frag : C.fragment) ~dst ~index ~stride
    ~layout =
  check_fragment frag ;
  if frag.C.frag_use <> C.Accumulator then
    refuse
      "store"
      "only an accumulator fragment can be stored (wmma has no store for A or \
       B)" ;
  let cf = declared alloc src in
  let a =
    emit_access buf alloc env ~what:"store" ~frag ~buf_name:dst ~index ~stride
  in
  let instr sp b addr stride =
    emit
      b
      "wmma.store.d.sync.aligned.%s.m16n16k16.%s.%s [%s], %s, %s;"
      (layout_suffix layout)
      sp
      (ptx_type frag.C.frag_component)
      addr
      (regs_text cf.cf_regs)
      stride
  in
  emit_guarded
    buf
    alloc
    ~op
    a
    ~layout
    ~load:false
    ~direct:(instr (ss a))
    ~staged:(instr "shared")

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
let emit_op buf alloc env (op : coopmat_op) =
  match op with
  | CM_decl {name; frag} -> emit_decl buf alloc name frag
  | CM_load {dst; frag; src; index; stride; layout} ->
      emit_load buf alloc env ~op ~dst ~frag ~src ~index ~stride ~layout
  | CM_store {src; frag; dst; index; stride; layout} ->
      emit_store buf alloc env ~op ~src ~frag ~dst ~index ~stride ~layout
  | CM_muladd {dst; a; b; c; cfg} -> emit_muladd buf alloc ~dst ~a ~b ~c cfg
  | CM_scale_add _ ->
      refuse "CM_scale_add" Sarek_ir_cuda_coopmat_scale.cuda_reason
