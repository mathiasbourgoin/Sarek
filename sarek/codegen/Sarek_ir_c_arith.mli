(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Wrapping signed integer arithmetic for the C-family emitters (CUDA C, which
    HIP also compiles, OpenCL C and MSL).

    {b Rule.} Sarek integers follow OCaml: [int32] and [int64] [+], [-], [*],
    unary [-] and [lsl] wrap modulo 2{^ 32} / 2{^ 64}, as the Interpreter and
    Native backends compute them ([Int32.add], [Int64.mul], ...). In C, CUDA C,
    OpenCL C and MSL signed [+], [-], [*] and unary [-] are undefined behaviour
    on overflow, which a compiler may assume never happens; so is [lsl] of a
    negative value in C and in C++ before C++20 (CUDA C, MSL). OpenCL C defines
    its shifts as bit operations on the low bits of the count, so [lsl] goes
    through the unsigned type there only for uniformity. The C-family emitters
    therefore print these operators on signed integer operands through the
    unsigned type of the same width, where they are defined modulo 2{^ n}, and
    convert the result back:

    - [a + b] on int32 (CUDA C): [((int)((unsigned int)a + (unsigned int)b))]
    - [- a] on int64 (OpenCL C): [as_long((-(ulong)a))]
    - [a lsl n] on int32 (MSL): [as_type<int>(((uint)a << n))]
    - a chain converts back once, at its root, and a non-negative literal takes
      the unsigned suffix: [i * 6 + 1] is
      [((int)(((unsigned int)i * 6u) + 1u))].

    The [Int32]/[Int64] stdlib intrinsics [add], [sub], [mul], [neg] and
    [shift_left] use the same forms in their device templates.

    The way back is a bit reinterpretation in OpenCL C ([as_int], [as_long]) and
    MSL ([as_type<int>], [as_type<long>]), so the result's bits are the unsigned
    result's. CUDA C uses a cast: converting an out-of-range unsigned value is
    implementation-defined before C++20, and nvcc/NVRTC and clang (HIP) define
    it as reduction modulo 2{^ n}, which C++20 made normative.

    PTX ([add.u32], [mul.lo.u32], [neg.s32] on registers) and GLSL/WGSL (whose
    integer arithmetic is defined to wrap) already wrap and do not use this
    module.

    {b Scope.} Every [Add]/[Sub]/[Mul]/[Shl] node and every [Neg] whose operand
    types are known signed integers, whoever built the node: the user's kernel
    or a compiler pass (fusion, matrix rewrite, ...); the IR does not tell them
    apart. Not affected:

    - floating-point operators (operand type float, or unknown: see
      {!binop_width});
    - the [for] loop counter increment ([i++]): it never steps past a bound that
      fits the type, the documented limit of [Sarek_ir_for_bound];
    - [Div]/[Mod]: [x / 0] raises [Division_by_zero] in OCaml and the
      Interpreter, which no device code reproduces, and [min_int / -1] (UB in C)
      stays the device's result; not rewritten, so constant divisors keep their
      strength reduction;
    - index arithmetic (array subscripts, cooperative-matrix offsets and
      strides, [for] bounds, lets used only there): see {i Index arithmetic}
      below;
    - shift counts outside [0 .. width-1]: unspecified in OCaml, so outside
      Sarek's portable contract whatever each dialect does with them (undefined
      in C, CUDA C and MSL; OpenCL C uses the low bits of the count); not
      masked. [asr] of a negative value: implementation-defined in C and
      arithmetic on every compiler these backends use; sign-extending by
      definition in OpenCL C;
    - the index operand of an atomic intrinsic
      ([atomic_add_global_int32 a (i + 1) v]): printed by the intrinsic's
      template through the general rule, so it wraps (correct, merely not
      exempt).

    {b Index arithmetic.} The [+], [-], [*] and [lsl] chain at the root of an
    array subscript ([a.(y * w + x - 1)]), of a cooperative-matrix load/store
    offset or stride, and of a [for] bound ([for i = 0 to k - 1]) keeps C's
    signed operators. The no-overflow assumption is what lets the compiler fold
    address arithmetic (clang merges a 3x3 stencil's adjacent loads into
    [global_load_b96] with immediate offsets on gfx1100 only with it: 3 loads
    instead of 6) and compute a loop's trip count; measured on the RX 7900 XTX
    (HIP), wrapping these positions slowed a 3x3 convolution by 10 % and the
    WMMA GEMMs by about 25 %, which this exemption removed. The rule is
    therefore the one [Sarek_ir_for_bound] already states for a [for] bound of
    [max_int]: a kernel whose index arithmetic overflows int32 is outside the
    contract on the C-family backends (an overflowing subscript is anyway an
    out-of-range index unless a later operation wraps it back).

    An immutable [let] whose every use is index arithmetic is index arithmetic
    too: in [let off = row * k + i * 4 in a.(off)] the definition of [off] keeps
    C's operators, as if it were written in the subscript (wrapping it cost an
    extra sign extension and 64-bit address add per iteration on gfx1100 instead
    of an immediate-offset load). The set is the greatest one closed under this
    rule (a let used only in another index let's definition qualifies); a name
    bound more than once anywhere in the kernel and its helpers, a mutable [let]
    and a let with any other use are excluded.

    Only the root chain is exempt: an operand that is anything else (a load, a
    call, [land], [/], a let with other uses) is printed by the general rule, so
    [a.((h * 31 + c) land mask)] still hashes with wrapping.

    {b Typing.} The IR does not annotate an operator with its type, so
    {!int_width} reconstructs it from variables, literals, casts, array element
    types (resolved in the function being emitted: the kernel and each helper
    have their own arrays), record fields, variant payloads read through
    [v.data.C_v], helper return types and intrinsic signatures. Each emitted
    node is typed once: an operator's typing is handed down to the children it
    prints (Sarek_ir_c_types). When an operand's type cannot be determined the
    operator is printed as before (plain C), which is the only safe choice when
    it might be a float. *)

open Sarek_ir_types

(** Signed integer widths. [W32] covers every operand C promotes to [int]
    ([int32], [bool], 8-bit storage). *)
type width = Sarek_ir_c_types.width = W32 | W64

(** Operand types of one function (its arrays) and of the kernel (record and
    variant layouts, helper return types, index lets). *)
type env = Sarek_ir_c_types.env

(** The kernel body's environment. [records] (the [~types] table the emitter
    declares its structs from) takes precedence over [k.kern_types]. *)
val env_of_kernel :
  ?records:(string * (string * elttype) list) list -> kernel -> env

(** The environment of helper [hf]: its own arrays, the kernel's layouts. *)
val for_helper : env -> helper_func -> env

(** No knowledge: only literals, variables and casts are typed. *)
val empty : unit -> env

(** Nodes typed so far with this environment (and the environments derived from
    it): each emitted node is typed once, so this grows linearly with the
    kernel. For tests. *)
val typing_steps : env -> int

(** [is_index_let env v]: [v] is bound once, by an immutable [let] of an integer
    value, and every use of it is index arithmetic (see {i Index arithmetic}).
*)
val is_index_let : env -> var -> bool

(** [int_width env e] is the signed integer width of [e]'s C type, or [None]
    when [e] is not an integer or its type cannot be determined. *)
val int_width : env -> expr -> width option

(** [binop_width env op a b] is [Some w] when [op a b] must be printed wrapping:
    [op] is [Add], [Sub], [Mul] or [Shl] and the operands are signed integers of
    known types ([w] is their common C type; for [Shl], the left operand's). *)
val binop_width : env -> binop -> expr -> expr -> width option

(** The type spellings of one C dialect, and how its unsigned result is turned
    back into the signed type: a cast in CUDA C, [as_int]/[as_long] in OpenCL C,
    [as_type<int>]/[as_type<long>] in MSL (bit reinterpretations, so no
    implementation-defined conversion is involved). *)
type spelling = {
  int32 : string;
  uint32 : string;
  int64 : string;
  uint64 : string;
  uint64_suffix : string;  (** of an unsigned 64-bit literal *)
  to_int32 : string;
  to_int64 : string;
}

val cuda : spelling

val opencl : spelling

val metal : spelling

(** Print the index expression [e] (a subscript, a cooperative-matrix offset or
    stride, a [for] bound, an index let's initialiser): the chain of signed [+],
    [-], [*] and [lsl] at its root in plain C, everything below it with
    [gen_expr]. See {i Index arithmetic} above. *)
val emit_index :
  env -> gen_expr:(Buffer.t -> expr -> unit) -> Buffer.t -> expr -> unit

(** The [EBinop] arm of a C emitter for the node [e]: wrapping when
    {!binop_width} says so, else [(a <operator_text> b)]. *)
val emit_binop_arm :
  env ->
  spelling ->
  gen_expr:(Buffer.t -> expr -> unit) ->
  operator_text:string ->
  Buffer.t ->
  expr ->
  unit

(** The [EUnop] arm of a C emitter for the node [e]: a negated integer literal
    folded to its constant, a wrapping [Neg] on a signed integer, else
    [Sarek_ir_intrinsic_dispatch.emit_unop]. *)
val emit_unop_arm :
  env ->
  spelling ->
  gen_expr:(Buffer.t -> expr -> unit) ->
  operator_text:string ->
  Buffer.t ->
  expr ->
  unit

(** The initialiser of [let v = e]: {!emit_index} when [v] is an index let,
    [gen_expr] otherwise. *)
val emit_let_init :
  env -> gen_expr:(Buffer.t -> expr -> unit) -> Buffer.t -> var -> expr -> unit
