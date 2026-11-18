(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Wrapping signed integer arithmetic for the C-family emitters (CUDA C, which
    HIP also compiles, OpenCL C and MSL).

    {b Rule.} Sarek integers follow OCaml: [int32] and [int64] [+], [-], [*],
    unary [-] and [lsl] wrap modulo 2{^ 32} / 2{^ 64}, as the Interpreter and
    Native backends compute them ([Int32.add], [Int64.mul], ...). In C, CUDA C,
    OpenCL C and MSL the same operators on a signed type are undefined behaviour
    on overflow (and [lsl] of a negative value is undefined before C++20/C23),
    which a compiler may assume never happens. The C-family emitters therefore
    print these operators on signed integer operands through the unsigned type
    of the same width, where they are defined modulo 2{^ n}, and convert the
    result back:

    - [a + b] on int32 (CUDA C): [((int)((unsigned int)a + (unsigned int)b))]
    - [- a] on int64 (OpenCL C, MSL): [((long)(-(ulong)a))]
    - [a lsl n] on int32: [((int)((uint)a << n))]
    - a chain converts back once, at its root, and a non-negative literal takes
      the unsigned suffix: [i * 6 + 1] is [((int)(((uint)i * 6u) + 1u))].

    The [Int32]/[Int64] stdlib intrinsics [add], [sub], [mul], [neg] and
    [shift_left] use the same forms in their device templates.

    The conversion back from an out-of-range unsigned value is
    implementation-defined in C and in C++ before C++20; every compiler these
    backends reach (clang for HIP, OpenCL and Metal, nvcc/NVRTC for CUDA)
    defines it as reduction modulo 2{^ n}, which C++20 made normative.

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
      strides, [for] bounds): see {i Index arithmetic} below;
    - shift counts outside [0 .. width-1]: unspecified in OCaml, undefined in C;
      not masked. [asr] of a negative value is implementation-defined in C and
      arithmetic on every compiler these backends use.

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
    out-of-range index unless a later operation wraps it back). Only the root
    chain is exempt: an operand that is anything else (a load, a call, [land],
    [/], a variable bound by [let]) is printed by the general rule, so
    [a.((h * 31 + c) land mask)] still hashes with wrapping.

    {b Typing.} The IR does not annotate an operator with its type, so
    {!int_width} reconstructs it from variables, literals, casts, array element
    types, record fields, helper return types and intrinsic signatures. When an
    operand's type cannot be determined the operator is printed as before (plain
    C), which is the only safe choice when it might be a float. *)

open Sarek_ir_types

(** Signed integer widths. [W32] covers every operand C promotes to [int]
    ([int32], [bool], 8-bit storage). *)
type width = W32 | W64

(** Array element types, record layouts and helper return types of one kernel
    (its parameters, locals, shared arrays, body and helpers). *)
type env

val env_of_kernel : kernel -> env

(** No knowledge: only literals, variables and casts are typed. *)
val empty : env

(** [int_width env e] is the signed integer width of [e]'s C type, or [None]
    when [e] is not an integer or its type cannot be determined. *)
val int_width : env -> expr -> width option

(** [binop_width env op a b] is [Some w] when [op a b] must be printed wrapping:
    [op] is [Add], [Sub], [Mul] or [Shl] and the operands are signed integers of
    known types ([w] is their common C type; for [Shl], the left operand's). *)
val binop_width : env -> binop -> expr -> expr -> width option

(** The type spellings of one C dialect. *)
type spelling = {
  int32 : string;
  uint32 : string;
  int64 : string;
  uint64 : string;
  uint64_suffix : string;  (** of an unsigned 64-bit literal *)
}

val cuda : spelling

(** OpenCL C and MSL share [int]/[uint]/[long]/[ulong]. *)
val opencl : spelling

val metal : spelling

(** Print [op a b] through the unsigned type of width [w]. *)
val emit_binop :
  env ->
  spelling ->
  width ->
  gen_expr:(Buffer.t -> expr -> unit) ->
  Buffer.t ->
  binop ->
  expr ->
  expr ->
  unit

(** Print [- a] through the unsigned type of width [w]. *)
val emit_neg :
  env ->
  spelling ->
  width ->
  gen_expr:(Buffer.t -> expr -> unit) ->
  Buffer.t ->
  expr ->
  unit

(** Print the index expression [e] (a subscript, a cooperative-matrix offset or
    stride, a [for] bound): the chain of signed [+], [-], [*] and [lsl] at its
    root in plain C, everything below it with [gen_expr]. See
    {i Index arithmetic} above. *)
val emit_index :
  env -> gen_expr:(Buffer.t -> expr -> unit) -> Buffer.t -> expr -> unit
