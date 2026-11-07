(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The 8-bit integer storage types [TUint8] and [TInt8], shared by every
    backend and by the interpreter.

    Both are STORAGE types, on the [TFloat16] model: an array element is one
    byte in memory, but there is no 8-bit value at run time. Every expression is
    int32-valued:

    - reading an element gives an int32, zero-extended for [TUint8] and
      sign-extended for [TInt8];
    - storing an int32 into an element keeps its low 8 bits;
    - [ECast (TUint8, e)] is the int32 [e land 0xff], [ECast (TInt8, e)] the low
      byte of [e] sign-extended. So a cast followed by a store stores what the
      store alone would, and a store followed by a read gives the cast.

    An 8-bit local scalar ([let], mutable or not, or [DLocal]) holds the
    narrowed value, as a C [unsigned char] local does (see {!lower_scalars}).

    What is not expressible, and is refused by {!refusal}: an 8-bit kernel or
    helper parameter, loop variable or helper result, an 8-bit record or variant
    field, an 8-bit array in [Local] memory, and an array name bound both to an
    8-bit and to a wider element type (the backends resolve an [EArrayRead] by
    name). *)

open Sarek_ir_types

(** [TUint8] or [TInt8]. *)
val is_8bit : elttype -> bool

(** [Some t] when the type is an array or vector of the 8-bit type [t]. *)
val byte_elt_of_type : elttype -> elttype option

(** Every array name the kernel binds to an 8-bit element type: kernel and
    helper parameters, [DShared] declarations and the binders of [EArrayCreate].
    Each name once. *)
val byte_arrays : kernel -> (string * elttype) list

(** The reason the kernel uses an 8-bit type outside the supported positions, if
    it does. Backends turn it into a codegen refusal, the interpreter into an
    interpreter error. *)
val refusal : kernel -> string option

(** The value an int32 takes once stored in an element of the given 8-bit type
    and read back: [land 0xff] for [TUint8], sign extension of the low byte for
    [TInt8]. The identity for any other type. *)
val narrow : elttype -> int32 -> int32

(** Rewrite every 8-bit local scalar ([SLet], [SLetMut], [DLocal] of type
    [TUint8]/[TInt8]) into an int32 binder holding the narrowed value:
    [SLet (v : TUint8, e, b)] becomes [SLet (v : TInt32, ECast (TUint8, e), b)]
    and an assignment to a mutable one is narrowed the same way. That is what
    such a binder means (a C [unsigned char] local); the rewrite is for the
    targets that have no implicit 8/32-bit conversion (GLSL, PTX) and for the
    interpreter. *)
val lower_scalars : kernel -> kernel

(** For the C-family emitters (CUDA/HIP, OpenCL, Metal): make every conversion
    of an int32 to the signed 8-bit type an explicit [ECast (TInt8, _)] (the
    value stored into an [int8] array element, and the [int8] local scalars,
    which {!lower_scalars} turns into int32 binders). Converting an out-of-range
    [int] to [signed char] is implementation-defined in C, C++ before C++20,
    OpenCL C and MSL; the emitters spell the cast with {!c_signed_narrowing},
    which is defined everywhere. Unsigned conversions are defined (modulo 2^8)
    and are left alone, so kernels without [int8] are unchanged. *)
val define_signed_narrowing : kernel -> kernel

(** [c_signed_narrowing e] is the C expression [((((e) & 0xFF) ^ 0x80) - 0x80)]:
    the low byte of [e] sign-extended, as an [int], with no
    implementation-defined conversion. *)
val c_signed_narrowing : string -> string

(** Make the storage semantics explicit: {!lower_scalars}, then every read of an
    8-bit array becomes [ECast (t, read)] and every store into one stores
    [ECast (t, value)] (left as is when the value already is that cast). The
    result has the same meaning under the semantics above; an evaluator that
    implements only [ECast] then implements the reads and stores too. Used by
    the interpreter. *)
val normalise : kernel -> kernel
