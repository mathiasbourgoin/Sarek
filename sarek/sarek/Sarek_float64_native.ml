(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_float64_native - CPU implementations of the Sarek Float64 stdlib
 *
 * Reachable as [Sarek.Sarek_cpu_runtime.Float64], the target that
 * sarek/ppx/Sarek_native_intrinsics.ml's [map_stdlib_path] gives to
 * [IntrinsicRef (["Float64"], name)]. The native backend copies the intrinsic
 * name VERBATIM onto that path, so this module must export EXACTLY the names
 * declared by sarek/Sarek_float64/Float64.ml — no more, no less.
 *
 * It previously mapped to OCaml's [Stdlib.Float], which shares most but not all
 * of that surface: [abs_float], [rsqrt], [fmod], [copysign], [of_int32],
 * [to_int32], [of_float32], [to_float32], the four [*_float64] arithmetic names
 * and the eight operator forms have no [Float.<name>] counterpart, so a native
 * kernel calling any of them failed to compile with "Unbound value
 * Float.<name>" pointing into PPX-generated code.
 *
 * Each body below is the [ocaml = ...] field of the corresponding
 * [let%sarek_intrinsic] in Sarek_float64/Float64.ml, copied verbatim: that
 * field IS the specification of the host-side semantics, and native/interpreter/
 * GPU agreement depends on not paraphrasing it. In particular [of_float32] and
 * [to_float32] are the identity — an OCaml [float] is always 64-bit, so the
 * float32 width narrowing is a device-side concern, exactly as the stdlib
 * declares.
 *
 * Reconciled against the stdlib surface by
 * sarek/tests/unit/test_intrinsic_surface.ml.
 ******************************************************************************)

(** {1 Arithmetic} *)

let add_float64 (x : float) (y : float) = x +. y

let sub_float64 (x : float) (y : float) = x -. y

let mul_float64 (x : float) (y : float) = x *. y

let div_float64 (x : float) (y : float) = x /. y

(** {1 Unary math} *)

let sin = Stdlib.sin

let cos = Stdlib.cos

let tan = Stdlib.tan

let asin = Stdlib.asin

let acos = Stdlib.acos

let atan = Stdlib.atan

let sinh = Stdlib.sinh

let cosh = Stdlib.cosh

let tanh = Stdlib.tanh

let exp = Stdlib.exp

let log = Stdlib.log

let log10 = Stdlib.log10

let sqrt = Stdlib.sqrt

let ceil = Stdlib.ceil

let floor = Stdlib.floor

let expm1 = Stdlib.expm1

let log1p = Stdlib.log1p

let abs_float = Stdlib.abs_float

let rsqrt x = 1.0 /. Stdlib.sqrt x

(** {1 Binary math} *)

let pow = Float.pow

let atan2 = Stdlib.atan2

let hypot = Stdlib.hypot

let copysign = Stdlib.copysign

(** C [fmod]: result has the sign of the dividend, magnitude < |divisor|.
    [Float.rem] is OCaml's [fmod]. *)
let fmod = Float.rem

(** {1 Conversions} *)

let of_int = Stdlib.float_of_int

let of_int32 = Int32.to_float

let to_int = Stdlib.int_of_float

let to_int32 = Int32.of_float

let of_float32 (x : float) = x

let to_float32 (x : float) = x

(** {1 Operator forms}

    Declared by the stdlib so that [let open Float64 in] shadows the float32
    operators inside a kernel. Only reachable here when written as a qualified
    call ([Float64.( +. ) a b]); infix syntax is lowered structurally by the PPX
    (Sarek_parse_helpers) and never becomes an [IntrinsicRef]. *)

let ( +. ) (x : float) (y : float) = Stdlib.( +. ) x y

let ( -. ) (x : float) (y : float) = Stdlib.( -. ) x y

let ( *. ) (x : float) (y : float) = Stdlib.( *. ) x y

let ( /. ) (x : float) (y : float) = Stdlib.( /. ) x y

let ( <= ) (x : float) (y : float) = Stdlib.( <= ) x y

let ( >= ) (x : float) (y : float) = Stdlib.( >= ) x y

let ( < ) (x : float) (y : float) = Stdlib.( < ) x y

let ( > ) (x : float) (y : float) = Stdlib.( > ) x y
