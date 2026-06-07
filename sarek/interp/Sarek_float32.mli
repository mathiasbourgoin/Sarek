(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Float32 - True 32-bit floating point operations
 *
 * Provides float32 semantics matching GPU behavior.
 ******************************************************************************)

(** {1 Constants} *)

(** Maximum representable float32 value (~3.4e38) *)
val max_float32 : float

(** Minimum positive normalized float32 value (~1.17e-38) *)
val min_positive_float32 : float

(** Machine epsilon for float32 (~1.19e-7) *)
val epsilon_float32 : float

(** Maximum input for exp before overflow (~88.7) *)
val max_exp_input : float

(** {1 Overflow Detection} *)

type overflow_mode =
  | Silent  (** Return infinity/-infinity silently (GPU behavior) *)
  | Warn  (** Print warning but continue *)
  | Exception  (** Raise exception *)

(** Set how overflow/underflow is handled. Default is Silent. *)
val set_overflow_mode : overflow_mode -> unit

(** Raised when overflow_mode is Exception and overflow occurs *)
exception Float32_overflow of string

(** Raised when overflow_mode is Exception and underflow occurs *)
exception Float32_underflow of string

(** {1 Conversion} *)

(** Truncate a float64 to float32 precision *)
val to_float32 : float -> float

(** Alias for to_float32 - convert float64 to float32 *)
val of_float : float -> float

val of_int32 : int32 -> float

val to_int32 : float -> int32

val of_int : int -> float

val to_int : float -> int

(** {1 Arithmetic} *)

val add : float -> float -> float

val sub : float -> float -> float

val mul : float -> float -> float

val div : float -> float -> float

val neg : float -> float

(** {1 Comparison} *)

val ( = ) : float -> float -> bool

val ( <> ) : float -> float -> bool

val ( < ) : float -> float -> bool

val ( > ) : float -> float -> bool

val ( <= ) : float -> float -> bool

val ( >= ) : float -> float -> bool

(** {1 Math Intrinsics} *)

val exp : float -> float

val log : float -> float

val log10 : float -> float

val pow : float -> float -> float

val sqrt : float -> float

(** Reciprocal square root: rsqrt x = 1 / sqrt(x) *)
val rsqrt : float -> float

val sin : float -> float

val cos : float -> float

val tan : float -> float

val asin : float -> float

val acos : float -> float

val atan : float -> float

val atan2 : float -> float -> float

val sinh : float -> float

val cosh : float -> float

val tanh : float -> float

val floor : float -> float

val ceil : float -> float

val abs : float -> float

(** Fused multiply-add: fma x y z = x * y + z *)
val fma : float -> float -> float -> float

(** {1 Min/Max with NaN handling} *)

val min : float -> float -> float

val max : float -> float -> float

val clamp : float -> float -> float -> float

(** {1 Predicates} *)

val is_finite : float -> bool

val is_nan : float -> bool

val is_inf : float -> bool

(** {1 Formatting} *)

val to_string : float -> string
