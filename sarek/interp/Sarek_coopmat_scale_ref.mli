(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The reference arithmetic of {!Sarek_ir_types.CM_scale_add}
    (docs/design/coopmat-scaled-accumulate.md §2.3), separated from the
    evaluator so that it can be tested on chosen values without building a
    kernel.

    {b Why not {!Sarek_float32}.} Its [to_float32] flushes binary32 subnormals
    to zero (and rounds every value above [max_float32] to infinity), which is
    not what a HIP device does: Sarek compiles HIP with
    [-fno-gpu-flush-denormals-to-zero], so a subnormal [t3] is kept, and an
    oracle that flushed it would disagree with correct hardware. Every step here
    rounds with {!round}, the IEEE binary32 conversion (round to nearest even,
    subnormals and overflow as IEEE 754 specifies). Computing each step in
    binary64 and rounding once to binary32 gives the correctly rounded binary32
    result for one multiplication, addition or subtraction of binary32 values
    (binary64 has at least [2 * 24 + 2] significant bits). *)

(** Round a binary64 value to the nearest binary32 value (ties to even). *)
val round : float -> float

(** The conversion [t1] of an s32 source element: exact when [|x| <= 2^24],
    correctly rounded otherwise. *)
val of_int32 : int32 -> float

(** One element: [t1 = src] (already binary32), [t2 = t1 * row], [t3 = t2 * col]
    ([t2] without [col]), [t4 = dst + t3] ([t3] when [accumulate = false], and
    [dst] is not read), [t5 = p * q], [result = t4 - t5] ([t4] without
    [offset]). *)
val element :
  accumulate:bool ->
  dst:float ->
  src:float ->
  row:float ->
  col:float option ->
  offset:(float * float) option ->
  float

(** The scales of one statement, already read from their buffers: [row] has [m]
    elements, [col] and [off_col] [n], [off_row] [m]. *)
type scales = {
  row : float array;
  col : float array option;
  offset : (float array * float array) option;  (** [(off_row, off_col)] *)
}

(** [apply ~m ~n ~accumulate ~scales ~dst ~src] is the new row-major [m x n]
    destination fragment; [dst] and [src] are row-major binary32 values (an s32
    source already converted with {!of_int32}). Fresh array: [dst] and [src] may
    be the same array. *)
val apply :
  m:int ->
  n:int ->
  accumulate:bool ->
  scales:scales ->
  dst:float array ->
  src:float array ->
  float array
