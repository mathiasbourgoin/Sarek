(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The scale operands of {!Sarek_ir_types.CM_scale_add}, in one place.

    [CM_scale_add] is the only cooperative-matrix statement that reads plain
    buffers through several index expressions (up to four scales). Every pass
    that renames buffers, rewrites expressions or collects the arrays a kernel
    reads must see all of them; these functions are how it does so, so that a
    pass cannot forget the offset's second scale. Every function is the identity
    (or empty) on the other members of {!Sarek_ir_types.coopmat_op}. *)

open Sarek_ir_types

(** The scales of the statement, in the order [row], [col], [off_row], [off_col]
    (absent ones skipped). *)
val scales : coopmat_op -> coopmat_scale list

(** The scale buffers the statement reads (it writes none). *)
val buffers : coopmat_op -> string list

(** The scale index expressions. *)
val exprs : coopmat_op -> expr list

(** Rewrite every scale buffer name with [buf] and index with [expr]. *)
val map :
  buf:(string -> string) -> expr:(expr -> expr) -> coopmat_op -> coopmat_op

(** [map ~buf:Fun.id ~expr:f]. *)
val map_exprs : (expr -> expr) -> coopmat_op -> coopmat_op

(** Fold over {!exprs}. *)
val fold_exprs : ('a -> expr -> 'a) -> 'a -> coopmat_op -> 'a
