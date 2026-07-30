(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test: The `let x : t = e` annotation must be READ (backlog-192).

   This is the positive half of the sweep, asserted the only way a negative-
   compile case can assert it: the annotation contradicts the expression, so if it
   is read there is a type error and if it is dropped the kernel compiles.

   Before the fix this file compiled at exit 0. Since OCaml 5.1 the annotation of
   `let x : t = e` lives in `pvb_constraint`, not in the pattern, and nothing read
   that field -- `extract_type_from_pattern` only ever saw the `let (x : t) = e`
   spelling. A kernel-local `let sum : float = ...` was therefore typed by
   inference with the declared width ignored. *)

let k =
  [%kernel
    fun (src : int32 vector) (dst : int32 vector) ->
      let tid = thread_idx_x in
      let x : float32 = src.(tid) in
      dst.(tid) <- x]
