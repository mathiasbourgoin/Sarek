(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

type float32 = float

(* Test that public functions are accessible in kernels.
   Private functions (marked with sarek.module_private) should not be
   visible to external modules - this is enforced by the registry. *)
let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector)
          (ys : float32 vector)
          (dst : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then dst.(tid) <- Visibility_lib.public_add xs.(tid) ys.(tid)]
  in

  let _native, _kirc = kernel in
  print_endline "=== Visibility kernel IR ===" ;
  print_endline "===========================" ;
  print_endline "Visibility test PASSED (public_add accessible in kernel)"
