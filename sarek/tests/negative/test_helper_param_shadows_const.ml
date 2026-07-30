(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile (backlog-180, H3).

   A helper PARAMETER named [scale] collides with a module constant [scale] that
   the helper genuinely needs. The collision is deliberately NOT a direct
   reference: [scale] is named by the INITIALIZER of [base], and [base] is what
   the helper body references. That routing matters, because the fix for H3 seeds
   the referenced-name scan with the parameter names, which removes a DIRECTLY
   named constant from the set — so a direct collision compiles cleanly and is
   covered by the e2e test instead. The transitive closure runs with an empty
   bound list, so a constant reached through another constant's initializer still
   enters the set, and this is the shape that reaches the refusal.

   Why a negative-compile case rather than an e2e assertion: the outcome is a
   [Location.raise_errorf] during lowering, so the kernel never reaches a device
   and nothing about it is observable from a running test. It is also the ONLY
   user-visible behaviour the H3 fix added — the seeding half makes previously
   miscompiled kernels correct, but pairing it with the parameter names in the
   binder table makes this shape a hard REJECTION of source that used to build.
   A rejection nothing asserts is a rejection that can silently widen.

   What is pinned is the message, not just the failure. Before this case existed
   the guard emitted the body-local wording for a parameter collision, which was
   false twice over: it called the parameter "a local", and it advised "pass the
   constant in as a parameter", which is precisely what collides. Asserting only
   "some error" would have kept both.

   DELIBERATELY SELF-CONTAINED. The sibling negative tests here open [Spoc] and
   reach for [Kirc], neither of which is in their library dependencies — they
   only get away with it because the refusal fires before those names are
   resolved. That makes the guard-stopped-firing case report "Unbound module
   Spoc" instead of the kernel having compiled, which is a real error in place of
   the informative one. Measured while proving this case red: with the guard
   reverted, that is exactly what the first draft of this file printed. So this
   file depends only on what it uses, and if the refusal ever stops firing it
   compiles and reaches the [print_endline] below. *)

type float32 = float

type ('a, 'b) vector = ('a, 'b) Spoc_core.Vector.t

let () =
  let _bad_kernel =
    [%kernel
      let open Std in
      let (scale : float32) = 100.0 in
      let (base : float32) = scale +. 1.0 in
      let shifted (scale : float32) : float32 = base +. scale in
      fun (out : float32 vector) (src : float32 vector) (n : int32) ->
        let t = thread_idx_x + (block_idx_x * block_dim_x) in
        if t < n then out.(t) <- shifted src.(t)]
  in
  print_endline "This should not print - test should have failed to compile"
