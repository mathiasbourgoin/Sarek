(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Negative test: [%%sarek.type] is not a supported syntax.
 *
 * Sarek registers a kernel-visible type through the [@@sarek.type] ATTRIBUTE.
 * An extension form was half-written and never wired into the driver, so
 * [%%sarek.type] failed as an uninterpreted extension while looking like a
 * public entry point. The half-implementation has been removed rather than
 * completed (see the commit that added this test for the argument), and this
 * test pins the outcome: the extension form is rejected, and the rejection is
 * ppxlib's, not a silent no-op.
 *
 * Expected error: "Uninterpreted extension 'sarek.type'"
 *
 * The supported spelling is the attribute, exercised by ~126 call sites and by
 * sarek/tests/e2e/test_ktype_record.ml, test_registered_variant.ml, etc:
 *
 *   type point = { x : float32; y : float32 } [@@sarek.type]
 ******************************************************************************)

[%%sarek.type type point = {x : float32; y : float32}]
