(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The TRIGGER CONDITIONS of Sarek_lower_ir's two ppx-time refusals (backlog-165).
 *
 * `Sarek_lower_ir` refuses two module-constant-in-helper shapes with a located
 * `Ppxlib.Location.raise_errorf`:
 *
 *   1. a constant whose initializer contains a synchronising intrinsic, because
 *      prefixing it into the helper would execute the barrier once per call site
 *      and change convergence;
 *   2. a helper that both REFERENCES a module constant and REBINDS that name,
 *      because the backends emit `SLet` flat and the prefix plus the rebinding
 *      would be two declarations of one identifier in one block.
 *
 * NEITHER WAS PINNED BY ANYTHING. Both are ppx-time errors and this tree has no
 * negative-compile infrastructure (no "this .ml must FAIL to build, with this
 * message" rule), so each was code that had been observed firing once, by hand,
 * during a mutation run. A refactor making either unreachable would have been
 * invisible — the same gate-that-cannot-fail shape the refusals themselves guard
 * against.
 *
 * WHAT THIS PINS, AND WHAT IT DOES NOT. It tests the PREDICATES that decide to
 * refuse — `expr_barrier`, and the free-names/binders pair whose intersection is
 * the collision condition — not the `raise_errorf` call itself. That boundary is
 * deliberate and worth stating plainly rather than leaving a reader to assume
 * more coverage than exists:
 *
 *   - covered: the decision. A predicate that stops discriminating (the barrier
 *     detector returning None for a real barrier; the free-name collector going
 *     back to over-approximating; the binder collector missing a binder) goes red
 *     here.
 *   - NOT covered: that a positive decision actually reaches the user as a
 *     located error. That needs a negative-compile rule, which does not exist.
 *
 * The first version of `expr_barrier` returned `false` unconditionally — a guard
 * that could not fire — so the detector is exactly the part with a history of
 * being wrong, and it is the part covered here.
 *
 * Reachable because `sarek_frontend` is a plain (wrapped false) library with no
 * .mli; `sarek/tests/e2e/test_stdlib_meta_proof` already links it directly.
 ******************************************************************************)

module Ir = Sarek_ir_ppx
module L = Sarek_lower_ir

let failures = ref 0

let expect label cond =
  Printf.printf "  %-64s %s\n%!" label (if cond then "OK" else "FAIL") ;
  if not cond then incr failures

let var name id : Ir.var =
  {Ir.var_name = name; var_id = id; var_type = Ir.TFloat32; var_mutable = false}

(* Free names of a statement, as the lowering computes them. *)
let free st =
  let acc = Hashtbl.create 8 in
  L.stmt_names st [] acc ;
  Hashtbl.fold (fun k () l -> k :: l) acc [] |> List.sort compare

let binders st =
  let acc = Hashtbl.create 8 in
  L.stmt_binders st acc ;
  Hashtbl.fold (fun k () l -> k :: l) acc [] |> List.sort compare

(* The refusal condition the fold applies: the constant is referenced AND the
   helper binds the same name. Expressed here exactly as the lowering expresses
   it, so the test tracks the real predicate rather than a paraphrase. *)
let collides ~const_name st =
  List.mem const_name (free st) && List.mem const_name (binders st)

let () =
  print_endline "=== Sarek_lower_ir refusal triggers (backlog-165) ===" ;

  (* ── trigger 2: reference + rebinding of one name ─────────────────────────
     `let c = c *. 2.` — the initializer reads the OUTER c (so c is free) and the
     binding introduces a local c (so c is a binder). Both hold => refuse. *)
  let c = var "c" 1 in
  let rebinds_after_use =
    Ir.SLet (c, Ir.EVar c, Ir.SExpr (Ir.EVar (var "other" 2)))
  in
  expect
    "rebinding a REFERENCED constant is a collision (refuse)"
    (collides ~const_name:"c" rebinds_after_use) ;

  (* The discriminator, and the regression that motivated the free-name fix: a
     local that merely SHARES the name, initialised from something else. `c` is
     bound but NOT free, so there is nothing to prefix and nothing to collide.
     Under the original over-approximating collector this was ALSO reported as a
     reference, which is what made a previously-compiling helper emit two
     declarations of `c`. *)
  let shadow_only = Ir.SLet (c, Ir.EVar (var "src" 3), Ir.SExpr (Ir.EVar c)) in
  expect
    "a local that only SHARES the name is NOT a collision"
    (not (collides ~const_name:"c" shadow_only)) ;
  expect
    "  and specifically: it is bound but not free"
    (List.mem "c" (binders shadow_only) && not (List.mem "c" (free shadow_only))) ;

  (* A plain reference with no rebinding: free, not bound => prefix it, no
     refusal. This is the ordinary working case, and without it "refuses a
     collision" and "refuses every reference" would be the same observation. *)
  let plain_use = Ir.SExpr (Ir.EVar c) in
  expect
    "a plain reference is free, not bound (prefix, do not refuse)"
    (List.mem "c" (free plain_use) && not (List.mem "c" (binders plain_use))) ;

  (* ── trigger 1: a synchronising intrinsic in the initializer ──────────────
     Each name is asserted individually rather than as a set, so dropping one
     arm of `synchronising_intrinsics` fails on that arm and names it. *)
  List.iter
    (fun intr ->
      let e = Ir.EIntrinsic ([], intr, []) in
      expect
        (Printf.sprintf "barrier detected: %s" intr)
        (L.expr_barrier e = Some intr))
    [
      "block_barrier";
      "warp_barrier";
      "memory_fence_block";
      "memory_fence_device";
    ] ;

  (* NESTED, because an initializer is an expression tree and the first version
     of this detector returned false unconditionally — a top-level-only check
     would still miss the realistic shape. *)
  let nested =
    Ir.EBinop
      (Ir.Add, Ir.EVar (var "x" 4), Ir.EIntrinsic ([], "block_barrier", []))
  in
  expect
    "barrier detected inside a nested expression"
    (L.expr_barrier nested = Some "block_barrier") ;

  (* The negative control. A non-synchronising intrinsic must NOT be reported,
     or every module constant with any intrinsic in its initializer would be
     refused and the feature would be unusable rather than safe. *)
  let benign = Ir.EIntrinsic ([], "global_thread_id", []) in
  expect
    "  (control) a non-synchronising intrinsic is NOT a barrier"
    (L.expr_barrier benign = None) ;
  expect
    "  (control) a barrier-free expression is NOT a barrier"
    (L.expr_barrier (Ir.EVar (var "y" 5)) = None) ;

  if !failures > 0 then (
    Printf.printf "\n%d failure(s)\n" !failures ;
    exit 1) ;
  print_endline "\nall refusal triggers behave as specified"
