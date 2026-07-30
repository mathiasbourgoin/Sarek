(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: a [@@sarek.type] record with a RECORD-TYPED field must have its
 * inner struct declared BEFORE the struct that uses it (backlog-203).
 *
 * The declaration-emission loops walked [kern_types] in list order, which is
 * not a dependency order: the PPX merges registered types ahead of the
 * payload's own, so a record whose field type is another record could be
 * emitted first and reference an as-yet-undeclared struct. Every C-family
 * backend (OpenCL/CUDA/HIP/Metal) then failed at compile time with
 * `unknown type name '<Inner>'`, and GLSL/WGSL with a parse error at the
 * field line. The values-carrying backends (Interpreter, Native) never see a
 * struct declaration and were unaffected.
 *
 * Three shapes are exercised, all of them beyond the trivial one-level pair:
 *   1. one-level nesting     — [outer { tag; mid : triple }]
 *   2. a THREE-level chain   — [chain_top { s : chain_mid { r : chain_leaf } }]
 *   3. two INDEPENDENT       — [twin { left : triple; right : chain_leaf }]
 *      nested types            (the sort must order both, deterministically)
 *
 * Each kernel is READ-ONLY on the nested field (`dst.(tid) <- src.(tid).f.g`).
 * That is deliberate: the defect is in the type-declaration emission, not in
 * the field-store path, so a read-only kernel is enough to reproduce it and
 * keeps this test independent of nested-field-store support.
 *
 * Every available device must PASS. There is no per-backend tolerance here:
 * a device that fails makes the process exit non-zero.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

(* --- Shape 1: one-level nesting ----------------------------------------- *)
type triple = {a : float32; b : float32; c : float32} [@@sarek.type]

type outer = {tag : float32; mid : triple} [@@sarek.type]

(* --- Shape 2: a three-level chain --------------------------------------- *)
type chain_leaf = {p : float32} [@@sarek.type]

type chain_mid = {r : chain_leaf} [@@sarek.type]

type chain_top = {s : chain_mid} [@@sarek.type]

(* --- Shape 3: two independent nested types in one record ---------------- *)
type twin = {left : triple; right : chain_leaf} [@@sarek.type]

(* Read `.mid.b` out of a nested record — no field store anywhere. *)
let k_outer =
  snd
    [%kernel
      fun (src : outer vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then dst.(tid) <- src.(tid).mid.b]

(* Read through a three-level chain. *)
let k_chain =
  snd
    [%kernel
      fun (src : chain_top vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then dst.(tid) <- src.(tid).s.r.p]

(* Read both independent nested fields and combine them. *)
let k_twin =
  snd
    [%kernel
      fun (src : twin vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then dst.(tid) <- src.(tid).left.a +. src.(tid).right.p]

let n = 64

let devices () =
  Device.init
    ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"]
    ()

let ir_of name kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith (name ^ ": kernel has no IR")

(* Run one kernel on one device and verify the float32 output vector. *)
let run_case ~dev ~name ~kirc ~make_src ~expected =
  Printf.printf "  [%s] %s / %s: %!" dev.Device.framework dev.Device.name name ;
  try
    let src = make_src () in
    let dst = Vector.create Vector.float32 n in
    for i = 0 to n - 1 do
      Vector.set dst i 0.0
    done ;
    let threads = min 64 n in
    let grid_x = (n + threads - 1) / threads in
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d threads)
      ~grid:(Sarek.Execute.dims1d grid_x)
      ~ir:(ir_of name kirc)
      ~args:
        [
          Sarek.Execute.Vec src;
          Sarek.Execute.Vec dst;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      () ;
    Transfer.flush dev ;
    let ok = ref true in
    for i = 0 to n - 1 do
      let got = Vector.get dst i in
      let want = expected i in
      if abs_float (got -. want) > 1e-3 then begin
        if !ok then
          Printf.printf
            "\n    mismatch at %d: got %.3f expected %.3f%!"
            i
            got
            want ;
        ok := false
      end
    done ;
    if !ok then print_endline "PASSED" else print_endline "FAILED" ;
    !ok
  with e ->
    Printf.printf "FAILED (%s)\n%!" (Printexc.to_string e) ;
    false

let make_outer () =
  let v = Vector.create_custom outer_custom n in
  for i = 0 to n - 1 do
    let f = float_of_int i in
    Vector.set v i {tag = f; mid = {a = f; b = f +. 0.5; c = f +. 1.0}}
  done ;
  v

let make_chain () =
  let v = Vector.create_custom chain_top_custom n in
  for i = 0 to n - 1 do
    Vector.set v i {s = {r = {p = float_of_int i *. 2.0}}}
  done ;
  v

let make_twin () =
  let v = Vector.create_custom twin_custom n in
  for i = 0 to n - 1 do
    let f = float_of_int i in
    Vector.set v i {left = {a = f; b = 0.0; c = 0.0}; right = {p = f *. 10.0}}
  done ;
  v

let () =
  print_endline "=== nested-record struct declaration order (backlog-203) ===" ;
  let devs = devices () in
  if Array.length devs = 0 then begin
    print_endline "No devices found - nothing to verify" ;
    exit 0
  end ;
  let any_failure = ref false in
  Array.iter
    (fun dev ->
      (* Each case has its own element type, so the three runs are spelled out
         rather than folded over a list (no existential to hide the type). *)
      let check b = if not b then any_failure := true in
      check
        (run_case
           ~dev
           ~name:"one-level (outer.mid.b)"
           ~kirc:k_outer
           ~make_src:make_outer
           ~expected:(fun i -> float_of_int i +. 0.5)) ;
      check
        (run_case
           ~dev
           ~name:"three-level (top.s.r.p)"
           ~kirc:k_chain
           ~make_src:make_chain
           ~expected:(fun i -> float_of_int i *. 2.0)) ;
      check
        (run_case
           ~dev
           ~name:"two independent (left.a + right.p)"
           ~kirc:k_twin
           ~make_src:make_twin
           ~expected:(fun i -> float_of_int i +. (float_of_int i *. 10.0))))
    devs ;
  if !any_failure then begin
    print_endline "FAILED: at least one device/case did not verify" ;
    exit 1
  end ;
  print_endline "ALL PASSED"
