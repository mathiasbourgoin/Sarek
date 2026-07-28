(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Helper-function scoping in the interpreter, against the GPU backends as the
 * reference.
 *
 * Three defects, all of which produced a SILENTLY WRONG NUMBER on the
 * interpreter — the cross-backend oracle — while Vulkan and Native computed the
 * right one:
 *
 *   A. the tail-recursion transform allocated its loop scaffolding from a
 *      counter starting at 0, the same range the typer uses for parameter ids.
 *      On `vsum acc v k n` the `_result` temporary took id 3 and so did the
 *      parameter `n`; since lookup_var resolves by id before name, the temporary
 *      answered every reference to `n`, the loop bound became 0, and the fold
 *      returned 0 instead of 4.
 *   B. a helper call inherited a COPY of the caller's scope (copy_env), so a
 *      caller binding could answer a callee reference. NOT covered here and NOT
 *      claimed as a repaired defect: reverting that change leaves this whole
 *      file green, so it is hardening whose absence nothing would notice. Said
 *      plainly rather than counted as a third fix.
 *   E. the same id collision as A, but with a body-local [let] in the helper.
 *      Seeding the transform above the PARAMETER ids alone did not fix it —
 *      max(param_id)+1 is exactly the first body-local's id — and the earlier
 *      appeal to case D as evidence was void, since case D is non-recursive and
 *      the transform never runs for it.
 *   C. get_array consulted `arrays` (the kernel's vectors) before the callee's
 *      own bindings, so a helper formal named like a kernel vector read the
 *      KERNEL's buffer. Folding `other` returned 400 — the contents of `src`.
 *
 * A FOURTH suspicion was probed and NOT confirmed, so nothing was changed for
 * it: `hf_params` gives each helper parameter the POSITIONAL INDEX as its id
 * while every use site inside the body carries the typer's id, which means the
 * same parameter has two ids and the name fallback in lookup_var is silently
 * load-bearing. Case D below was written to expose that (four parameters, two
 * body bindings, no tail recursion) and it passes identically with and without a
 * fix, on all five devices. Filed rather than fixed: an unobservable change is
 * not shippable on the strength of looking more correct.
 *
 * Each case asserts BOTH an exact expected value per device AND agreement
 * across devices, and reports which backend deviates. An earlier version of this
 * header claimed the cross-device comparison while the code only compared each
 * device to a hardcoded constant — the claim is now implemented rather than
 * deleted, because it is the useful half: a constant says a run is wrong, the
 * comparison says which side is.
 *
 * Values are exact — small integers in binary64 — so there is no tolerance to
 * hide behind, and NaN would fail rather than pass.
 *
 * Case E carries a WALL-CLOCK guard. Its failure mode on the base revision is
 * not a wrong number but non-termination: the loop scaffolding overwrote the
 * counter it was testing, so `_continue` never cleared. Which of the two symptoms
 * appears — 0, or a hang — depends on how many kernels the process compiled
 * first, because the transform's id counter is global and persists. A test whose
 * red is a hang reads as a stuck CI job rather than a failure, so it must time
 * out on its own.
 *
 * Run with: dune exec sarek/tests/e2e/test_helper_scoping.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* A — single helper, tail-recursive fold over a vector parameter. One helper is
   deliberate: the three-helper shape in test_tailrec_vector_param happened to
   get a non-colliding numbering and passed while this one returned 0. *)
let single_helper_kernel =
  [%kernel
    let open Std in
    let open Sarek_float64 in
    let rec vsum (acc : float64) (v : float64 vector) (k : int32) (n : int32) :
        float64 =
      if k >= n then acc else vsum (acc +. v.(k)) v (k + 1l) n
    in
    fun (out : float64 vector) (src : float64 vector) (n : int32) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- vsum 0.0G src 0l n]

(* C — the helper's formal is named like a DIFFERENT kernel vector. The call
   passes `other`; a lookup that prefers the kernel's `arrays` folds `src`. *)
let shadowing_kernel =
  [%kernel
    let open Std in
    let open Sarek_float64 in
    let rec fold_src (acc : float64) (src : float64 vector) (k : int32)
        (n : int32) : float64 =
      if k >= n then acc else fold_src (acc +. src.(k)) src (k + 1l) n
    in
    fun (out : float64 vector)
        (src : float64 vector)
        (other : float64 vector)
        (n : int32)
      ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- fold_src 0.0G other 0l n]

(* D — a NON-recursive helper with several parameters and a body binding. The
   positional-index scheme gave the parameters ids 0..n-1; a body [let] carries a
   typer id from an independent space, so it can land on one of them. No tail
   recursion here on purpose: this probes the id scheme itself, not the loop
   scaffolding. *)
let multi_param_kernel =
  [%kernel
    let open Std in
    let open Sarek_float64 in
    let combine (a : float64) (b : float64) (c : float64) (d : float64) :
        float64 =
      let z = a +. b in
      let w = z +. c in
      w +. d
    in
    fun (out : float64 vector) (src : float64 vector) (n : int32) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- combine src.(0l) src.(1l) src.(2l) src.(3l)]

(* E — the case that seeding above the parameter ids alone did NOT fix. *)
let body_local_kernel =
  [%kernel
    let open Std in
    let open Sarek_float64 in
    let rec vsum2 (acc : float64) (v : float64 vector) (k : int32) (n : int32) :
        float64 =
      if k >= n then acc
      else
        let x = v.(k) in
        vsum2 (acc +. x) v (k + 1l) n
    in
    fun (out : float64 vector) (src : float64 vector) (n : int32) ->
      let t = global_thread_id in
      if t < 1l then out.(0l) <- vsum2 0.0G src 0l n]

let n = 4

let describe = function
  | Sarek_interp.Interp_error.Interpreter_error e ->
      Sarek_interp.Interp_error.error_to_string e
  | e -> Printexc.to_string e

let ir_of k =
  let _, kirc = k in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let fp64_devices () =
  Array.to_list (Device.init ())
  |> List.filter (fun (d : Device.t) ->
      let f = d.Device.framework in
      f = "Native" || f = "Interpreter" || Device.allows_fp64 d)

(* [fills] gives each input vector its constant value, in argument order after
   [out]. Returns out.(0) or an error string. *)
let run (dev : Device.t) ir ~fills =
  try
    let out = Vector.create Vector.float64 1 in
    Vector.set out 0 0.0 ;
    let vecs =
      List.map
        (fun v ->
          let a = Vector.create Vector.float64 n in
          for i = 0 to n - 1 do
            Vector.set a i v
          done ;
          a)
        fills
    in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        ((Execute.Vec out :: List.map (fun a -> Execute.Vec a) vecs)
        @ [Execute.Int n])
      ~block:(Execute.dims1d 1)
      ~grid:(Execute.dims1d 1)
      () ;
    Transfer.flush dev ;
    Ok (Vector.to_array out).(0)
  with e -> Error (describe e)

let failures = ref 0

let case ?(timeout_s = 30) label ir ~fills ~expected =
  Printf.printf "  %s (expect %g on every device)\n%!" label expected ;
  let devs = fp64_devices () in
  if devs = [] then begin
    print_endline "    no fp64-capable device — proves nothing" ;
    incr failures
  end ;
  let results =
    List.map
      (fun (dev : Device.t) ->
        (* Wall-clock guard: the pre-fix failure mode of case E is
           non-termination, and a hung run is not a reported failure. *)
        let alarm_fired = ref false in
        let prev =
          Sys.signal
            Sys.sigalrm
            (Sys.Signal_handle
               (fun _ ->
                 alarm_fired := true ;
                 Printf.printf
                   "    %-11s %-40s TIMEOUT after %ds\n%!"
                   dev.Device.framework
                   dev.Device.name
                   timeout_s ;
                 exit 1))
        in
        ignore (Unix.alarm timeout_s) ;
        let r = run dev ir ~fills in
        ignore (Unix.alarm 0) ;
        Sys.set_signal Sys.sigalrm prev ;
        ignore !alarm_fired ;
        (dev, r))
      devs
  in
  List.iter
    (fun ((dev : Device.t), r) ->
      match r with
      | Ok got ->
          Printf.printf
            "    %-11s %-40s got=%g %s\n%!"
            dev.Device.framework
            dev.Device.name
            got
            (if got = expected then "" else "<-- WRONG") ;
          if got <> expected then incr failures
      | Error msg ->
          Printf.printf
            "    %-11s %-40s ERROR %s\n%!"
            dev.Device.framework
            dev.Device.name
            msg ;
          incr failures)
    results ;
  (* Cross-device agreement, reported separately: it says WHICH backend deviates,
     which the per-device constant check cannot. *)
  let oks =
    List.filter_map
      (fun (d, r) -> match r with Ok v -> Some (d, v) | _ -> None)
      results
  in
  let values = List.sort_uniq compare (List.map snd oks) in
  if List.length values > 1 then begin
    let tally =
      List.map
        (fun v ->
          ( v,
            List.filter_map
              (fun ((d : Device.t), w) ->
                if w = v then Some d.Device.framework else None)
              oks ))
        values
    in
    let majority, _ =
      List.fold_left
        (fun (bv, bn) (v, ds) ->
          let n = List.length ds in
          if n > bn then (v, n) else (bv, bn))
        (expected, 0)
        tally
    in
    Printf.printf
      "    DISAGREEMENT — majority %g, deviating: %s\n%!"
      majority
      (String.concat
         ", "
         (List.concat_map
            (fun (v, ds) ->
              if v = majority then []
              else List.map (fun f -> Printf.sprintf "%s(%g)" f v) ds)
            tally)) ;
    incr failures
  end

let () =
  print_endline "=== helper-function scoping in the interpreter ===" ;
  (* src = four 1.0s, so the fold is 4. *)
  case
    "A: single-helper tail-recursive fold over a vector parameter"
    (ir_of single_helper_kernel)
    ~fills:[1.0]
    ~expected:4.0 ;
  (* src = 100.0 each, other = 1.0 each. Folding `other` is 4; folding the
     kernel's `src` instead is 400, which is what the wrong precedence gave. *)
  case
    "C: helper formal named like a DIFFERENT kernel vector"
    (ir_of shadowing_kernel)
    ~fills:[100.0; 1.0]
    ~expected:4.0 ;
  (* src = four 1.0s, so a + b + c + d = 4. *)
  case
    "D: non-recursive helper, four params and two body bindings"
    (ir_of multi_param_kernel)
    ~fills:[1.0]
    ~expected:4.0 ;
  (* E — same as A plus one body-local `let`. src = four 1.0s, fold = 4. *)
  case
    "E: tail-recursive helper with a body-local let"
    (ir_of body_local_kernel)
    ~fills:[1.0]
    ~expected:4.0 ;
  if !failures > 0 then exit 1
