(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A helper parameter has ONE identity (backlog-158).
 *
 * THE DEFECT. `Sarek_lower_ir` built `hf_params` with `List.mapi`, giving each
 * helper parameter its POSITIONAL INDEX as `var_id` while `p.tparam_id` — the
 * typer's id, the one every use site inside the body carries — was destructured
 * and thrown away. The interpreter's `lookup_var` resolves by id BEFORE name, so
 * a reference resolved to whichever parameter happened to occupy that positional
 * slot. The name fallback was silently load-bearing: it only saved the cases
 * where the id lookup missed entirely.
 *
 * THE WINDOW, derived from the code rather than guessed. Let `c` be the global
 * typer counter when the helper's parameters are allocated and `n` the parameter
 * count. The wrong parameter is selected exactly for `1 <= c <= n-1`:
 *   c = 0    -> positional and typer ids coincide, accidental identity, no bug
 *   c >= n   -> every id lookup misses and the name fallback answers, no bug
 * So the defect needs something to have consumed ids first, and a MODULE
 * CONSTANT declared ahead of the helper does precisely that: it takes id 0, the
 * parameters start at 1, and every reference is off by one.
 *
 * WHY A NEW FILE, and this is not tidiness. The typer's counter is global and
 * persists across kernels in one process, so `c` depends on how many kernels
 * were compiled before. Putting this probe in an existing file would place it
 * outside the window and it would pass with and without the fix. It is the FIRST
 * kernel here for the same reason.
 *
 * WHY THE PREVIOUS PROBE COULD NOT FAIL. `test_helper_scoping.ml` case D was
 * written for this defect and cited as the reason backlog-158 was filed without
 * a fix. It cannot go red, for two independent reasons: it is the third kernel in
 * its file (outside the window), and it passes `~fills:[1.0]` so all four
 * arguments are EQUAL — any permutation of them sums to the same thing. "Passes
 * identically with and without the fix" was a fact about that test, not about
 * the fix.
 *
 * ARGUMENTS ARE DISTINCT, AND THAT IS THE ASSERTION. With `combine a b c d`
 * returning `a + b + c` and arguments 1, 2, 3, 4:
 *   correct  a+b+c = 1+2+3 = 6
 *   off-by-one (a->b, b->c, c->d) = 2+3+4 = 9
 * A probe with equal arguments cannot tell those apart, which is exactly how the
 * defect survived being probed.
 *
 * Native is the control: it lowers the helper to an OCaml function whose
 * parameters bind by name, so it is correct either way. The interesting output is
 * the DISAGREEMENT between it and the Interpreter.
 *
 * Run with: dune exec sarek/tests/e2e/test_hf_param_identity.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

(* FIRST kernel in this file, deliberately — see the header. The module constant
   is what puts the parameter ids inside the failure window; that is its only job.
   It is added into the RESULT (and is 0.0, so the expected value is unchanged)
   purely so it is not an unused binding — and it is referenced from the KERNEL
   body rather than from the helper, which keeps the helper's situation exactly as
   it was. *)
let off_by_one_kernel =
  [%kernel
    let open Std in
    let (bias : float32) = 0.0 in
    let combine (a : float32) (b : float32) (c : float32) (d : float32) :
        float32 =
      a +. b +. c
    in
    fun (out : float32 vector) (src : float32 vector) ->
      let t = global_thread_id in
      if t < 1l then
        out.(0l) <- combine src.(0l) src.(1l) src.(2l) src.(3l) +. bias]

let describe = function
  | Sarek_interp.Interp_error.Interpreter_error e ->
      Sarek_interp.Interp_error.error_to_string e
  | e -> Printexc.to_string e

let ir_of k =
  let _, kirc = k in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let fp32_devices () =
  Array.to_list (Device.init ())
  |> List.filter (fun (d : Device.t) ->
      let f = d.Device.framework in
      (* Metal included after review on #363. The defect is in lowering, so it
         reaches every backend; omitting Metal dropped the one platform where
         this would first be seen by a user on Apple hardware. Matches the
         framework lists in test_defunc and test_ematch_payload_e2e. Not
         observable from this host — Device.init returns no Metal device on
         Linux, so the addition is inert here and verified only by matching the
         convention. *)
      f = "Native" || f = "Interpreter" || f = "OpenCL" || f = "CUDA"
      || f = "Vulkan" || f = "Metal")

let run (dev : Device.t) ir =
  try
    let out = Vector.create Vector.float32 1 in
    Vector.set out 0 0.0 ;
    let src = Vector.create Vector.float32 4 in
    (* DISTINCT, and that is the whole assertion — see the header. *)
    List.iteri (fun i v -> Vector.set src i v) [1.0; 2.0; 3.0; 4.0] ;
    Execute.run_vectors
      ~device:dev
      ~ir
        (* Exactly the kernel's two parameters. A third argument made the first
         run of this probe fail on every device with an arg-count mismatch — red,
         but for a reason that proves nothing about the defect. *)
      ~args:[Execute.Vec out; Execute.Vec src]
      ~block:(Execute.dims1d 1)
      ~grid:(Execute.dims1d 1)
      () ;
    Transfer.flush dev ;
    Ok (Vector.to_array out).(0)
  with e -> Error (describe e)

let failures = ref 0

let () =
  print_endline "=== a helper parameter has one identity (backlog-158) ===" ;
  print_endline "  combine a b c d = a +. b +. c, args 1 2 3 4" ;
  print_endline "  correct 6.0 — off-by-one (a->b, b->c, c->d) would give 9.0" ;
  let devs = fp32_devices () in
  if devs = [] then begin
    print_endline "    no device at all — proves nothing" ;
    incr failures
  end ;
  let results =
    List.map
      (fun (dev : Device.t) -> (dev, run dev (ir_of off_by_one_kernel)))
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
            (if got = 6.0 then ""
             else if got = 9.0 then "<-- OFF BY ONE"
             else "<-- WRONG") ;
          if got <> 6.0 then incr failures
      | Error msg ->
          Printf.printf
            "    %-11s %-40s ERROR %s\n%!"
            dev.Device.framework
            dev.Device.name
            msg ;
          incr failures)
    results ;
  (* Cross-device agreement, reported separately because it names WHICH side is
     wrong. Native binds helper parameters by name and is correct either way, so
     a disagreement here localises the defect to the id-resolving side rather
     than merely saying a number was wrong. *)
  let oks =
    List.filter_map
      (fun (d, r) -> match r with Ok v -> Some (d, v) | _ -> None)
      results
  in
  (match List.sort_uniq compare (List.map snd oks) with
  | [] | [_] -> ()
  | values ->
      Printf.printf
        "    DISAGREEMENT across devices: %s\n%!"
        (String.concat
           ", "
           (List.map
              (fun v ->
                Printf.sprintf
                  "%g on [%s]"
                  v
                  (String.concat
                     " "
                     (List.filter_map
                        (fun ((d : Device.t), w) ->
                          if w = v then Some d.Device.framework else None)
                        oks)))
              values)) ;
      incr failures) ;
  if !failures > 0 then exit 1
