(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tail-recursive helper functions taking a VECTOR parameter, on every device.
 *
 * Writing an accumulation as a tail-recursive fold - the functional spelling of
 * a loop - puts the vector being folded over in the helper's parameter list.
 * Nothing in the DSL forbids that, and Sarek_tailrec rewrites the self call
 * into a loop before codegen, so a backend only ever sees an ordinary function
 * whose body is a loop.
 *
 * That path had no coverage. test_bounded_recursion, the only other
 * tail-recursion test, runs on devs.(0) ONLY - a single device, whichever comes
 * first - and all three of its helpers are int32-only with no vector
 * parameter. Two independent defects lived here undetected:
 *
 *   - the interpreter raised "Unbound variable '<param>' in get_array" when a
 *     helper indexed a vector parameter, because [eval_app] binds arguments
 *     into [vars] while [get_array] looked only in [arrays]/[shared]. The
 *     cross-backend ORACLE was the one backend that could not run these
 *     kernels;
 *   - [Float64.of_int32] had no arm in the interpreter or in the GLSL backend,
 *     though it is declared in the stdlib and type-checks in the DSL.
 *
 * Three helpers isolate the failing axis rather than merging it: A is int32
 * with no vector parameter, B adds a float64 accumulator, C adds the vector
 * parameter. A and B passed throughout; only C ever failed, which is what
 * localised the interpreter bug to argument binding.
 *
 * The test also asserts, on the emitted CUDA, that the self call is GONE
 * rather than only that the result is right: a correct answer on Native would
 * not distinguish "compiled to a loop" from "evaluated as recursion".
 *
 * Run with: dune exec sarek/tests/e2e/test_tailrec_vector_param.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

let probe_kernel =
  [%kernel
    let open Std in
    let open Sarek_float64 in
    (* A: int32 accumulator, no vector parameter. *)
    let rec isum (acc : int32) (k : int32) (n : int32) : int32 =
      if k >= n then acc else isum (acc + k) (k + 1l) n
    in
    (* B: float64 accumulator, no vector parameter. Also covers
       Float64.of_int32. *)
    let rec fsum (acc : float64) (k : int32) (n : int32) : float64 =
      if k >= n then acc else fsum (acc +. Float64.of_int32 k) (k + 1l) n
    in
    (* C: float64 accumulator WITH a vector parameter - the case under test. *)
    let rec vsum (acc : float64) (v : float64 vector) (k : int32) (n : int32) :
        float64 =
      if k >= n then acc else vsum (acc +. v.(k)) v (k + 1l) n
    in
    fun (out : float64 vector) (src : float64 vector) (n : int32) ->
      let t = global_thread_id in
      if t < 1l then begin
        out.(0l) <- Float64.of_int32 (isum 0l 0l n) ;
        out.(1l) <- fsum 0.0G 0l n ;
        out.(2l) <- vsum 0.0G src 0l n
      end]

let n = 10

(* src.(i) = i, so all three helpers compute 0+1+...+9 = 45. Every value is a
   small integer, so binary64 represents each partial sum exactly and the
   comparison is bit-for-bit on EVERY device - no tolerance, and none
   warranted. A tolerance here would let a genuinely wrong fold pass. *)
let expected = [|45.0; 45.0; 45.0|]

let labels =
  [|
    "A int32 acc, no vector param";
    "B float64 acc, no vector param";
    "C float64 acc, WITH vector param";
  |]

let slots = Array.length labels

let is_interpreter framework = framework = "Interpreter"

let describe = function
  | Sarek_interp.Interp_error.Interpreter_error e ->
      Sarek_interp.Interp_error.error_to_string e
  | e -> Printexc.to_string e

(* Each helper must appear exactly twice in the emitted CUDA: its definition,
   and the single call from the kernel body. A third occurrence would be a
   surviving self call, i.e. tail-recursion elimination did not fire. *)
let assert_tailrec_eliminated ir =
  let src = Sarek_codegen.Sarek_ir_cuda.generate ir in
  let occurrences needle =
    let nh = String.length src and nn = String.length needle in
    let rec go i acc =
      if i + nn > nh then acc
      else go (i + 1) (if String.sub src i nn = needle then acc + 1 else acc)
    in
    go 0 0
  in
  let bad =
    List.filter
      (fun name -> occurrences (name ^ "(") <> 2)
      ["isum"; "fsum"; "vsum"]
  in
  if bad = [] then begin
    print_endline "  tailrec->loop: no self call survives in the emitted CUDA" ;
    true
  end
  else begin
    Printf.printf
      "  tailrec->loop: FAILED - self call survives for %s\n%!"
      (String.concat ", " bad) ;
    false
  end

let run_on_device (dev : Device.t) ir =
  let out = Vector.create Vector.float64 slots in
  let src = Vector.create Vector.float64 n in
  for i = 0 to slots - 1 do
    Vector.set out i 0.0
  done ;
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int i)
  done ;
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec out; Execute.Vec src; Execute.Int n]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let () =
  let _, kirc = probe_kernel in
  let ir =
    match kirc.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "tail-recursion probe kernel has no IR"
  in
  print_endline "=== tail-recursive helpers, vector parameter ===" ;
  let codegen_ok = assert_tailrec_eliminated ir in
  let devs = Device.init () in
  let ran = ref 0 in
  let failures = ref 0 in
  let interpreters_run = ref 0 in
  let gap_seen = ref 0 in
  Array.iter
    (fun (dev : Device.t) ->
      let framework = dev.Device.framework in
      let native = framework = "Native" || framework = "Interpreter" in
      (* Every helper returns float64, so a device without fp64 cannot run this
         kernel at all - skipping it is correct, not a silent pass. *)
      if native || Device.allows_fp64 dev then begin
        incr ran ;
        if is_interpreter framework then incr interpreters_run ;
        try
          let got = run_on_device dev ir in
          let bad = ref 0 in
          Array.iteri
            (fun i l ->
              if got.(i) <> expected.(i) then begin
                Printf.printf
                  "    %-34s got=%.17g want=%.17g <-- WRONG\n%!"
                  l
                  got.(i)
                  expected.(i) ;
                incr bad
              end)
            labels ;
          Printf.printf
            "  %-11s %-40s %s\n%!"
            framework
            dev.Device.name
            (if !bad = 0 then "PASS" else "FAIL") ;
          if !bad <> 0 then incr failures
        with e ->
          (* DOCUMENTED GAP, not a failure. Helper C passes a vector as a helper
             PARAMETER; the interpreter binds arguments into [vars] while
             [get_array] looks only in [arrays]/[shared], so it refuses. That is
             the LOUD behaviour, and it is deliberately still in place: the
             one-line fallback that makes it run is not sufficient on its own,
             because helper parameter ids and body-local ids come from two
             overlapping numbering spaces and a tail-recursion temporary can
             clobber a parameter — a single-helper fold then returns 0 on the
             interpreter where Vulkan and Native return 4, silently. Trading a
             loud refusal for a sometimes-wrong number is a regression, so the
             repair lands with the id fix rather than ahead of it.

             When that lands this branch stops being taken and the test goes
             RED — on purpose. That is the reminder to delete this arm and let
             helper C be asserted on every device like A and B. *)
          let expected_gap =
            (* Match the CONSTRUCTOR, not the rendered text: a reworded
               error_to_string would otherwise silently reclassify this
               documented gap as a generic ERROR and fail for an unrelated
               reason. *)
            is_interpreter framework
            &&
            match e with
            | Sarek_interp.Interp_error.Interpreter_error
                (Sarek_interp.Interp_error.Unbound_variable
                   {name = "v"; context = "get_array"}) ->
                true
            | _ -> false
          in
          if expected_gap then incr gap_seen ;
          Printf.printf
            "  %-11s %-40s %s (%s)\n%!"
            framework
            dev.Device.name
            (if expected_gap then "KNOWN-GAP" else "ERROR")
            (describe e) ;
          if not expected_gap then incr failures
      end)
    devs ;
  if !gap_seen = 0 && !interpreters_run > 0 then begin
    print_endline
      "test_tailrec_vector_param: the interpreter no longer refuses a vector \
       helper parameter — the id-allocation fix has landed. Delete the \
       KNOWN-GAP arm and assert helper C on every device." ;
    exit 1
  end ;
  (* A run that reached no device proves nothing; report it as a failure rather
     than exiting 0 on an empty device list. *)
  if !ran = 0 then begin
    print_endline
      "test_tailrec_vector_param: FAILED - no fp64-capable device available" ;
    exit 1
  end ;
  Printf.printf "  %d device(s), %d failure(s)\n%!" !ran !failures ;
  if !failures > 0 || not codegen_ok then exit 1
