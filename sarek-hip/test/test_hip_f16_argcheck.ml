(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #57 slice 1 review: an f16-vector kernel launched with FLOAT32 vectors.
 *
 * [Execute.vector_arg]'s [Vec] constructor is existential, so the element type
 * is erased before the launch. Before the launch-time check in
 * Execute.check_launch_args this compiled clean and, on gfx1100, read
 * and wrote 2N bytes of a 4N-byte buffer: input [1 2 3 4] came back [1 2 0 0].
 * The Native path caught it by accident (type-id comparison), so the corruption
 * was GPU-only and silent.
 *
 * Self-skips with exit 0 when no device is present.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

let () = Sarek_native.Native_plugin.init ()

let f16_double =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) *. 2.0)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let ir = ir_of f16_double

let n = 4

let failures = ref 0

let report label ok detail =
  if not ok then incr failures ;
  Printf.printf "    %-40s : %s%s\n" label (if ok then "OK" else "FAIL") detail

let launch dev out inp =
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec out; Vec inp]
    ~block:(Sarek.Execute.dims1d n)
    ~grid:(Sarek.Execute.dims1d 1)
    () ;
  Transfer.flush dev

(* The mismatch: float32 vectors into a float16-declared kernel. *)
let test_mismatch_rejected dev =
  let out = Vector.create Vector.float32 n in
  let inp = Vector.create Vector.float32 n in
  Array.iteri (fun i x -> Vector.set inp i x) [|1.0; 2.0; 3.0; 4.0|] ;
  let label = Printf.sprintf "%s f32-into-f16 rejected" dev.Device.framework in
  match launch dev out inp with
  | () ->
      let got = Vector.to_array out in
      report
        label
        false
        (Printf.sprintf
           " — LAUNCHED with no error; out = [%s] (silent corruption)"
           (String.concat
              " "
              (Array.to_list (Array.map (Printf.sprintf "%g") got))))
  (* ANCHORED on the exception SHAPE, not on a substring. The previous version
     accepted any message mentioning both "float16" and "float32", which the
     Native backend's unrelated `vec_get_custom: vector element type mismatch`
     failure could also have satisfied — i.e. it did not actually prove that
     Execute's launch-time check was the thing that rejected the launch. *)
  | exception
      Sarek.Execute_error.Execution_error
        (Sarek.Execute_error.Type_mismatch {expected; actual; context}) ->
      let has needle =
        let nl = String.length needle and hl = String.length context in
        let rec go i =
          i + nl <= hl && (String.sub context i nl = needle || go (i + 1))
        in
        go 0
      in
      (* Argument 0 is [out], declared `float16 vector` and supplied as f32. *)
      let ok =
        expected = "float16 vector"
        && actual = "float32 vector" && has "argument 0"
        && has "(parameter \"out\")"
      in
      if ok then report label true " (element-type check)"
      else
        report
          label
          false
          (Printf.sprintf
             " — right exception, wrong detail: expected=%S actual=%S \
              context=%S"
             expected
             actual
             context)
  | exception e ->
      report
        label
        false
        (Printf.sprintf " — wrong diagnostic: %s" (Printexc.to_string e))

(* Control: matching f16 vectors must still run and be correct. *)
let test_match_still_runs dev =
  let out = Vector.create Vector.float16 n in
  let inp = Vector.create Vector.float16 n in
  Array.iteri (fun i x -> Vector.set inp i x) [|1.0; 2.0; 3.0; 4.0|] ;
  let label =
    Printf.sprintf "%s matching f16 still runs" dev.Device.framework
  in
  match launch dev out inp with
  | () ->
      let got = Vector.to_array out in
      let ok = got = [|2.0; 4.0; 6.0; 8.0|] in
      report
        label
        ok
        (if ok then ""
         else
           Printf.sprintf
             " — out = [%s]"
             (String.concat
                " "
                (Array.to_list (Array.map (Printf.sprintf "%g") got))))
  | exception e -> report label false (" — " ^ Printexc.to_string e)

let () =
  Printf.printf "test_hip_f16_argcheck (#57 slice 1 review)\n" ;
  let devices = Device.init () in
  let exercised = ref 0 in
  if Array.length devices = 0 then print_endline "    [SKIP] no devices"
  else
    Array.iter
      (fun dev ->
        (* Only backends that implement f16 device-side can reach the launch. *)
        if
          List.mem dev.Device.framework ["HIP"; "CUDA"; "Native"; "Interpreter"]
        then begin
          Printf.printf "  %s: %s\n" dev.Device.framework dev.Device.name ;
          incr exercised ;
          test_mismatch_rejected dev ;
          test_match_still_runs dev
        end)
      devices ;
  (* Without this the suite prints PASSED after checking nothing at all: the
     per-device filter can exclude every enumerated device and the loop then
     runs zero assertions, which is indistinguishable from a real pass. *)
  if !exercised = 0 && Array.length devices > 0 then
    print_endline
      "    [SKIP] no enumerated device implements device-side f16 (HIP / CUDA \
       / Native / Interpreter) — 0 checks ran" ;
  if !failures = 0 && !exercised = 0 then
    print_endline "test_hip_f16_argcheck SKIPPED (no eligible device)"
  else if !failures = 0 then print_endline "test_hip_f16_argcheck PASSED"
  else begin
    Printf.printf "test_hip_f16_argcheck FAILED (%d)\n" !failures ;
    exit 1
  end
