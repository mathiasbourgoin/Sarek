(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #57 slice 1 review: an f16-vector kernel launched with FLOAT32 vectors.
 *
 * [Execute.vector_arg]'s [Vec] constructor is existential, so the element type
 * is erased before the launch. Before the launch-time check in
 * Execute.check_vector_element_types this compiled clean and, on gfx1100, read
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
  | exception e ->
      let msg = Printexc.to_string e in
      let mentions =
        let has needle =
          let nl = String.length needle and hl = String.length msg in
          let rec go i =
            i + nl <= hl && (String.sub msg i nl = needle || go (i + 1))
          in
          go 0
        in
        has "float16" && has "float32"
      in
      if mentions then report label true " (element-type check)"
      else report label false (Printf.sprintf " — wrong diagnostic: %s" msg)

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
  if Array.length devices = 0 then print_endline "    [SKIP] no devices"
  else
    Array.iter
      (fun dev ->
        (* Only backends that implement f16 device-side can reach the launch. *)
        if
          List.mem dev.Device.framework ["HIP"; "CUDA"; "Native"; "Interpreter"]
        then begin
          Printf.printf "  %s: %s\n" dev.Device.framework dev.Device.name ;
          test_mismatch_rejected dev ;
          test_match_still_runs dev
        end)
      devices ;
  if !failures = 0 then print_endline "test_hip_f16_argcheck PASSED"
  else begin
    Printf.printf "test_hip_f16_argcheck FAILED (%d)\n" !failures ;
    exit 1
  end
