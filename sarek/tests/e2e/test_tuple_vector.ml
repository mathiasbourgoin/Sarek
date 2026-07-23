(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * L13 — E2E test for tuple-typed vectors: [(float32 * int32) vector].
 *
 * A tuple vector element is lowered to a synthesized packed record (fields
 * _0, _1) and consumed by the existing record/aggregate codegen. The host
 * builds the matching [custom_type] with [Sarek_tuple_vec], so the same bytes
 * are marshalled on both sides. The kernel reads a tuple element (via [match])
 * and writes a modified tuple element back; results are checked against a pure
 * OCaml reference on every available device.
 *
 * Device support tier: CUDA/PTX, OpenCL, Vulkan, Native AND the Interpreter
 * must pass. The Interpreter decodes a tuple element from the raw composite
 * bytes into a positional record ([_0], [_1], ...) using the shape layout
 * resolved from [Sarek_tuple_vec], and re-encodes it on writeback — the same
 * value model the record/aggregate path already uses.
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

(* Kernel: dst[i] = (src[i]._0 +. 1.0, src[i]._1) — reads and writes both
   components of the tuple element, exercising field offsets for a mixed
   (float32, int32) layout. *)
let tuple_copy_kirc =
  snd
    [%kernel
      fun (src : (float32 * int32) vector)
          (dst : (float32 * int32) vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then match src.(tid) with a, b -> dst.(tid) <- (a +. 1.0, b)]

(* Devices where the tuple-vector capability must work in this tier. *)
let must_pass fw =
  match fw with
  | "CUDA" | "OpenCL" | "Vulkan" | "Metal" | "Native" | "Interpreter" -> true
  | _ -> false

let () =
  print_endline "=== L13 tuple-vector E2E: (float32 * int32) vector ===" ;
  let devs =
    Device.init
      ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No runtime devices found - build/lowering test passed" ;
    exit 0
  end ;

  let n = 64 in
  let tup =
    Sarek_tuple_vec.pair Sarek_tuple_vec.float32 Sarek_tuple_vec.int32
  in
  (* Pure OCaml reference. *)
  let ref_src i = (float_of_int i, Int32.of_int (n - i)) in
  let ref_dst i =
    let a, b = ref_src i in
    (a +. 1.0, b)
  in

  let any_failure = ref false in
  let pass_count = ref 0 in
  let unsupported = ref 0 in
  Array.iter
    (fun dev ->
      let fw = dev.Device.framework in
      Printf.printf "runtime [%s] %s: %!" fw dev.Device.name ;
      let run () =
        let src = Vector.create_custom tup n in
        let dst = Vector.create_custom tup n in
        for i = 0 to n - 1 do
          Vector.set src i (ref_src i) ;
          Vector.set dst i (0.0, 0l)
        done ;
        let threads = min 64 n in
        let grid_x = (n + threads - 1) / threads in
        let ir =
          match tuple_copy_kirc.Sarek.Kirc_types.body_ir with
          | Some ir -> ir
          | None -> failwith "Kernel has no IR"
        in
        Sarek.Execute.run_vectors
          ~device:dev
          ~block:(Sarek.Execute.dims1d threads)
          ~grid:(Sarek.Execute.dims1d grid_x)
          ~ir
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
          let ea, eb = ref_dst i in
          let da, db = Vector.get dst i in
          if abs_float (da -. ea) > 1e-3 || db <> eb then begin
            ok := false ;
            if i < 5 then
              Printf.printf
                "\n  Mismatch at %d: got (%.2f, %ld) expected (%.2f, %ld)%!"
                i
                da
                db
                ea
                eb
          end
        done ;
        !ok
      in
      try
        if run () then begin
          incr pass_count ;
          print_endline "PASSED"
        end
        else if must_pass fw then begin
          any_failure := true ;
          print_endline "FAILED"
        end
        else begin
          incr unsupported ;
          print_endline "NOT-YET-SUPPORTED (result mismatch; see L13 findings)"
        end
      with e ->
        let msg =
          match e with
          | Sarek_backend_error.Backend_error.Backend_error err ->
              Sarek_backend_error.Backend_error.to_string err
          | e -> Printexc.to_string e
        in
        if must_pass fw then begin
          any_failure := true ;
          Printf.printf "FAIL (%s)\n%!" msg
        end
        else begin
          incr unsupported ;
          Printf.printf
            "NOT-YET-SUPPORTED (%s; see L13 findings)\n%!"
            (Printexc.to_string e)
        end)
    devs ;
  Printf.printf
    "\n=== tuple-vector: %d passed, %d not-yet-supported ===\n"
    !pass_count
    !unsupported ;
  if !any_failure then exit 1
