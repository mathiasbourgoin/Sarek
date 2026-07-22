(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for the L8 aligned host aggregate ABI: a MIXED-ALIGNMENT record
 * [{tag : int32; value : float64}] round-trips through a device kernel.
 *
 * Under the packed ABI this type was rejected on PTX and read/wrote the wrong
 * bytes on the C-family backends (the pre-existing landmine). With the aligned
 * ABI the host PPX get/set, the PTX layout, and the C-compiler-aligned
 * [typedef struct { int; double; }] all agree byte-for-byte:
 *   tag at offset 0, value at offset 8 (4 bytes of padding), element stride 16.
 *
 * The kernel reads every field and writes it back mutated; the host fills and
 * verifies. If the host and device disagree on the aligned layout, the read-back
 * values are wrong and the process exits non-zero (so CI catches it) on EVERY
 * available device, including CUDA under ZLUDA.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

type float32 = float

type float64 = float

(* Mixed-alignment record: int32 field (align 4) + float64 field (align 8).
   Aligned host ABI: tag@0, value@8, size 16 (only valid under L8). *)
type boxed = {tag : int32; value : float64} [@@sarek.type]

let boxed_copy_kirc =
  snd
    [%kernel
      fun (src : boxed vector) (dst : boxed vector) (n : int32) ->
        let tid = thread_idx_x in
        if tid < n then
          let b = src.(tid) in
          (* Mutate the int32 field; copy the float64 field verbatim. This
             round-trips the aligned layout (tag@0, value@8, stride 16): if host
             and device disagreed on the offsets, [value] would read/write the
             wrong bytes. Float64 arithmetic is out of scope for the ABI test. *)
          let next : boxed = {tag = b.tag + 1l; value = b.value} in
          dst.(tid) <- next]

let () =
  print_endline "=== ktype mixed-alignment {i32;f64} round-trip (L8) ===" ;
  let devs =
    Device.init ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"] ()
  in
  if Array.length devs = 0 then begin
    print_endline "No runtime devices found - IR generation test passed" ;
    exit 0
  end ;
  let any_failure = ref false in
  (* The float64 field requires device fp64 support. CPU backends
     (Native/Interpreter) are always fp64-capable and are the authoritative
     reference for the aligned host ABI, so they are hard-verified. GPU devices
     without fp64 (or whose fp64 path is a separate, pre-existing gap) are
     skipped rather than failed — L8 is a layout change, not fp64 enablement. *)
  let is_cpu dev =
    dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"
  in
  Array.iter
    (fun dev ->
      Printf.printf "runtime [%s] %s: %!" dev.Device.framework dev.Device.name ;
      if (not (is_cpu dev)) && not (Device.allows_fp64 dev) then
        print_endline "SKIP (no device fp64 support)"
      else
        try
          let n = 64 in
          let src = Vector.create_custom boxed_custom n in
          let dst = Vector.create_custom boxed_custom n in
          for i = 0 to n - 1 do
            Vector.set
              src
              i
              {tag = Int32.of_int i; value = float_of_int (n - i) +. 0.5} ;
            Vector.set dst i {tag = 0l; value = 0.0}
          done ;
          let threads = min 64 n in
          let grid_x = (n + threads - 1) / threads in
          let ir =
            match boxed_copy_kirc.Sarek.Kirc_types.body_ir with
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
            let s = Vector.get src i in
            let d = Vector.get dst i in
            let expected_tag = Int32.add s.tag 1l in
            let expected_value = s.value in
            if
              s.tag <> Int32.of_int i
              || d.tag <> expected_tag
              || abs_float (d.value -. expected_value) > 1e-9
            then (
              ok := false ;
              if i < 5 then
                Printf.printf
                  "\n\
                  \  Mismatch at %d: got {tag=%ld, value=%.3f} expected \
                   {tag=%ld, value=%.3f}%!"
                  i
                  d.tag
                  d.value
                  expected_tag
                  expected_value)
          done ;
          if !ok then print_endline "PASSED"
          else begin
            any_failure := true ;
            print_endline "FAILED"
          end
        with e ->
          any_failure := true ;
          Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e))
    devs ;
  if !any_failure then exit 1
