(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * In-place record-field store on a vector element: `v.(i).f <- e`.
 *
 * backlog-172. The construct is documented, is used by a shipped kernel
 * (p3_scale_y_kernel in test_soa_emitter_equiv), and worked on CUDA/PTX — while
 * the two CPU backends did something else entirely:
 *
 *   Interpreter: REFUSED, raising Unsupported_operation "record field
 *                assignment" / "not fully supported".
 *   Native:      ACCEPTED and silently dropped the store. The generated OCaml
 *                was a setfield on the fresh record Vector.get had just
 *                marshalled out of storage, so the write hit a temporary. No
 *                error on any path; the vector simply kept its old values.
 *
 * The Native half is the dangerous one, and it is why this test asserts on
 * EVERY available device rather than on the one that was broken loudly. A
 * silently-dropped store is indistinguishable from a kernel that did not run,
 * so the only thing that catches it is reading the values back and comparing.
 *
 * What is checked, per device:
 *   1. The written field holds the new value.
 *   2. The OTHER fields are untouched — a read-modify-write that rebuilt the
 *      record from defaults would satisfy (1) and destroy the rest.
 *   3. A store into the SECOND field of a mixed record lands in that field and
 *      not in the first, which is what a wrong field index looks like.
 *
 * Every device failure makes the process exit non-zero.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Explicit registration: linking a plugin does not enumerate its devices. *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

(* Three same-width fields: the store target plus two witnesses on either side,
   so a store that overruns in either direction is visible. *)
type triple = {a : float32; b : float32; c : float32} [@@sarek.type]

(* The SAME record with mutable fields, because Native's pre-fix behaviour was
   TWO different failures and only this one is silent.

   With immutable fields the old codegen did not compile at all: the emitted
   setfield produced "The record field b is not mutable" — loud, but
   misdiagnosed, since the problem was never mutability. That error is what
   pushed a user to add [mutable], and THEN the store compiled and was silently
   discarded (point3d in test_soa_emitter_equiv carries exactly that [mutable]
   with a comment describing it as necessary "to write a leaf in place").

   So the immutable case alone would prove-red as a build failure and never
   exercise the silent path. Both are pinned, and after the fix [mutable] is not
   required for either. *)
type mtriple = {
  mutable ma : float32;
  mutable mb : float32;
  mutable mc : float32;
}
[@@sarek.type]

let scale_b =
  snd
    [%kernel
      fun (v : triple vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).b <- v.(tid).b *. 2.0]

let scale_mb =
  snd
    [%kernel
      fun (v : mtriple vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mb <- v.(tid).mb *. 2.0]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let n = 64

let orig i =
  {a = float_of_int i; b = float_of_int (i + 1); c = float_of_int (i + 2)}

let morig i =
  {ma = float_of_int i; mb = float_of_int (i + 1); mc = float_of_int (i + 2)}

(* One checker over both record shapes. [read] returns the three field values in
   (target, witness1, witness2) order so the assertions below are shape-agnostic
   and cannot drift apart between the two cases. *)
let run_case (dev : Device.t) ~(label : string) ~kernel ~make ~read : bool =
  Printf.printf
    "field-store %-9s [%s] %s: %!"
    label
    dev.Device.framework
    dev.Device.name ;
  let v = make () in
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of kernel)
      ~args:[Vec v; Int n]
      ~block:(Sarek.Execute.dims1d (min 64 n))
      ~grid:(Sarek.Execute.dims1d ((n + 63) / 64))
      ()
  with
  | exception e ->
      (* A refusal is a FAILURE here, not a skip. The construct is part of the
         DSL and every backend in this list executes ordinary custom-vector
         kernels; a backend that cannot do this one is the defect. *)
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok = ref true in
      let reported = ref 0 in
      for i = 0 to n - 1 do
        let tgt, w1, w2 = read v i in
        let bad name got want =
          if Float.abs (got -. want) > 1e-4 then begin
            ok := false ;
            if !reported < 3 then begin
              incr reported ;
              Printf.printf "\n  @%d field %s: got %g want %g" i name got want
            end
          end
        in
        (* The written field doubled; the two witnesses untouched. Checking the
           witnesses is not padding: a read-modify-write that rebuilt the record
           from defaults would satisfy the first assertion and destroy the
           rest. *)
        bad "target" tgt (float_of_int (i + 1) *. 2.0) ;
        bad "witness 1 (must be untouched)" w1 (float_of_int i) ;
        bad "witness 2 (must be untouched)" w2 (float_of_int (i + 2))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let run_on (dev : Device.t) : bool =
  let immutable_ok =
    run_case
      dev
      ~label:"immutable"
      ~kernel:scale_b
      ~make:(fun () ->
        let v = Vector.create_custom triple_custom n in
        for i = 0 to n - 1 do
          Vector.set v i (orig i)
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.b, p.a, p.c))
  in
  let mutable_ok =
    run_case
      dev
      ~label:"mutable"
      ~kernel:scale_mb
      ~make:(fun () ->
        let v = Vector.create_custom mtriple_custom n in
        for i = 0 to n - 1 do
          Vector.set v i (morig i)
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mb, p.ma, p.mc))
  in
  immutable_ok && mutable_ok

let () =
  let devs =
    Device.init
      ~frameworks:
        ["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"; "HIP"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found — nothing asserted, and that is a gap" ;
    (* Exit non-zero: a run that asserted nothing must not read as a pass. *)
    exit 1
  end ;
  let any_failure = ref false in
  Array.iter (fun dev -> if not (run_on dev) then any_failure := true) devs ;
  Printf.printf "%d device(s) exercised\n%!" (Array.length devs) ;
  if !any_failure then exit 1
