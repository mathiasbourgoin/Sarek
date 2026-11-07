(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host cost of rewriting two small vectors (8 int32 + 8 float32) between
 * kernel launches that read them, the decode-loop pattern of a MoE router
 * (a downstream LLM engine: selected expert ids and their weights, once per layer).
 *
 * Modes, all producing the same kernel inputs:
 *   none         no rewrite (kernel time alone)
 *   vector_set   Vector.set on each element. After a launch the vectors are
 *                Stale_CPU, so the first set downloads each vector (waiting for
 *                the queue), and the next launch re-uploads it synchronously
 *                from pageable memory (waiting for the queue again).
 *   staged       Transfer.stage_to_device (this change).
 *
 * Reports wall microseconds per iteration (one flush at the end) and the
 * host time spent in the rewrite itself. The final output vector must be identical in every mode; a mismatch
 * exits 1. INFORMATION ONLY otherwise (alias e2e-hip-bench).
 *
 *   SPOC_BENCH_ITERS=480 SPOC_BENCH_SPIN=1000 SPOC_BENCH_HOST_US=300 dune exec .../bench_hip_small_writes.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std

let () = Sarek_hip.Hip_plugin.register ()

let consume =
  [%kernel
    fun (out : float32 vector)
        (slots : int32 vector)
        (w : float32 vector)
        (n : int32)
        (spin : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let acc = mut 0.0 in
        let i = mut 0l in
        while i < spin do
          acc := (acc *. 0.5) +. (w.(i land 7l) *. float slots.(i land 7l)) ;
          i := i + 1l
        done ;
        out.(tid) <- out.(tid) +. acc
      end]

let consume_ir =
  let _, kirc = consume in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "consume: no IR"

let env_int name default =
  match Option.bind (Sys.getenv_opt name) int_of_string_opt with
  | Some v -> v
  | None -> default

let iters = env_int "SPOC_BENCH_ITERS" 480

let spin = env_int "SPOC_BENCH_SPIN" 1000

let host_us = env_int "SPOC_BENCH_HOST_US" 300

let launches = env_int "SPOC_BENCH_LAUNCHES" 4

(* Host work between a rewrite and the launches that read it (downstream:
   fetching the selected experts' weights): a busy wait. When the rewrite does
   not make the host wait for the GPU, this overlaps the previous launches. *)
let host_work () =
  let until = Unix.gettimeofday () +. (float_of_int host_us *. 1e-6) in
  while Unix.gettimeofday () < until do
    ()
  done

let n = 1 lsl 16

let k = 8

let slot_value i j = Int32.of_int (((i * 7) + j) mod 13)

let weight_value i j = 1.0 /. float_of_int (j + 1 + (i mod 3))

let run dev mode =
  let out = Vector.create Vector.float32 n in
  Vector.fill out 0.0 ;
  let slots = Vector.create Vector.int32 k in
  let w = Vector.create Vector.float32 k in
  for j = 0 to k - 1 do
    Vector.set slots j (slot_value 0 j) ;
    Vector.set w j (weight_value 0 j)
  done ;
  Transfer.to_device out dev ;
  Transfer.to_device slots dev ;
  Transfer.to_device w dev ;
  let host_s = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout k in
  let host_w = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout k in
  let rewrite i =
    match mode with
    | "none" -> ()
    | "vector_set" ->
        for j = 0 to k - 1 do
          Vector.set slots j (slot_value i j) ;
          Vector.set w j (weight_value i j)
        done
    | "staged" ->
        for j = 0 to k - 1 do
          Bigarray.Array1.set host_s j (slot_value i j) ;
          Bigarray.Array1.set host_w j (weight_value i j)
        done ;
        Transfer.stage_to_device host_s ~dst:slots ~dst_offset:0 ;
        Transfer.stage_to_device host_w ~dst:w ~dst_offset:0
    | m -> failwith ("unknown mode " ^ m)
  in
  let launch () =
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:consume_ir
      ~args:
        Sarek.Execute.
          [
            Vec out;
            Vec slots;
            Vec w;
            Int32 (Int32.of_int n);
            Int32 (Int32.of_int spin);
          ]
      ~block:(Sarek.Execute.dims1d 256)
      ~grid:(Sarek.Execute.dims1d (n / 256))
      ()
  in
  (* Warm-up: compile and settle allocations; not timed, but it is part of
     the result every mode must agree on. *)
  for i = 1 to 4 do
    rewrite i ;
    launch ()
  done ;
  Transfer.flush dev ;
  let t0 = Unix.gettimeofday () in
  let rw = ref 0.0 in
  for i = 5 to iters + 4 do
    let t = Unix.gettimeofday () in
    rewrite i ;
    rw := !rw +. (Unix.gettimeofday () -. t) ;
    host_work () ;
    for _ = 1 to launches do
      launch ()
    done
  done ;
  Transfer.flush dev ;
  let dt = Unix.gettimeofday () -. t0 in
  let result = Array.init n (fun i -> Vector.get out i) in
  let per x = x *. 1e6 /. float_of_int iters in
  (per dt, per !rw, result)

let modes =
  match Sys.getenv_opt "SPOC_BENCH_MODES" with
  | Some s -> String.split_on_char ',' s
  | None -> ["none"; "vector_set"; "staged"]

let () =
  let devices = Device.init () in
  match
    Array.to_list devices |> List.find_opt (fun d -> d.Device.framework = "HIP")
  with
  | None -> print_endline "bench_hip_small_writes: [SKIP] no HIP device present"
  | Some dev ->
      Printf.printf
        "%s: %d iterations of (rewrite, %d us host work, %d launches), kernel \
         spin %d, %d threads\n\
         %!"
        dev.Device.name
        iters
        host_us
        launches
        spin
        n ;
      let reference = ref None in
      let mismatch = ref false in
      List.iter
        (fun mode ->
          let us, rewrite_us, result = run dev mode in
          Printf.printf
            "  %-11s %9.1f us/iteration (rewrite %7.1f us)\n%!"
            mode
            us
            rewrite_us ;
          if mode <> "none" then
            match !reference with
            | None -> reference := Some (mode, result)
            | Some (ref_mode, r) ->
                if r <> result then begin
                  Printf.printf
                    "  RESULT MISMATCH: %s differs from %s\n%!"
                    mode
                    ref_mode ;
                  mismatch := true
                end)
        modes ;
      if !mismatch then exit 1
