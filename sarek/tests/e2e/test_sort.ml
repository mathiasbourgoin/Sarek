(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Sorting algorithms
 *
 * Tests bitonic sort and odd-even merge sort - parallel sorting algorithms
 * that work well on GPUs due to their regular communication patterns.
 * GPU runtime only.
 ******************************************************************************)

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
module Benchmarks = Test_helpers.Benchmarks

(* Backends auto-register when linked; Benchmarks.init() ensures initialization *)

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_sort arr _n =
  let a = Array.copy arr in
  Array.sort Int32.compare a ;
  a

(* ========== Shared test data ========== *)

let input_bitonic_global = ref [||]

let sort_size_global = ref 0

let input_odd_even = ref [||]

let sort_size_odd_even = ref 0

let init_sort_data size =
  let log2n = int_of_float (log (float_of_int size) /. log 2.0) in
  let n = 1 lsl log2n in
  sort_size_global := n ;
  Random.init 42 ;
  input_bitonic_global :=
    Array.init n (fun _ -> Int32.of_int (Random.int 10000)) ;

  let n_oe = min 256 size in
  sort_size_odd_even := n_oe ;
  Random.init 42 ;
  input_odd_even := Array.init n_oe (fun _ -> Int32.of_int (Random.int 10000))

(* ========== Sarek kernels ========== *)

(** Bitonic sort step - one comparison/swap pass *)
let bitonic_sort_step_kernel =
  [%kernel
    fun (data : int32 vector) (j : int32) (k : int32) (n : int32) ->
      let open Std in
      let i = global_thread_id in
      if i < n then begin
        let ij = i lxor j in
        if ij > i then begin
          let di = data.(i) in
          let dij = data.(ij) in
          let ascending = i land k = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            data.(i) <- dij ;
            data.(ij) <- di
          end
        end
      end]

(** Odd-even transposition sort step *)
let odd_even_step_kernel =
  [%kernel
    fun (data : int32 vector) (phase : int32) (n : int32) ->
      let open Std in
      let i = global_thread_id in
      let idx = (2l * i) + phase in
      if idx + 1l < n then begin
        let a = data.(idx) in
        let b = data.(idx + 1l) in
        if a > b then begin
          data.(idx) <- b ;
          data.(idx + 1l) <- a
        end
      end]

(* ========== runtime test runners ========== *)

(** Run global bitonic sort on runtime *)
let run_bitonic_sort_global (dev : Device.t) _size _block_size =
  let n = !sort_size_global in
  let inp = !input_bitonic_global in
  let _, kirc = bitonic_sort_step_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let data = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set data i inp.(i)
  done ;

  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

  let t0 = Unix.gettimeofday () in
  let k = ref 2 in
  while !k <= n do
    let j = ref (!k / 2) in
    while !j > 0 do
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:
          [
            Sarek.Execute.Vec data;
            Sarek.Execute.Int32 (Int32.of_int !j);
            Sarek.Execute.Int32 (Int32.of_int !k);
            Sarek.Execute.Int32 (Int32.of_int n);
          ]
        ~block
        ~grid
        () ;
      Transfer.flush dev ;
      j := !j / 2
    done ;
    k := !k * 2
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array data)

(** Run odd-even transposition sort on runtime *)
let run_odd_even_sort (dev : Device.t) _size _block_size =
  let n = !sort_size_odd_even in
  let inp = !input_odd_even in
  let _, kirc = odd_even_step_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  let data = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set data i inp.(i)
  done ;

  let block_size = 256 in
  let grid_size = ((n / 2) + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d (max 1 grid_size) in

  let t0 = Unix.gettimeofday () in
  for phase = 0 to n - 1 do
    let phase_mod = Int32.of_int (phase mod 2) in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec data;
          Sarek.Execute.Int32 phase_mod;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;
    (* Explicitly synchronize every 16 iterations to prevent queue overflow on Vulkan *)
    if phase mod 16 = 0 then Device.synchronize dev
  done ;
  Device.synchronize dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  (time_ms, Vector.to_array data)

(* ========== Main ========== *)

let () =
  Benchmarks.init () ;
  let size = Benchmarks.config.size in
  init_sort_data size ;

  let verify result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    for i = 0 to n - 2 do
      if result.(i) > result.(i + 1) then incr errors
    done ;
    for i = 0 to min 10 (n - 1) do
      if result.(i) <> expected.(i) then incr errors
    done ;
    !errors = 0
  in

  (* Bitonic - too slow on interpreter *)
  let baseline_bitonic _size =
    let n = !sort_size_global in
    ocaml_sort !input_bitonic_global n
  in
  Benchmarks.run
    ~filter:Benchmarks.no_interpreter
    ~baseline:baseline_bitonic
    ~verify
    "Bitonic Sort (global)"
    run_bitonic_sort_global ;

  (* Odd-Even - uses shared memory, interpreter doesn't support it *)
  let baseline_odd_even _size =
    let n = !sort_size_odd_even in
    ocaml_sort !input_odd_even n
  in
  Benchmarks.run
    ~filter:Benchmarks.no_interpreter
    ~baseline:baseline_odd_even
    ~verify
    "Odd-Even Sort"
    run_odd_even_sort ;
  Benchmarks.exit ()
