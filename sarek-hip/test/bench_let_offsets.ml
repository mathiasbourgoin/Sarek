(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Offsets bound by [let] inside a loop, the shape the index-let rule of
   Sarek_ir_c_arith is about: [let off = ... in a.(off) +. a.(off + 1) ...].
   With C's signed operators the compiler folds the [+ 1 .. + 3] into
   immediate load offsets (or one wide load); with wrapping arithmetic it
   recomputes each address.

   - rows: each thread sums its own row of [k] floats, four per iteration
     ([let off = (row * k) + (i * 4)]);
   - cols: the same sum, coalesced ([let off = ((i * n) + row) * 4]);
   - stencil: a 3x3 box sum with [let up = ((y - 1) * w) + x] and two more.

   GPU events around batches of [batch] launches, [samples] batches per round,
   [rounds] rounds; median and min-max per launch. The GPU is shared.

   Usage: bench_let_offsets.exe [rounds] [samples] [batch] *)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Profiling = Spoc_core.Profiling

let () = Sarek_hip.Hip_plugin.init ()

let arg i default =
  if Array.length Sys.argv > i then int_of_string Sys.argv.(i) else default

let rounds = arg 1 3

let samples = arg 2 10

let batch = arg 3 10

let rows_kernel =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) (n : int32) (k : int32) ->
      let row = global_idx_x in
      if row < n then begin
        let sum = mut 0.0 in
        for i = 0 to (k / 4l) - 1l do
          let off = (row * k) + (i * 4l) in
          sum := sum +. a.(off) +. a.(off + 1l) +. a.(off + 2l) +. a.(off + 3l)
        done ;
        out.(row) <- sum
      end]

let cols_kernel =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) (n : int32) (k : int32) ->
      let row = global_idx_x in
      if row < n then begin
        let sum = mut 0.0 in
        for i = 0 to (k / 4l) - 1l do
          let off = ((i * n) + row) * 4l in
          sum := sum +. a.(off) +. a.(off + 1l) +. a.(off + 2l) +. a.(off + 3l)
        done ;
        out.(row) <- sum
      end]

let stencil_kernel =
  [%kernel
    fun (a : float32 vector) (out : float32 vector) (w : int32) (h : int32) ->
      let x = global_idx_x in
      let y = global_idx_y in
      if x > 0l && x < w - 1l && y > 0l && y < h - 1l then begin
        let up = ((y - 1l) * w) + x in
        let mid = (y * w) + x in
        let down = ((y + 1l) * w) + x in
        out.(mid) <-
          a.(up - 1l)
          +. a.(up)
          +. a.(up + 1l)
          +. a.(mid - 1l)
          +. a.(mid)
          +. a.(mid + 1l)
          +. a.(down - 1l)
          +. a.(down)
          +. a.(down + 1l)
      end]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let measure dev launch =
  launch () ;
  Transfer.flush dev ;
  let timer = Profiling.create_timer dev in
  let all =
    List.concat
      (List.init rounds (fun _ ->
           List.init samples (fun _ ->
               Profiling.timer_start timer ;
               for _ = 1 to batch do
                 launch ()
               done ;
               Profiling.timer_stop timer ;
               Profiling.timer_elapsed_ms timer *. 1000.0 /. float_of_int batch)))
  in
  Profiling.timer_destroy timer ;
  all

let report name all =
  Printf.printf
    "%-10s median %9.1f us  min %9.1f max %9.1f us\n%!"
    name
    (median all)
    (List.fold_left min infinity all)
    (List.fold_left max 0.0 all)

let floats len =
  let v = Vector.create Vector.float32 len in
  for i = 0 to len - 1 do
    Vector.set v i (float_of_int (i mod 7))
  done ;
  v

let i n = Execute.Int n

let bench dev =
  let n = 65536 and k = 1024 in
  let a = floats (n * k) and out = floats n in
  let launch ir () =
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Execute.Vec a; Execute.Vec out; i n; i k]
      ~block:(Execute.dims1d 256)
      ~grid:(Execute.dims1d (n / 256))
      ()
  in
  report "rows" (measure dev (launch (ir_of rows_kernel))) ;
  report "cols" (measure dev (launch (ir_of cols_kernel))) ;
  let w = 4096 and h = 4096 in
  let a = floats (w * h) and out = floats (w * h) in
  report
    "stencil"
    (measure dev (fun () ->
         Execute.run_vectors
           ~device:dev
           ~ir:(ir_of stencil_kernel)
           ~args:[Execute.Vec a; Execute.Vec out; i w; i h]
           ~block:(Execute.dims2d 32 8)
           ~grid:(Execute.dims2d (w / 32) (h / 8))
           ()))

let () =
  match
    List.filter
      (fun d -> d.Device.framework = "HIP")
      (Array.to_list (Device.init ()))
  with
  | [] -> print_endline "[SKIP] no HIP device"
  | dev :: _ ->
      Printf.printf
        "device: %s; rounds=%d samples=%d batch=%d\n%!"
        dev.Device.name
        rounds
        samples
        batch ;
      bench dev
