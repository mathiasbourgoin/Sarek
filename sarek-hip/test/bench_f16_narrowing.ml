(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Cost of the f32 -> f16 narrowing barrier in a dequantise-to-LDS kernel
   (specs/f16-narrowing-barrier.md), on the RX 7900 XTX.

   Each work-group of 256 threads loops over [steps] steps; per step every
   thread dequantises one unit of 32 values into a shared f16 array, the group
   synchronises, and every thread reads one value written by a neighbour (so the
   stores stay live). Three dequantisation kernels:

     q4f  Q4_K as a downstream fused GEMM writes it: f32 scales read as bits,
          d * q - dmin * m (inexact: the barrier stays without an opt-in)
     q8f  Q8_0 with an f32 scale read as bits, d * int8 (not provably exact)
     q8h  Q8_0 with a binary16 scale, d * int8 (exact in f32)

   Two columns, both launched through [Execute.run_source] so that the host
   cost per launch is the same: [generated] is the source the code generator
   emits by default; [no-asm] is that source with the barrier's asm statement
   deleted, i.e. the code the compiler produces when nothing stops the fusion
   (what an elided or opted-out barrier gives); [nonvolatile] keeps the
   barrier but drops [volatile] from its asm; [opt-in] is the source generated
   under Sarek_f16_policy.accept_single_rounding.

   Method as bench_matrix_rewrite: GPU events around batches of [batch]
   back-to-back launches, [samples] batches per round, [rounds] rounds
   interleaving all kernels and columns; median and min-max per launch. The
   GPU is shared: the 1-minute load average is printed before and after.

   Usage: bench_f16_narrowing.exe [rounds] [samples] [batch]
   (BENCH_F16_DUMP=<dir> also writes each launched source there). *)

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

let groups = 1024

let steps = 32

(* Q4_K record: d, dmin (f32 bits), 3 words of 6-bit scales, 32 words of
   4-bit quants; one record is 8 units of 32 values. *)
let rec_words = 37

(* Units of input data; work-groups wrap around them (the kernels mask the
   unit index with [units - 1]), so the timing is that of the arithmetic. *)
let units = 1 lsl 20

let blocks = units / 8

let k_q4f =
  [%kernel
    fun (er : int32 vector) (sink : float32 vector) (steps : int32) ->
      let%shared (lds : float16) = 8192l in
      let open Sarek_stdlib.Std in
      let t = thread_idx_x in
      let acc = mut 0.0 in
      for s = 0 to steps - 1l do
        let u = ((((block_idx_x * steps) + s) * 256l) + t) land 1048575l in
        let blk = u / 8l in
        let j = u - (blk * 8l) in
        let base = blk * 37l in
        let ja = j + 4l in
        let jb = if j < 4l then j else j - 4l in
        let w_j = er.(base + 2l + (j / 4l)) in
        let w_ja = er.(base + 2l + (ja / 4l)) in
        let w_jb = er.(base + 2l + (jb / 4l)) in
        let b_j = (w_j asr ((j - (j / 4l * 4l)) * 8l)) land 255l in
        let b_ja = (w_ja asr ((ja - (ja / 4l * 4l)) * 8l)) land 255l in
        let b_jb = (w_jb asr ((jb - (jb / 4l * 4l)) * 8l)) land 255l in
        let sc = mut 0l in
        let m = mut 0l in
        if j < 4l then begin
          sc := b_j land 63l ;
          m := b_ja land 63l
        end
        else begin
          sc := b_ja land 15l lor ((b_jb asr 6l) lsl 4l) ;
          m := (b_ja asr 4l) lor ((b_j asr 6l) lsl 4l)
        end ;
        let ds = Int32.float_of_bits er.(base) *. float sc in
        let dm = Int32.float_of_bits er.(base + 1l) *. float m in
        let p = j / 2l in
        let nsh = (j - (p * 2l)) * 4l in
        let qb = base + 5l + (p * 8l) in
        for wi = 0 to 7l do
          let qw = (er.(qb + wi) asr nsh) land 0x0F0F0F0Fl in
          let o = (t * 32l) + (wi * 4l) in
          lds.(o) <- float16_of_float32 ((ds *. float (qw land 255l)) -. dm) ;
          lds.(o + 1l) <-
            float16_of_float32 ((ds *. float ((qw asr 8l) land 255l)) -. dm) ;
          lds.(o + 2l) <-
            float16_of_float32 ((ds *. float ((qw asr 16l) land 255l)) -. dm) ;
          lds.(o + 3l) <-
            float16_of_float32 ((ds *. float ((qw asr 24l) land 255l)) -. dm)
        done ;
        block_barrier () ;
        acc := acc +. float32_of_float16 lds.((t lxor 1l * 32l) + (s land 31l)) ;
        block_barrier ()
      done ;
      sink.((block_idx_x * 256l) + t) <- acc]

(* Q8_0: one unit is 8 words of int8 quants; [er] holds the f32 scale bits
   after the quants (word [units * 8 + unit]). *)
let k_q8f =
  [%kernel
    fun (er : int32 vector) (sink : float32 vector) (steps : int32) ->
      let%shared (lds : float16) = 8192l in
      let open Sarek_stdlib.Std in
      let t = thread_idx_x in
      let acc = mut 0.0 in
      for s = 0 to steps - 1l do
        let u = ((((block_idx_x * steps) + s) * 256l) + t) land 1048575l in
        let d = Int32.float_of_bits er.(8388608l + u) in
        for wi = 0 to 7l do
          let word = er.((u * 8l) + wi) in
          let o = (t * 32l) + (wi * 4l) in
          lds.(o) <- float16_of_float32 (d *. float ((word lsl 24l) asr 24l)) ;
          lds.(o + 1l) <-
            float16_of_float32 (d *. float ((word lsl 16l) asr 24l)) ;
          lds.(o + 2l) <- float16_of_float32 (d *. float ((word lsl 8l) asr 24l)) ;
          lds.(o + 3l) <- float16_of_float32 (d *. float (word asr 24l))
        done ;
        block_barrier () ;
        acc := acc +. float32_of_float16 lds.((t lxor 1l * 32l) + (s land 31l)) ;
        block_barrier ()
      done ;
      sink.((block_idx_x * 256l) + t) <- acc]

(* Q8_0 with the scale stored as binary16, as GGUF stores it. *)
let k_q8h =
  [%kernel
    fun (er : int32 vector)
        (sink : float32 vector)
        (steps : int32)
        (scales : float16 vector) ->
      let%shared (lds : float16) = 8192l in
      let open Sarek_stdlib.Std in
      let t = thread_idx_x in
      let acc = mut 0.0 in
      for s = 0 to steps - 1l do
        let u = ((((block_idx_x * steps) + s) * 256l) + t) land 1048575l in
        let d = float32_of_float16 scales.(u) in
        for wi = 0 to 7l do
          let word = er.((u * 8l) + wi) in
          let o = (t * 32l) + (wi * 4l) in
          lds.(o) <- float16_of_float32 (d *. float ((word lsl 24l) asr 24l)) ;
          lds.(o + 1l) <-
            float16_of_float32 (d *. float ((word lsl 16l) asr 24l)) ;
          lds.(o + 2l) <- float16_of_float32 (d *. float ((word lsl 8l) asr 24l)) ;
          lds.(o + 3l) <- float16_of_float32 (d *. float (word asr 24l))
        done ;
        block_barrier () ;
        acc := acc +. float32_of_float16 lds.((t lxor 1l * 32l) + (s land 31l)) ;
        block_barrier ()
      done ;
      sink.((block_idx_x * 256l) + t) <- acc]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* ---- inputs ---- *)

let rng = Random.State.make [|3|]

let scale_bits () =
  Int32.bits_of_float (ldexp (1.0 +. Random.State.float rng 1.0) (-6))

let er_q4 () =
  let v = Vector.create Vector.int32 (blocks * rec_words) in
  for b = 0 to blocks - 1 do
    for w = 0 to rec_words - 1 do
      let x = if w < 2 then scale_bits () else Random.State.bits32 rng in
      Vector.set v ((b * rec_words) + w) x
    done
  done ;
  v

let er_q8 () =
  let v = Vector.create Vector.int32 ((units * 8) + units) in
  for i = 0 to (units * 8) - 1 do
    Vector.set v i (Random.State.bits32 rng)
  done ;
  for u = 0 to units - 1 do
    Vector.set v ((units * 8) + u) (scale_bits ())
  done ;
  v

let scales_f16 () =
  let v = Vector.create Vector.float16 units in
  for u = 0 to units - 1 do
    Vector.set v u (ldexp (1.0 +. Random.State.float rng 1.0) (-6))
  done ;
  v

(* ---- launching ---- *)

let block = Execute.dims1d 256

let grid = Execute.dims1d groups

type variant = {
  name : string;
  ir : Sarek_ir_types.kernel;
  args : Execute.vector_arg list;
}

let barrier_stmt = {|asm volatile("" : "+v"(x));|}

let without_asm src =
  let nl = String.length barrier_stmt and hl = String.length src in
  let rec find i =
    if i + nl > hl then None
    else if String.sub src i nl = barrier_stmt then Some i
    else find (i + 1)
  in
  match find 0 with
  | None -> src
  | Some i -> String.sub src 0 i ^ String.sub src (i + nl) (hl - i - nl)

(* The barrier as a non-volatile asm: still opaque to value analysis, but no
   longer a side effect the scheduler must keep in place. *)
let non_volatile src =
  let needle = "asm volatile(" in
  let nl = String.length needle in
  let buf = Buffer.create (String.length src) in
  let rec go i =
    if i >= String.length src then ()
    else if i + nl <= String.length src && String.sub src i nl = needle then (
      Buffer.add_string buf "asm(" ;
      go (i + nl))
    else (
      Buffer.add_char buf src.[i] ;
      go (i + 1))
  in
  go 0 ;
  Buffer.contents buf

let source_args args =
  List.map
    (function Execute.Int n -> Execute.Int32 (Int32.of_int n) | a -> a)
    args

(* With BENCH_F16_DUMP=<dir>, each launched source is also written to
   <dir>/<label>.hip, for disassembly. *)
let dump label source =
  match Sys.getenv_opt "BENCH_F16_DUMP" with
  | None -> ()
  | Some dir ->
      let oc = open_out (Filename.concat dir (label ^ ".hip")) in
      output_string oc source ;
      close_out oc

(* The variant under its own kernel name, covered by
   Sarek_f16_policy.accept_single_rounding: generated without barriers. *)
let single_rounding v =
  let kern_name = "bench_f16_single_" ^ v.name in
  (match
     Sarek_f16_policy.accept_single_rounding
       (Kernel kern_name)
       ~reason:"benchmark"
   with
  | Ok () -> ()
  | Error r -> failwith (Sarek_f16_policy.refusal_message r)) ;
  {v with ir = {v.ir with Sarek_ir_types.kern_name}}

let from_source dev v ~label ~edit =
  let source =
    edit
      (Sarek_codegen.Sarek_ir_cuda.generate_with_types
         ~types:v.ir.Sarek_ir_types.kern_types
         v.ir)
  in
  dump (v.name ^ "_" ^ label) source ;
  let args = source_args v.args in
  fun () ->
    Execute.run_source
      ~device:dev
      ~source
      ~lang:Execute.CUDA_Source
      ~kernel_name:v.ir.Sarek_ir_types.kern_name
      ~block
      ~grid
      args

(* ---- measuring ---- *)

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a ;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let one_round dev timer launch =
  List.init samples (fun _ ->
      Profiling.timer_start timer ;
      for _ = 1 to batch do
        launch ()
      done ;
      Profiling.timer_stop timer ;
      Profiling.timer_elapsed_ms timer *. 1000.0 /. float_of_int batch)
  |> fun xs ->
  Transfer.flush dev ;
  xs

let loadavg () =
  try
    let ic = open_in "/proc/loadavg" in
    let l = input_line ic in
    close_in ic ;
    List.hd (String.split_on_char ' ' l)
  with _ -> "?"

let report label all =
  Printf.printf
    "%-16s median %8.1f us  min %8.1f  max %8.1f  (n=%d)\n%!"
    label
    (median all)
    (List.fold_left min infinity all)
    (List.fold_left max 0.0 all)
    (List.length all)

let () =
  let dev =
    match
      Device.init () |> Array.to_list
      |> List.filter (fun d -> d.Device.framework = "HIP")
    with
    | d :: _ -> d
    | [] ->
        print_endline "[SKIP] no HIP device" ;
        exit 0
  in
  Printf.printf
    "bench_f16_narrowing on %s: %d groups x %d steps, rounds=%d samples=%d \
     batch=%d, load(1min) %s\n\
     %!"
    dev.Device.name
    groups
    steps
    rounds
    samples
    batch
    (loadavg ()) ;
  let sink () = Vector.create Vector.float32 (groups * 256) in
  let er4 = er_q4 () and er8 = er_q8 () and sc = scales_f16 () in
  let st = Execute.Int steps in
  let variants =
    [
      {
        name = "q4f";
        ir = ir_of k_q4f;
        args = Execute.[Vec er4; Vec (sink ()); st];
      };
      {
        name = "q8f";
        ir = ir_of k_q8f;
        args = Execute.[Vec er8; Vec (sink ()); st];
      };
      {
        name = "q8h";
        ir = ir_of k_q8h;
        args = Execute.[Vec er8; Vec (sink ()); st; Vec sc];
      };
    ]
  in
  let columns =
    List.concat_map
      (fun v ->
        [
          ( v.name ^ " generated",
            from_source dev v ~label:"generated" ~edit:Fun.id );
          ( v.name ^ " no-asm",
            from_source dev v ~label:"no-asm" ~edit:without_asm );
          ( v.name ^ " nonvolatile",
            from_source dev v ~label:"nonvolatile" ~edit:non_volatile );
          ( v.name ^ " opt-in",
            from_source dev (single_rounding v) ~label:"opt-in" ~edit:Fun.id );
        ])
      variants
  in
  List.iter
    (fun (_, launch) ->
      launch () ;
      Transfer.flush dev)
    columns ;
  let timer = Profiling.create_timer dev in
  let results = Hashtbl.create 8 in
  for _ = 1 to rounds do
    List.iter
      (fun (label, launch) ->
        let prev = Option.value ~default:[] (Hashtbl.find_opt results label) in
        Hashtbl.replace results label (prev @ one_round dev timer launch))
      columns
  done ;
  Profiling.timer_destroy timer ;
  List.iter
    (fun (label, _) -> report label (Hashtbl.find results label))
    columns ;
  Printf.printf "load(1min) after: %s\n" (loadavg ())
