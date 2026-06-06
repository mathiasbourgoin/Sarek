(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** SPIKE (throwaway) — Milestone B worker-side jsoo program.

    Runs INSIDE a Web Worker (loaded via importScripts). Exposes a genuinely
    SYNCHRONOUS OCaml function
    [vector_add_sync : float array -> float array -> float array] that obtains a
    GPU-computed result from the main thread via SharedArrayBuffer +
    Atomics.wait — no Lwt, no callback. The worker thread blocks; the
    non-blocked main thread runs WebGPU and Atomics.notify's.

    SAB layout (matches the Milestone-A byte map):
    {v
      Int32 control @ byte 0 : slot0=state (IDLE=0,REQUEST=1,READY=2,ERROR=3), slot1=errcode
      Float32 data  @ byte 8 : a (0..N) then b (N..2N) then c (2N..3N)
    v} *)

open Js_of_ocaml

let n = 256

let state_idx = 0

let request = 1

let ready = 2

let error = 3

let timeout_ms = 10000

let g = Js.Unsafe.global

let jget name = Js.Unsafe.get g (Js.string name)

let jint (i : int) = Js.Unsafe.inject (Js.number_of_float (float_of_int i))

let jflo (f : float) = Js.Unsafe.inject (Js.number_of_float f)

(* SAB typed-array views, set on 'init'. *)
let ctrl = ref (Js.Unsafe.inject Js.null)

let data = ref (Js.Unsafe.inject Js.null)

let atomics_store idx v =
  ignore
    (Js.Unsafe.meth_call (jget "Atomics") "store" [|!ctrl; jint idx; jint v|])

let atomics_load idx =
  int_of_float
    (Js.float_of_number
       (Js.Unsafe.meth_call (jget "Atomics") "load" [|!ctrl; jint idx|]))

(* Atomics.wait returns "ok" | "not-equal" | "timed-out". *)
let atomics_wait idx expected =
  Js.to_string
    (Js.Unsafe.meth_call
       (jget "Atomics")
       "wait"
       [|!ctrl; jint idx; jint expected; jint timeout_ms|])

let post (o : Js.Unsafe.any) =
  ignore (Js.Unsafe.fun_call (jget "postMessage") [|Js.Unsafe.inject o|])

let data_set i v =
  Js.Unsafe.set !data (Js.number_of_float (float_of_int i)) (jflo v)

let data_get i =
  Js.float_of_number (Js.Unsafe.get !data (Js.number_of_float (float_of_int i)))

(* The synchronous bridge: write inputs, request, block on Atomics.wait, read output. *)
let vector_add_sync (a : float array) (b : float array) : float array =
  Array.iteri (fun i v -> data_set i v) a ;
  Array.iteri (fun i v -> data_set (n + i) v) b ;
  (* Store the sentinel BEFORE postMessage and wait on that same value to avoid
     a lost wakeup (if main notifies first, state <> REQUEST -> immediate return). *)
  atomics_store state_idx request ;
  post (Js.Unsafe.obj [|("t", Js.Unsafe.inject (Js.string "go"))|]) ;
  let w = atomics_wait state_idx request in
  if w = "timed-out" then
    failwith "vector_add_sync: Atomics.wait timed out (deadlock)" ;
  let st = atomics_load state_idx in
  if st = error then
    failwith
      (Printf.sprintf
         "vector_add_sync: main reported ERROR %d"
         (atomics_load 1))
  else if st <> ready then
    failwith (Printf.sprintf "vector_add_sync: unexpected state %d" st) ;
  Array.init n (fun i -> data_get ((2 * n) + i))

let on_run () =
  let a = Array.init n (fun i -> float_of_int i *. 0.5) in
  let b = Array.init n (fun i -> float_of_int i *. 2.0) in
  let result =
    try
      let c = vector_add_sync a b in
      let bad = ref 0 in
      Array.iteri
        (fun i ci -> if abs_float (ci -. (a.(i) +. b.(i))) > 1e-4 then incr bad)
        c ;
      Js.Unsafe.obj
        [|
          ("t", Js.Unsafe.inject (Js.string "result"));
          ("pass", Js.Unsafe.inject (Js.bool (!bad = 0)));
          ("bad", jint !bad);
        |]
    with Failure msg ->
      Js.Unsafe.obj
        [|
          ("t", Js.Unsafe.inject (Js.string "result"));
          ("pass", Js.Unsafe.inject Js._false);
          ("error", Js.Unsafe.inject (Js.string msg));
        |]
  in
  post result

let on_message e =
  let m = Js.Unsafe.get e "data" in
  let t = Js.to_string (Js.Unsafe.get m "t") in
  if t = "init" then begin
    let sab = Js.Unsafe.get m "sab" in
    ctrl :=
      Js.Unsafe.new_obj
        (jget "Int32Array")
        [|Js.Unsafe.inject sab; jint 0; jint 2|] ;
    data :=
      Js.Unsafe.new_obj
        (jget "Float32Array")
        [|Js.Unsafe.inject sab; jint 8; jint (3 * n)|] ;
    post (Js.Unsafe.obj [|("t", Js.Unsafe.inject (Js.string "ready"))|])
  end
  else if t = "run" then on_run ()

let () = Js.Unsafe.set g "onmessage" (Js.Unsafe.callback on_message)
