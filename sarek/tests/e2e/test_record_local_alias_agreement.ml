(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A record bound to a LOCAL is a COPY, on every backend.
 *
 *   let e = v.(tid) in e.p <- 42.0
 *
 * The store lands in the local — it is readable back through [e] — and vector
 * storage is NOT touched. That is what all five device backends do, because
 * that is what the construct MEANS on each of them: the C-family emits a
 * struct-copy local (`l1 e = v[tid]; e.p = 42.0f;`) and Native marshals a fresh
 * OCaml record out of storage through [Vector.get]. Writing back is
 * `v.(tid) <- e`, and nothing here does that.
 *
 * This file exists because the backlog-172 field-store fix broke exactly this
 * on the INTERPRETER and no test noticed. Measured on the pre-fix tip
 * (1b18e090), with 4 elements and `let e = v.(tid) in e.p <- 42.0`:
 *
 *   Native 0 1 2 3 | OpenCL x2 0 1 2 3 | Vulkan x2 0 1 2 3 |
 *   CUDA-PTX x2 0 1 2 3 | Interpreter 42 42 42 42
 *
 * The CUDA-PTX pair is the one this file's gate could not reproduce on its own:
 * the dune rule now puts ZLUDA on LD_LIBRARY_PATH so the two CUDA/PTX devices
 * are enumerated, and the run prints a named NOT-MEASURED-HERE line for that
 * framework when they are not — rather than exiting 0 having exercised 7 of 9
 * devices with nothing to say about it. Every CUDA/PTX row here is
 * ZLUDA-on-AMD; there is no NVIDIA hardware on this machine.
 *
 * The interpreter wrote THROUGH the local into vector storage: [read_lvalue]'s
 * [LVar] arm hands back the [VRecord]'s [value array] itself, and for the
 * `v.(i).f <- e` target that sharing is exactly what makes the store land — but
 * a local binding must not inherit it. Before the fix landed, the interpreter
 * REFUSED this shape outright (`Unsupported operation 'record field
 * assignment'`), so the regression traded a loud refusal for a silent write
 * into memory five devices leave alone.
 *
 * The fix is in the binding, not in the read: [SLet]/[SLetMut] deep-copy a
 * [VRecord] as they bind it. Copying at the READ instead would make the store
 * vanish entirely, and `e.p` would read back the old value — which is NOT what
 * the other backends do either.
 *
 * Two depths, because a shallow copy passes the first and fails the second:
 *
 *   A. depth 1 — `let e = v.(tid) in e.p <- 42.0`. Runs on EVERY device.
 *   B. nested  — `let e = v.(tid) in e.sub.p <- 42.0`. Also runs on EVERY device,
 *      with nothing skipped and no source inspection deciding whether to try.
 *
 *      It did not always. [l2] names [l1] as a field type, record typedefs were
 *      emitted in [kern_types] order with no dependency sort (backlog-203), and
 *      for THIS kernel that order put l2 before l1 — so OpenCL and Vulkan could
 *      not build the nested kernel and this file skipped them, deciding per
 *      backend from the generated source. That ordering bug is fixed on `main`
 *      ([Sarek_ir_codegen.sort_record_types_by_dependency], #394), so the skip
 *      predicate, the two-dialect source inspector behind it and its self-check
 *      are deleted rather than left to be satisfied by nothing. A skip that
 *      outlives its reason is how a test stops testing without ever going red.
 *
 *      What is still pinned is that the nested case actually RAN on both CPU
 *      frameworks — with no skip left, that can now only fail if a plugin did not
 *      enumerate, but a nested case that quietly exercised nothing would still
 *      read the same as one that agreed.
 *
 * Every disagreement makes the process exit non-zero.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer
module Execute = Sarek.Execute

(* Explicit registration: linking a plugin does not enumerate its devices. *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

(* [p] is mutable because Native lowers a store into a LOCAL record to an OCaml
   setfield, which an immutable field rejects at compile time ("The record field
   p is not mutable"). The mutability is a Native codegen requirement, not part
   of what is being asserted. *)
type l1 = {mutable p : float32; q : float32} [@@sarek.type]

(* Nested: the local copy must be deep, or a store through [e.sub] reaches the
   inner record still shared with vector storage. *)
type l2 = {tag : float32; sub : l1} [@@sarek.type]

let n = 64

(* A: depth-1 local store. [out] carries the local's own field back so the store
   is checked to have LANDED, not merely to have missed storage. *)
let k_local_depth1 =
  snd
    [%kernel
      fun (v : l1 vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then begin
          let e = v.(tid) in
          e.p <- 42.0 ;
          out.(tid) <- e.p
        end]

(* B: nested local store, one level down. *)
let k_local_nested =
  snd
    [%kernel
      fun (v : l2 vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then begin
          let e = v.(tid) in
          e.sub.p <- 42.0 ;
          out.(tid) <- e.sub.p
        end]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let launch dev ir args =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d (min 64 n))
    ~grid:(Execute.dims1d ((n + 63) / 64))
    ()

(* Reports at most three mismatches per case, then keeps counting. *)
let make_reporter () =
  let ok = ref true in
  let reported = ref 0 in
  let bad i name got want =
    if Float.abs (got -. want) > 1e-4 then begin
      ok := false ;
      if !reported < 3 then begin
        incr reported ;
        Printf.printf "\n  @%d %s: got %g want %g" i name got want
      end
    end
  in
  (ok, bad)

let case_depth1 (dev : Device.t) : bool =
  Printf.printf
    "local-alias depth1 [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let v = Vector.create_custom l1_custom n in
  for i = 0 to n - 1 do
    Vector.set v i {p = float_of_int i; q = float_of_int (i + 100)}
  done ;
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set out i 0.0
  done ;
  match
    launch
      dev
      (ir_of k_local_depth1)
      [Execute.Vec v; Execute.Vec out; Execute.Int n]
  with
  | exception e ->
      (* A refusal is a FAILURE: every backend in this list runs ordinary
         custom-vector kernels, and this shape is plain DSL. *)
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok, bad = make_reporter () in
      for i = 0 to n - 1 do
        let e = Vector.get v i in
        (* The store landed IN THE LOCAL. *)
        bad i "out (local's own field after the store)" (Vector.get out i) 42.0 ;
        (* And vector storage is untouched — the regression this file exists
           for wrote 42 here. *)
        bad i "v.(i).p (storage, must be untouched)" e.p (float_of_int i) ;
        bad
          i
          "v.(i).q (storage, must be untouched)"
          e.q
          (float_of_int (i + 100))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let case_nested (dev : Device.t) : bool =
  Printf.printf
    "local-alias nested [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let v = Vector.create_custom l2_custom n in
  for i = 0 to n - 1 do
    Vector.set
      v
      i
      {
        tag = float_of_int (1000 + i);
        sub = {p = float_of_int i; q = float_of_int (i + 100)};
      }
  done ;
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set out i 0.0
  done ;
  match
    launch
      dev
      (ir_of k_local_nested)
      [Execute.Vec v; Execute.Vec out; Execute.Int n]
  with
  | exception e ->
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok, bad = make_reporter () in
      for i = 0 to n - 1 do
        let e = Vector.get v i in
        bad i "out (local's own nested field)" (Vector.get out i) 42.0 ;
        bad
          i
          "v.(i).sub.p (storage, must be untouched)"
          e.sub.p
          (float_of_int i) ;
        bad
          i
          "v.(i).sub.q (storage, must be untouched)"
          e.sub.q
          (float_of_int (i + 100)) ;
        bad
          i
          "v.(i).tag (storage, must be untouched)"
          e.tag
          (float_of_int (1000 + i))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let nested_must_run_on = ["Interpreter"; "Native"]

let () =
  let devs =
    Device.init
      ~frameworks:
        ["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"; "HIP"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found — nothing asserted, and that is a gap" ;
    exit 1
  end ;
  let any_failure = ref false in
  let nested_ran = ref [] in
  Array.iter
    (fun (dev : Device.t) ->
      if not (case_depth1 dev) then any_failure := true ;
      (* The nested case is launched on EVERY device, with no source inspection
         deciding whether to try.

         It used to be gated by a [nested_can_compile] predicate: [l2] names [l1]
         as a field type, record typedefs were emitted in [kern_types] order with
         no dependency sort (backlog-203), and for this kernel that order put [l2]
         first — so OpenCL and Vulkan could not compile it and the case was skipped
         there with a printed reason. That ordering bug is fixed on `main`
         ([Sarek_ir_codegen.sort_record_types_by_dependency], #394), so the gate,
         the two-dialect source inspector behind it and its self-check are all
         deleted. A skip predicate outliving the reason for the skip is how a test
         stops testing without going red: this one would have declined to launch on
         both shader backends for as long as it lived, whatever the codegen did. *)
      nested_ran := dev.Device.framework :: !nested_ran ;
      if not (case_nested dev) then any_failure := true)
    devs ;
  (* Kept as a non-vacuity floor even though nothing is skipped any more: it now
     fails only if a CPU plugin did not enumerate at all, which is the remaining
     way this half could assert nothing while exiting 0. *)
  List.iter
    (fun fw ->
      if not (List.mem fw !nested_ran) then begin
        Printf.printf
          "nested local-alias case never ran on %s — the deep-copy half of \
           this file asserted nothing\n\
           %!"
          fw ;
        any_failure := true
      end)
    nested_must_run_on ;
  (* Same rule as test_record_field_store.ml: a device class this file's header
     makes a claim ABOUT must not be able to go missing quietly.

     The header's measurement line names CUDA-PTX x2 among the backends that
     read back [0 1 2 3], and the nested case's scope is stated as "the two CPU
     backends plus CUDA/PTX". Without ZLUDA on the loader path this file
     enumerated 7 devices, printed no CUDA/PTX row at all, and exited 0. The
     dune rule now sets LD_LIBRARY_PATH; this line reports the absence where
     there is genuinely no such device.

     A loud named skip rather than a failure, for the reason given in the
     sibling file: a host with neither ZLUDA nor a CUDA driver has no such
     device, and the honest report is "not reproduced here", not a hard stop. *)
  let frameworks_seen =
    Array.to_list devs |> List.map (fun (d : Device.t) -> d.Device.framework)
  in
  List.iter
    (fun fw ->
      if not (List.mem fw frameworks_seen) then
        Printf.printf
          "NOT MEASURED HERE: no %s device was enumerated, so the header's %s \
           rows (storage untouched at both depths, nested case included) are \
           NOT reproduced by this run. Put ZLUDA on LD_LIBRARY_PATH, or run on \
           a CUDA host, to exercise it.\n\
           %!"
          fw
          fw)
    ["CUDA/PTX"] ;
  Printf.printf
    "%d device(s) exercised (frameworks: %s)\n%!"
    (Array.length devs)
    (String.concat ", " (List.sort_uniq String.compare frameworks_seen)) ;
  if !any_failure then exit 1
