(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1b device-side SoA emitter equivalence.
 *
 * Tier 1a proved the host SoA transpose + scalar transfer path. This test
 * proves the Tier 1b EMITTER: a single custom (record) vector kernel parameter
 * lowered as Structure-of-Arrays (Sarek_ir_ptx.generate ~soa_params) — N
 * per-leaf base pointers + coalesced per-leaf scalar loads — computes the same
 * result on CUDA/PTX as the default AoS lowering of the very same kernel IR,
 * and as a pure-OCaml reference.
 *
 * Mechanics: the same [%kernel] IR is compiled twice.
 *   - AoS: run via Execute.run_vectors with the single custom vector argument
 *     (backend generate_source, default packed layout).
 *   - SoA: Sarek_ir_ptx.generate ~soa_params:[<the custom vector param>] emits
 *     N pointer params + one length; the AoS host buffer is transposed into N
 *     contiguous leaf vectors (Spoc_core.Soa.scatter) and fed positionally via
 *     Execute.run_source ~inject_lengths:false — exactly the N-base-pointer ABI
 *     the emitter now produces. (The user-facing Vector.create ~layout:SoA +
 *     automatic launch expansion is Tier 1c; this drives the emitter directly.)
 *
 * SoA is PTX-only in this deliverable, so the SoA leg runs on CUDA/PTX devices
 * only; the AoS leg + reference run everywhere and are always checked. f32 and
 * f64 leaves are exercised end-to-end here; i32/i64 leaf codegen is covered at
 * the PTX-instruction + ptxas-assembly level in tests/unit/test_ptx_snapshot.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Soa = Spoc_core.Soa
module Soa_vector = Spoc_core.Soa_vector
module Soa_launch = Sarek.Soa_launch
module Benchmarks = Test_helpers.Benchmarks
open Sarek_codegen

type ('a, 'b) vector = ('a, 'b) Vector.t

type float32 = float

type float64 = float

type point3d = {x : float32; y : float32; z : float32} [@@sarek.type]

type dpair = {u : float64; v : float64} [@@sarek.type]

(* f32 headline case: reads three fields of a custom vector and sums them. *)
let p3_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = pts.(tid) in
          out.(tid) <- p.x +. p.y +. p.z]

(* f64 case: two 8-byte leaves. *)
let dpair_kernel =
  snd
    [%kernel
      fun (pv : dpair vector) (out : float64 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then out.(tid) <- pv.(tid).u +. pv.(tid).v]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* Name of the first (custom vector) kernel parameter — what we lower as SoA. *)
let first_param_name (ir : Sarek_ir_types.kernel) =
  match ir.Sarek_ir_types.kern_params with
  | Sarek_ir_types.DParam (v, _) :: _ -> v.Sarek_ir_types.var_name
  | _ -> failwith "kernel has no parameters"

let is_ptx (dev : Device.t) = dev.Device.framework = "CUDA/PTX"

let dims threads = Sarek.Execute.dims1d threads

(* Launch the SoA compilation of [ir] whose first param (a flat 2/3-field record
   vector) is lowered SoA. [leaves] are the per-leaf scalar vectors (declaration
   order); [out] the scalar output. Arg order mirrors the emitted param block:
   leaf pointers, the shared length, then (out ptr, out length), then n — all
   with inject_lengths:false so we control every slot. *)
let run_soa dev ir ~leaves ~out ~n ~block ~grid =
  let ptx = Sarek_ir_ptx.generate ~soa_params:[first_param_name ir] ir in
  let leaf_args = List.map (fun v -> Sarek.Execute.Vec v) leaves in
  let len = Sarek.Execute.Int32 (Int32.of_int n) in
  let args =
    leaf_args @ [len; Sarek.Execute.Vec out; len; Sarek.Execute.Int n]
  in
  Sarek.Execute.run_source
    ~device:dev
    ~source:ptx
    ~lang:Sarek.Execute.PTX
    ~kernel_name:ir.Sarek_ir_types.kern_name
    ~block
    ~grid
    ~inject_lengths:false
    args ;
  Transfer.flush dev

(* Tier 1c: the SAME kernel driven through the real user-facing API —
   Soa_vector storage + Soa_launch.run_soa. Unlike run_soa above (which pokes
   the emitter directly), this exercises the whole host path: SoA storage
   allocation, host AoS->leaf scatter, per-leaf H2D transfer, N-base-pointer
   launch expansion, and the CUDA/PTX gate. [sv] is the SoA input vector (kernel
   param 0), [out] the scalar output, [n] the length. *)
let run_soa_via_api dev ir ~sv ~out ~n ~block ~grid =
  Soa_launch.run_soa
    ~device:dev
    ~ir
    ~args:
      [
        Soa_launch.SA_Soa sv;
        Soa_launch.SA_Reg (Sarek.Execute.Vec out);
        Soa_launch.SA_Reg (Sarek.Execute.Int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev

(* ---- point3d (f32) ---- *)

let run_p3 dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let src = Vector.create_custom point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {
        x = float_of_int i;
        y = (float_of_int i *. 0.5) +. 1.0;
        z = float_of_int (n - i);
      }
  done ;
  let ir = ir_of p3_kernel in
  (* AoS *)
  let out_aos = Vector.create Vector.float32 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec src; Vec out_aos; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* SoA (PTX only) *)
  let out_soa =
    if not (is_ptx dev) then None
    else begin
      let plan =
        Soa.plan
          ~name:"point3d"
          Sarek_ir_types.[("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
      in
      let xs = Vector.create Vector.float32 n in
      let ys = Vector.create Vector.float32 n in
      let zs = Vector.create Vector.float32 n in
      Soa.scatter
        plan
        ~aos:(Vector.to_ctypes_ptr src)
        ~length:n
        ~leaves:
          [|
            Vector.to_ctypes_ptr xs;
            Vector.to_ctypes_ptr ys;
            Vector.to_ctypes_ptr zs;
          |] ;
      let out = Vector.create Vector.float32 n in
      run_soa dev ir ~leaves:[xs; ys; zs] ~out ~n ~block ~grid ;
      Some out
    end
  in
  (* SoA via the real user-facing API (Soa_vector + Soa_launch.run_soa). *)
  let out_api =
    if not (is_ptx dev) then None
    else begin
      let sv =
        Soa_vector.create
          point3d_custom
          ~fields:
            Sarek_ir_types.[("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
          n
      in
      for i = 0 to n - 1 do
        Soa_vector.set
          sv
          i
          {
            x = float_of_int i;
            y = (float_of_int i *. 0.5) +. 1.0;
            z = float_of_int (n - i);
          }
      done ;
      let out = Vector.create Vector.float32 n in
      run_soa_via_api dev ir ~sv ~out ~n ~block ~grid ;
      Some out
    end
  in
  let reference i =
    let p = Vector.get src i in
    p.x +. p.y +. p.z
  in
  (out_aos, out_soa, out_api, reference)

(* ---- dpair (f64) ---- *)

let run_dpair dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let src = Vector.create_custom dpair_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {u = float_of_int i *. 1.5; v = float_of_int (n - i) -. 0.25}
  done ;
  let ir = ir_of dpair_kernel in
  let out_aos = Vector.create Vector.float64 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec src; Vec out_aos; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let out_soa =
    if not (is_ptx dev) then None
    else begin
      let plan =
        Soa.plan ~name:"dpair" Sarek_ir_types.[("u", TFloat64); ("v", TFloat64)]
      in
      let us = Vector.create Vector.float64 n in
      let vs = Vector.create Vector.float64 n in
      Soa.scatter
        plan
        ~aos:(Vector.to_ctypes_ptr src)
        ~length:n
        ~leaves:[|Vector.to_ctypes_ptr us; Vector.to_ctypes_ptr vs|] ;
      let out = Vector.create Vector.float64 n in
      run_soa dev ir ~leaves:[us; vs] ~out ~n ~block ~grid ;
      Some out
    end
  in
  let out_api =
    if not (is_ptx dev) then None
    else begin
      let sv =
        Soa_vector.create
          dpair_custom
          ~fields:Sarek_ir_types.[("u", TFloat64); ("v", TFloat64)]
          n
      in
      for i = 0 to n - 1 do
        Soa_vector.set
          sv
          i
          {u = float_of_int i *. 1.5; v = float_of_int (n - i) -. 0.25}
      done ;
      let out = Vector.create Vector.float64 n in
      run_soa_via_api dev ir ~sv ~out ~n ~block ~grid ;
      Some out
    end
  in
  let reference i =
    let p = Vector.get src i in
    p.u +. p.v
  in
  (out_aos, out_soa, out_api, reference)

let check name dev n runner =
  Printf.printf
    "SoA-emitter %s [%s] %s: %!"
    name
    dev.Device.framework
    dev.Device.name ;
  try
    let out_aos, out_soa, out_api, reference = runner dev n in
    let ok = ref true in
    let check_leg label o a r i =
      match o with
      | None -> ()
      | Some o ->
          let s = Vector.get o i in
          if abs_float (s -. r) > 1e-3 || abs_float (s -. a) > 1e-4 then begin
            ok := false ;
            if i < 5 then
              Printf.printf
                "\n  %s mismatch @%d: %s=%f aos=%f ref=%f%!"
                label
                i
                label
                s
                a
                r
          end
    in
    for i = 0 to n - 1 do
      let r = reference i in
      let a = Vector.get out_aos i in
      if abs_float (a -. r) > 1e-3 then begin
        ok := false ;
        if i < 5 then
          Printf.printf "\n  AoS mismatch @%d: aos=%f ref=%f%!" i a r
      end ;
      (* Direct-emitter SoA leg. *)
      check_leg "SoA" out_soa a r i ;
      (* Real user-facing API leg (Soa_vector + Soa_launch.run_soa). *)
      check_leg "SoA-API" out_api a r i
    done ;
    let soa_note =
      match out_soa with None -> " (SoA skipped: non-PTX)" | Some _ -> ""
    in
    if !ok then (
      Printf.printf "PASSED%s\n%!" soa_note ;
      true)
    else (
      Printf.printf "FAILED\n%!" ;
      false)
  with e ->
    Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e) ;
    false

(* Item 3 gate: run_soa on a non-PTX device MUST raise a located error rather
   than binding the SoA N-pointer ABI to an AoS kernel signature (which would
   read wrong data). This is the "never wrong data" guarantee, checked
   concretely on whatever non-PTX backends are present. *)
let check_gate dev =
  if is_ptx dev then true
  else begin
    Printf.printf "SoA-gate [%s] %s: %!" dev.Device.framework dev.Device.name ;
    let ir = ir_of p3_kernel in
    let sv =
      Soa_vector.create
        point3d_custom
        ~fields:
          Sarek_ir_types.[("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
        16
    in
    let out = Vector.create Vector.float32 16 in
    match
      run_soa_via_api dev ir ~sv ~out ~n:16 ~block:(dims 16) ~grid:(dims 1)
    with
    | () | (exception Not_found) ->
        Printf.printf "FAILED (run_soa did not reject a non-PTX device)\n%!" ;
        false
    | exception Sarek.Execute_error.Execution_error _ ->
        Printf.printf "rejected (located error) OK\n%!" ;
        true
  end

let () =
  Benchmarks.init () ;
  let n = 1024 in
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline "test_soa_emitter_equiv: no device - SKIPPED" ;
    exit 0) ;
  let any_ptx = Array.exists is_ptx devs in
  if not any_ptx then
    print_endline
      "test_soa_emitter_equiv: no CUDA/PTX device - SoA leg skipped (AoS + \
       reference still checked)" ;
  let ok = ref true in
  Array.iter
    (fun dev ->
      (* point3d (f32) runs everywhere: cross-backend AoS + reference, plus the
         PTX SoA leg. *)
      if not (check "point3d(f32)" dev n run_p3) then ok := false ;
      (* dpair (f64) exists to prove the f64 SoA leaf on PTX; run it on CUDA/PTX
         only. (Some non-PTX backends — e.g. OpenCL/radeonsi — have an unrelated
         f64 custom-vector gap that is out of scope for this emitter test and is
         exercised elsewhere.) *)
      if is_ptx dev && not (check "dpair(f64)" dev n run_dpair) then ok := false ;
      (* Item 3: SoA launch must be rejected (never wrong data) on non-PTX. *)
      if not (check_gate dev) then ok := false)
    devs ;
  if not !ok then exit 1
