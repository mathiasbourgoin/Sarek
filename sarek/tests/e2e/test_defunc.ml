(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for L12 Tier-0 defunctionalization (Sarek_defunc).
 *
 * Exercises the two capabilities the pass unlocks, end-to-end, on every
 * available device, self-verifying against a pure-OCaml reference:
 *
 *   1. Static single candidate:  `let f = addf in ... f x 1l`
 *      The function-valued binding resolves to a single named helper; the
 *      pass drops the binding and calls `addf` directly.
 *
 *   2. Genuinely runtime-dynamic: `let f = if op = 0l then addf else mulf in
 *                                  ... f x x`
 *      `op` is a *runtime kernel parameter*, so the choice cannot be resolved
 *      at compile time. The pass distributes the application into the `if`
 *      branches -> `if op = 0l then addf x x else mulf x x`. No function value
 *      reaches lowering and no tag variant is synthesized, so every backend
 *      handles it with its existing `if` + direct-call paths (zero emitter
 *      changes). Both kernels would fail to compile before L12 with
 *      "EApp to unknown function 'f'".
 *
 * The candidate leaves are kernel-local `let`-bound helpers (a Tier-0
 * candidate class per L12 6). Only int32 vectors + scalars are used (no
 * custom types), so the kernels run on the always-available
 * Native/Interpreter backends and on every GPU backend without the
 * custom-vector caveats that gate test_klet_variant.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
open Sarek
module Std = Sarek_stdlib.Std

(* Backend init goes through the shared conditional loader so this test
   stays buildable when optional GPU plugins (CUDA/OpenCL/Vulkan/Metal) are
   absent, and enumerates every backend that IS present. *)
let () = Test_helpers.Benchmarks.init ()

(* Case 1: static single candidate. `let f = addf` resolves to one helper. *)
let static_kernel =
  [%kernel
    let open Std in
    let addf (a : int32) (b : int32) : int32 = a + b in
    fun (data : int32 vector) (out : int32 vector) ->
      let idx = global_idx_x in
      let f = addf in
      out.(idx) <- f data.(idx) 1l]

(* Case 2: runtime-dynamic dispatch. `op` is a kernel parameter, so the
   selection is only known at runtime. *)
let dynamic_kernel =
  [%kernel
    let open Std in
    let addf (a : int32) (b : int32) : int32 = a + b in
    let mulf (a : int32) (b : int32) : int32 = a * b in
    fun (data : int32 vector) (op : int32) (out : int32 vector) ->
      let idx = global_idx_x in
      let f = if op = 0l then addf else mulf in
      out.(idx) <- f data.(idx) data.(idx)]

let n = 1024

let block_size = 256

let ir_of kernel =
  let _, kirc = kernel in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let make_data () =
  let data = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set data i (Int32.of_int (i mod 97))
  done ;
  data

(* Returns true if this device produced correct results for both kernels. *)
let run_on_device dev =
  Printf.printf
    "--- device: %s (%s) ---\n%!"
    dev.Device.name
    dev.Device.framework ;
  let block = Execute.dims1d block_size in
  let grid = Execute.dims1d (n / block_size) in
  let ok = ref true in

  (* Case 1: static. out[i] = data[i] + 1 *)
  let data = make_data () in
  let out = Vector.create Vector.int32 n in
  Execute.run_vectors
    ~device:dev
    ~ir:(ir_of static_kernel)
    ~args:[Execute.Vec data; Execute.Vec out]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  for i = 0 to n - 1 do
    let expected = Int32.add (Vector.get data i) 1l in
    let got = Vector.get out i in
    if got <> expected then begin
      if !ok then
        Printf.printf
          "  [static] FAIL at %d: got %ld expected %ld\n%!"
          i
          got
          expected ;
      ok := false
    end
  done ;
  if !ok then Printf.printf "  [static] PASS\n%!" ;

  (* Case 2: dynamic, both selectors. op=0 -> add (x+x), op=1 -> mul (x*x) *)
  let run_dynamic op reference =
    let data = make_data () in
    let out = Vector.create Vector.int32 n in
    Execute.run_vectors
      ~device:dev
      ~ir:(ir_of dynamic_kernel)
      ~args:[Execute.Vec data; Execute.Int32 (Int32.of_int op); Execute.Vec out]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;
    let dyn_ok = ref true in
    for i = 0 to n - 1 do
      let x = Vector.get data i in
      let expected = reference x in
      let got = Vector.get out i in
      if got <> expected then begin
        if !dyn_ok then
          Printf.printf
            "  [dynamic op=%d] FAIL at %d: got %ld expected %ld\n%!"
            op
            i
            got
            expected ;
        dyn_ok := false ;
        ok := false
      end
    done ;
    if !dyn_ok then Printf.printf "  [dynamic op=%d] PASS\n%!" op
  in
  run_dynamic 0 (fun x -> Int32.add x x) ;
  run_dynamic 1 (fun x -> Int32.mul x x) ;
  !ok

let () =
  print_endline "=== L12 Tier-0 defunctionalization E2E ===" ;
  let devs =
    Device.init
      ~frameworks:["CUDA"; "OpenCL"; "Vulkan"; "Metal"; "Native"; "Interpreter"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No device found - nothing to run" ;
    exit 0
  end ;
  let all_ok = ref true in
  Array.iter (fun dev -> if not (run_on_device dev) then all_ok := false) devs ;
  print_endline "" ;
  if !all_ok then begin
    print_endline "test_defunc: PASS (all devices)" ;
    exit 0
  end
  else begin
    print_endline "test_defunc: FAIL" ;
    exit 1
  end
