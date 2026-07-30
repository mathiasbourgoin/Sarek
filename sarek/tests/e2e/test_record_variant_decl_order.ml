(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: a dependency edge that CROSSES between a record declaration and a
 * variant declaration must be ordered (backlog-211).
 *
 * backlog-203 sorted record declarations among THEMSELVES, and each backend
 * family runs that sort inside its own emission loop. The two loops are
 * separate, and the families disagree on which one runs first, so a cross-kind
 * edge was ordered by neither:
 *
 *   - the C family (OpenCL/CUDA/HIP/Metal) emits VARIANTS then RECORDS, so a
 *     variant with a RECORD payload names a struct declared later;
 *   - GLSL and WGSL emit RECORDS then VARIANTS, so a record with a
 *     VARIANT-TYPED field names a struct declared later.
 *
 * Each family is therefore red on exactly the shape the other family is green
 * on. That is why a reproducer built on one shape alone reports the gap fixed,
 * and why BOTH shapes are here, as two separate kernels:
 *
 *   Shape A — variant with a record payload  (`At of probe_pt`)
 *             red on the C family, green on GLSL/WGSL before the fix.
 *   Shape B — record with a variant field    (`{gk : flagv; ...}`)
 *             red on GLSL/WGSL, green on the C family before the fix.
 *
 * Both kernels are self-contained: the aggregate is built INSIDE the kernel and
 * only a float32 vector crosses the host boundary. That is deliberate — the
 * defect is in type-DECLARATION emission, so nothing here depends on a
 * host-side byte layout for a record with a variant field (which device
 * backends do not have; see test_ktype_record_variant_field.ml).
 *
 * Every available device must PASS both shapes. There is no per-backend
 * tolerance: a device that fails makes the process exit non-zero.
 *
 * DEVICE COVERAGE IS WHAT THE HOST HAS. This host has OpenCL and Vulkan (two
 * devices each) plus Interpreter/Native; it has no NVIDIA, HIP or Metal
 * hardware, and there is no WGSL device at all. The emission ORDER for those
 * families is pinned device-independently in
 * sarek/tests/codegen_golden/test_decl_order_all_backends.ml, which is where a
 * regression is caught on a machine with zero devices.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

(* --- Shape A: a variant whose payload is a record ------------------------- *)
type probe_pt = {px : float32; py : float32} [@@sarek.type]

type probe = Nowhere | At of probe_pt [@@sarek.type]

(* --- Shape B: a record whose field type is a variant --------------------- *)
type flagv = Off | On | Level of float32 [@@sarek.type]

type gauge = {gk : flagv; gv : float32} [@@sarek.type]

(* Shape A. The record literal registers [probe_pt] in the kernel's record
   table and the constructor application registers [probe] in its variant
   table, so both declarations are emitted and the cross edge
   (probe -> probe_pt) is live.

   THE CONSTRUCTOR IS RUNTIME-SELECTED ON PURPOSE, and that is the whole
   difficulty of writing this test. Static tag erasure (L14/S1, see
   Sarek_tag_erasure.ml) reduces a variant-typed immutable local written once by
   a LITERAL constructor and read only as a match scrutinee, and a
   literal-constructor version of this kernel is erased before lowering: the
   variant never reaches [kern_variants], no variant struct is emitted at all,
   and the test is green on every device while covering nothing. Selecting the
   constructor behind an [if] leaves the slot tagged, which is what puts the
   cross edge in the emitted source. [n > 0] always takes the [At] branch, so
   the expected value stays a constant. *)
let k_variant_payload =
  snd
    [%kernel
      fun (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let q = {px = 1.5; py = 2.25} in
          let p = if n > 0 then At q else Nowhere in
          let v = match p with Nowhere -> 0.0 | At r -> r.px +. r.py in
          dst.(tid) <- v
        end]

(* Shape B. Mirror: the record literal registers [gauge] (whose [gk] field type
   is the variant) and the constructor application registers [flagv], so the
   cross edge (gauge -> flagv) is live.

   Runtime-selected for the same reason as shape A — here it is S2, the
   variant-typed record FIELD erasure, which requires every variant field of a
   let-bound record literal to be a literal constructor application. An [if]
   in the field position leaves [gauge] with a genuine [TVariant] field. *)
let k_record_variant_field =
  snd
    [%kernel
      fun (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let g = {gk = (if n > 0 then Level 2.0 else Off); gv = 4.75} in
          let a = match g.gk with Off -> 0.0 | On -> 1.0 | Level x -> x in
          dst.(tid) <- g.gv +. a
        end]

let n = 64

let devices () =
  Device.init
    ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"]
    ()

let ir_of name kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith (name ^ ": kernel has no IR")

let run_case ~dev ~name ~kirc ~expected =
  Printf.printf "  [%s] %s / %s: %!" dev.Device.framework dev.Device.name name ;
  try
    let dst = Vector.create Vector.float32 n in
    for i = 0 to n - 1 do
      Vector.set dst i 0.0
    done ;
    let threads = min 64 n in
    let grid_x = (n + threads - 1) / threads in
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d threads)
      ~grid:(Sarek.Execute.dims1d grid_x)
      ~ir:(ir_of name kirc)
      ~args:[Sarek.Execute.Vec dst; Sarek.Execute.Int32 (Int32.of_int n)]
      () ;
    Transfer.flush dev ;
    let ok = ref true in
    for i = 0 to n - 1 do
      let got = Vector.get dst i in
      if abs_float (got -. expected) > 1e-3 then begin
        if !ok then
          Printf.printf
            "\n    mismatch at %d: got %.3f expected %.3f%!"
            i
            got
            expected ;
        ok := false
      end
    done ;
    if !ok then print_endline "PASSED" else print_endline "FAILED" ;
    !ok
  with e ->
    Printf.printf "FAILED (%s)\n%!" (Printexc.to_string e) ;
    false

let () =
  print_endline
    "=== record/variant cross-kind declaration order (backlog-211) ===" ;
  let devs = devices () in
  if Array.length devs = 0 then begin
    print_endline "No devices found - nothing to verify" ;
    exit 0
  end ;
  let any_failure = ref false in
  Array.iter
    (fun dev ->
      let check b = if not b then any_failure := true in
      check
        (run_case
           ~dev
           ~name:"shape A: variant with a record payload"
           ~kirc:k_variant_payload
           ~expected:3.75) ;
      check
        (run_case
           ~dev
           ~name:"shape B: record with a variant field"
           ~kirc:k_record_variant_field
           ~expected:6.75))
    devs ;
  if !any_failure then begin
    print_endline "FAILED: at least one device/case did not verify" ;
    exit 1
  end ;
  print_endline "ALL PASSED"
