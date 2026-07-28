(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Per-backend emission of the synchronisation statements.
 *
 * SBarrier / SWarpBarrier / SMemFence are lowered by a hand-written match arm
 * in each of the six backends, and nothing held those arms to the instruction
 * each target actually has. They drifted: the Metal arm emitted
 * `sub_group_threadgroup_barrier(...)`, which is not an MSL function — Metal
 * spells the SIMD-group barrier `simdgroup_barrier`, which is what this repo's
 * OWN Metal plugin table (sarek-metal/Metal_plugin.ml) has always said, and
 * what sarek-metal/README.md documented incorrectly.
 *
 * That went unnoticed because no PPX syntax constructs SWarpBarrier or
 * SMemFence today (see the note at the bottom of this file), so the arm has
 * never reached a Metal compiler. This test pins all three statements on all
 * six backends at the IR level, which is the only level at which they are
 * currently reachable.
 *
 * These are STRING assertions against each target's documented spelling, not
 * device-verified behaviour: no Metal, CUDA or WebGPU device was available.
 * They are a drift guard — "this arm still says what we decided it says" —
 * and they are why the Metal arm's disagreement with the Metal plugin table
 * became visible at all.
 ******************************************************************************)

open Sarek_ir_types

let sync_kernel stmt : kernel =
  let out =
    {var_name = "out"; var_id = 0; var_type = TVec TInt32; var_mutable = false}
  in
  let tid =
    {var_name = "tid"; var_id = 1; var_type = TInt32; var_mutable = false}
  in
  {
    default_kernel with
    kern_name = "sync_probe";
    kern_params =
      [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SSeq
            [stmt; SAssign (LArrayElem ("out", EVar tid), EConst (CInt32 1l))]
        );
  }

let backends =
  [
    ( "CUDA",
      fun k ->
        Sarek_codegen.Sarek_ir_cuda.current_framework := None ;
        Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k );
    ( "OpenCL",
      fun k -> Sarek_codegen.Sarek_ir_opencl.generate_with_types ~types:[] k );
    ( "Metal",
      fun k -> Sarek_codegen.Sarek_ir_metal.generate_with_types ~types:[] k );
    ( "GLSL",
      fun k -> Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types:[] k );
    ( "WGSL",
      fun k -> Sarek_codegen.Sarek_ir_wgsl.generate_with_types ~types:[] k );
    ("PTX", fun k -> Sarek_codegen.Sarek_ir_ptx.generate k);
  ]

let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  go 0

(* (statement, per-backend expected substring) *)
let expectations =
  [
    ( "SBarrier",
      SBarrier,
      [
        ("CUDA", "__syncthreads()");
        ("OpenCL", "barrier(CLK_LOCAL_MEM_FENCE)");
        ("Metal", "threadgroup_barrier(mem_flags::mem_threadgroup)");
        ("GLSL", "barrier()");
        ("WGSL", "workgroupBarrier()");
        ("PTX", "bar.sync");
      ] );
    ( "SWarpBarrier",
      SWarpBarrier,
      [
        ("CUDA", "__syncwarp()");
        ("OpenCL", "sub_group_barrier(CLK_LOCAL_MEM_FENCE)");
        (* MSL's SIMD-group barrier. `sub_group_threadgroup_barrier` is not an
           MSL function; it was emitted here until #70. *)
        ("Metal", "simdgroup_barrier(mem_flags::mem_threadgroup)");
        ("GLSL", "subgroupBarrier()");
        ("WGSL", "subgroupBarrier()");
        ("PTX", "bar.warp.sync");
      ] );
    ( "SMemFence",
      SMemFence,
      [
        ("CUDA", "__threadfence()");
        ("OpenCL", "mem_fence(CLK_GLOBAL_MEM_FENCE)");
        ("Metal", "threadgroup_barrier(mem_flags::mem_device)");
        ("GLSL", "memoryBarrier()");
        ("WGSL", "storageBarrier()");
        ("PTX", "membar");
      ] );
  ]

let check_stmt (label, stmt, per_backend) () =
  let k = sync_kernel stmt in
  List.iter
    (fun (backend, gen) ->
      match List.assoc_opt backend per_backend with
      | None -> ()
      | Some expected -> (
          match gen k with
          | src ->
              Alcotest.(check bool)
                (Printf.sprintf "%s on %s emits %S" label backend expected)
                true
                (contains src expected)
          | exception e ->
              Alcotest.failf
                "%s on %s raised %s"
                label
                backend
                (Printexc.to_string e)))
    backends

(** The Metal arm is the one that had drifted, so assert its old text is gone
    rather than only asserting the new text is present: a backend emitting both
    would satisfy the check above. *)
let check_metal_no_stale_name () =
  let src =
    Sarek_codegen.Sarek_ir_metal.generate_with_types
      ~types:[]
      (sync_kernel SWarpBarrier)
  in
  Alcotest.(check bool)
    "Metal no longer emits sub_group_threadgroup_barrier (not an MSL function; \
     the repo's own Metal plugin table says simdgroup_barrier)"
    false
    (contains src "sub_group_threadgroup_barrier")

(** Scope note, asserted rather than left as prose: no PPX surface syntax
    constructs [SWarpBarrier] or [SMemFence] today. [block_barrier] is the only
    sync name the front end declares ([Sarek_core_primitives], [Gpu.ml]); the
    [warp_barrier] and [memory_fence] names that several READMEs present as
    callable user API are not declared anywhere, so the arms pinned above are
    reachable only from hand-built IR such as this test. This check pins that
    fact, so if a future change makes them callable it lands together with a
    deliberate update here rather than silently. *)
let check_warp_barrier_not_callable () =
  Alcotest.(check bool)
    "warp_barrier is not a declared core primitive"
    false
    (Sarek_core_primitives.is_core_primitive "warp_barrier") ;
  Alcotest.(check bool)
    "block_barrier IS a declared core primitive (positive control: the lookup \
     above is not vacuously false)"
    true
    (Sarek_core_primitives.is_core_primitive "block_barrier")

let () =
  Alcotest.run
    "sync-stmt-emission"
    [
      ( "per-backend emission",
        List.map
          (fun ((label, _, _) as e) ->
            Alcotest.test_case label `Quick (check_stmt e))
          expectations );
      ( "drift guards",
        [
          Alcotest.test_case
            "Metal stale name gone"
            `Quick
            check_metal_no_stale_name;
          Alcotest.test_case
            "warp_barrier is not user-callable"
            `Quick
            check_warp_barrier_not_callable;
        ] );
    ]
