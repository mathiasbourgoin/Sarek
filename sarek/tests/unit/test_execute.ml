(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Execute - kernel execution dispatcher *)

open Sarek.Execute
open Spoc_core
open Spoc_framework
open Alcotest

type point = {x : float; y : float}

let point_type_id : point Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let point_vector_type_id : (point, unit) Vector.t Sarek_ir_types.Type_id.t =
  Sarek_ir_types.Type_id.create ()

let point_custom : point Vector.custom_type =
  let get ptr i =
    let off = i * 8 in
    {
      x = Vector.Custom_helpers.read_float32 ptr off;
      y = Vector.Custom_helpers.read_float32 ptr (off + 4);
    }
  in
  let set ptr i p =
    let off = i * 8 in
    Vector.Custom_helpers.write_float32 ptr off p.x ;
    Vector.Custom_helpers.write_float32 ptr (off + 4) p.y
  in
  {
    elem_size = 8;
    type_id = point_type_id;
    vector_type_id = point_vector_type_id;
    get;
    set;
    name = "point";
    (* Hand-written descriptor: [get]/[set] above are written by hand, not
       derived from a layout, so nothing keeps a field list in step with them.
       [None] is the correct answer for that shape even though this particular
       layout happens to be SoA-derivable. *)
    ir_fields = None;
  }

(** {1 Tests for vector argument types} *)

let test_vec_arg_int () =
  let arg = Int 42 in
  ignore arg ;
  check bool "int arg created" true true

let test_vec_arg_int32 () =
  let arg = Int32 42l in
  ignore arg ;
  check bool "int32 arg created" true true

let test_vec_arg_float32 () =
  let arg = Float32 3.14 in
  ignore arg ;
  check bool "float32 arg created" true true

let test_vec_arg_float64 () =
  let arg = Float64 2.71828 in
  ignore arg ;
  check bool "float64 arg created" true true

(** {1 Tests for dimension helpers} *)

let test_dims1d () =
  let d = dims1d 1024 in
  check int "1d dims x" 1024 d.Framework_sig.x ;
  check int "1d dims y" 1 d.Framework_sig.y ;
  check int "1d dims z" 1 d.Framework_sig.z

let test_dims2d () =
  let d = dims2d 32 64 in
  check int "2d dims x" 32 d.Framework_sig.x ;
  check int "2d dims y" 64 d.Framework_sig.y ;
  check int "2d dims z" 1 d.Framework_sig.z

let test_dims3d () =
  let d = dims3d 16 32 8 in
  check int "3d dims x" 16 d.Framework_sig.x ;
  check int "3d dims y" 32 d.Framework_sig.y ;
  check int "3d dims z" 8 d.Framework_sig.z

(** {1 Tests for grid calculation} *)

let test_grid_for_size_exact () =
  (* Problem size 1024, block size 256 → grid size 4 *)
  let grid = grid_for_size ~problem_size:1024 ~block_size:256 in
  check int "grid exact division" 4 grid

let test_grid_for_size_remainder () =
  (* Problem size 1000, block size 256 → grid size 4 (rounds up) *)
  let grid = grid_for_size ~problem_size:1000 ~block_size:256 in
  check int "grid with remainder" 4 grid

let test_grid_for_size_small () =
  (* Problem size 100, block size 256 → grid size 1 *)
  let grid = grid_for_size ~problem_size:100 ~block_size:256 in
  check int "grid smaller than block" 1 grid

let test_grid_for_size_zero () =
  (* Problem size 0 → grid size 0 *)
  let grid = grid_for_size ~problem_size:0 ~block_size:256 in
  check int "grid zero size" 0 grid

(** {1 Tests for vector creation (integration)} *)

let test_create_int32_vector () =
  let v = Vector.create Vector.int32 10 in
  check int "vector length" 10 (Vector.length v) ;
  (* Set and get a value *)
  Vector.set v 5 42l ;
  check int32 "vector get/set" 42l (Vector.get v 5)

let test_create_float32_vector () =
  let v = Vector.create Vector.float32 8 in
  check int "vector length" 8 (Vector.length v) ;
  Vector.set v 3 3.14 ;
  check (float 0.001) "vector get/set" 3.14 (Vector.get v 3)

(** {1 Tests for vector wrapping} *)

let test_vec_wrapper () =
  let v = Vector.create Vector.int32 5 in
  let arg = Vec v in
  ignore arg ;
  check bool "vec wrapper created" true true

let test_multiple_args () =
  let v1 = Vector.create Vector.int32 10 in
  let v2 = Vector.create Vector.float32 20 in
  let args = [Vec v1; Int 42; Vec v2; Float32 3.14] in
  check int "arg list length" 4 (List.length args)

let test_custom_exec_vector_get_set () =
  let v = Vector.create_custom point_custom 2 in
  Vector.set v 0 {x = 1.5; y = 2.5} ;
  Vector.set v 1 {x = 0.0; y = 0.0} ;
  match vector_args_to_exec_array [Vec v] with
  | [|Framework_sig.EA_Vec (module V)|] -> (
      match V.get 0 with
      | Typed_value.TV_Composite (Typed_value.CV ((module C), p)) ->
          check string "custom type name" "point" C.name ;
          check int "custom byte size" 8 C.size ;
          check int "serialized length" 8 (Bytes.length (C.to_bytes p)) ;
          V.set 1 (Typed_value.TV_Composite (Typed_value.CV ((module C), p))) ;
          let got = Vector.get v 1 in
          check (float 0.001) "custom set x" 1.5 got.x ;
          check (float 0.001) "custom set y" 2.5 got.y
      | _ -> fail "expected composite typed value")
  | _ -> fail "expected single exec vector"

(** {1 Device capability gate (#142)} *)

(* A device is only as capable as its [device_features] list says. These tests
   drive [check_device_capabilities] directly rather than through [run], because
   [run] needs a registered backend and the property under test is entirely
   about the device/IR pair.

   NOTE ON READING THE RESULT: Alcotest captures printf in this suite, so a
   check that silently does nothing still prints [OK]. Every case below either
   asserts an exception was raised or asserts one was not — none of them can
   pass by not executing. *)

let contains_sub haystack needle =
  let hl = String.length haystack and nl = String.length needle in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  go 0

let caps_with (features : Sarek_ir_analysis.feature list) :
    Framework_sig.capabilities =
  {
    Framework_sig.max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = features;
    (* backlog-62: no cooperative-matrix probe on this backend. [None] is
       "not probed", which Sarek_coopmat.verdict maps to Unknown and therefore
       refuses; an empty list would be a positive claim nobody measured. *)
    coopmat = None;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = false;
  }

let device_with features : Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "Fake Capability Device";
    framework = "Vulkan";
    capabilities = caps_with features;
  }

(* dst[0] = <a literal of element type [elt]>. Enough for [kernel_uses] to see
   the width through the parameter's element type. *)
let kernel_over elt : Sarek_ir_types.kernel =
  let open Sarek_ir_types in
  let dst =
    {var_name = "dst"; var_id = 0; var_type = TVec elt; var_mutable = false}
  in
  {
    default_kernel with
    kern_name = "cap_probe";
    kern_params =
      [DParam (dst, Some {arr_elttype = elt; arr_memspace = Global})];
  }

let refusal_of ~device ir =
  match check_device_capabilities ~device ir with
  | () -> None
  | exception e -> Some (Printexc.to_string e)

let test_int64_refused_without_device_support () =
  let device = device_with [Sarek_ir_analysis.Float64] in
  match refusal_of ~device (kernel_over Sarek_ir_types.TInt64) with
  | None ->
      fail
        "an int64 kernel on a device that does not provide int64 must be \
         refused at launch"
  | Some msg ->
      check bool "diagnostic names int64" true (contains_sub msg "int64") ;
      check
        bool
        "diagnostic names the device"
        true
        (contains_sub msg "Fake Capability Device") ;
      check
        bool
        "diagnostic names the missing Vulkan feature"
        true
        (contains_sub msg "shaderInt64")

(* The control that makes the test above mean something. An unconditional raise
   would satisfy it; this requires the gate to DISCRIMINATE. *)
let test_int64_allowed_with_device_support () =
  let device = device_with [Sarek_ir_analysis.Int64] in
  match refusal_of ~device (kernel_over Sarek_ir_types.TInt64) with
  | None -> ()
  | Some msg -> fail ("int64 kernel must launch on an int64 device, got: " ^ msg)

(* And a kernel that needs neither must be unaffected by either list. *)
let test_int32_never_gated () =
  let device = device_with [] in
  match refusal_of ~device (kernel_over Sarek_ir_types.TInt32) with
  | None -> ()
  | Some msg -> fail ("an int32 kernel must never be gated, got: " ^ msg)

let test_float64_refused_without_device_support () =
  let device = device_with [Sarek_ir_analysis.Int64] in
  match refusal_of ~device (kernel_over Sarek_ir_types.TFloat64) with
  | None -> fail "an f64 kernel on a device without fp64 must be refused"
  | Some msg ->
      check bool "diagnostic names float64" true (contains_sub msg "float64")

(** {1 The interpreter's own capability gate (backlog-154)}

    [run_interpreter_vectors] applied [check_launch_args] and no capability
    gate, so the cross-backend numeric ORACLE was the one execution path with
    none. The asymmetry was visible in a single run of
    [test_coopmat_integer_e2e] on this workstation: [check_device_capabilities]
    refused the Interpreter device for a kernel that [run_interpreter_vectors]
    then evaluated to 65536 bit-exact results.

    These cases drive [Sarek_interp_capability] directly rather than through
    [run_interpreter_vectors], which needs live vectors; the property under test
    is entirely the kernel/capability pair. The e2e coopmat suite covers the
    wired path.

    Read {!Sarek_interp_capability} for the argument about what an interpreter
    should advertise. In one line: what its evaluator implements, stated by
    construction — never "unprobed", and never everything. *)

let interp_refusal ir =
  Option.map
    (fun v ->
      match v with
      | Sarek_capability.Unavailable cap ->
          Sarek_capability.explain ~target:"CPU Interpreter" cap
      | Sarek_capability.Unknown why -> why
      | Sarek_capability.Available -> fail "first_refusal returned Available")
    (Sarek_interp.Sarek_interp_capability.first_refusal ir)

let coopmat_kernel (comp : Sarek_coopmat.component_type) : Sarek_ir_types.kernel
    =
  let open Sarek_ir_types in
  let shape = {Sarek_coopmat.m = 16; n = 16; k = 16} in
  let cfg =
    {
      Sarek_coopmat.cfg_shape = shape;
      cfg_a = comp;
      cfg_b = comp;
      cfg_c = Sarek_coopmat.Sint32;
      cfg_result = Sarek_coopmat.Sint32;
      cfg_saturating = false;
      cfg_scope = Sarek_coopmat.Subgroup;
    }
  in
  {
    default_kernel with
    kern_name = "interp_coopmat_probe";
    kern_body =
      SCoopmat (CM_muladd {dst = "fd"; a = "fa"; b = "fb"; c = "fc"; cfg});
  }

(* THE POSITIVE CONTROL, and it is the load-bearing one. A gate observed only
   refusing is indistinguishable from one that refuses unconditionally — and
   refusing unconditionally is exactly what the DEVICE gate does to the
   interpreter (coopmat = None -> Unknown), which is the defect. This case is
   the assertion that the new gate did not simply reproduce it. *)
let test_interp_permits_integer_coopmat () =
  match interp_refusal (coopmat_kernel Sarek_coopmat.Uint8) with
  | None -> ()
  | Some msg ->
      fail
        (Printf.sprintf
           "the interpreter evaluates integer cooperative matrices \
            (Sarek_ir_interp_eval.exec_coopmat) and is the oracle the GPU \
            backends are compared against, yet its own gate refused: %s"
           msg)

(* And it still refuses what the evaluator refuses — at the launch gate now,
   rather than from inside exec_coopmat after a partial writeback. *)
let test_interp_refuses_float_coopmat () =
  match interp_refusal (coopmat_kernel Sarek_coopmat.Float16) with
  | None ->
      fail
        "float cooperative-matrix accumulation has no single right answer \
         (SPV_KHR_cooperative_matrix leaves the addition order to the \
         implementation), so a strict oracle must refuse it"
  | Some msg ->
      check bool "diagnostic names f16" true (contains_sub msg "f16") ;
      check
        bool
        "diagnostic names the interpreter"
        true
        (contains_sub msg "CPU Interpreter")

(* The second instance of the same gap, and the reason this is not a
   coopmat-specific fix. Both plugins reported [Float64; Int64], omitting an
   f16 they implement; nothing refused only because check_device_capabilities
   excludes Float16 from its gated list for an unrelated reason. The
   interpreter gate DOES gate f16, against its own by-construction claim, so
   this assertion has teeth: drop Float16 from device_features and it goes
   red. *)
let test_interp_declares_float16 () =
  check
    bool
    "the interpreter advertises the f16 that Sarek_float16 implements"
    true
    (List.mem
       Sarek_ir_analysis.Float16
       Sarek_interp.Sarek_interp_capability.device_features) ;
  match interp_refusal (kernel_over Sarek_ir_types.TFloat16) with
  | None -> ()
  | Some msg -> fail ("an f16 kernel must reach the interpreter: " ^ msg)

(* THE GATE MUST BE WIRED, not merely correct.

   Every case above drives Sarek_interp_capability directly, and every one of
   them stays green if the call in run_interpreter_vectors is deleted — which
   is the whole defect restored. That is the "unwired declaration" shape: a
   checker that is right about everything and consulted by nobody. This case is
   the only one that goes red on that deletion, so it is the one that actually
   pins backlog-154.

   It also pins WHICH error: before the gate, this kernel reached
   Sarek_ir_interp_eval.coopmat_refuse_float from inside the evaluation, after
   the run had begun and could have written back. Now it is refused at the
   launch gate, as an Execute Backend_error, before anything runs. *)
let test_run_interpreter_vectors_is_gated () =
  match
    run_interpreter_vectors
      ~ir:(coopmat_kernel Sarek_coopmat.Float16)
      ~args:[]
      ~block:(dims1d 1)
      ~grid:(dims1d 1)
      ~parallel:false
  with
  | () ->
      fail
        "run_interpreter_vectors ran a float cooperative-matrix kernel: the \
         interpreter capability gate is not wired into the oracle entry point \
         (backlog-154)"
  | exception
      Sarek.Execute_error.Execution_error
        (Sarek.Execute_error.Backend_error {backend; message}) ->
      check string "refused as the Interpreter backend" "Interpreter" backend ;
      check
        bool
        "diagnostic names the capability"
        true
        (contains_sub message "coopmat")
  | exception e ->
      fail
        ("expected the launch gate's Backend_error, got a deeper failure — \
          which means the kernel started running: " ^ Printexc.to_string e)

(* The gate is not vacuous on the feature axis either: a feature the
   interpreter does NOT provide is refused. Uses a deliberately-emptied
   provided-list through the same code path the real one uses. *)
let test_interp_feature_gate_can_refuse () =
  match
    Sarek_capability.device_verdict
      ~provided:(Some [Sarek_ir_analysis.Int64])
      Sarek_ir_analysis.Float64
  with
  | Sarek_capability.Available ->
      fail "device_verdict permitted a feature absent from the provided list"
  | _ -> ()

(** {1 Test suite} *)

let () =
  run
    "Execute"
    [
      ( "interpreter_capability_gate",
        [
          test_case
            "integer coopmat permitted (positive control)"
            `Quick
            test_interp_permits_integer_coopmat;
          test_case
            "float coopmat refused at the launch gate"
            `Quick
            test_interp_refuses_float_coopmat;
          test_case
            "f16 declared and permitted"
            `Quick
            test_interp_declares_float16;
          test_case
            "feature gate can refuse"
            `Quick
            test_interp_feature_gate_can_refuse;
          test_case
            "run_interpreter_vectors is gated (the wiring)"
            `Quick
            test_run_interpreter_vectors_is_gated;
        ] );
      ( "device_capability_gate",
        [
          test_case
            "int64 refused without device support"
            `Quick
            test_int64_refused_without_device_support;
          test_case
            "int64 allowed with device support"
            `Quick
            test_int64_allowed_with_device_support;
          test_case "int32 never gated" `Quick test_int32_never_gated;
          test_case
            "float64 refused without device support"
            `Quick
            test_float64_refused_without_device_support;
        ] );
      ( "vector_arg_types",
        [
          test_case "int" `Quick test_vec_arg_int;
          test_case "int32" `Quick test_vec_arg_int32;
          test_case "float32" `Quick test_vec_arg_float32;
          test_case "float64" `Quick test_vec_arg_float64;
        ] );
      ( "dimension_helpers",
        [
          test_case "dims1d" `Quick test_dims1d;
          test_case "dims2d" `Quick test_dims2d;
          test_case "dims3d" `Quick test_dims3d;
        ] );
      ( "grid_calculation",
        [
          test_case "exact_division" `Quick test_grid_for_size_exact;
          test_case "with_remainder" `Quick test_grid_for_size_remainder;
          test_case "smaller_than_block" `Quick test_grid_for_size_small;
          test_case "zero_size" `Quick test_grid_for_size_zero;
        ] );
      ( "vector_operations",
        [
          test_case "create_int32" `Quick test_create_int32_vector;
          test_case "create_float32" `Quick test_create_float32_vector;
        ] );
      ( "vector_wrapping",
        [
          test_case "vec_wrapper" `Quick test_vec_wrapper;
          test_case "multiple_args" `Quick test_multiple_args;
          test_case
            "custom_exec_vector_get_set"
            `Quick
            test_custom_exec_vector_get_set;
        ] );
    ]
