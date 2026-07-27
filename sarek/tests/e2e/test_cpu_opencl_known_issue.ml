(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Truth-table test for the CPU-OpenCL float32-math KNOWN-ISSUE classifier
 * (Test_helpers.is_cpu_opencl_device / classify_cpu_opencl_math_result).
 *
 * This is the strictness guard for the #74 flake fix: the classifier must
 * annotate a wrong float32 math result ONLY on an OpenCL device whose
 * CL_DEVICE_TYPE is CPU, and must still `Fail on every other device -
 * including a GPU-OpenCL device whose CL_DEVICE_NAME is a CPU model string
 * (which is exactly what rusticl reports for an APU: "AMD Ryzen 9 7950X
 * 16-Core Processor (radeonsi, raphael_mendocino, ...)"). Device records are
 * fabricated so the truth table is checked on every machine, with or without
 * a CPU-OpenCL ICD installed.
 *
 * Device identity is NOT sufficient (audit finding #74 / F1): the failure SHAPE
 * is a second, independent gate. On the known-issue device the classifier must
 * still `Fail when the result is wrong from element 0 (an intrinsic-mapping or
 * operand regression), wrong from element 1 (a kernel that never executed,
 * leaving the 0.0 pre-fill), non-finite, or wrong in EVERY element. Only the
 * documented signature - first bad index at or beyond the 4-wide SSE lane
 * boundary, finite, and partial - is excused.
 *
 * Run with: dune exec sarek/tests/e2e/test_cpu_opencl_known_issue.exe
 ******************************************************************************)

module Device = Spoc_core.Device

let caps ~is_cpu : Spoc_framework.Framework_sig.capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 32768;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = [Sarek_ir_analysis.Float64; Sarek_ir_analysis.Int64];
    (* backlog-62: no cooperative-matrix probe on this backend. [None] is
       "not probed", which Sarek_coopmat.verdict maps to Unknown and therefore
       refuses; an empty list would be a positive claim nobody measured. *)
    coopmat = None;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 0;
    clock_rate_khz = 1000000;
    multiprocessor_count = 8;
    is_cpu;
  }

let device ~name ~framework ~is_cpu : Device.t =
  {id = 0; backend_id = 0; name; framework; capabilities = caps ~is_cpu}

(* The CI runner's device: Intel oneAPI CPU OpenCL runtime, CL_DEVICE_TYPE=CPU,
   and a name that contains none of the rusticl/radeonsi tokens. *)
let ci_cpu_opencl =
  device
    ~name:"AMD EPYC 9V45 96-Core Processor"
    ~framework:"OpenCL"
    ~is_cpu:true

(* pocl's CPU device. Not in the CI image today (see Test_helpers.is_pocl_device
   for why the jammy build was reverted), but CL_DEVICE_TYPE=CPU, so it would
   satisfy is_cpu_opencl_device the moment one appears — and the carve-out in
   classify_cpu_opencl_math_result is what would keep its failures hard. Pinning
   the behaviour now means the experimental pocl PR cannot land the coverage
   regression silently. Two names, one per pocl naming era: "pthread-" (1.x,
   Ubuntu 22.04) and "cpu-" (3.x+). *)
let pocl_cpu_pthread =
  device ~name:"pthread-znver3" ~framework:"OpenCL" ~is_cpu:true

let pocl_cpu_modern =
  device ~name:"cpu-znver4-AMD EPYC 9V45" ~framework:"OpenCL" ~is_cpu:true

(* The campaign workstation's iGPU under rusticl: CL_DEVICE_TYPE=GPU, but named
   after the CPU socket. A name heuristic would misclassify this one. *)
let igpu_named_like_a_cpu =
  device
    ~name:
      "AMD Ryzen 9 7950X 16-Core Processor (radeonsi, raphael_mendocino, ACO)"
    ~framework:"OpenCL"
    ~is_cpu:false

let dgpu_opencl =
  device
    ~name:"AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO)"
    ~framework:"OpenCL"
    ~is_cpu:false

(* A GPU whose CL_DEVICE_NAME happens to start with a pocl prefix. Contrived as
   a product name, but it is the case that separates the two possible readings
   of is_pocl_device: name-only (would say "pocl") versus name-AND-device-type
   (says "not pocl", which is what the predicate's contract states). Without
   this row, dropping the CL_DEVICE_TYPE test from is_pocl_device passes the
   whole suite. *)
let gpu_named_like_pocl =
  device ~name:"cpu-znver4 compatibility GPU" ~framework:"OpenCL" ~is_cpu:false

let native_cpu = device ~name:"CPU Native" ~framework:"Native" ~is_cpu:true

let interpreter =
  device ~name:"CPU Interpreter" ~framework:"Interpreter" ~is_cpu:true

let vulkan_gpu =
  device
    ~name:"AMD Radeon RX 7900 XTX (RADV NAVI31)"
    ~framework:"Vulkan"
    ~is_cpu:false

let failures = ref 0

let check name expected actual =
  if expected = actual then Printf.printf "  OK   %s\n" name
  else begin
    Printf.printf "  FAIL %s\n" name ;
    incr failures
  end

let verdict_name = function
  | `Pass -> "Pass"
  | `Known_issue _ -> "Known_issue"
  | `Fail -> "Fail"

let shape ?(total = 256) ?(non_finite = false) ~first_bad ~bad_count () :
    Test_helpers.float_check_shape =
  {first_bad_index = first_bad; bad_count; total; non_finite}

(* The real flake signature: first bad element at the 4-wide SSE lane boundary,
   finite values, only part of the buffer damaged. *)
let flake_shape = shape ~first_bad:(Some 4) ~bad_count:60 ()

(* A clean result. *)
let clean_shape = shape ~first_bad:None ~bad_count:0 ()

(* A wrong result carrying the real flake shape, on each device. *)
let classify dev =
  verdict_name
    (Test_helpers.classify_cpu_opencl_math_result ~dev ~shape:flake_shape ())

(* A correct result must always be `Pass, even on the known-issue device. *)
let classify_ok dev =
  verdict_name
    (Test_helpers.classify_cpu_opencl_math_result ~dev ~shape:clean_shape ())

(* Shape discrimination, always on the known-issue device: only the shape
   varies, so any `Fail here is attributable to the shape gate alone. *)
let classify_shape s dev =
  verdict_name (Test_helpers.classify_cpu_opencl_math_result ~dev ~shape:s ())

(* Run [f] with stdout redirected to a temp file and return what it printed.
   Needed to assert on the F2 expiry NOTE, which is a side effect on the `Pass
   path rather than part of the verdict. *)
let capture_stdout f =
  let tmp = Filename.temp_file "cpu_opencl_known_issue" ".out" in
  let fd =
    Unix.openfile tmp [Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC] 0o600
  in
  let saved = Unix.dup Unix.stdout in
  flush stdout ;
  Unix.dup2 fd Unix.stdout ;
  let restore () =
    flush stdout ;
    Unix.dup2 saved Unix.stdout ;
    Unix.close fd ;
    Unix.close saved
  in
  (try f ()
   with e ->
     restore () ;
     Sys.remove tmp ;
     raise e) ;
  restore () ;
  let ic = open_in_bin tmp in
  let out = really_input_string ic (in_channel_length ic) in
  close_in ic ;
  Sys.remove tmp ;
  out

let note_printed ~dev ~shape =
  let out =
    capture_stdout (fun () ->
        ignore (Test_helpers.classify_cpu_opencl_math_result ~dev ~shape ()))
  in
  Test_helpers.string_contains
    ~needle:"re-evaluate the CPU-OpenCL suppression"
    out

let () =
  print_endline "=== CPU-OpenCL KNOWN-ISSUE classifier truth table ===" ;
  print_endline "is_cpu_opencl_device:" ;
  check
    "CI CPU-OpenCL device (AMD EPYC 9V45) is CPU-OpenCL"
    true
    (Test_helpers.is_cpu_opencl_device ci_cpu_opencl) ;
  check
    "GPU-OpenCL device named after a CPU is NOT CPU-OpenCL"
    false
    (Test_helpers.is_cpu_opencl_device igpu_named_like_a_cpu) ;
  check
    "discrete GPU-OpenCL device is NOT CPU-OpenCL"
    false
    (Test_helpers.is_cpu_opencl_device dgpu_opencl) ;
  check
    "Native CPU device is NOT CPU-OpenCL"
    false
    (Test_helpers.is_cpu_opencl_device native_cpu) ;
  check
    "Interpreter device is NOT CPU-OpenCL"
    false
    (Test_helpers.is_cpu_opencl_device interpreter) ;
  check
    "Vulkan GPU device is NOT CPU-OpenCL"
    false
    (Test_helpers.is_cpu_opencl_device vulkan_gpu) ;

  print_endline
    "classify_cpu_opencl_math_result, wrong result with the flake shape:" ;
  check "CPU-OpenCL -> Known_issue" "Known_issue" (classify ci_cpu_opencl) ;
  check
    "GPU-OpenCL named like a CPU -> Fail"
    "Fail"
    (classify igpu_named_like_a_cpu) ;
  check "discrete GPU-OpenCL -> Fail" "Fail" (classify dgpu_opencl) ;
  check "Native -> Fail" "Fail" (classify native_cpu) ;
  check "Interpreter -> Fail" "Fail" (classify interpreter) ;
  check "Vulkan -> Fail" "Fail" (classify vulkan_gpu) ;

  (* Failure-SHAPE gate (#74 / F1). Device is held constant at the known-issue
     device, so every `Fail below is the shape gate doing its job. *)
  print_endline
    "classify_cpu_opencl_math_result, shape gate on the CPU-OpenCL device:" ;
  check
    "first bad index 0 (intrinsic-mapping / operand regression) -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:(Some 0) ~bad_count:256 ()) ci_cpu_opencl) ;
  check
    "first bad index 0, partial -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:(Some 0) ~bad_count:60 ()) ci_cpu_opencl) ;
  check
    "first bad index 1 (kernel never executed, 0.0 pre-fill) -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:(Some 1) ~bad_count:255 ()) ci_cpu_opencl) ;
  check
    "first bad index 3 (still inside the scalar prologue) -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:(Some 3) ~bad_count:60 ()) ci_cpu_opencl) ;
  check
    "non-finite (NaN/inf) result -> Fail"
    "Fail"
    (classify_shape
       (shape ~first_bad:(Some 4) ~bad_count:60 ~non_finite:true ())
       ci_cpu_opencl) ;
  check
    "all elements wrong (total failure) -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:(Some 4) ~bad_count:256 ()) ci_cpu_opencl) ;
  check
    "first bad index >= 4, finite, partial (the flake) -> Known_issue"
    "Known_issue"
    (classify_shape (shape ~first_bad:(Some 4) ~bad_count:60 ()) ci_cpu_opencl) ;
  check
    "first bad index 128, finite, partial -> Known_issue"
    "Known_issue"
    (classify_shape
       (shape ~first_bad:(Some 128) ~bad_count:128 ())
       ci_cpu_opencl) ;
  check
    "bad_count > 0 but no first_bad_index (inconsistent shape) -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:None ~bad_count:60 ()) ci_cpu_opencl) ;
  (* Ordering guard: non_finite must be tested BEFORE the bad_count = 0 -> Pass
     branch. This shape is unreachable today, because a non-finite value always
     lands in bad_count too - but only as an incidental property of
     compute_float_check_shape (drop its Float.is_nan guard and this is exactly
     the shape a NaN buffer produces). Pins the structural guarantee instead of
     relying on that implication holding forever. *)
  check
    "non_finite with bad_count 0 (unreachable today) -> Fail, not Pass"
    "Fail"
    (classify_shape
       (shape ~first_bad:None ~bad_count:0 ~non_finite:true ())
       ci_cpu_opencl) ;
  check
    "non_finite with bad_count 0 on a GPU device -> Fail"
    "Fail"
    (classify_shape
       (shape ~first_bad:None ~bad_count:0 ~non_finite:true ())
       dgpu_opencl) ;
  (* The flake shape on a non-known-issue device is still a hard failure: both
     gates are required, neither alone suffices. *)
  check
    "flake shape on the discrete GPU -> Fail"
    "Fail"
    (classify_shape (shape ~first_bad:(Some 4) ~bad_count:60 ()) dgpu_opencl) ;

  (* pocl carve-out (#79). pocl is CL_DEVICE_TYPE=CPU on OpenCL, so it passes
     is_cpu_opencl_device; if it were not carved out, the exact flake shape
     would be excused on it and adding a conformant ICD to the CI image would
     buy no coverage. Both naming eras must Fail. *)
  print_endline "pocl (conformant CPU ICD) is never excused:" ;
  check
    "flake shape on pocl (pthread- name) -> Fail"
    "Fail"
    (classify_shape
       (shape ~first_bad:(Some 4) ~bad_count:60 ())
       pocl_cpu_pthread) ;
  check
    "flake shape on pocl (cpu- name) -> Fail"
    "Fail"
    (classify_shape
       (shape ~first_bad:(Some 4) ~bad_count:60 ())
       pocl_cpu_modern) ;
  check
    "is_pocl_device on pocl (pthread-) -> true"
    true
    (Test_helpers.is_pocl_device pocl_cpu_pthread) ;
  check
    "is_pocl_device on pocl (cpu-) -> true"
    true
    (Test_helpers.is_pocl_device pocl_cpu_modern) ;
  check
    "is_pocl_device on the Intel oneAPI CPU device -> false"
    false
    (Test_helpers.is_pocl_device ci_cpu_opencl) ;
  check
    "is_pocl_device on Native -> false"
    false
    (Test_helpers.is_pocl_device native_cpu) ;
  check
    "is_pocl_device on a GPU whose name starts with a pocl prefix -> false"
    false
    (Test_helpers.is_pocl_device gpu_named_like_pocl) ;
  check
    "is_pocl_device on the iGPU named like a CPU -> false"
    false
    (Test_helpers.is_pocl_device igpu_named_like_a_cpu) ;

  print_endline "classify_cpu_opencl_math_result, correct result:" ;
  check "CPU-OpenCL -> Pass" "Pass" (classify_ok ci_cpu_opencl) ;
  check "Native -> Pass" "Pass" (classify_ok native_cpu) ;
  check
    "not-verified shape (--no-verify) -> Pass"
    "Pass"
    (classify_shape Test_helpers.float_check_not_verified ci_cpu_opencl) ;

  (* F2 expiry signal. The NOTE is what lets the suppression die once the CI ICD
     is fixed, so its presence AND its absence are both part of the contract. *)
  print_endline "expiry NOTE on the `Pass path:" ;
  check
    "correct result on the known-issue device -> NOTE printed"
    true
    (note_printed ~dev:ci_cpu_opencl ~shape:clean_shape) ;
  check
    "correct result on a GPU device -> no NOTE"
    false
    (note_printed ~dev:dgpu_opencl ~shape:clean_shape) ;
  check
    "correct result on Native -> no NOTE"
    false
    (note_printed ~dev:native_cpu ~shape:clean_shape) ;
  (* pocl is not the device whose defect the suppression tracks, so a correct
     result from it is not an expiry signal for the Intel flake. *)
  check
    "correct result on pocl -> no NOTE"
    false
    (note_printed ~dev:pocl_cpu_pthread ~shape:clean_shape) ;
  check
    "--no-verify on the known-issue device -> no NOTE (nothing was compared)"
    false
    (note_printed
       ~dev:ci_cpu_opencl
       ~shape:Test_helpers.float_check_not_verified) ;
  check
    "flake shape on the known-issue device -> no NOTE (not a correct result)"
    false
    (note_printed ~dev:ci_cpu_opencl ~shape:flake_shape) ;

  if !failures = 0 then print_endline "\nAll classifier checks PASSED"
  else begin
    Printf.printf "\n%d classifier check(s) FAILED\n" !failures ;
    exit 1
  end
