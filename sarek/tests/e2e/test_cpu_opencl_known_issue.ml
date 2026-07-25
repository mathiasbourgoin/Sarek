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
    supports_fp64 = true;
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

(* A wrong result (within_tol:false) on each device. *)
let classify dev =
  verdict_name
    (Test_helpers.classify_cpu_opencl_math_result ~dev ~within_tol:false ())

(* A correct result must always be `Pass, even on the known-issue device. *)
let classify_ok dev =
  verdict_name
    (Test_helpers.classify_cpu_opencl_math_result ~dev ~within_tol:true ())

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

  print_endline "classify_cpu_opencl_math_result, wrong result:" ;
  check "CPU-OpenCL -> Known_issue" "Known_issue" (classify ci_cpu_opencl) ;
  check
    "GPU-OpenCL named like a CPU -> Fail"
    "Fail"
    (classify igpu_named_like_a_cpu) ;
  check "discrete GPU-OpenCL -> Fail" "Fail" (classify dgpu_opencl) ;
  check "Native -> Fail" "Fail" (classify native_cpu) ;
  check "Interpreter -> Fail" "Fail" (classify interpreter) ;
  check "Vulkan -> Fail" "Fail" (classify vulkan_gpu) ;

  print_endline "classify_cpu_opencl_math_result, correct result:" ;
  check "CPU-OpenCL -> Pass" "Pass" (classify_ok ci_cpu_opencl) ;
  check "Native -> Pass" "Pass" (classify_ok native_cpu) ;

  if !failures = 0 then print_endline "\nAll classifier checks PASSED"
  else begin
    Printf.printf "\n%d classifier check(s) FAILED\n" !failures ;
    exit 1
  end
