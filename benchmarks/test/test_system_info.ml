(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for System_info.gpu_vendor_of -- the vendor half of the machine
 * label.
 *
 * Why this is worth testing at all: since backlog-168 the machine label is
 * what replaced the hostname. It is the dedup key AND the leading token of
 * every result filename. If it is not a stable function of the hardware, one
 * machine reappears under several labels, dedup stops working, and the
 * filenames stop grouping -- with no error anywhere.
 *
 * Two failure modes are pinned here, both of which the original implementation
 * had:
 *   1. a CPU exposed as an OpenCL device speaking for the machine, because it
 *      was excluded only by exact name equality with cpu_model;
 *   2. dependence on device ENUMERATION ORDER, because the first matching
 *      device won.
 ******************************************************************************)

open Benchmark_common.System_info

let dev ?(backend = "OpenCL") ?(id = 0) name =
  {
    id;
    name;
    framework = backend;
    compute_capability = None;
    memory_gb = 1.0;
    driver_version = None;
    runtime_version = None;
  }

let failures = ref 0

let expect label ~devices ~cpu_model ~want =
  let got = gpu_vendor_of devices cpu_model in
  if got = want then Printf.printf "  ok   %s -> %s\n" label got
  else begin
    Printf.printf "  FAIL %s -> got %S, wanted %S\n" label got want ;
    incr failures
  end

(* --- a CPU device must not speak for the machine ------------------------- *)

(* The exact-equality exclusion this replaces let every one of these through:
   an OpenCL CPU device's name and /proc/cpuinfo's "model name" routinely
   differ by whitespace runs or trailing text, and the CPU brand string
   contains a vendor token, so the CPU outvoted the real GPU. *)
let test_cpu_device_does_not_outvote_a_gpu () =
  let cpu_model = "AMD Ryzen 9 7950X 16-Core Processor" in
  expect
    "cpu device named exactly as cpu_model"
    ~devices:[dev cpu_model; dev "NVIDIA GeForce RTX 3090"]
    ~cpu_model
    ~want:"nvidia" ;
  expect
    "cpu device differing by whitespace runs"
    ~devices:
      [
        dev "  AMD  Ryzen 9   7950X 16-Core Processor ";
        dev "NVIDIA GeForce RTX 3090";
      ]
    ~cpu_model
    ~want:"nvidia" ;
  expect
    "cpu device with trailing text"
    ~devices:
      [
        dev "AMD Ryzen 9 7950X 16-Core Processor (with SSE4.2)";
        dev "NVIDIA GeForce RTX 3090";
      ]
    ~cpu_model
    ~want:"nvidia" ;
  expect
    "cpu-backed device excluded by backend, whatever its name"
    ~devices:
      [dev ~backend:"Native" "Intel Arc Pretender"; dev "Radeon RX 7900 XTX"]
    ~cpu_model:"some other cpu"
    ~want:"amd" ;
  expect
    "interpreter device excluded by backend"
    ~devices:
      [
        dev ~backend:"Interpreter" "Intel Arc Pretender";
        dev "Radeon RX 7900 XTX";
      ]
    ~cpu_model:"some other cpu"
    ~want:"amd"

(* --- the label must not depend on enumeration order ---------------------- *)

(* Drivers do not promise a stable device order between runs. When the first
   match won, a discrete + integrated box derived a different label depending
   on which device was enumerated first -- so the SAME machine split across two
   labels, two filename groups and two dedup keys. *)
let test_order_independence () =
  let discrete = dev ~id:0 "NVIDIA GeForce RTX 3090" in
  let integrated = dev ~id:1 "Intel Arc Graphics" in
  let cpu_model = "Intel(R) Core(TM) i9-13900K" in
  expect
    "discrete first"
    ~devices:[discrete; integrated]
    ~cpu_model
    ~want:"nvidia" ;
  expect
    "integrated first -- must be the SAME label"
    ~devices:[integrated; discrete]
    ~cpu_model
    ~want:"nvidia" ;
  (* Every permutation of a three-device set must agree. *)
  let amd = dev ~id:2 "Radeon RX 7900 XTX" in
  List.iter
    (fun (name, devices) -> expect name ~devices ~cpu_model ~want:"nvidia")
    [
      ("perm nvidia,intel,amd", [discrete; integrated; amd]);
      ("perm nvidia,amd,intel", [discrete; amd; integrated]);
      ("perm intel,nvidia,amd", [integrated; discrete; amd]);
      ("perm intel,amd,nvidia", [integrated; amd; discrete]);
      ("perm amd,nvidia,intel", [amd; discrete; integrated]);
      ("perm amd,intel,nvidia", [amd; integrated; discrete]);
    ]

(* --- the ordinary cases must keep working ------------------------------- *)

(* Pinned green so the CPU exclusion above cannot be tightened into excluding
   everything: on Apple Silicon the GPU name legitimately resembles the CPU. *)
let test_single_vendor_machines () =
  expect
    "apple silicon: gpu name resembles the cpu but is a real GPU device"
    ~devices:[dev ~backend:"Metal" "Apple M4 Max"]
    ~cpu_model:"Apple M4 Max"
    ~want:"apple" ;
  expect
    "nvidia only"
    ~devices:[dev ~backend:"CUDA" "NVIDIA GeForce RTX 4090"]
    ~cpu_model:"Intel(R) Core(TM) i9-13900K"
    ~want:"nvidia" ;
  expect
    "no devices at all"
    ~devices:[]
    ~cpu_model:"Intel(R) Core(TM) i9-13900K"
    ~want:"unknown" ;
  expect
    "no recognisable vendor"
    ~devices:[dev "Some Unlisted Accelerator"]
    ~cpu_model:"Intel(R) Core(TM) i9-13900K"
    ~want:"unknown" ;
  (* "unknown" is get_cpu_info's FAILURE value, not a CPU name. Matching on it
     would exclude every device on a box whose CPU probe failed, turning one
     probe failure into a label change. *)
  expect
    "cpu probe failed: devices are still considered"
    ~devices:[dev "NVIDIA GeForce RTX 3090"]
    ~cpu_model:"unknown"
    ~want:"nvidia"

let () =
  print_endline "test_system_info: gpu_vendor_of" ;
  test_cpu_device_does_not_outvote_a_gpu () ;
  test_order_independence () ;
  test_single_vendor_machines () ;
  if !failures > 0 then begin
    Printf.printf "test_system_info: %d case(s) FAILED\n" !failures ;
    exit 1
  end ;
  print_endline "test_system_info: all cases passed"
