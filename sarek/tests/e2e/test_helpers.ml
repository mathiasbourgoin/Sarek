(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Test helpers for Sarek E2E tests
 *
 * Shared utilities for device selection, verification, and benchmarking.
 * runtime-only version using Spoc_core.
 ******************************************************************************)

open Spoc_core

(** Command line options *)
type config = {
  mutable dev_id : int;
  mutable use_interpreter : bool;
  mutable use_native : bool;
  mutable use_vulkan : bool;
  mutable use_metal : bool;
  mutable benchmark_all : bool;
  mutable benchmark_devices : int list option;
      (** None = all, Some ids = specific *)
  mutable verify : bool;
  mutable size : int;
  mutable block_size : int;
}

let default_config () =
  {
    dev_id = -1;
    (* -1 means run on all devices *)
    use_interpreter = false;
    use_native = false;
    use_vulkan = false;
    use_metal = false;
    benchmark_all = false;
    benchmark_devices = None;
    verify = true;
    size = 1024;
    block_size = 256;
  }

let usage name extra_opts =
  Printf.printf "Usage: %s [options]\n" name ;
  Printf.printf "Options:\n" ;
  Printf.printf "  -d <id>       Device ID (default: all GPU devices)\n" ;
  Printf.printf "  --interpreter Use CPU interpreter device\n" ;
  Printf.printf "  --native      Use native CPU runtime device\n" ;
  Printf.printf "  --vulkan      Use Vulkan device\n" ;
  Printf.printf "  --metal       Use Metal device\n" ;
  Printf.printf "  --benchmark, --benchmark-all  Run on all devices\n" ;
  Printf.printf "  --benchmark-devices <0,1,4>  Run on specific devices\n" ;
  Printf.printf "  -s, --size <size>  Problem size (default: 1024)\n" ;
  Printf.printf "  -b <size>     Block/work-group size (default: 256)\n" ;
  Printf.printf "  -no-verify    Skip result verification\n" ;
  Printf.printf "  -h            Show this help\n" ;
  extra_opts () ;
  exit 0

let parse_args ?(extra = fun _ _ -> false) ?(extra_usage = fun () -> ()) name =
  let cfg = default_config () in
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    let consumed = extra cfg !i in
    (if not consumed then
       match Sys.argv.(!i) with
       | "-d" ->
           incr i ;
           cfg.dev_id <- int_of_string Sys.argv.(!i)
       | "--interpreter" -> cfg.use_interpreter <- true
       | "--native" -> cfg.use_native <- true
       | "--vulkan" -> cfg.use_vulkan <- true
       | "--metal" -> cfg.use_metal <- true
       | "--benchmark" | "--benchmark-all" -> cfg.benchmark_all <- true
       | "--benchmark-devices" ->
           incr i ;
           let ids =
             String.split_on_char ',' Sys.argv.(!i)
             |> List.map String.trim |> List.map int_of_string
           in
           cfg.benchmark_all <- true ;
           cfg.benchmark_devices <- Some ids
       | "-s" | "--size" ->
           incr i ;
           cfg.size <- int_of_string Sys.argv.(!i)
       | "-b" ->
           incr i ;
           cfg.block_size <- int_of_string Sys.argv.(!i)
       | "-no-verify" -> cfg.verify <- false
       | "-h" | "--help" -> usage name extra_usage
       | _ -> ()) ;
    incr i
  done ;
  cfg

(** Initialize runtime devices - uses all registered backends *)
let init_devices _cfg = Device.init ()

(** Get device based on config *)
let get_device cfg devs =
  if cfg.use_native then (
    match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
    | Some d -> d
    | None ->
        print_endline "No native CPU device found" ;
        exit 1)
  else if cfg.use_interpreter then (
    match Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs with
    | Some d -> d
    | None ->
        print_endline "No interpreter device found" ;
        exit 1)
  else if cfg.use_vulkan then (
    match Array.find_opt (fun d -> d.Device.framework = "Vulkan") devs with
    | Some d -> d
    | None ->
        print_endline "No Vulkan device found" ;
        exit 1)
  else if cfg.use_metal then (
    match Array.find_opt (fun d -> d.Device.framework = "Metal") devs with
    | Some d -> d
    | None ->
        print_endline "No Metal device found" ;
        exit 1)
  else if cfg.dev_id >= 0 then devs.(cfg.dev_id)
  else devs.(0)
(* default to first device *)

(** Print available devices *)
let print_devices devs =
  Printf.printf "Available devices:\n" ;
  Array.iteri
    (fun i d ->
      Printf.printf "  [%d] %s (%s)\n" i d.Device.name d.Device.framework)
    devs ;
  flush stdout

(** Run benchmark on selected devices (None = all, Some ids = specific) *)
let benchmark_all ?(device_ids = None) devs run_test name =
  (match device_ids with
  | None -> Printf.printf "\nBenchmark: %s (all devices)\n" name
  | Some ids ->
      Printf.printf
        "\nBenchmark: %s (devices: %s)\n"
        name
        (String.concat ", " (List.map string_of_int ids))) ;
  Printf.printf "%-40s %12s %10s\n" "Device" "Time (ms)" "Status" ;
  Printf.printf "%s\n" (String.make 64 '-') ;
  Array.iteri
    (fun i dev ->
      let should_run =
        match device_ids with None -> true | Some ids -> List.mem i ids
      in
      if should_run then begin
        let dev_name = dev.Device.name in
        flush stdout ;
        let time_ms, ok = run_test dev in
        let status = if ok then "OK" else "FAIL" in
        Printf.printf "%-40s %12.4f %10s\n%!" dev_name time_ms status
      end)
    devs ;
  print_endline "\nBenchmark complete."

(** Run benchmark with pure OCaml baseline, showing speedups *)
let benchmark_with_baseline ?(device_ids = None) devs ~baseline run_test name =
  (match device_ids with
  | None -> Printf.printf "\nBenchmark: %s (all devices)\n" name
  | Some ids ->
      Printf.printf
        "\nBenchmark: %s (devices: %s)\n"
        name
        (String.concat ", " (List.map string_of_int ids))) ;
  Printf.printf "%-40s %12s %10s %10s\n" "Device" "Time (ms)" "Status" "Speedup" ;
  Printf.printf "%s\n" (String.make 76 '-') ;
  (* Run baseline first *)
  let baseline_time, baseline_ok = baseline () in
  Printf.printf
    "%-40s %12.4f %10s %10s\n%!"
    "Pure OCaml (baseline)"
    baseline_time
    (if baseline_ok then "OK" else "FAIL")
    "1.00x" ;
  (* Run on devices *)
  Array.iteri
    (fun i dev ->
      let should_run =
        match device_ids with None -> true | Some ids -> List.mem i ids
      in
      if should_run then begin
        let dev_name = dev.Device.name in
        flush stdout ;
        let time_ms, ok = run_test dev in
        let status = if ok then "OK" else "FAIL" in
        let speedup = if time_ms > 0.0 then baseline_time /. time_ms else 0.0 in
        Printf.printf
          "%-40s %12.4f %10s %9.2fx\n%!"
          dev_name
          time_ms
          status
          speedup
      end)
    devs ;
  print_endline "\nBenchmark complete."

(** Get appropriate block size for device *)
let get_block_size cfg (dev : Device.t) =
  match dev.framework with
  | "OpenCL" ->
      (* CPU OpenCL can use larger work-groups for barrier-based kernels.
         Use cfg.block_size if specified, otherwise default to reasonable size. *)
      if cfg.block_size > 1 then cfg.block_size else 64
  | _ -> cfg.block_size

(** Verify float arrays are approximately equal *)
let verify_float_array expected actual tolerance =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let e = expected.(i) in
    let a = actual.(i) in
    if abs_float (e -. a) > tolerance then begin
      if !errors < 10 then
        Printf.printf "  Mismatch at %d: expected %.6f, got %.6f\n" i e a ;
      incr errors
    end
  done ;
  if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors ;
  !errors = 0

(** Verify int32 arrays are equal *)
let verify_int32_array expected actual =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let e = expected.(i) in
    let a = actual.(i) in
    if e <> a then begin
      if !errors < 10 then
        Printf.printf "  Mismatch at %d: expected %ld, got %ld\n" i e a ;
      incr errors
    end
  done ;
  if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors ;
  !errors = 0

(** Time a function and return (result, time_ms) *)
let time_it f =
  let t0 = Unix.gettimeofday () in
  let result = f () in
  let t1 = Unix.gettimeofday () in
  (result, (t1 -. t0) *. 1000.0)

(* ========================================================================== *)
(* fp64 result classification (shared by the float64 / real64 E2E tests)      *)
(* ========================================================================== *)

(** [true] iff [haystack] contains [needle] as a substring. *)
let string_contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  if nl = 0 then true
  else begin
    let rec go i =
      i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
    in
    go 0
  end

(** [true] iff [(framework, device)] identifies a rusticl (Mesa) OpenCL device.

    The rusticl fp64 KNOWN-ISSUE (div/sqrt at ~single precision, see
    [classify_fp64_result]) is a limitation of exactly this one ICD, so the
    annotation must be gated on rusticl IDENTITY rather than on the generic
    ["OpenCL"] framework tag. Otherwise a conformant NON-rusticl ICD that
    regressed its fp64 div/sqrt into the tolerance envelope would be silently
    masked as "known" instead of FAILing (audit finding #52 / F5).

    We key on the OpenCL device name: rusticl reports its Gallium driver as
    ["radeonsi"] in CL_DEVICE_NAME on the campaign hardware (observed: "AMD
    Radeon RX 7900 XTX (radeonsi, navi31, ...)" and the CPU-socket-named
    "raphael_mendocino" device, which despite its name is the integrated GPU and
    reports CL_DEVICE_TYPE=GPU — verified with clinfo during the #74
    investigation), and the ICD/platform identifies itself as ["rusticl"]. We
    match either token, case-insensitively.

    Why the device name and NOT the [RUSTICL_FEATURES] env var: the run-rules
    export [RUSTICL_FEATURES=fp64] for the WHOLE test process, so the variable
    is present for every device in the run and cannot discriminate a
    co-installed non-rusticl ICD from rusticl within the same process. The
    device name is per-device and can. Name sniffing is admittedly driver-string
    dependent; if a future rusticl build changes CL_DEVICE_NAME this predicate
    must be revisited (a bare non-rusticl over-tolerance simply FAILs, which is
    the safe direction). *)
let is_rusticl_device ~framework ~device =
  framework = "OpenCL"
  &&
  let d = String.lowercase_ascii device in
  string_contains ~needle:"rusticl" d || string_contains ~needle:"radeonsi" d

(** Default relative-error envelope for the rusticl fp64 div/sqrt KNOWN-ISSUE.

    rusticl / Mesa fp64 (with RUSTICL_FEATURES=fp64) computes fp64 division and
    sqrt at only ~single precision while +, -, * stay exact. Measured directly
    with a hand-written OpenCL C repro (briefs/opencl-f64-while-loop-impl.md,
    harness clprobe.c) on both rusticl devices: sqrt/div rel err ~1.8e-8, mul
    ~4e-16. Vulkan/RADV on the same GPU is exact, so this is a driver
    limitation, not a Sarek codegen artefact (evidence: PR #266). *)
let opencl_fp64_transcendental_envelope = 1e-5

(** Classify one fp64 result as [`Pass], the documented rusticl fp64 div/sqrt
    KNOWN-ISSUE, or a genuine [`Fail]. Single source of truth for the fp64 E2E
    tests (audit finding #52 / F4), replacing the constant + classifier + label
    previously copy-pasted across test_real64, test_real64_single_source,
    test_float64_kernel_arith and test_ktype_record_f64_arith.

    - [framework], [device]: the running device's framework tag and name; used
      only to decide rusticl identity via [is_rusticl_device] (F5 gating).
    - [within_tol]: the result met its normal tolerance -> [`Pass] outright.
    - [transcendental]: this result depends on fp64 div/sqrt (the ops rusticl
      computes at ~single precision). Only such results are eligible for the
      KNOWN-ISSUE annotation. A result that uses only +,-,* is never eligible
      and, over tolerance, always [`Fail]s.
    - [exact_ok]: the companion parts that MUST stay exact (e.g. escape-loop
      iteration counts, or add/sub/mul in the same pass) are exact. A violation
      here is a real regression -> [`Fail].
    - [max_rel]: worst relative error of the transcendental part.
    - [non_finite]: a non-finite (NaN/inf) result was observed. This forces
      [`Fail] independently of [max_rel] (a NaN must never fit the envelope).
    - [envelope]: KNOWN-ISSUE ceiling (default
      [opencl_fp64_transcendental_envelope] = 1e-5).
    - [label]: the KNOWN-ISSUE text surfaced (and printed) when annotated.

    A result is annotated [`Known_issue] iff it is over tolerance AND on a
    rusticl device AND transcendental AND its exact companions are exact AND the
    error is finite AND within [envelope]; everything else [`Fail]s. *)
let classify_fp64_result ~framework ~device ~within_tol ~transcendental
    ~exact_ok ~max_rel ~non_finite
    ?(envelope = opencl_fp64_transcendental_envelope) ~label () =
  if within_tol then `Pass
  else if
    is_rusticl_device ~framework ~device
    && transcendental && exact_ok && (not non_finite) && Float.is_finite max_rel
    && max_rel <= envelope
  then `Known_issue label
  else `Fail

(* ========================================================================== *)
(* CPU-OpenCL float32 math-intrinsic classification                           *)
(* ========================================================================== *)

(** [true] iff [dev] is an OpenCL device whose CL_DEVICE_TYPE is CPU.

    Unlike [is_rusticl_device], this predicate does NOT sniff the device name:
    it uses the real OpenCL device-type query. [capabilities.is_cpu] is filled
    from [CL_DEVICE_TYPE & CL_DEVICE_TYPE_CPU] in [sarek-opencl/Opencl_api.ml],
    so the predicate holds for ANY CPU-OpenCL ICD (Intel oneAPI CPU runtime,
    pocl, rusticl-on-llvmpipe, ...) and never for a real GPU, whatever its
    CL_DEVICE_NAME string happens to say. That matters here: the CI runner's
    device reports itself as "AMD EPYC 9V45 96-Core Processor", which matches
    none of the rusticl/radeonsi name tokens, while the two OpenCL devices on
    the campaign workstation are named after the CPU socket ("AMD Ryzen 9 7950X
    16-Core Processor (radeonsi, raphael_mendocino)") yet are
    CL_DEVICE_TYPE=GPU. Name matching would get both cases backwards; the
    device-type query gets both right. *)
let is_cpu_opencl_device (dev : Device.t) =
  Device.is_opencl dev && Device.is_cpu dev

(** KNOWN-ISSUE label for the CPU-OpenCL float32 transcendental flake. *)
let cpu_opencl_float32_math_label =
  "CPU-OpenCL float32 transcendentals (sin/cos/exp/sqrt) are miscompiled by \
   the CI CPU OpenCL runtime on an unrecognised host CPU"

(** Shape of one verified float32 array comparison — what went wrong and where,
    not merely whether anything did.

    [classify_cpu_opencl_math_result] needs this because a device-identity-only
    gate is unsound (audit finding #74 / F1): on a CPU-OpenCL device it would
    turn ANY wrongness of ANY extent into a non-blocking KNOWN-ISSUE, including
    an all-zeros buffer — exactly what a kernel that never ran leaves behind,
    since both tests pre-fill their output vector with 0.0.

    - [first_bad_index]: index of the first element outside tolerance, [None] if
      none.
    - [bad_count]: how many elements are outside tolerance.
    - [total]: how many elements were compared.
    - [non_finite]: a NaN or infinity was seen in the produced or expected
      values. NaN comparisons are false-y, so a verifier must flag them
      explicitly rather than let [diff > tol] silently accept them. *)
type float_check_shape = {
  first_bad_index : int option;
  bad_count : int;
  total : int;
  non_finite : bool;
}

(** The shape of a comparison that was not performed (e.g. [--no-verify]):
    treated exactly like a clean result. *)
let float_check_not_verified =
  {first_bad_index = None; bad_count = 0; total = 0; non_finite = false}

(** How many individual mismatches a verifier prints before going quiet. *)
let max_reported_mismatches = 5

(** Single source of truth for computing a [float_check_shape]: compare [total]
    elements of [got] against [expected] at absolute tolerance [tolerance],
    calling [report] for the first [max_reported_mismatches] mismatches.

    Factored out of the two E2E verifiers (test_float32_sin_pure.verify_result
    and test_math_intrinsics.verify_float_arrays) so the shape that gates the
    CPU-OpenCL KNOWN-ISSUE is computed in exactly ONE place, and so it can be
    fixture-tested device-free (test_float_check_shape.ml). If this miscomputed
    [first_bad_index], [bad_count] or [non_finite] the classifier would silently
    be fed garbage and the #74 / F1 masking hole would reopen with no test
    failing. [expected] and [got] are index functions so callers can compare an
    array against an array or against a computed reference without materialising
    one.

    The [Float.is_nan diff] guard is load-bearing: every comparison against a
    NaN is false, so [diff > tolerance] alone silently ACCEPTS a NaN buffer.
    Non-finite values on either side also raise [non_finite], which the
    classifier treats as never-excusable. *)
let compute_float_check_shape ~total ~tolerance ~expected ~got ~report =
  let errors = ref 0 in
  let first_bad = ref None in
  let non_finite = ref false in
  for i = 0 to total - 1 do
    let e = expected i and g = got i in
    if not (Float.is_finite g && Float.is_finite e) then non_finite := true ;
    let diff = Float.abs (g -. e) in
    if Float.is_nan diff || diff > tolerance then begin
      if !first_bad = None then first_bad := Some i ;
      if !errors < max_reported_mismatches then
        report ~index:i ~expected:e ~got:g ~diff ;
      incr errors
    end
  done ;
  {
    first_bad_index = !first_bad;
    bad_count = !errors;
    total;
    non_finite = !non_finite;
  }

(** Vector lane width of the miscompiled CPU-OpenCL math library (SSE, 4 x
    float32). The observed flake never damages the scalar prologue, so a genuine
    wrong result at an index below this bound is NEVER the known issue. *)
let cpu_opencl_math_lane_width = 4

(** Classify one float32 math-intrinsic result, tolerating the documented
    CPU-OpenCL KNOWN-ISSUE.

    Evidence (PR #282, run 30134740526): the GitHub runner's CPU-OpenCL device
    ("AMD EPYC 9V45 96-Core Processor", Intel oneAPI CPU runtime, which logs
    "SYCL CPU RT Warning: Unknown host CPU") intermittently returns wrong values
    from Float32 sin/cos/exp/sqrt kernels — zeros, neighbouring input elements,
    or garbage magnitudes (got 3041634.0 for an expected -2.34). Both failures
    start at element 4, an SSE 4-wide lane boundary: the scalar prologue is
    correct and the vectorised body is not, i.e. the runtime mis-JITs its vector
    math library when it cannot identify the host CPU. The same commit passed on
    its parent, on a plain re-run, and on Native / Interpreter / GPU devices, so
    it is a device-runtime defect and not a Sarek codegen regression.

    Device identity alone is NOT sufficient to annotate (audit finding #74 / F1,
    and the same discipline [classify_fp64_result] applies for #52 / F5): the
    failure must also match the flake's SHAPE. [`Known_issue] iff ALL of

    - [is_cpu_opencl_device dev] — the ICD really reports CL_DEVICE_TYPE=CPU;
    - [first_bad_index >= cpu_opencl_math_lane_width] (= 4) — the scalar
      prologue is intact. A wrong intrinsic-name mapping or a swapped operand in
      an emitter is wrong from element 0, and a kernel that never executed is
      wrong from element 1 (element 0 accidentally matches for sin, since sin 0
      = 0 = the pre-fill); both must FAIL;
    - [not non_finite] — a NaN/inf result is never this flake;
    - [bad_count < total] — a partially wrong buffer. An all-elements-wrong
      result (dead kernel, dead queue, wrong buffer bound) must always FAIL.

    Everything else [`Fail]s, on every device.

    Suppression review: this KNOWN-ISSUE exists only until the CI OpenCL ICD is
    fixed or replaced (task #74). Re-evaluate by 2027-01-01, or earlier if the
    [`Pass]-on-known-issue-device note below starts appearing in CI logs — that
    note means the device produced a CORRECT result and the suppression may
    already be dead weight masking future regressions. *)
let classify_cpu_opencl_math_result ~dev ~(shape : float_check_shape)
    ?(label = cpu_opencl_float32_math_label) () =
  if shape.bad_count = 0 then begin
    (* F2: the suppression must be able to expire. Announce every correct
       result from a device we would otherwise excuse, so a fixed ICD is
       visible in the log instead of silently keeping the annotation alive. *)
    if is_cpu_opencl_device dev then
      Printf.printf
        "NOTE: known-issue device produced a CORRECT result — re-evaluate the \
         CPU-OpenCL suppression (task #74)\n\
         %!" ;
    `Pass
  end
  else if
    is_cpu_opencl_device dev && (not shape.non_finite)
    && shape.bad_count < shape.total
    &&
    match shape.first_bad_index with
    | Some i -> i >= cpu_opencl_math_lane_width
    | None -> false
  then `Known_issue label
  else `Fail

(** Emit a GitHub Actions warning annotation, so a KNOWN-ISSUE suppression is
    visible on the checks page and not only to whoever opens the raw log (audit
    finding #74 / F3). Outside Actions this is just a line of stdout. *)
let github_warning ~title msg =
  Printf.printf "::warning title=%s::%s\n%!" title msg

module Benchmarks = Benchmarks
