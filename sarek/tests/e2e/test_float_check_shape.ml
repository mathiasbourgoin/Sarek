(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Fixture test for Test_helpers.compute_float_check_shape - the shape
 * computation that feeds the CPU-OpenCL KNOWN-ISSUE classifier (#74 / F1).
 *
 * test_cpu_opencl_known_issue.ml covers the CLASSIFIER: given a shape, does it
 * reach the right verdict. Nothing covered the other half: does the verifier
 * compute the right SHAPE in the first place. If first_bad_index, bad_count or
 * non_finite were miscomputed, the classifier would be fed garbage and the
 * masking hole would reopen silently, with no test failing - e.g. a
 * first_bad_index that reported the last mismatch instead of the first would
 * let an all-wrong buffer look like the lane-boundary flake and be excused.
 *
 * compute_float_check_shape is the single source of truth: both
 * test_float32_sin_pure.verify_result and
 * test_math_intrinsics.verify_float_arrays delegate to it, contributing only
 * their tolerance, their reference values and their print format. So testing it
 * once covers both verifiers' shape logic.
 *
 * Array-in / record-out: no device, no backend, runs everywhere.
 *
 * Run with: dune exec sarek/tests/e2e/test_float_check_shape.exe
 ******************************************************************************)

let failures = ref 0

let check name expected actual =
  if expected = actual then Printf.printf "  OK   %s\n" name
  else begin
    Printf.printf "  FAIL %s: expected %s, got %s\n" name expected actual ;
    incr failures
  end

let show (s : Test_helpers.float_check_shape) =
  Printf.sprintf
    "{first_bad_index=%s; bad_count=%d; total=%d; non_finite=%b}"
    (match s.first_bad_index with None -> "None" | Some i -> string_of_int i)
    s.bad_count
    s.total
    s.non_finite

let expect ~first_bad ~bad_count ~total ~non_finite =
  show {first_bad_index = first_bad; bad_count; total; non_finite}

let tolerance = 1e-4

(* Reference values: an arbitrary but non-degenerate signal, so a mismatch is
   never masked by expected = got = 0. *)
let total = 16

let reference = Array.init total (fun i -> sin (float_of_int i *. 0.37) +. 1.5)

(** Run the shape computation over [got] vs [reference]. Also counts the
    [report] callbacks, to pin the "print at most [max_reported_mismatches]"
    contract that keeps a totally-wrong buffer from flooding a CI log. *)
let shape_of ?(tolerance = tolerance) got =
  let reported = ref 0 in
  let s =
    Test_helpers.compute_float_check_shape
      ~total
      ~tolerance
      ~expected:(fun i -> reference.(i))
      ~got:(fun i -> got.(i))
      ~report:(fun ~index:_ ~expected:_ ~got:_ ~diff:_ -> incr reported)
  in
  (s, !reported)

let case ?tolerance name ~mutate ~want =
  let got = Array.copy reference in
  mutate got ;
  let s, _ = shape_of ?tolerance got in
  check name want (show s)

let () =
  print_endline "=== compute_float_check_shape fixture table ===" ;

  case
    "all elements correct -> bad_count 0, no first_bad_index, finite"
    ~mutate:(fun _ -> ())
    ~want:(expect ~first_bad:None ~bad_count:0 ~total ~non_finite:false) ;

  case
    "single wrong element at index 0"
    ~mutate:(fun a -> a.(0) <- a.(0) +. 1.0)
    ~want:(expect ~first_bad:(Some 0) ~bad_count:1 ~total ~non_finite:false) ;

  case
    "single wrong element at index 4"
    ~mutate:(fun a -> a.(4) <- a.(4) +. 1.0)
    ~want:(expect ~first_bad:(Some 4) ~bad_count:1 ~total ~non_finite:false) ;

  (* The real CPU-OpenCL flake shape: scalar prologue intact, vectorised body
     damaged from the 4-wide lane boundary onward. *)
  case
    "wrong from index 4 onward (partial, the flake shape)"
    ~mutate:(fun a ->
      for i = 4 to total - 1 do
        a.(i) <- a.(i) +. 1.0
      done)
    ~want:
      (expect
         ~first_bad:(Some 4)
         ~bad_count:(total - 4)
         ~total
         ~non_finite:false) ;

  (* first_bad_index must be the FIRST mismatch, not the last: this is what
     distinguishes a dead kernel from the flake. *)
  case
    "wrong at indices 1 and 9 -> first_bad_index is 1, not 9"
    ~mutate:(fun a ->
      a.(1) <- a.(1) +. 1.0 ;
      a.(9) <- a.(9) +. 1.0)
    ~want:(expect ~first_bad:(Some 1) ~bad_count:2 ~total ~non_finite:false) ;

  case
    "all elements wrong -> bad_count = total"
    ~mutate:(fun a -> Array.iteri (fun i v -> a.(i) <- v +. 1.0) a)
    ~want:(expect ~first_bad:(Some 0) ~bad_count:total ~total ~non_finite:false) ;

  (* A zeroed buffer is what a kernel that never executed leaves behind. *)
  case
    "all-zeros buffer (kernel never executed) -> bad_count = total"
    ~mutate:(fun a -> Array.fill a 0 total 0.0)
    ~want:(expect ~first_bad:(Some 0) ~bad_count:total ~total ~non_finite:false) ;

  (* NaN: every comparison against NaN is false, so `diff > tolerance` alone
     silently ACCEPTS a NaN. The is_nan guard must both count it as wrong AND
     raise non_finite. *)
  case
    "NaN in the result -> counted wrong AND non_finite"
    ~mutate:(fun a -> a.(6) <- Float.nan)
    ~want:(expect ~first_bad:(Some 6) ~bad_count:1 ~total ~non_finite:true) ;

  case
    "all-NaN result -> bad_count = total AND non_finite"
    ~mutate:(fun a -> Array.fill a 0 total Float.nan)
    ~want:(expect ~first_bad:(Some 0) ~bad_count:total ~total ~non_finite:true) ;

  case
    "+inf in the result -> counted wrong AND non_finite"
    ~mutate:(fun a -> a.(7) <- Float.infinity)
    ~want:(expect ~first_bad:(Some 7) ~bad_count:1 ~total ~non_finite:true) ;

  case
    "-inf in the result -> counted wrong AND non_finite"
    ~mutate:(fun a -> a.(7) <- Float.neg_infinity)
    ~want:(expect ~first_bad:(Some 7) ~bad_count:1 ~total ~non_finite:true) ;

  (* Tolerance boundary, approached from a real signal value (magnitude ~1.5).
     Note this cannot pin the boundary EXACTLY: at that magnitude
     [v +. tolerance -. v] does not recover [tolerance] bit-for-bit, so the
     exact-boundary case below uses exact arithmetic instead. *)
  case
    "diff just inside tolerance -> counted correct"
    ~mutate:(fun a -> a.(3) <- a.(3) +. (tolerance *. 0.99))
    ~want:(expect ~first_bad:None ~bad_count:0 ~total ~non_finite:false) ;

  case
    "diff just outside tolerance -> counted wrong"
    ~mutate:(fun a -> a.(3) <- a.(3) +. (tolerance *. 1.01))
    ~want:(expect ~first_bad:(Some 3) ~bad_count:1 ~total ~non_finite:false) ;

  case
    "negative-direction diff just outside tolerance -> counted wrong"
    ~mutate:(fun a -> a.(3) <- a.(3) -. (tolerance *. 1.01))
    ~want:(expect ~first_bad:(Some 3) ~bad_count:1 ~total ~non_finite:false) ;

  (* Exact boundary. The predicate is [diff > tolerance], so a diff of exactly
     the tolerance is CORRECT and the very next representable float above it is
     WRONG. Compared against 0.0 so the subtraction is exact and the diff really
     is the intended value, bit-for-bit. *)
  print_endline "exact tolerance boundary (expected = 0.0):" ;
  let boundary name got want =
    let s =
      Test_helpers.compute_float_check_shape
        ~total:1
        ~tolerance
        ~expected:(fun _ -> 0.0)
        ~got:(fun _ -> got)
        ~report:(fun ~index:_ ~expected:_ ~got:_ ~diff:_ -> ())
    in
    check name want (show s)
  in
  boundary
    "diff exactly = tolerance -> correct (predicate is diff > tol)"
    tolerance
    (expect ~first_bad:None ~bad_count:0 ~total:1 ~non_finite:false) ;
  boundary
    "diff one ulp above tolerance -> wrong"
    (Float.succ tolerance)
    (expect ~first_bad:(Some 0) ~bad_count:1 ~total:1 ~non_finite:false) ;
  boundary
    "diff one ulp below tolerance -> correct"
    (Float.pred tolerance)
    (expect ~first_bad:None ~bad_count:0 ~total:1 ~non_finite:false) ;
  boundary
    "negative diff exactly = tolerance -> correct (absolute value)"
    (-.tolerance)
    (expect ~first_bad:None ~bad_count:0 ~total:1 ~non_finite:false) ;
  boundary
    "negative diff one ulp above tolerance -> wrong (absolute value)"
    (-.Float.succ tolerance)
    (expect ~first_bad:(Some 0) ~bad_count:1 ~total:1 ~non_finite:false) ;

  (* The tolerance is honoured as given, not hard-coded. *)
  case
    "a wider tolerance accepts what the default rejects"
    ~tolerance:1.0
    ~mutate:(fun a -> a.(3) <- a.(3) +. 0.5)
    ~want:(expect ~first_bad:None ~bad_count:0 ~total ~non_finite:false) ;

  (* Mismatch reporting is capped, so a totally-wrong buffer cannot flood CI. *)
  print_endline "report callback cap:" ;
  let all_wrong = Array.map (fun v -> v +. 1.0) reference in
  let s, reported = shape_of all_wrong in
  check
    "all 16 wrong -> bad_count 16 but only max_reported_mismatches reported"
    (Printf.sprintf "16/%d" Test_helpers.max_reported_mismatches)
    (Printf.sprintf "%d/%d" s.bad_count reported) ;
  let s1, reported1 = shape_of (Array.copy reference) in
  check
    "no mismatches -> nothing reported"
    "0/0"
    (Printf.sprintf "%d/%d" s1.bad_count reported1) ;

  (* An empty comparison must not be mistaken for a failure. *)
  print_endline "degenerate input:" ;
  let empty =
    Test_helpers.compute_float_check_shape
      ~total:0
      ~tolerance
      ~expected:(fun _ -> 0.0)
      ~got:(fun _ -> 0.0)
      ~report:(fun ~index:_ ~expected:_ ~got:_ ~diff:_ -> ())
  in
  check
    "total 0 -> clean shape"
    (expect ~first_bad:None ~bad_count:0 ~total:0 ~non_finite:false)
    (show empty) ;

  if !failures = 0 then print_endline "\nAll float_check_shape checks PASSED"
  else begin
    Printf.printf "\n%d float_check_shape check(s) FAILED\n" !failures ;
    exit 1
  end
