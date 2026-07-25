(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for the f16 DSL element type (#57 slice 1).
 *
 * Three groups:
 *   1. Host storage — an f16 vector really is binary16-backed.
 *   2. Rounding      — Sarek_float16.to_float16 is the shared narrowing used by
 *                      the interpreter, the native path and the Bigarray store,
 *                      so it must agree with the store exactly.
 *   3. Type system   — `float16` resolves as an annotation AND is excluded from
 *                      the numeric/float predicates, which is what forces
 *                      "compute in f32" at the type level.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module F16 = Sarek_interp.Sarek_float16

(* ------------------------------------------------------------------ *)
(* 1. Host storage                                                    *)
(* ------------------------------------------------------------------ *)

let test_host_roundtrip () =
  let v = Vector.create Vector.float16 4 in
  Vector.set v 0 3.14159 ;
  (* 3.14159 is not representable in binary16; the nearest value is 3.140625.
     Reading back 3.14159 would prove the vector is NOT f16-backed. *)
  Alcotest.(check (float 1e-9))
    "3.14159 stores as binary16 3.140625"
    3.140625
    (Vector.get v 0) ;
  Vector.set v 1 1.0 ;
  Vector.set v 2 0.5 ;
  Vector.set v 3 (-2.5) ;
  (* Exactly representable values must survive untouched. *)
  Alcotest.(check (float 0.)) "1.0 exact" 1.0 (Vector.get v 1) ;
  Alcotest.(check (float 0.)) "0.5 exact" 0.5 (Vector.get v 2) ;
  Alcotest.(check (float 0.)) "-2.5 exact" (-2.5) (Vector.get v 3)

let test_host_elem_size () =
  let v = Vector.create Vector.float16 1 in
  Alcotest.(check int)
    "f16 element is 2 bytes"
    2
    (Vector.elem_size (Vector.kind v)) ;
  (* Contrast with f32, so this is a discrimination test and not a tautology. *)
  let f32 = Vector.create Vector.float32 1 in
  Alcotest.(check int)
    "f32 element is 4 bytes"
    4
    (Vector.elem_size (Vector.kind f32))

let test_host_range_edges () =
  let v = Vector.create Vector.float16 4 in
  (* 65504 is the largest finite binary16; above it, binary16 saturates to
     infinity rather than wrapping. *)
  Vector.set v 0 65504.0 ;
  Alcotest.(check (float 0.)) "max finite binary16" 65504.0 (Vector.get v 0) ;
  Vector.set v 1 70000.0 ;
  Alcotest.(check bool) "overflow -> +inf" true (Vector.get v 1 = infinity) ;
  Vector.set v 2 (-70000.0) ;
  Alcotest.(check bool) "overflow -> -inf" true (Vector.get v 2 = neg_infinity) ;
  (* Below the smallest subnormal, binary16 flushes to zero. *)
  Vector.set v 3 1e-10 ;
  Alcotest.(check (float 0.)) "underflow -> 0" 0.0 (Vector.get v 3)

(* ------------------------------------------------------------------ *)
(* 2. Rounding helper agrees with the storage path                    *)
(* ------------------------------------------------------------------ *)

let test_round_matches_store () =
  (* This is the load-bearing invariant of the whole slice: the narrowing the
     interpreter and native paths apply at an ECast MUST be the same narrowing
     the Bigarray.Float16 store applies. If these ever diverge, the interpreter
     stops being a faithful oracle for GPU f16 kernels. *)
  let v = Vector.create Vector.float16 1 in
  let samples =
    [
      3.14159;
      0.1;
      1.0 /. 3.0;
      -0.7;
      1e-5;
      6.0e-8;
      65504.0;
      70000.0;
      -70000.0;
      0.0;
      -0.0;
      1e-10;
      2.7182818284;
      1023.5;
      1024.5;
    ]
  in
  List.iter
    (fun x ->
      Vector.set v 0 x ;
      let stored = Vector.get v 0 in
      let rounded = F16.to_float16 x in
      if not (stored = rounded || (stored <> stored && rounded <> rounded)) then
        Alcotest.failf
          "narrowing disagrees for %.17g: store gave %.17g, to_float16 gave \
           %.17g"
          x
          stored
          rounded)
    samples

let test_round_is_idempotent () =
  (* Rounding an already-binary16 value must be a no-op — otherwise repeated
     store/load cycles would drift. *)
  List.iter
    (fun x ->
      let once = F16.to_float16 x in
      let twice = F16.to_float16 once in
      if not (once = twice || (once <> once && twice <> twice)) then
        Alcotest.failf
          "to_float16 not idempotent at %.17g: %.17g then %.17g"
          x
          once
          twice)
    [3.14159; 0.1; -0.7; 1e-5; 65504.0; 1024.5]

let test_round_is_lossy_where_expected () =
  (* Guard against a to_float16 that silently became the identity function. *)
  Alcotest.(check bool)
    "3.14159 is changed by narrowing"
    true
    (F16.to_float16 3.14159 <> 3.14159) ;
  Alcotest.(check (float 1e-9))
    "and lands on the binary16 neighbour"
    3.140625
    (F16.to_float16 3.14159) ;
  Alcotest.(check (float 0.))
    "an exact value is untouched"
    0.5
    (F16.to_float16 0.5)

(* ------------------------------------------------------------------ *)
(* 2c. Rounding is ROUND-TO-NEAREST-EVEN, not just "lossy"            *)
(* ------------------------------------------------------------------ *)

(* Why these specific constants, and why hard-coded.

   Every assertion above compares one narrowing arm against ANOTHER arm of the
   same shared semantics ([F16.to_float16] vs the [Bigarray.Float16] store), or
   asserts only that narrowing is lossy. That makes them tautological with
   respect to the rounding MODE: two mutant narrowings — ties-away-from-zero and
   round-toward-zero — pass every hard literal in this file, including
   3.14159 -> 3.140625 (truncation lands there too). Nothing pinned a TIE.

   These constants are independent, verified binary16 RNE values. Each is an
   exact tie or a boundary where RNE, ties-away and truncation DISAGREE:

   - 1024.5   -> 1024   : in [1024, 2048) the binary16 spacing is 1, so 1024.5 is
                          an exact tie between 1024 and 1025. RNE picks the even
                          significand (1024); ties-away picks 1025.
   - 2049     -> 2048   : in [2048, 4096) the spacing is 2, so 2049 is an exact
                          tie between 2048 and 2050. RNE picks 2048 (even);
                          ties-away picks 2050.
   - 2051     -> 2052   : also a tie (between 2050 and 2052), but here the EVEN
                          neighbour is the upper one. This is the assertion that
                          separates RNE from "ties-toward-zero": a rule that
                          always rounded ties down would give 2050.
   - 1+2^-11  -> 1      : 2^-11 is exactly half the gap above 1.0 (which is
                          2^-10), so this is a tie between 1.0 and 1 + 2^-10.
                          RNE picks 1.0; ties-away picks 1.0009765625.
   - -0.0     -> -0.0   : sign preservation, checked on the BITS. A narrowing that
                          lost the sign of zero would read as 0.0 = -0.0 under a
                          float comparison. *)

let check_exact name expected got =
  if got <> expected then
    Alcotest.failf
      "%s: expected %.17g, got %.17g (binary16 RNE)"
      name
      expected
      got

let test_round_to_nearest_even_ties () =
  check_exact "1024.5 ties to even 1024" 1024.0 (F16.to_float16 1024.5) ;
  check_exact "2049 ties to even 2048" 2048.0 (F16.to_float16 2049.0) ;
  (* Upper neighbour is the even one here, so this cannot be explained by
     "ties round down". *)
  check_exact "2051 ties to even 2052" 2052.0 (F16.to_float16 2051.0) ;
  check_exact "1 + 2^-11 ties to even 1.0" 1.0 (F16.to_float16 (1.0 +. 0x1p-11)) ;
  (* Non-tie controls immediately either side, so the above cannot be a
     coincidence of some blanket rounding direction. *)
  check_exact "1024.4 rounds down" 1024.0 (F16.to_float16 1024.4) ;
  check_exact "1024.6 rounds up" 1025.0 (F16.to_float16 1024.6) ;
  check_exact "2049.5 rounds up" 2050.0 (F16.to_float16 2049.5)

let test_round_preserves_negative_zero () =
  let z = F16.to_float16 (-0.0) in
  (* Compared on the SIGN BIT: -0.0 = 0.0 as floats, so a float comparison
     cannot see this. *)
  Alcotest.(check bool)
    "-0.0 stays negative zero"
    true
    (Int64.bits_of_float z = Int64.bits_of_float (-0.0)) ;
  Alcotest.(check bool)
    "+0.0 stays positive zero"
    true
    (Int64.bits_of_float (F16.to_float16 0.0) = Int64.bits_of_float 0.0)

let test_store_agrees_on_ties () =
  (* The same ties through the STORAGE path, so the "cast-then-store ==
     store" invariant is pinned at the ties too, not only at non-ties. *)
  let v = Vector.create Vector.float16 1 in
  List.iter
    (fun (x, want) ->
      Vector.set v 0 x ;
      check_exact (Printf.sprintf "store %.17g" x) want (Vector.get v 0))
    [
      (1024.5, 1024.0); (2049.0, 2048.0); (2051.0, 2052.0); (1.0 +. 0x1p-11, 1.0);
    ]

(* ------------------------------------------------------------------ *)
(* 2b. Interpreter ECast narrowing                                     *)
(* ------------------------------------------------------------------ *)

(* This is asserted DIRECTLY on eval_expr rather than through a kernel, and
   deliberately so. End-to-end, an [ECast (TFloat16, _)] whose result is stored
   straight into an f16 vector is indistinguishable from no cast at all, because
   the Bigarray.Float16 store narrows anyway. Amplifying the difference with
   catastrophic cancellation is not a usable alternative either: it would equally
   amplify the interpreter's f64-vs-GPU-f32 intermediate difference and produce
   spurious divergence unrelated to f16.

   So the arm is pinned here, at the one place where it is observable in
   isolation: an f16-typed IR value must BE a binary16 value the moment the cast
   is evaluated, not merely by the time it is stored. *)

module Interp = Sarek.Sarek_ir_interp

let interp_env () =
  {
    Interp.vars = Hashtbl.create 4;
    vars_by_name = Hashtbl.create 4;
    arrays = Hashtbl.create 4;
    shared = Hashtbl.create 4;
    funcs = Hashtbl.create 4;
  }

let interp_state () =
  {
    Interp.thread_idx = (0, 0, 0);
    block_idx = (0, 0, 0);
    block_dim = (1, 1, 1);
    grid_dim = (1, 1, 1);
  }

let eval_f16_cast x =
  let e =
    Sarek_ir_types.ECast
      ( Sarek_ir_types.TFloat16,
        Sarek_ir_types.EConst (Sarek_ir_types.CFloat32 x) )
  in
  match Interp.eval_expr (interp_state ()) (interp_env ()) e with
  | Interp.VFloat32 f -> f
  | _ -> Alcotest.fail "ECast (TFloat16, _) did not evaluate to a float value"

let test_interp_cast_narrows () =
  Alcotest.(check (float 1e-9))
    "ECast to f16 narrows 3.14159 to 3.140625"
    3.140625
    (eval_f16_cast 3.14159) ;
  (* Agreement with the shared narrowing helper, for several magnitudes. *)
  List.iter
    (fun x ->
      let got = eval_f16_cast x in
      let want = F16.to_float16 x in
      if not (got = want || (got <> got && want <> want)) then
        Alcotest.failf
          "ECast (TFloat16) at %.17g gave %.17g, expected %.17g"
          x
          got
          want)
    [0.1; 1.0 /. 3.0; -0.7; 1e-5; 1024.5; 70000.0] ;
  (* And it must NOT be the identity. *)
  Alcotest.(check bool)
    "ECast to f16 is not the identity"
    true
    (eval_f16_cast 3.14159 <> 3.14159)

let test_interp_cast_f32_does_not_narrow () =
  (* Discrimination: an f32 cast must leave the value alone, so the arm above is
     specific to TFloat16 and not a blanket rounding of every cast. *)
  let e =
    Sarek_ir_types.ECast
      ( Sarek_ir_types.TFloat32,
        Sarek_ir_types.EConst (Sarek_ir_types.CFloat32 3.14159) )
  in
  match Interp.eval_expr (interp_state ()) (interp_env ()) e with
  | Interp.VFloat32 f ->
      Alcotest.(check (float 0.)) "f32 cast is value-preserving" 3.14159 f
  | _ -> Alcotest.fail "ECast (TFloat32, _) did not evaluate to a float value"

(* ------------------------------------------------------------------ *)
(* 2d. End-to-end through the interpreter, with an f32 SINK            *)
(* ------------------------------------------------------------------ *)

(* The slice-1 claim that ECast (TFloat16, _) narrowing is "not observable
   end-to-end" is false; it is only unobservable when the sink is itself an f16
   vector, because the Bigarray.Float16 store re-applies the same narrowing and
   masks a missing cast.

   Point the SAME kernel at an f32 vector and the difference is loud. Body:

     out_f32[i] <- float32_of_float16 (float16_of_float32 inp_f32[i]) *. 100.0

   with inp[0] = 3.14159:
     - narrowing present : 3.140625            *. 100 = 314.0625
     - narrowing missing : 3.14159012...(f32)  *. 100 = 314.15899658203125

   That is ~245 binary16 ulps apart at this magnitude (spacing 0.25 in
   [256, 512)) — no tolerance can straddle it. Runs on the DEFAULT runtest alias
   and needs no device, unlike test_hip_f16 which hangs off e2e-hip and builds
   its CPU reference from to_float16 itself. *)

module Ir = Sarek_ir_types

let f16_narrow_to_f32_sink_ir () =
  let out =
    {
      Ir.var_name = "out";
      var_id = 0;
      var_type = Ir.TVec Ir.TFloat32;
      var_mutable = false;
    }
  in
  let inp =
    {
      Ir.var_name = "inp";
      var_id = 1;
      var_type = Ir.TVec Ir.TFloat32;
      var_mutable = false;
    }
  in
  let idx =
    {Ir.var_name = "idx"; var_id = 2; var_type = Ir.TInt32; var_mutable = false}
  in
  let body =
    Ir.SLet
      ( idx,
        Ir.EIntrinsic ([], "global_thread_id", []),
        Ir.SAssign
          ( Ir.LArrayElem ("out", Ir.EVar idx),
            Ir.EBinop
              ( Ir.Mul,
                Ir.ECast
                  ( Ir.TFloat32,
                    Ir.ECast (Ir.TFloat16, Ir.EArrayRead ("inp", Ir.EVar idx))
                  ),
                Ir.EConst (Ir.CFloat32 100.0) ) ) )
  in
  {
    Ir.kern_name = "f16_narrow_f32_sink";
    kern_params =
      [
        Ir.DParam
          (out, Some {Ir.arr_elttype = Ir.TFloat32; arr_memspace = Ir.Global});
        Ir.DParam
          (inp, Some {Ir.arr_elttype = Ir.TFloat32; arr_memspace = Ir.Global});
      ];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let test_interp_e2e_f32_sink () =
  let n = 4 in
  let inp = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  let samples = [|3.14159; 0.1; 1.0 /. 3.0; 1024.5 /. 100.0|] in
  Array.iteri (fun i x -> Vector.set inp i x) samples ;
  for i = 0 to n - 1 do
    Vector.set out i (-1.0)
  done ;
  Sarek.Execute.run_interpreter_vectors
    ~ir:(f16_narrow_to_f32_sink_ir ())
    ~args:[Vec out; Vec inp]
    ~block:(Sarek.Execute.dims1d 4)
    ~grid:(Sarek.Execute.dims1d 1)
    ~parallel:false ;
  let got = Vector.to_array out in
  (* The pinned constant. Independent of to_float16: 3.140625 is the binary16
     neighbour of 3.14159 and 3.140625 *. 100. is exact in f32. *)
  check_exact "3.14159 narrowed then scaled" 314.0625 got.(0) ;
  (* And it is NOT the un-narrowed value. *)
  Alcotest.(check bool)
    "differs from the un-narrowed f32 result"
    true
    (abs_float (got.(0) -. (3.14159 *. 100.0)) > 0.05) ;
  (* The remaining lanes must agree with the narrowing helper scaled in f32, so
     the whole kernel (not just lane 0) went through the cast. *)
  Array.iteri
    (fun i x ->
      if i > 0 then
        let want = F16.to_float16 x *. 100.0 in
        if abs_float (got.(i) -. want) > 1e-4 then
          Alcotest.failf "lane %d: got %.17g, expected %.17g" i got.(i) want)
    samples

(* ------------------------------------------------------------------ *)
(* 2e. GC-root canary for the f16 host pointer                        *)
(* ------------------------------------------------------------------ *)

(* [Spoc_core.Memory.bigarray_void_ptr] rests on ctypes INTERNALS that no public
   API covers, and on a SEMANTIC property of them that a type signature cannot
   express: the fat pointer it builds for a Float16 bigarray must keep that
   bigarray GC-rooted for as long as the pointer is live, exactly as
   [Ctypes.bigarray_start] does for every other kind.

   ctypes' kind GADT has no Float16 arm, so [bigarray_start] raises
   Failure "Unsupported bigarray kind" and the f16 arm has to reconstruct the
   pointer over the kind-independent [Ctypes_bigarray.unsafe_address]. The first
   version of that arm used [Ctypes.ptr_of_raw_address], which is
   [make_unmanaged] — it produced a correct ADDRESS and silently dropped the
   root, which is a device->host write into possibly-freed memory. The fix
   rebuilds ctypes' own shape,
   [Fat.make ~managed:(Some (Obj.repr ba)) ~reftyp:Void addr].

   If a future ctypes changes what [Fat.make]'s [managed] field does while
   keeping its signature, nothing else in the tree notices: the code still
   compiles and still produces the right address. This canary is what notices.

   The ASYMMETRY is the whole claim, so both halves are asserted: managed must
   survive a major GC, unmanaged must not. Without the negative half, a test
   environment where finalisers simply never run would pass vacuously. *)

let alloc_f16 () = Bigarray.Array1.create Bigarray.Float16 Bigarray.c_layout 64

(* Each pointer is built in its own function so the bigarray cannot stay live in
   a caller stack slot: the ONLY thing that can root it afterwards is the
   returned pointer itself. *)

let[@inline never] ptr_via_memory collected =
  let ba = alloc_f16 () in
  Gc.finalise (fun _ -> collected := true) ba ;
  Spoc_core.Memory.bigarray_void_ptr ba

let[@inline never] ptr_unmanaged collected =
  let ba = alloc_f16 () in
  Gc.finalise (fun _ -> collected := true) ba ;
  Ctypes.ptr_of_raw_address (Ctypes_bigarray.unsafe_address ba)

let[@inline never] no_ptr_at_all collected =
  let ba = alloc_f16 () in
  Gc.finalise (fun _ -> collected := true) ba ;
  Bigarray.Array1.dim ba

(* Drop every OCaml reference to the bigarray, force collection, and report
   whether it was finalised WHILE [keep] is still live — i.e. at the exact
   moment a transfer would be handing the address to the FFI. *)
let collected_with_live_pointer make =
  let collected = ref false in
  let keep = make collected in
  Gc.full_major () ;
  Gc.full_major () ;
  let verdict = !collected in
  ignore (Sys.opaque_identity keep) ;
  verdict

let test_f16_pointer_roots_its_bigarray () =
  (* Positive control: the finaliser mechanism really does fire here, so the
     managed assertion below is not vacuous. *)
  Alcotest.(check bool)
    "control: with no pointer held, the bigarray IS collected"
    true
    (collected_with_live_pointer (fun c -> no_ptr_at_all c)) ;
  (* The negative half: an unmanaged pointer does NOT root its bigarray. This is
     the bug that shipped. *)
  Alcotest.(check bool)
    "ptr_of_raw_address does NOT root the bigarray"
    true
    (collected_with_live_pointer (fun c -> ptr_unmanaged c)) ;
  (* The property under test: the pointer Spoc_core hands to every backend DOES
     root it. *)
  Alcotest.(check bool)
    "Memory.bigarray_void_ptr roots the f16 bigarray across a major GC"
    false
    (collected_with_live_pointer (fun c -> ptr_via_memory c))

let[@inline never] ptr_via_memory_f32 collected =
  let ba = Bigarray.Array1.create Bigarray.Float32 Bigarray.c_layout 64 in
  Gc.finalise (fun _ -> collected := true) ba ;
  Spoc_core.Memory.bigarray_void_ptr ba

let test_non_f16_pointer_still_roots () =
  (* The f16 arm must match the behaviour of the untouched [bigarray_start] path
     it sits next to — that parity is the point of reconstructing ctypes' own
     fat-pointer shape rather than inventing a new one. *)
  Alcotest.(check bool)
    "Memory.bigarray_void_ptr roots an f32 bigarray too (unchanged path)"
    false
    (collected_with_live_pointer (fun c -> ptr_via_memory_f32 c))

let test_ctypes_still_lacks_a_float16_kind () =
  (* WHY the f16 arm exists at all. The day ctypes grows a Float16 arm this
     fails, and the whole function collapses back to [bigarray_start]. *)
  let raised =
    match Ctypes.typ_of_bigarray_kind Bigarray.Float16 with
    | (_ : _ Ctypes.typ) -> false
    | exception Failure _ -> true
  in
  Alcotest.(check bool)
    "ctypes has no Float16 bigarray kind (if this fails, simplify \
     Memory.bigarray_void_ptr)"
    true
    raised ;
  (* Contrast: the kinds ctypes does know still work, so the check is specific. *)
  Alcotest.(check bool)
    "ctypes does know Float32"
    true
    (match Ctypes.typ_of_bigarray_kind Bigarray.Float32 with
    | (_ : _ Ctypes.typ) -> true
    | exception Failure _ -> false)

(* ------------------------------------------------------------------ *)
(* 2f. Launch-time argument check                                      *)
(* ------------------------------------------------------------------ *)

(* [Execute.vector_arg]'s [Vec] is existential, so element types are erased
   before a launch and no OCaml constraint can police them. These assertions
   pin [Execute.check_launch_args] directly — device-free, because the check is
   pure and runs before any backend is touched. *)

module Ex = Sarek.Execute
module Irt = Sarek_ir_types

let mk_param name elt =
  Irt.DParam
    ( {
        Irt.var_name = name;
        var_id = 0;
        var_type = Irt.TVec elt;
        var_mutable = false;
      },
      Some {Irt.arr_elttype = elt; arr_memspace = Irt.Global} )

let mk_scalar_param name ty =
  Irt.DParam
    ({Irt.var_name = name; var_id = 0; var_type = ty; var_mutable = false}, None)

let kern params =
  {
    Irt.kern_name = "argcheck";
    kern_params = params;
    kern_locals = [];
    kern_body = Irt.SEmpty;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let rejects what ir args =
  match Ex.check_launch_args ~kernel:"argcheck" ir args with
  | () -> Alcotest.failf "%s: expected a launch-time rejection, got none" what
  | exception Sarek.Execute_error.Execution_error _ -> ()
  | exception e ->
      Alcotest.failf "%s: wrong exception: %s" what (Printexc.to_string e)

let accepts what ir args =
  match Ex.check_launch_args ~kernel:"argcheck" ir args with
  | () -> ()
  | exception e ->
      Alcotest.failf "%s: unexpected rejection: %s" what (Printexc.to_string e)

let test_argcheck_arity () =
  let ir = kern [mk_param "out" Irt.TFloat32; mk_param "inp" Irt.TFloat32] in
  let v () = Vector.create Vector.float32 4 in
  accepts "matching arity" ir [Ex.Vec (v ()); Ex.Vec (v ())] ;
  (* SHORT list: this is the memory-safety case. Cuda_api/Hip_api size the
     kernel-argument array from List.length args and pass cuLaunchKernel /
     hipModuleLaunchKernel a bare pointer with NO count, so the driver reads as
     many entries as the compiled signature declares — past the end of the
     array. *)
  rejects "short argument list" ir [Ex.Vec (v ())] ;
  rejects "long argument list" ir [Ex.Vec (v ()); Ex.Vec (v ()); Ex.Int 3]

let test_argcheck_arity_does_not_disable_the_rest () =
  (* Regression: the element-type check used to run only [if] the counts
     matched, so a wrong count silently disabled it. Arity is now an error in
     its own right, and a matching count still gets fully checked. *)
  let ir = kern [mk_param "out" Irt.TFloat16; mk_param "inp" Irt.TFloat16] in
  let f32 () = Vector.create Vector.float32 4 in
  let f16 () = Vector.create Vector.float16 4 in
  rejects "f32 for f16, matching arity" ir [Ex.Vec (f32 ()); Ex.Vec (f32 ())] ;
  accepts "f16 for f16" ir [Ex.Vec (f16 ()); Ex.Vec (f16 ())]

let test_argcheck_shape () =
  let ir = kern [mk_param "out" Irt.TFloat32; mk_scalar_param "n" Irt.TInt32] in
  let v () = Vector.create Vector.float32 4 in
  accepts "vector then scalar" ir [Ex.Vec (v ()); Ex.Int 4] ;
  rejects "scalar where a vector is declared" ir [Ex.Int 1; Ex.Int 4] ;
  rejects "vector where a scalar is declared" ir [Ex.Vec (v ()); Ex.Vec (v ())]

let test_argcheck_width_fallback_for_unmappable_kinds () =
  (* [Vector.Char] has no IR constructor. The checker must NOT silently pass it:
     Char holds 1-byte elements while source `char` lowers to TInt32
     (Sarek_lower_ir.elttype_of_typ, PRE-EXISTING), so the device would access
     the buffer through a 4-byte int*. Caught by physical width, not by an IR
     element-type comparison. *)
  let ir = kern [mk_param "buf" Irt.TInt32] in
  let c = Vector.create Vector.char 4 in
  Alcotest.(check int)
    "Char elements are 1 byte"
    1
    (Vector.elem_size (Vector.kind c)) ;
  rejects "Char vector against a TInt32 parameter" ir [Ex.Vec c] ;
  (* An int32 vector against the same parameter is fine, so the rejection is
     about the width and not about the parameter. *)
  accepts
    "int32 vector against a TInt32 parameter"
    ir
    [Ex.Vec (Vector.create Vector.int32 4)]

(* ------------------------------------------------------------------ *)
(* 3. Type system                                                     *)
(* ------------------------------------------------------------------ *)

open Sarek_types

let test_annotation_resolves () =
  (* `float16` as written in a [%kernel] parameter annotation. *)
  let t = type_of_type_expr (Sarek_ast.TEConstr ("float16", [])) in
  Alcotest.(check bool)
    "float16 resolves to TReg Float16"
    true
    (match repr t with TReg Float16 -> true | _ -> false) ;
  (* `float16 vector`, the surface slice 1 actually delivers. *)
  let tv =
    type_of_type_expr
      (Sarek_ast.TEConstr ("vector", [Sarek_ast.TEConstr ("float16", [])]))
  in
  Alcotest.(check bool)
    "float16 vector resolves to TVec (TReg Float16)"
    true
    (match repr tv with
    | TVec e -> ( match repr e with TReg Float16 -> true | _ -> false)
    | _ -> false)

let test_half_is_not_an_alias () =
  (* Deliberate decision: only `float16` is accepted. `half` stays reserved so
     it can be added later without breaking anything, but it must NOT silently
     resolve to something else today. *)
  let t = type_of_type_expr (Sarek_ast.TEConstr ("half", [])) in
  Alcotest.(check bool)
    "half does not resolve to float16"
    false
    (match repr t with TReg Float16 -> true | _ -> false)

let test_f16_is_not_numeric () =
  (* The enforcement mechanism for "storage type, compute in f32": because f16
     is outside these predicates, f16 values cannot be added or fed to math
     intrinsics — a conversion is mandatory. *)
  Alcotest.(check bool) "f16 is not numeric" false (is_numeric t_float16) ;
  Alcotest.(check bool) "f16 is not float" false (is_float t_float16) ;
  Alcotest.(check bool) "f16 is not integer" false (is_integer t_float16) ;
  (* Sanity: the predicates still hold for the widths that DO compute. *)
  Alcotest.(check bool) "f32 is numeric" true (is_numeric t_float32) ;
  Alcotest.(check bool) "f64 is numeric" true (is_numeric t_float64)

let test_bare_float_literal_cannot_be_f16 () =
  (* A bare float literal defaults into the f32/f64 lattice only. Allowing it to
     link to f16 would reintroduce implicit narrowing through the back door. *)
  Alcotest.(check bool)
    "float literal cannot link to f16"
    false
    (float_literal_can_link t_float16) ;
  Alcotest.(check bool)
    "float literal can link to f32"
    true
    (float_literal_can_link t_float32)

let test_f16_unifies_only_with_itself () =
  let ok = function Ok () -> true | Error _ -> false in
  Alcotest.(check bool) "f16 ~ f16" true (ok (unify t_float16 (TReg Float16))) ;
  Alcotest.(check bool)
    "f16 does not unify with f32"
    false
    (ok (unify t_float16 t_float32)) ;
  Alcotest.(check bool)
    "f16 does not unify with f64"
    false
    (ok (unify t_float16 t_float64))

let test_pretty_printer () =
  Alcotest.(check string)
    "float16 prints as float16"
    "float16"
    (Format.asprintf "%a" pp_registered Float16)

let () =
  Alcotest.run
    "float16"
    [
      ( "host_storage",
        [
          Alcotest.test_case "binary16 round-trip" `Quick test_host_roundtrip;
          Alcotest.test_case "element size" `Quick test_host_elem_size;
          Alcotest.test_case
            "overflow/underflow edges"
            `Quick
            test_host_range_edges;
        ] );
      ( "rounding",
        [
          Alcotest.test_case
            "to_float16 agrees with the store"
            `Quick
            test_round_matches_store;
          Alcotest.test_case "idempotent" `Quick test_round_is_idempotent;
          Alcotest.test_case
            "lossy where expected"
            `Quick
            test_round_is_lossy_where_expected;
          Alcotest.test_case
            "ties round to nearest EVEN (independent constants)"
            `Quick
            test_round_to_nearest_even_ties;
          Alcotest.test_case
            "negative zero keeps its sign"
            `Quick
            test_round_preserves_negative_zero;
          Alcotest.test_case
            "the store agrees at the ties too"
            `Quick
            test_store_agrees_on_ties;
        ] );
      ( "interpreter_cast",
        [
          Alcotest.test_case
            "ECast to f16 narrows immediately"
            `Quick
            test_interp_cast_narrows;
          Alcotest.test_case
            "ECast to f32 does not narrow"
            `Quick
            test_interp_cast_f32_does_not_narrow;
        ] );
      ( "interp_e2e",
        [
          Alcotest.test_case
            "f32 sink observes ECast (TFloat16) narrowing"
            `Quick
            test_interp_e2e_f32_sink;
        ] );
      ( "gc_roots",
        [
          Alcotest.test_case
            "f16 host pointer roots its bigarray (managed vs unmanaged)"
            `Quick
            test_f16_pointer_roots_its_bigarray;
          Alcotest.test_case
            "non-f16 pointer still roots"
            `Quick
            test_non_f16_pointer_still_roots;
          Alcotest.test_case
            "ctypes still lacks a Float16 kind"
            `Quick
            test_ctypes_still_lacks_a_float16_kind;
        ] );
      ( "launch_argcheck",
        [
          Alcotest.test_case
            "arity is rejected, not assumed"
            `Quick
            test_argcheck_arity;
          Alcotest.test_case
            "a matching arity is still fully checked"
            `Quick
            test_argcheck_arity_does_not_disable_the_rest;
          Alcotest.test_case "vector/scalar shape" `Quick test_argcheck_shape;
          Alcotest.test_case
            "unmappable kinds fall back to physical width (Char)"
            `Quick
            test_argcheck_width_fallback_for_unmappable_kinds;
        ] );
      ( "type_system",
        [
          Alcotest.test_case
            "float16 annotation resolves"
            `Quick
            test_annotation_resolves;
          Alcotest.test_case
            "half is not an alias"
            `Quick
            test_half_is_not_an_alias;
          Alcotest.test_case
            "f16 is excluded from numeric predicates"
            `Quick
            test_f16_is_not_numeric;
          Alcotest.test_case
            "bare float literal cannot be f16"
            `Quick
            test_bare_float_literal_cannot_be_f16;
          Alcotest.test_case
            "f16 unifies only with itself"
            `Quick
            test_f16_unifies_only_with_itself;
          Alcotest.test_case "pretty printer" `Quick test_pretty_printer;
        ] );
    ]
