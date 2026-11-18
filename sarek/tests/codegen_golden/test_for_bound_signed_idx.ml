(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * C-8 / C-9 codegen pins, COMPILE-ONLY (no device, no compiler is run; only
 * the pure source generators are called). The run-time differential on HIP
 * lives in sarek-hip/test/test_hip_cfamily_semantics.ml.
 *
 *   C-8  a [for] upper bound that the loop body can change is evaluated once
 *        into a [sarek_for_hi_N] temporary before the loop on CUDA/HIP,
 *        OpenCL, Metal, GLSL and WGSL; a bound the body cannot change keeps its
 *        plain spelling (so existing generated source is stable).
 *   C-9  on CUDA/HIP, OpenCL and Metal every int32 thread/block/grid index is
 *        spelled with a signed [(int)] cast (GLSL/WGSL already cast).
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

let var ?(mutable_ = false) name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = mutable_}

let i32 n = EConst (CInt32 (Int32.of_int n))

let intrinsic name = EIntrinsic ([], name, [])

(* [let m = mut 3 in let count = mut 0 in
    for j = 0 to m do m := m - 1; count := count + 1 done;
    for k = 0 to n - 1 do count := count + 1 done;
    out.(0) <- (thread_idx_x - 1 < 0 ? 1 : 0);
    out.(1) <- (thread_idx_x - 5) / 2] *)
let probe_kernel () =
  let out = var "out" (TVec TInt32) in
  let n = var "n" TInt32 in
  let m = var ~mutable_:true "m" TInt32 in
  let count = var ~mutable_:true "count" TInt32 in
  let j = var ~mutable_:true "j" TInt32 in
  let k = var ~mutable_:true "k" TInt32 in
  let ev v = EVar {v with var_mutable = false} in
  let bump_count = SAssign (LVar count, EBinop (Add, ev count, i32 1)) in
  let mutating_bound_loop =
    SFor
      ( j,
        i32 0,
        ev m,
        Upto,
        SSeq [SAssign (LVar m, EBinop (Sub, ev m, i32 1)); bump_count] )
  in
  let stable_bound_loop =
    SFor (k, i32 0, EBinop (Sub, ev n, i32 1), Upto, bump_count)
  in
  let tid = intrinsic "thread_idx_x" in
  let body =
    SLetMut
      ( m,
        i32 3,
        SLetMut
          ( count,
            i32 0,
            SSeq
              [
                mutating_bound_loop;
                stable_bound_loop;
                SAssign
                  ( LArrayElem ("out", i32 0),
                    EIf
                      ( EBinop (Lt, EBinop (Sub, tid, i32 1), i32 0),
                        i32 1,
                        i32 0 ) );
                SAssign
                  ( LArrayElem ("out", i32 1),
                    EBinop (Div, EBinop (Sub, tid, i32 5), i32 2) );
                SAssign
                  ( LArrayElem ("out", i32 2),
                    EBinop
                      ( Add,
                        intrinsic "global_thread_id",
                        EBinop
                          (Add, intrinsic "block_dim_x", intrinsic "grid_dim_x")
                      ) );
                SAssign (LArrayElem ("out", i32 3), intrinsic "global_size");
              ] ) )
  in
  {
    default_kernel with
    kern_name = "for_bound_probe";
    kern_params =
      [
        DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (n, None);
      ];
    kern_body = body;
  }

let contains hay needle =
  let hl = String.length hay and nl = String.length needle in
  let rec go i = i + nl <= hl && (String.sub hay i nl = needle || go (i + 1)) in
  nl = 0 || go 0

let check_has label src needle =
  if not (contains src needle) then
    Alcotest.failf "%s: expected [%s] in:\n%s" label needle src

let check_lacks label src needle =
  if contains src needle then
    Alcotest.failf "%s: did not expect [%s] in:\n%s" label needle src

(* C-8 on one backend: the mutating bound goes through a temporary, the stable
   one does not. A [for] bound keeps C's [-] in the C dialects (index
   arithmetic, Sarek_ir_c_arith); the loop body's int32 arithmetic wraps.
   [decl] is the temporary's declaration, [cond] the loop test. *)
let check_bound_hoist label src ~decl ~cond ~stable ~raw =
  check_has label src decl ;
  check_has label src cond ;
  check_has label src stable ;
  check_lacks label src raw ;
  check_lacks label src "sarek_for_hi_2"

let test_cuda_bound () =
  let src = Sarek_ir_cuda.generate (probe_kernel ()) in
  check_bound_hoist
    "cuda"
    src
    ~decl:"int sarek_for_hi_1 = m;"
    ~cond:"j <= sarek_for_hi_1"
    ~stable:"k <= (n - 1)"
    ~raw:"j <= m"

let test_opencl_bound () =
  let src = Sarek_ir_opencl.generate (probe_kernel ()) in
  check_bound_hoist
    "opencl"
    src
    ~decl:"int sarek_for_hi_1 = m;"
    ~cond:"j <= sarek_for_hi_1"
    ~stable:"k <= (n - 1)"
    ~raw:"j <= m"

let test_metal_bound () =
  let src = Sarek_ir_metal.generate (probe_kernel ()) in
  check_bound_hoist
    "metal"
    src
    ~decl:"int sarek_for_hi_1 = m;"
    ~cond:"j <= sarek_for_hi_1"
    ~stable:"k <= (n - 1)"
    ~raw:"j <= m"

let test_glsl_bound () =
  let src = Sarek_ir_glsl.generate (probe_kernel ()) in
  check_bound_hoist
    "glsl"
    src
    ~decl:"int sarek_for_hi_1 = m;"
    ~cond:"j <= sarek_for_hi_1"
    ~stable:"k <= (n - 1)"
    ~raw:"j <= m"

let test_wgsl_bound () =
  let src = Sarek_ir_wgsl.generate (probe_kernel ()) in
  check_bound_hoist
    "wgsl"
    src
    ~decl:"let sarek_sarek_for_hi_1"
    ~cond:"j <= sarek_sarek_for_hi_1"
    ~stable:"k <= (params.n - 1i)"
    ~raw:"j <= m"

(* C-9. The subtraction itself wraps (Sarek_ir_c_arith): it is computed in
   the unsigned type and converted back, so the operand must already be the
   signed [int] for [x - 1 < 0] to hold at thread 0. *)
let test_cuda_signed () =
  let src = Sarek_ir_cuda.generate (probe_kernel ()) in
  check_has "cuda" src "((int)((unsigned int)((int)threadIdx.x) - 1u))" ;
  check_has "cuda" src "((int)(threadIdx.x + blockIdx.x * blockDim.x))" ;
  check_has "cuda" src "((int)blockDim.x)" ;
  check_has "cuda" src "((int)gridDim.x)" ;
  check_has "cuda" src "((int)(blockDim.x * gridDim.x))"

let test_opencl_signed () =
  let src = Sarek_ir_opencl.generate (probe_kernel ()) in
  check_has "opencl" src "((int)((uint)((int)get_local_id(0)) - 1u))" ;
  check_has "opencl" src "((int)get_global_id(0))" ;
  check_has "opencl" src "((int)get_local_size(0))" ;
  check_has "opencl" src "((int)get_num_groups(0))" ;
  check_has "opencl" src "((int)get_global_size(0))"

let test_metal_signed () =
  let src = Sarek_ir_metal.generate (probe_kernel ()) in
  check_has "metal" src "((int)((uint)((int)__metal_tid.x) - 1u))" ;
  check_has "metal" src "((int)__metal_gid.x)" ;
  check_has "metal" src "((int)__metal_tpg.x)" ;
  check_has "metal" src "((int)__metal_num_groups.x)" ;
  check_has "metal" src "((int)(__metal_tpg.x * __metal_num_groups.x))"

(* GLSL/WGSL were already signed; pin it so they stay that way. *)
let test_glsl_wgsl_signed () =
  check_has
    "glsl"
    (Sarek_ir_glsl.generate (probe_kernel ()))
    "int(gl_LocalInvocationID.x)" ;
  check_has "wgsl" (Sarek_ir_wgsl.generate (probe_kernel ())) "i32(sarek_lid.x)"

(* Unit-level: the stability rule. *)
let test_stable_rule () =
  let x = var "x" TInt32 in
  let ev = EVar x in
  Alcotest.(check bool)
    "constant"
    true
    (Sarek_ir_for_bound.stable ~assigned:[] (i32 4)) ;
  Alcotest.(check bool)
    "unassigned variable"
    true
    (Sarek_ir_for_bound.stable ~assigned:["y"] (EBinop (Sub, ev, i32 1))) ;
  Alcotest.(check bool)
    "assigned variable"
    false
    (Sarek_ir_for_bound.stable ~assigned:["x"] (EBinop (Sub, ev, i32 1))) ;
  Alcotest.(check bool)
    "array element"
    false
    (Sarek_ir_for_bound.stable ~assigned:[] (EArrayRead ("a", i32 0)))

(* A user loop nested in an outer loop whose bound reads the outer variable
   keeps its plain spelling; a hoisted pragma stays on its loop. *)
let test_nested_and_pragma () =
  let i = var ~mutable_:true "i" TInt32 in
  let j = var ~mutable_:true "j" TInt32 in
  let inner =
    SFor (j, i32 0, EVar {i with var_mutable = false}, Upto, SEmpty)
  in
  (match
     Sarek_ir_for_bound.hoist_stmt (SFor (i, i32 0, i32 7, Upto, inner))
   with
  | SFor (_, _, EConst _, Upto, SFor (_, _, EVar v, Upto, SEmpty)) ->
      Alcotest.(check string) "inner bound is the outer variable" "i" v.var_name
  | _ -> Alcotest.fail "nested loop with a stable bound must not be rewritten") ;
  let m = var "m" TInt32 in
  let body = SAssign (LVar m, i32 0) in
  let loop = SFor (j, i32 0, EVar m, Upto, body) in
  match Sarek_ir_for_bound.hoist_stmt (SPragma (["unroll"], loop)) with
  | SLet (tmp, EVar _, SPragma (["unroll"], SFor (_, _, EVar b, Upto, _))) ->
      Alcotest.(check string) "loop reads temporary" tmp.var_name b.var_name
  | _ -> Alcotest.fail "pragma must stay attached to the hoisted loop"

(* Each statement shape that can hide an assignment to the bound variable [m]
   must make the bound unstable: the loop is then wrapped in a temporary. *)
let hoisted_by_assignment_in label body =
  let m = var "m" TInt32 in
  let j = var ~mutable_:true "j" TInt32 in
  match
    Sarek_ir_for_bound.hoist_stmt (SFor (j, i32 0, EVar m, Upto, body m))
  with
  | SLet (tmp, EVar _, SFor (_, _, EVar b, Upto, _)) ->
      Alcotest.(check string) label tmp.var_name b.var_name
  | _ -> Alcotest.failf "%s: bound assigned in the body was not hoisted" label

let assign_m m = SAssign (LVar m, i32 0)

let test_assignment_shapes () =
  let wild = PWild in
  hoisted_by_assignment_in "plain" assign_m ;
  hoisted_by_assignment_in "nested for" (fun m ->
      SFor (var "q" TInt32, i32 0, i32 1, Upto, assign_m m)) ;
  hoisted_by_assignment_in "match arm" (fun m ->
      SMatch (i32 0, [(wild, assign_m m)])) ;
  hoisted_by_assignment_in "if then" (fun m -> SIf (i32 1, assign_m m, None)) ;
  hoisted_by_assignment_in "else branch" (fun m ->
      SIf (i32 1, SEmpty, Some (assign_m m))) ;
  hoisted_by_assignment_in "record field root" (fun m ->
      SAssign (LRecordField (LVar m, "f"), i32 0)) ;
  hoisted_by_assignment_in "while" (fun m -> SWhile (i32 1, assign_m m)) ;
  hoisted_by_assignment_in "block" (fun m -> SBlock (assign_m m)) ;
  hoisted_by_assignment_in "let body" (fun m ->
      SLet (var "t" TInt32, i32 0, assign_m m)) ;
  hoisted_by_assignment_in "pragma" (fun m -> SPragma (["x"], assign_m m))

let test_helper_body_hoisted () =
  let a = var "a" TInt32 in
  let j = var ~mutable_:true "j" TInt32 in
  let loop = SFor (j, i32 0, EArrayRead ("arr", i32 0), Upto, SEmpty) in
  let hf =
    {hf_name = "h"; hf_params = [a]; hf_ret_type = TUnit; hf_body = loop}
  in
  let k = {default_kernel with kern_name = "k"; kern_funcs = [hf]} in
  match (Sarek_ir_for_bound.hoist k).kern_funcs with
  | [{hf_body = SLet (_, _, SFor _); _}] -> ()
  | _ -> Alcotest.fail "helper function loop bound was not hoisted"

(* The start runs before the bound: an impure start is hoisted first, a pure one
   stays in the loop header. *)
let test_start_before_bound () =
  let j = var ~mutable_:true "j" TInt32 in
  let impure = EIntrinsic (["Gpu"], "atomic_add_global_int32", [i32 0]) in
  (match
     Sarek_ir_for_bound.hoist_stmt
       (SFor (j, impure, EArrayRead ("c", i32 0), Upto, SEmpty))
   with
  | SLet (lo, _, SLet (hi, _, SFor (_, EVar l, EVar h, Upto, _))) ->
      Alcotest.(check string) "start temp" lo.var_name l.var_name ;
      Alcotest.(check string) "bound temp" hi.var_name h.var_name ;
      Alcotest.(check bool)
        "start declared before bound"
        true
        (lo.var_name <> hi.var_name)
  | _ -> Alcotest.fail "impure start must be hoisted before the bound") ;
  match
    Sarek_ir_for_bound.hoist_stmt
      (SFor (j, i32 0, EArrayRead ("c", i32 0), Upto, SEmpty))
  with
  | SLet (_, _, SFor (_, EConst _, EVar _, Upto, _)) -> ()
  | _ -> Alcotest.fail "pure start must stay inline"

let test_temp_uses_loop_type () =
  let j = var ~mutable_:true "j" TInt64 in
  match
    Sarek_ir_for_bound.hoist_stmt
      (SFor (j, EConst (CInt64 0L), EArrayRead ("c", i32 0), Upto, SEmpty))
  with
  | SLet (tmp, _, _) ->
      Alcotest.(check bool)
        "temporary has the loop variable type"
        true
        (tmp.var_type = TInt64)
  | _ -> Alcotest.fail "expected a hoisted bound"

(* Every thread/block/grid name, on every C-family emitter, is a signed cast. *)
let test_every_intrinsic_signed () =
  List.iter
    (fun name ->
      let k =
        {
          default_kernel with
          kern_name = "t";
          kern_params =
            [
              DParam
                ( var "out" (TVec TInt32),
                  Some {arr_elttype = TInt32; arr_memspace = Global} );
            ];
          kern_body = SAssign (LArrayElem ("out", i32 0), intrinsic name);
        }
      in
      List.iter
        (fun (label, gen) ->
          check_has (label ^ " " ^ name) (gen k) "out[0] = ((int)")
        [
          ("cuda", Sarek_ir_cuda.generate);
          ("opencl", Sarek_ir_opencl.generate);
          ("metal", Sarek_ir_metal.generate);
        ])
    Sarek_ir_intrinsic_dispatch.thread_intrinsic_names

let () =
  Alcotest.run
    "for_bound_signed_idx"
    [
      ( "C-8 bound evaluated once (compile-only)",
        [
          Alcotest.test_case "cuda" `Quick test_cuda_bound;
          Alcotest.test_case "opencl" `Quick test_opencl_bound;
          Alcotest.test_case "metal" `Quick test_metal_bound;
          Alcotest.test_case "glsl" `Quick test_glsl_bound;
          Alcotest.test_case "wgsl" `Quick test_wgsl_bound;
          Alcotest.test_case "stability rule" `Quick test_stable_rule;
          Alcotest.test_case "nested and pragma" `Quick test_nested_and_pragma;
          Alcotest.test_case "assignment shapes" `Quick test_assignment_shapes;
          Alcotest.test_case "helper body" `Quick test_helper_body_hoisted;
          Alcotest.test_case "start before bound" `Quick test_start_before_bound;
          Alcotest.test_case "temp type" `Quick test_temp_uses_loop_type;
        ] );
      ( "C-9 signed thread intrinsics (compile-only)",
        [
          Alcotest.test_case "cuda" `Quick test_cuda_signed;
          Alcotest.test_case "opencl" `Quick test_opencl_signed;
          Alcotest.test_case "metal" `Quick test_metal_signed;
          Alcotest.test_case
            "every intrinsic"
            `Quick
            test_every_intrinsic_signed;
          Alcotest.test_case
            "glsl and wgsl unchanged"
            `Quick
            test_glsl_wgsl_signed;
        ] );
    ]
