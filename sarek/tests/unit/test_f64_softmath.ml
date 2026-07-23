(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Correctness of the PTX software f64 transcendentals (Sarek_ir_ptx_softmath),
    checked against the OCaml stdlib at the IR level.

    The helper bodies are pure scalar IR (SLet/SIf/SReturn over
    EBinop/EIntrinsic/ECast/EIf); this test evaluates them directly with a small
    IEEE-754-faithful interpreter (OCaml floats are f64; fma is Float.fma; the
    bitcasts are Int64.bits_of_float/float_of_bits — the same semantics the PTX
    instructions mov.b64/fma.rn.f64/shr.s64 have) on ~1000 domain points per
    function, asserting max relative error ≤ 1e-12 (≤ 1e-11 for tan). *)

open Sarek_ir_types

(** {1 Scalar IR evaluator} *)

type value = VF of float | VI32 of int32 | VI64 of int64

exception Ret of value

let funcs : (string, helper_func) Hashtbl.t = Hashtbl.create 16

let () =
  List.iter
    (fun hf -> Hashtbl.replace funcs hf.hf_name hf)
    (Sarek_codegen.Sarek_ir_ptx_softmath.all_helpers ())

let as_f = function VF x -> x | _ -> Alcotest.fail "expected f64 value"

let as_i32 = function VI32 n -> n | _ -> Alcotest.fail "expected i32 value"

let vbool b = VI32 (if b then 1l else 0l)

let eval_arith op a b =
  match (op, a, b) with
  | Add, VF x, VF y -> VF (x +. y)
  | Sub, VF x, VF y -> VF (x -. y)
  | Mul, VF x, VF y -> VF (x *. y)
  | Div, VF x, VF y -> VF (x /. y)
  | Add, VI32 x, VI32 y -> VI32 (Int32.add x y)
  | Sub, VI32 x, VI32 y -> VI32 (Int32.sub x y)
  | _ -> Alcotest.fail "eval_arith: unsupported operand types"

let eval_bits op a b =
  match (op, a, b) with
  | Shl, VI64 x, VI32 n -> VI64 (Int64.shift_left x (Int32.to_int n))
  (* PTX shr.s64 is arithmetic, matching Int64.shift_right. *)
  | Shr, VI64 x, VI32 n -> VI64 (Int64.shift_right x (Int32.to_int n))
  | BitAnd, VI64 x, VI64 y -> VI64 (Int64.logand x y)
  | BitOr, VI64 x, VI64 y -> VI64 (Int64.logor x y)
  | BitAnd, VI32 x, VI32 y -> VI32 (Int32.logand x y)
  | BitOr, VI32 x, VI32 y -> VI32 (Int32.logor x y)
  | _ -> Alcotest.fail "eval_bits: unsupported operand types"

let eval_cmp op a b =
  match (a, b) with
  | VF x, VF y -> (
      match op with
      | Gt -> vbool (x > y)
      | Lt -> vbool (x < y)
      | Ge -> vbool (x >= y)
      | Le -> vbool (x <= y)
      | Eq -> vbool (x = y)
      | _ -> Alcotest.fail "eval_cmp: unsupported op")
  | VI32 x, VI32 y -> (
      match op with
      | Gt -> vbool (x > y)
      | Lt -> vbool (x < y)
      | Eq -> vbool (x = y)
      | _ -> Alcotest.fail "eval_cmp: unsupported op")
  | _ -> Alcotest.fail "eval_cmp: mixed operand types"

let eval_cast ty v =
  match (ty, v) with
  (* cvt.rzi.s32.f64: truncation toward zero, like Int32.of_float. *)
  | TInt32, VF x -> VI32 (Int32.of_float x)
  (* cvt.u32.u64: low 32 bits. *)
  | TInt32, VI64 x -> VI32 (Int64.to_int32 x)
  (* cvt.s64.s32: sign extension. *)
  | TInt64, VI32 x -> VI64 (Int64.of_int32 x)
  (* cvt.rn.f64.s32 *)
  | TFloat64, VI32 x -> VF (Int32.to_float x)
  | _ -> Alcotest.fail "eval_cast: unsupported cast"

let eval_intrinsic name args =
  match (name, args) with
  | "fma", [VF a; VF b; VF c] -> VF (Float.fma a b c)
  | "floor", [VF a] -> VF (Float.floor a)
  | "fabs", [VF a] -> VF (Float.abs a)
  | "min", [VF a; VF b] -> VF (Float.min a b)
  (* sqrt.rn.f64 is correctly rounded, same as OCaml's sqrt. *)
  | "sqrt", [VF a] -> VF (Stdlib.sqrt a)
  (* IR copysign(mag, sgn) follows the OCaml argument order. *)
  | "copysign", [VF m; VF s] -> VF (Float.copy_sign m s)
  | "f64_bits", [VF a] -> VI64 (Int64.bits_of_float a)
  | "bits_f64", [VI64 a] -> VF (Int64.float_of_bits a)
  | _ -> Alcotest.fail ("eval_intrinsic: unsupported intrinsic " ^ name)

let rec eval_expr env e : value =
  match e with
  | EConst (CFloat64 x) -> VF x
  | EConst (CInt32 n) -> VI32 n
  | EConst (CInt64 n) -> VI64 n
  | EVar v -> (
      match Hashtbl.find_opt env v.var_name with
      | Some x -> x
      | None -> Alcotest.fail ("unbound variable " ^ v.var_name))
  | EBinop (((Add | Sub | Mul | Div) as op), a, b) ->
      eval_arith op (eval_expr env a) (eval_expr env b)
  | EBinop (((Shl | Shr | BitAnd | BitOr | BitXor) as op), a, b) ->
      eval_bits op (eval_expr env a) (eval_expr env b)
  | EBinop (op, a, b) -> eval_cmp op (eval_expr env a) (eval_expr env b)
  | EUnop (Neg, a) -> VF (-.as_f (eval_expr env a))
  | ECast (ty, a) -> eval_cast ty (eval_expr env a)
  | EIf (c, t, f) ->
      if as_i32 (eval_expr env c) <> 0l then eval_expr env t
      else eval_expr env f
  | EIntrinsic (_, name, args) ->
      eval_intrinsic name (List.map (eval_expr env) args)
  | EApp (EVar f, args) -> apply f.var_name (List.map (eval_expr env) args)
  | _ -> Alcotest.fail "eval_expr: unsupported IR construct"

and apply fname arg_vals : value =
  match Hashtbl.find_opt funcs fname with
  | None -> Alcotest.fail ("unknown helper " ^ fname)
  | Some hf -> (
      let env = Hashtbl.create 16 in
      List.iter2
        (fun p v -> Hashtbl.replace env p.var_name v)
        hf.hf_params
        arg_vals ;
      try
        eval_stmt env hf.hf_body ;
        Alcotest.fail ("helper " ^ fname ^ " did not return")
      with Ret v -> v)

and eval_stmt env s : unit =
  match s with
  | SLet (v, e, body) ->
      Hashtbl.replace env v.var_name (eval_expr env e) ;
      eval_stmt env body
  | SIf (c, t, f) ->
      if as_i32 (eval_expr env c) <> 0l then eval_stmt env t
      else Option.iter (eval_stmt env) f
  | SReturn e -> raise (Ret (eval_expr env e))
  | SSeq ss -> List.iter (eval_stmt env) ss
  | _ -> Alcotest.fail "eval_stmt: unsupported IR construct"

(** {1 Accuracy harness} *)

let call1 fname x = as_f (apply fname [VF x])

let call2 fname x y = as_f (apply fname [VF x; VF y])

let rel_err ~got ~expect =
  if got = expect then 0.0
  else abs_float (got -. expect) /. Float.max (abs_float expect) 1e-300

(** Max relative error of [f] vs [reference] over [n] points spanning [lo, hi]
    (linear grid, endpoints included). *)
let max_rel_on ~lo ~hi ~n f reference =
  let worst = ref 0.0 in
  let worst_x = ref lo in
  for i = 0 to n - 1 do
    let x = lo +. ((hi -. lo) *. float_of_int i /. float_of_int (n - 1)) in
    let e = rel_err ~got:(f x) ~expect:(reference x) in
    if e > !worst then begin
      worst := e ;
      worst_x := x
    end
  done ;
  (!worst, !worst_x)

let check_unary name ?(tol = 1e-12) ~domains reference =
  let hname =
    match Sarek_codegen.Sarek_ir_ptx_softmath.helper_name name with
    | Some h -> h
    | None -> Alcotest.fail ("no softmath helper for " ^ name)
  in
  List.iter
    (fun (lo, hi) ->
      let worst, at = max_rel_on ~lo ~hi ~n:1001 (call1 hname) reference in
      Printf.printf
        "%-6s [%.3g, %.3g]: max rel err %.3e (at x = %.17g)\n%!"
        name
        lo
        hi
        worst
        at ;
      if worst > tol then
        Alcotest.failf
          "%s: max rel err %.3e > %.0e on [%g, %g] (worst at x = %.17g)"
          name
          worst
          tol
          lo
          hi
          at)
    domains

let test_exp () =
  check_unary "exp" Stdlib.exp ~domains:[(-2.0, 2.0); (-700.0, 700.0)]

(** Out-of-domain saturation (audit finding M3): the (n + 1023) << 52 scale
    construction used to wrap into the sign/exponent bits outside [-708, 709.78]
    and return garbage (exp -1000 came back huge). Now it flushes to 0 / -1 /
    +inf per the documented tier. *)
let test_exp_expm1_saturation () =
  let hname name =
    match Sarek_codegen.Sarek_ir_ptx_softmath.helper_name name with
    | Some h -> h
    | None -> Alcotest.fail ("no softmath helper for " ^ name)
  in
  let exp_h = hname "exp" and expm1_h = hname "expm1" in
  let checkf label want got =
    if got <> want then
      Alcotest.failf "%s: expected %.17g, got %.17g" label want got
  in
  checkf "exp(-1000)" 0.0 (call1 exp_h (-1000.0)) ;
  checkf "exp(-750)" 0.0 (call1 exp_h (-750.0)) ;
  checkf "exp(710)" infinity (call1 exp_h 710.0) ;
  checkf "exp(1e10)" infinity (call1 exp_h 1e10) ;
  checkf "expm1(-1000)" (-1.0) (call1 expm1_h (-1000.0)) ;
  checkf "expm1(710)" infinity (call1 expm1_h 710.0)

let test_log () =
  check_unary
    "log"
    Stdlib.log
    ~domains:[(0.01, 5.0); (1e-300, 1.0); (1.0, 1e300)]

let test_sin () =
  check_unary "sin" Stdlib.sin ~domains:[(-3.0, 3.0); (-1e6, 1e6)]

let test_cos () =
  check_unary "cos" Stdlib.cos ~domains:[(-3.0, 3.0); (-1e6, 1e6)]

let test_tan () =
  check_unary
    "tan"
    Stdlib.tan
    ~tol:1e-11
    ~domains:[(-1.0, 1.0); (-1.5, 1.5); (-20.0, 20.0)]

let test_log10 () =
  check_unary "log10" Stdlib.log10 ~domains:[(0.01, 5.0); (1e-300, 1e300)]

let test_sinh () =
  check_unary
    "sinh"
    Stdlib.sinh
    ~domains:[(-2.0, 2.0); (-0.04, 0.04); (-700.0, 700.0)]

let test_cosh () =
  check_unary "cosh" Stdlib.cosh ~domains:[(-2.0, 2.0); (-700.0, 700.0)]

let test_tanh () =
  check_unary
    "tanh"
    Stdlib.tanh
    ~domains:[(-2.0, 2.0); (-0.04, 0.04); (-25.0, 25.0)]

let test_asin () = check_unary "asin" Stdlib.asin ~domains:[(-1.0, 1.0)]

let test_acos () = check_unary "acos" Stdlib.acos ~domains:[(-1.0, 1.0)]

let test_atan () =
  check_unary "atan" Stdlib.atan ~domains:[(-1.0, 1.0); (-1e6, 1e6)]

(** expm1/log1p exist for relative precision near 0: sweep ±10^-e down to 1e-300
    on top of the linear-grid domains. *)
let test_near_zero name reference =
  let hname =
    match Sarek_codegen.Sarek_ir_ptx_softmath.helper_name name with
    | Some h -> h
    | None -> Alcotest.fail ("no softmath helper for " ^ name)
  in
  let worst = ref 0.0 in
  let worst_x = ref 0.0 in
  for e = 1 to 300 do
    List.iter
      (fun x ->
        let err = rel_err ~got:(call1 hname x) ~expect:(reference x) in
        if err > !worst then begin
          worst := err ;
          worst_x := x
        end)
      [10.0 ** float_of_int (-e); -.(10.0 ** float_of_int (-e))]
  done ;
  Printf.printf
    "%-6s ±1e-300..1e-1: max rel err %.3e (at x = %.17g)\n%!"
    name
    !worst
    !worst_x ;
  if !worst > 1e-12 then
    Alcotest.failf
      "%s: near-zero max rel err %.3e > 1e-12 (worst at x = %.17g)"
      name
      !worst
      !worst_x

let test_expm1 () =
  check_unary
    "expm1"
    Stdlib.expm1
    ~domains:[(-0.9, 20.0); (-0.1, 0.1); (-700.0, 700.0)] ;
  test_near_zero "expm1" Stdlib.expm1

let test_log1p () =
  check_unary "log1p" Stdlib.log1p ~domains:[(-0.9, 20.0); (-0.1, 0.1)] ;
  test_near_zero "log1p" Stdlib.log1p

(** atan2 over all four quadrants (32x32 grid chosen to avoid exact zeros, which
    linear 1D grids would hit) plus the axis special cases. *)
let test_atan2 () =
  let worst = ref 0.0 in
  let worst_at = ref (0.0, 0.0) in
  let check y x =
    let e =
      rel_err ~got:(call2 "__sarek_f64_atan2" y x) ~expect:(Stdlib.atan2 y x)
    in
    if e > !worst then begin
      worst := e ;
      worst_at := (y, x)
    end
  in
  for i = 0 to 31 do
    for j = 0 to 31 do
      let y = -3.0 +. (6.0 *. float_of_int i /. 31.0) in
      let x = -3.0 +. (6.0 *. float_of_int j /. 31.0) in
      check y x
    done
  done ;
  (* Axes: y = 0 (x > 0 and x < 0), x = 0 (both y signs). *)
  List.iter
    (fun (y, x) -> check y x)
    [(0.0, 2.0); (0.0, -2.0); (2.0, 0.0); (-2.0, 0.0); (-0.0, 2.0)] ;
  let y, x = !worst_at in
  Printf.printf
    "atan2  [-3,3]x[-3,3]: max rel err %.3e (at y=%g, x=%g)\n%!"
    !worst
    y
    x ;
  if !worst > 1e-12 then
    Alcotest.failf
      "atan2: max rel err %.3e > 1e-12 (worst at y=%g x=%g)"
      !worst
      y
      x

let test_pow () =
  let worst = ref 0.0 in
  let worst_at = ref (0.0, 0.0) in
  for i = 0 to 31 do
    for j = 0 to 31 do
      let x = 0.1 +. (2.9 *. float_of_int i /. 31.0) in
      let y = -2.0 +. (4.0 *. float_of_int j /. 31.0) in
      let e =
        rel_err ~got:(call2 "__sarek_f64_pow" x y) ~expect:(Float.pow x y)
      in
      if e > !worst then begin
        worst := e ;
        worst_at := (x, y)
      end
    done
  done ;
  let x, y = !worst_at in
  Printf.printf
    "pow    [0.1,3]x[-2,2]: max rel err %.3e (at %g, %g)\n%!"
    !worst
    x
    y ;
  if !worst > 1e-12 then
    Alcotest.failf
      "pow: max rel err %.3e > 1e-12 (worst at x=%g y=%g)"
      !worst
      x
      y

let () =
  Alcotest.run
    "f64_softmath"
    [
      ( "accuracy",
        [
          Alcotest.test_case "exp" `Quick test_exp;
          Alcotest.test_case
            "exp/expm1 saturation"
            `Quick
            test_exp_expm1_saturation;
          Alcotest.test_case "log" `Quick test_log;
          Alcotest.test_case "sin" `Quick test_sin;
          Alcotest.test_case "cos" `Quick test_cos;
          Alcotest.test_case "tan" `Quick test_tan;
          Alcotest.test_case "log10" `Quick test_log10;
          Alcotest.test_case "sinh" `Quick test_sinh;
          Alcotest.test_case "cosh" `Quick test_cosh;
          Alcotest.test_case "tanh" `Quick test_tanh;
          Alcotest.test_case "pow" `Quick test_pow;
          Alcotest.test_case "asin" `Quick test_asin;
          Alcotest.test_case "acos" `Quick test_acos;
          Alcotest.test_case "atan" `Quick test_atan;
          Alcotest.test_case "atan2" `Quick test_atan2;
          Alcotest.test_case "expm1" `Quick test_expm1;
          Alcotest.test_case "log1p" `Quick test_log1p;
        ] );
    ]
