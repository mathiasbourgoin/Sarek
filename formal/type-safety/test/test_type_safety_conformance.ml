(******************************************************************************)
(* test_type_safety_conformance.ml
 *
 * Smoke / conformance tests for the extracted TypeSafetyModel.
 *
 * The Coq spec models a post-unification, simplified type checker over a
 * 3-constructor expression type (ELit / EVar / ELet).  Sarek_typer.infer
 * operates on the full texpr AST, so end-to-end differential comparison is
 * deferred to T2.  This file exercises the extracted model itself.
 *
 * Note: the extracted model's [string] type is Coq's ascii-encoded String,
 * not OCaml's native string.  A conversion helper is provided so that EVar /
 * ELet tests can be written with ordinary string literals.
 ******************************************************************************)

module M = Type_safety_model.TypeSafetyModel

(* ----- helpers -------------------------------------------------------------- *)

(** Convert an OCaml [string] to the Coq-extracted [string] type. Defined before
    [open M] to avoid shadowing OCaml's [String] module. *)
let coq_string_of_string (s : string) : M.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    M.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (M.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) M.EmptyString

open M

(** Build an EVar node from an OCaml string literal. *)
let evar s = EVar (coq_string_of_string s)

(** Build an ELet node from OCaml string literal for the binder. *)
let elet x e1 e2 = ELet (coq_string_of_string x, e1, e2)

(** Build an environment entry from an OCaml string literal. *)
let env_entry x t = (coq_string_of_string x, t)

(* ----- literal tests -------------------------------------------------------- *)

let test_lit_int () =
  let result = infer_type [] (ELit (LInt 42)) in
  assert (result = Inl (TPrim TInt32)) ;
  Printf.printf "  ELit (LInt 42) -> TPrim TInt32 [ok]\n"

let test_lit_float () =
  let result = infer_type [] (ELit (LFloat 0)) in
  assert (result = Inl (TReg RFloat32)) ;
  Printf.printf "  ELit (LFloat 0) -> TReg RFloat32 [ok]\n"

let test_lit_bool_true () =
  let result = infer_type [] (ELit (LBool true)) in
  assert (result = Inl (TPrim TBool)) ;
  Printf.printf "  ELit (LBool true) -> TPrim TBool [ok]\n"

let test_lit_bool_false () =
  let result = infer_type [] (ELit (LBool false)) in
  assert (result = Inl (TPrim TBool)) ;
  Printf.printf "  ELit (LBool false) -> TPrim TBool [ok]\n"

let test_lit_unit () =
  let result = infer_type [] (ELit LUnit) in
  assert (result = Inl (TPrim TUnit)) ;
  Printf.printf "  ELit LUnit -> TPrim TUnit [ok]\n"

(* ----- variable tests ------------------------------------------------------- *)

let test_var_bound () =
  let env = [env_entry "x" (TPrim TInt32)] in
  let result = infer_type env (evar "x") in
  assert (result = Inl (TPrim TInt32)) ;
  Printf.printf "  EVar \"x\" (bound) -> TPrim TInt32 [ok]\n"

let test_var_unbound () =
  let result = infer_type [] (evar "y") in
  (match result with
  | Inr (UnboundVar _) -> ()
  | _ -> failwith "expected UnboundVar error") ;
  Printf.printf "  EVar \"y\" (unbound) -> UnboundVar [ok]\n"

(* ----- let tests ------------------------------------------------------------ *)

let test_let_simple () =
  (* let x = 1 in x  -->  TPrim TInt32 *)
  let e = elet "x" (ELit (LInt 1)) (evar "x") in
  let result = infer_type [] e in
  assert (result = Inl (TPrim TInt32)) ;
  Printf.printf "  ELet x=1 in x -> TPrim TInt32 [ok]\n"

let test_let_shadow () =
  (* let x = true in (let x = 1 in x)  -->  TPrim TInt32 (inner binding wins) *)
  let inner = elet "x" (ELit (LInt 1)) (evar "x") in
  let e = elet "x" (ELit (LBool true)) inner in
  let result = infer_type [] e in
  assert (result = Inl (TPrim TInt32)) ;
  Printf.printf
    "  ELet x=true in (ELet x=1 in x) -> TPrim TInt32 (shadow) [ok]\n"

let test_let_propagates_error () =
  (* let x = (unbound y) in 0  -->  error *)
  let e = elet "x" (evar "y") (ELit (LInt 0)) in
  let result = infer_type [] e in
  (match result with
  | Inr (UnboundVar _) -> ()
  | _ -> failwith "expected UnboundVar propagation") ;
  Printf.printf "  ELet x=(unbound y) in 0 -> error propagated [ok]\n"

(* ----- lookup_env tests ----------------------------------------------------- *)

let test_lookup_hit () =
  let env = [env_entry "a" (TReg RFloat32)] in
  let result = lookup_env env (coq_string_of_string "a") in
  assert (result = Some (TReg RFloat32)) ;
  Printf.printf "  lookup_env [a:float32] \"a\" = Some RFloat32 [ok]\n"

let test_lookup_miss () =
  let env = [env_entry "a" (TReg RFloat32)] in
  let result = lookup_env env (coq_string_of_string "b") in
  assert (result = None) ;
  Printf.printf "  lookup_env [a:float32] \"b\" = None [ok]\n"

let test_lookup_first_wins () =
  (* First binding for x should win (left-to-right assoc list). *)
  let env = [env_entry "x" (TPrim TInt32); env_entry "x" (TPrim TBool)] in
  let result = lookup_env env (coq_string_of_string "x") in
  assert (result = Some (TPrim TInt32)) ;
  Printf.printf
    "  lookup_env [x:int32; x:bool] \"x\" = Some TInt32 (first wins) [ok]\n"

(* ----- runner --------------------------------------------------------------- *)

let () =
  Printf.printf "=== TypeSafetyModel smoke tests ===\n" ;
  Printf.printf "--- Literals ---\n" ;
  test_lit_int () ;
  test_lit_float () ;
  test_lit_bool_true () ;
  test_lit_bool_false () ;
  test_lit_unit () ;
  Printf.printf "--- Variables ---\n" ;
  test_var_bound () ;
  test_var_unbound () ;
  Printf.printf "--- Let bindings ---\n" ;
  test_let_simple () ;
  test_let_shadow () ;
  test_let_propagates_error () ;
  Printf.printf "--- lookup_env ---\n" ;
  test_lookup_hit () ;
  test_lookup_miss () ;
  test_lookup_first_wins () ;
  Printf.printf "=== All tests passed ===\n"
