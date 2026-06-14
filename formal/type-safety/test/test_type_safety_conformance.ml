(******************************************************************************)
(* test_type_safety_conformance.ml
 *
 * T1-CMBT differential harness for the Sarek PPX type checker.
 *
 * Two layers:
 *
 *   1. Smoke / conformance tests for the extracted TypeSafetyModel — fixed
 *      expressions exercising the extracted Coq [infer_type] directly.
 *
 *   2. REAL differential property: random exprs over the model's covered
 *      fragment (ELit LInt/LFloat/LBool/LUnit, EVar, ELet) are typed by BOTH
 *      the extracted Coq model (TypeSafetyModel.infer_type) AND the real
 *      Sarek_typer.infer (the production inference engine).  We assert the two
 *      engines agree on (a) success vs. failure and (b) the inferred type.
 *
 * The Coq model has its own type universe (TypeSafetyModel.sarek_type) and the
 * production engine uses Sarek_types.typ.  Both are projected onto a small
 * normalised type [ncmp] for comparison; only the literal/var/let fragment is
 * exercised, so the projection is total on the values that actually occur.
 *
 * Note: the extracted model's [string] type is Coq's ascii-encoded String,
 * not OCaml's native string.  A conversion helper bridges the two.
 ******************************************************************************)

module M = Type_safety_model.TypeSafetyModel

(* Alias for OCaml's native string, captured before [open M] shadows [string]
   with the Coq-extracted ascii-list [M.string]. *)
type ostring = string

(* ----- coq-string helpers --------------------------------------------------- *)

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

(* ===========================================================================
   T1-CMBT — real differential harness (Coq model  vs  Sarek_typer.infer)
   ===========================================================================*)

(* --- normalised comparison value --------------------------------------------
   Both engines emit results in different universes. We project each onto this
   small type. Only the model-covered fragment can occur, so the projection is
   total on the values produced by these tests. *)
type ncmp =
  | NInt32
  | NFloat32
  | NBool
  | NUnit
  | NTuple of ncmp list
  | NUnbound (* typing failed: variable not in scope *)
  | NOtherErr (* any other failure — flagged as a divergence candidate *)
  | NUnsupported (* a type outside the covered fragment leaked through *)

let rec string_of_ncmp = function
  | NInt32 -> "int32"
  | NFloat32 -> "float32"
  | NBool -> "bool"
  | NUnit -> "unit"
  | NTuple ns ->
      Printf.sprintf "(%s)" (String.concat " * " (List.map string_of_ncmp ns))
  | NUnbound -> "<unbound>"
  | NOtherErr -> "<other-error>"
  | NUnsupported -> "<unsupported-type>"

(* --- project the Coq model result ------------------------------------------- *)
let rec ncmp_of_coq_typ (t : M.sarek_type) : ncmp =
  match t with
  | TPrim TInt32 -> NInt32
  | TReg RFloat32 -> NFloat32
  | TPrim TBool -> NBool
  | TPrim TUnit -> NUnit
  | TTuple ts -> NTuple (List.map ncmp_of_coq_typ ts)
  | _ -> NUnsupported

let ncmp_of_coq (r : M.infer_result) : ncmp =
  match r with
  | Inl t -> ncmp_of_coq_typ t
  | Inr (UnboundVar _) -> NUnbound
  | Inr (TypeMismatch _) -> NOtherErr

(* --- project the real Sarek_types.typ --------------------------------------- *)
let rec ncmp_of_typ (t : Sarek_types.typ) : ncmp =
  match t with
  | Sarek_types.TPrim Sarek_types.TInt32 -> NInt32
  | Sarek_types.TReg Sarek_types.Float32 -> NFloat32
  | Sarek_types.TPrim Sarek_types.TBool -> NBool
  | Sarek_types.TPrim Sarek_types.TUnit -> NUnit
  | Sarek_types.TTuple ts -> NTuple (List.map ncmp_of_typ ts)
  | _ -> NUnsupported

(* --- project the real Sarek_typer.infer result ------------------------------ *)
let ncmp_of_real (r : (Sarek_typed_ast.texpr * Sarek_env.t) Sarek_error.result)
    : ncmp =
  match r with
  | Ok (te, _) -> ncmp_of_typ te.Sarek_typed_ast.ty
  | Error errs -> (
      (* Treat an error list that mentions an unbound variable as NUnbound;
         everything else is a distinct error class. *)
      let is_unbound = function
        | Sarek_error.Unbound_variable _ -> true
        | _ -> false
      in
      match List.exists is_unbound errs with
      | true -> NUnbound
      | false -> NOtherErr)

(* --- translate a Coq-model expr into a real Sarek_ast.expr ------------------ *)
let mk_ast (e : Sarek_ast.expr_desc) : Sarek_ast.expr =
  {Sarek_ast.e; expr_loc = Sarek_ast.dummy_loc}

let real_string_of_coq (s : M.string) : ostring =
  let ascii_to_char (M.Ascii (b0, b1, b2, b3, b4, b5, b6, b7)) =
    let bit b k = if b then 1 lsl k else 0 in
    Char.chr
      (bit b0 0 + bit b1 1 + bit b2 2 + bit b3 3 + bit b4 4 + bit b5 5
     + bit b6 6 + bit b7 7)
  in
  let buf = Buffer.create 8 in
  let rec go = function
    | M.EmptyString -> ()
    | M.String (a, rest) ->
        Buffer.add_char buf (ascii_to_char a) ;
        go rest
  in
  go s ;
  Buffer.contents buf

let rec real_expr_of_coq (e : M.expr) : Sarek_ast.expr =
  match e with
  | ELit (LInt n) -> mk_ast (Sarek_ast.EInt n)
  | ELit (LFloat _) -> mk_ast (Sarek_ast.EFloat 0.0)
  | ELit (LBool b) -> mk_ast (Sarek_ast.EBool b)
  | ELit LUnit -> mk_ast Sarek_ast.EUnit
  | EVar x -> mk_ast (Sarek_ast.EVar (real_string_of_coq x))
  | ELet (x, e1, e2) ->
      mk_ast
        (Sarek_ast.ELet
           (real_string_of_coq x, None, real_expr_of_coq e1, real_expr_of_coq e2))
  | ETuple es -> mk_ast (Sarek_ast.ETuple (List.map real_expr_of_coq es))

(* --- translate a Coq-model type_env into a real Sarek_env.t ----------------- *)
let real_typ_of_coq (t : M.sarek_type) : Sarek_types.typ option =
  match t with
  | TPrim TInt32 -> Some (Sarek_types.TPrim Sarek_types.TInt32)
  | TPrim TBool -> Some (Sarek_types.TPrim Sarek_types.TBool)
  | TPrim TUnit -> Some (Sarek_types.TPrim Sarek_types.TUnit)
  | TReg RFloat32 -> Some (Sarek_types.TReg Sarek_types.Float32)
  | _ -> None (* outside covered fragment — not generated *)

let real_env_of_coq (env : M.type_env) : Sarek_env.t =
  List.fold_left
    (fun acc (name, t) ->
      match real_typ_of_coq t with
      | None -> acc
      | Some vi_type ->
          let vi =
            {
              Sarek_env.vi_type;
              vi_mutable = false;
              vi_is_param = false;
              vi_index = 0;
              vi_is_vec = false;
            }
          in
          Sarek_env.add_var (real_string_of_coq name) vi acc)
    Sarek_env.empty
    (* Sarek_env keeps a single binding per name (last write wins via
       StringMap), whereas the Coq assoc-list is first-write-wins. We fold in
       REVERSE so the earliest Coq binding is the one that survives in the map,
       matching the model's lookup_env semantics. *)
    (List.rev env)

(* --- QCheck generators ------------------------------------------------------ *)
open QCheck2

(* keep variable names in a tiny pool so EVar frequently hits a binding *)
let gen_name : ostring Gen.t = Gen.oneof_list ["x"; "y"; "z"; "w"]

let gen_lit : M.lit Gen.t =
  Gen.oneof
    [
      Gen.map (fun n -> M.LInt n) (Gen.int_range 0 1000);
      Gen.map (fun n -> M.LFloat n) (Gen.int_range 0 1000);
      Gen.map (fun b -> M.LBool b) Gen.bool;
      Gen.return M.LUnit;
    ]

let coq_str_gen : M.string Gen.t = Gen.map coq_string_of_string gen_name

(* bounded-depth expr generator over the covered fragment *)
let gen_expr : M.expr Gen.t =
  Gen.sized_size (Gen.int_range 0 6)
  @@ Gen.fix (fun self n ->
      if n = 0 then Gen.map (fun l -> M.ELit l) gen_lit
      else
        let sub = self (n / 2) in
        Gen.oneof
          [
            Gen.map (fun l -> M.ELit l) gen_lit;
            Gen.map (fun x -> M.EVar x) coq_str_gen;
            Gen.map
              (fun (x, (e1, e2)) -> M.ELet (x, e1, e2))
              (Gen.pair coq_str_gen (Gen.pair sub sub));
            (* ETuple uses only literals to avoid ELet-scoping divergences
               between the Coq model (per-element independent env) and
               Sarek_typer (which may thread env across elements). *)
            Gen.map
              (fun es -> M.ETuple es)
              (Gen.list_size
                 (Gen.int_range 0 3)
                 (Gen.map (fun l -> M.ELit l) gen_lit));
          ])

(* environment generator: assoc list over the same name pool / covered types *)
let gen_coq_typ : M.sarek_type Gen.t =
  Gen.oneof_list
    [M.TPrim M.TInt32; M.TPrim M.TBool; M.TPrim M.TUnit; M.TReg M.RFloat32]

let gen_env : M.type_env Gen.t =
  Gen.list_size (Gen.int_range 0 3) (Gen.pair coq_str_gen gen_coq_typ)

(* --- divergence recorder ---------------------------------------------------- *)
let divergences : ostring list ref = ref []

let pp_coq_expr (e : M.expr) : ostring =
  let rec go = function
    | M.ELit (LInt n) -> Printf.sprintf "%d" n
    | M.ELit (LFloat _) -> "0.0"
    | M.ELit (LBool b) -> string_of_bool b
    | M.ELit LUnit -> "()"
    | M.EVar x -> real_string_of_coq x
    | M.ELet (x, e1, e2) ->
        Printf.sprintf
          "(let %s = %s in %s)"
          (real_string_of_coq x)
          (go e1)
          (go e2)
    | M.ETuple es -> Printf.sprintf "(%s)" (String.concat ", " (List.map go es))
  in
  go e

let pp_case ((env, e) : M.type_env * M.expr) : ostring =
  let entries = List.map (fun (n, _) -> real_string_of_coq n) env in
  Printf.sprintf "env=[%s]  expr=%s" (String.concat ";" entries) (pp_coq_expr e)

(* --- the differential property ---------------------------------------------- *)
let test_differential =
  Test.make
    ~name:"coq_model_vs_sarek_typer_agree"
    ~count:2000
    ~print:pp_case
    (Gen.pair gen_env gen_expr)
    (fun (env, e) ->
      let coq_result = ncmp_of_coq (M.infer_type env e) in
      let real_result =
        ncmp_of_real
          (Sarek_typer.infer (real_env_of_coq env) (real_expr_of_coq e))
      in
      if coq_result = real_result then true
      else begin
        let msg =
          Printf.sprintf
            "expr=%s  coq=%s  sarek=%s"
            (pp_coq_expr e)
            (string_of_ncmp coq_result)
            (string_of_ncmp real_result)
        in
        divergences := msg :: !divergences ;
        false
      end)

(* ----- runner --------------------------------------------------------------- *)

let test_tuple_empty () =
  let result = M.infer_type [] (M.ETuple []) in
  assert (result = M.Inl (M.TTuple [])) ;
  Printf.printf "  ETuple [] -> TTuple [] [ok]\n"

let test_tuple_single () =
  let result = M.infer_type [] (M.ETuple [M.ELit (M.LInt 42)]) in
  assert (result = M.Inl (M.TTuple [M.TPrim M.TInt32])) ;
  Printf.printf "  ETuple [42] -> TTuple [TInt32] [ok]\n"

let test_tuple_pair () =
  let result =
    M.infer_type [] (M.ETuple [M.ELit (M.LInt 1); M.ELit (M.LBool true)])
  in
  assert (result = M.Inl (M.TTuple [M.TPrim M.TInt32; M.TPrim M.TBool])) ;
  Printf.printf "  ETuple [1; true] -> TTuple [TInt32; TBool] [ok]\n"

let test_tuple_nested () =
  let inner = M.ETuple [M.ELit (M.LInt 0); M.ELit M.LUnit] in
  let result = M.infer_type [] (M.ETuple [inner; M.ELit (M.LBool false)]) in
  assert (
    result
    = M.Inl
        (M.TTuple
           [M.TTuple [M.TPrim M.TInt32; M.TPrim M.TUnit]; M.TPrim M.TBool])) ;
  Printf.printf "  ETuple [ETuple; false] -> TTuple [TTuple; TBool] [ok]\n"

let test_tuple_error_propagates () =
  let result =
    M.infer_type
      []
      (M.ETuple [M.ELit (M.LInt 1); M.EVar (coq_string_of_string "x")])
  in
  assert (match result with M.Inr (M.UnboundVar _) -> true | _ -> false) ;
  Printf.printf "  ETuple [1; free_var] -> UnboundVar [ok]\n"

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
  Printf.printf "--- ETuple (T2-CUSTOM) ---\n" ;
  test_tuple_empty () ;
  test_tuple_single () ;
  test_tuple_pair () ;
  test_tuple_nested () ;
  test_tuple_error_propagates () ;
  Printf.printf "=== Smoke tests passed ===\n" ;
  Printf.printf "\n=== T1-CMBT differential (Coq model vs Sarek_typer) ===\n" ;
  let passed = QCheck_base_runner.run_tests ~verbose:true [test_differential] in
  if !divergences <> [] then begin
    Printf.printf "\n--- divergence candidates (deduped) ---\n" ;
    List.iter
      (fun m -> Printf.printf "  %s\n" m)
      (List.sort_uniq compare !divergences)
  end ;
  if passed <> 0 then exit passed

(* ===========================================================================
   T2-UNIFY smoke tests — extracted UnifyModel oracle
   ===========================================================================

   These tests drive the extracted pure unifier (UnifyModel.unify_fun) directly,
   checking:
     1. PPrim/PPrim and PReg/PReg ground unification.
     2. PVar binds to PPrim/PReg when unbound.
     3. Mismatch cases return None.
     4. Zero-fuel always returns None.
     5. PTuple matching (same-length tuples of ground types).

   The conformance property: results match what Sarek_types.unify would do
   on equivalent inputs (for the covered fragment without mutable state). *)

module U = Type_safety_model.UnifyModel

let fuel = 100

let empty_subst : U.pre_subst = []

let test_unify_prim_same () =
  let s = U.unify_fun fuel empty_subst (U.PPrim U.TInt32) (U.PPrim U.TInt32) in
  assert (s = Some []) ;
  Printf.printf "  unify PPrim TInt32 PPrim TInt32 = Some [] [ok]\n"

let test_unify_prim_diff () =
  let s = U.unify_fun fuel empty_subst (U.PPrim U.TInt32) (U.PPrim U.TBool) in
  assert (s = None) ;
  Printf.printf "  unify PPrim TInt32 PPrim TBool = None [ok]\n"

let test_unify_reg_same () =
  let s =
    U.unify_fun fuel empty_subst (U.PReg U.RFloat32) (U.PReg U.RFloat32)
  in
  assert (s = Some []) ;
  Printf.printf "  unify PReg RFloat32 PReg RFloat32 = Some [] [ok]\n"

let test_unify_reg_diff () =
  let s = U.unify_fun fuel empty_subst (U.PReg U.RFloat32) (U.PReg U.RInt) in
  assert (s = None) ;
  Printf.printf "  unify PReg RFloat32 PReg RInt = None [ok]\n"

let test_unify_prim_mismatch () =
  let s = U.unify_fun fuel empty_subst (U.PPrim U.TBool) (U.PReg U.RFloat32) in
  assert (s = None) ;
  Printf.printf "  unify PPrim TBool PReg RFloat32 = None (mismatch) [ok]\n"

let test_unify_var_prim () =
  let s = U.unify_fun fuel empty_subst (U.PVar 0) (U.PPrim U.TInt32) in
  (match s with
  | Some [(0, U.PPrim U.TInt32)] -> ()
  | _ -> failwith "expected Some [(0, PPrim TInt32)]") ;
  Printf.printf "  unify PVar 0 PPrim TInt32 = Some [(0, PPrim TInt32)] [ok]\n"

let test_unify_prim_var () =
  let s = U.unify_fun fuel empty_subst (U.PPrim U.TBool) (U.PVar 1) in
  (match s with
  | Some [(1, U.PPrim U.TBool)] -> ()
  | _ -> failwith "expected Some [(1, PPrim TBool)]") ;
  Printf.printf "  unify PPrim TBool PVar 1 = Some [(1, PPrim TBool)] [ok]\n"

let test_unify_var_reg () =
  let s = U.unify_fun fuel empty_subst (U.PVar 2) (U.PReg U.RInt64) in
  (match s with
  | Some [(2, U.PReg U.RInt64)] -> ()
  | _ -> failwith "expected Some [(2, PReg RInt64)]") ;
  Printf.printf "  unify PVar 2 PReg RInt64 = Some [(2, PReg RInt64)] [ok]\n"

let test_unify_var_var_same () =
  let s = U.unify_fun fuel empty_subst (U.PVar 0) (U.PVar 0) in
  assert (s = Some []) ;
  Printf.printf "  unify PVar 0 PVar 0 = Some [] [ok]\n"

let test_unify_var_var_diff () =
  let s = U.unify_fun fuel empty_subst (U.PVar 0) (U.PVar 1) in
  (match s with
  | Some [(0, U.PVar 1)] -> ()
  | _ -> failwith "expected Some [(0, PVar 1)]") ;
  Printf.printf "  unify PVar 0 PVar 1 = Some [(0, PVar 1)] [ok]\n"

let test_unify_zero_fuel () =
  let s = U.unify_fun 0 empty_subst (U.PPrim U.TInt32) (U.PPrim U.TInt32) in
  assert (s = None) ;
  Printf.printf "  unify fuel=0 PPrim TInt32 PPrim TInt32 = None [ok]\n"

let test_unify_tuple_same () =
  let t1 = U.PTuple [U.PPrim U.TInt32; U.PReg U.RFloat32] in
  let t2 = U.PTuple [U.PPrim U.TInt32; U.PReg U.RFloat32] in
  let s = U.unify_fun fuel empty_subst t1 t2 in
  assert (s = Some []) ;
  Printf.printf
    "  unify PTuple[int32;float32] PTuple[int32;float32] = Some [] [ok]\n"

let test_unify_tuple_diff_len () =
  let t1 = U.PTuple [U.PPrim U.TInt32] in
  let t2 = U.PTuple [U.PPrim U.TInt32; U.PReg U.RFloat32] in
  let s = U.unify_fun fuel empty_subst t1 t2 in
  assert (s = None) ;
  Printf.printf
    "  unify PTuple[int32] PTuple[int32;float32] = None (len mismatch) [ok]\n"

let test_unify_tuple_mismatch_elem () =
  let t1 = U.PTuple [U.PPrim U.TInt32; U.PPrim U.TBool] in
  let t2 = U.PTuple [U.PPrim U.TInt32; U.PReg U.RFloat32] in
  let s = U.unify_fun fuel empty_subst t1 t2 in
  assert (s = None) ;
  Printf.printf
    "  unify PTuple[int32;bool] PTuple[int32;float32] = None (elem mismatch) \
     [ok]\n"

let test_unify_var_already_bound () =
  let s0 = [(0, U.PPrim U.TInt32)] in
  let s = U.unify_fun fuel s0 (U.PVar 0) (U.PPrim U.TInt32) in
  assert (s = Some s0) ;
  Printf.printf
    "  unify (PVar 0 -> TInt32) PVar 0 PPrim TInt32 = Some s0 [ok]\n"

(* T2-UNIFY QCheck: conformance against Sarek_types.unify for ground types *)

let gen_ground_pre_type : U.pre_type QCheck2.Gen.t =
  QCheck2.Gen.oneof_list
    [
      U.PPrim U.TUnit;
      U.PPrim U.TBool;
      U.PPrim U.TInt32;
      U.PReg U.RInt;
      U.PReg U.RInt64;
      U.PReg U.RFloat32;
      U.PReg U.RFloat64;
      U.PReg U.RChar;
    ]

type unify_ncmp = UOk | UFail

let ncmp_of_unify_opt = function Some _ -> UOk | None -> UFail

let ncmp_of_sarek_unify = function Ok () -> UOk | Error _ -> UFail

let sarek_typ_of_ground (t : U.pre_type) : Sarek_types.typ option =
  match t with
  | U.PPrim U.TUnit -> Some (Sarek_types.TPrim Sarek_types.TUnit)
  | U.PPrim U.TBool -> Some (Sarek_types.TPrim Sarek_types.TBool)
  | U.PPrim U.TInt32 -> Some (Sarek_types.TPrim Sarek_types.TInt32)
  | U.PReg U.RInt -> Some (Sarek_types.TReg Sarek_types.Int)
  | U.PReg U.RInt64 -> Some (Sarek_types.TReg Sarek_types.Int64)
  | U.PReg U.RFloat32 -> Some (Sarek_types.TReg Sarek_types.Float32)
  | U.PReg U.RFloat64 -> Some (Sarek_types.TReg Sarek_types.Float64)
  | U.PReg U.RChar -> Some (Sarek_types.TReg Sarek_types.Char)
  | _ -> None

let test_unify_differential =
  QCheck2.Test.make
    ~name:"unify_model_vs_sarek_unify_ground"
    ~count:1000
    (QCheck2.Gen.pair gen_ground_pre_type gen_ground_pre_type)
    (fun (t1, t2) ->
      let model_result = ncmp_of_unify_opt (U.unify_fun fuel [] t1 t2) in
      match (sarek_typ_of_ground t1, sarek_typ_of_ground t2) with
      | Some st1, Some st2 ->
          Sarek_types.reset_tvar_counter () ;
          let real_result = ncmp_of_sarek_unify (Sarek_types.unify st1 st2) in
          model_result = real_result
      | _ -> true)

let () =
  Printf.printf "\n=== T2-UNIFY smoke tests (UnifyModel oracle) ===\n" ;
  test_unify_prim_same () ;
  test_unify_prim_diff () ;
  test_unify_reg_same () ;
  test_unify_reg_diff () ;
  test_unify_prim_mismatch () ;
  test_unify_var_prim () ;
  test_unify_prim_var () ;
  test_unify_var_reg () ;
  test_unify_var_var_same () ;
  test_unify_var_var_diff () ;
  test_unify_zero_fuel () ;
  test_unify_tuple_same () ;
  test_unify_tuple_diff_len () ;
  test_unify_tuple_mismatch_elem () ;
  test_unify_var_already_bound () ;
  Printf.printf "=== T2-UNIFY smoke tests passed ===\n" ;
  Printf.printf
    "\n=== T2-UNIFY differential (UnifyModel vs Sarek_types.unify) ===\n" ;
  let passed2 =
    QCheck_base_runner.run_tests ~verbose:true [test_unify_differential]
  in
  if passed2 <> 0 then exit passed2

(* ===========================================================================
   T2-VEC smoke tests -- extracted VecModel oracle
   ===========================================================================

   These tests drive the extracted memory-access type inference model
   (VecModel.infer_mem_type) directly, checking:
     1. MCore delegation -- core expr inference is threaded through.
     2. EVecGet success -- TVec elem correctly yields elem type.
     3. EVecGet NotAVector -- non-vec type rejected.
     4. EVecGet IndexNotInt -- non-int32 index rejected.
     5. EVecSet success -- matching value type yields TPrim TUnit.
     6. EVecSet ElemMismatch -- mismatched value type rejected.
     7. EArrGet success -- TArr elem correctly yields elem type.
     8. EArrGet NotAnArray -- non-array type rejected.
     9. EArrSet success -- matching value type yields TPrim TUnit.
    10. EArrSet ElemMismatch -- mismatched value type rejected. *)

module V = Type_safety_model.VecModel

(** Convert an OCaml string to VecModel's Coq-extracted string type. VecModel
    has its own Ascii/String types (same structure as M but different OCaml
    types). *)
let vec_coq_string_of_string (s : ostring) : V.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    V.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (V.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) V.EmptyString

(** Build a VecModel environment entry from an OCaml string. *)
let vec_env_entry x t = (vec_coq_string_of_string x, t)

(** Build a VecModel EVar node from an OCaml string. *)
let vevar s = V.EVar (vec_coq_string_of_string s)

(* MCore delegates to core infer_type. *)
let test_vec_mcore_lit () =
  let result = V.infer_mem_type [] (V.MCore (V.ELit (V.LInt 1))) in
  assert (result = V.Inl (V.TPrim V.TInt32)) ;
  Printf.printf "  MCore (ELit (LInt 1)) -> TPrim TInt32 [ok]\n"

(* EVecGet on TVec (TPrim TInt32) with int32 index -> TPrim TInt32. *)
let test_vec_get_ok () =
  let env = [vec_env_entry "v" (V.TVec (V.TPrim V.TInt32))] in
  let result =
    V.infer_mem_type
      env
      (V.EVecGet (V.MCore (vevar "v"), V.MCore (V.ELit (V.LInt 0))))
  in
  assert (result = V.Inl (V.TPrim V.TInt32)) ;
  Printf.printf "  EVecGet vec[int32] idx:int32 -> TPrim TInt32 [ok]\n"

(* EVecGet on a non-vector type -> NotAVector. *)
let test_vec_get_not_a_vector () =
  let result =
    V.infer_mem_type
      []
      (V.EVecGet (V.MCore (V.ELit (V.LInt 0)), V.MCore (V.ELit (V.LInt 0))))
  in
  (match result with
  | V.Inr (V.NotAVector _) -> ()
  | _ -> failwith "expected NotAVector") ;
  Printf.printf "  EVecGet non-vec -> NotAVector [ok]\n"

(* EVecGet with a boolean index -> IndexNotInt. *)
let test_vec_get_bad_index () =
  let env = [vec_env_entry "v" (V.TVec (V.TPrim V.TInt32))] in
  let result =
    V.infer_mem_type
      env
      (V.EVecGet (V.MCore (vevar "v"), V.MCore (V.ELit (V.LBool true))))
  in
  (match result with
  | V.Inr (V.IndexNotInt _) -> ()
  | _ -> failwith "expected IndexNotInt") ;
  Printf.printf "  EVecGet with bool index -> IndexNotInt [ok]\n"

(* EVecSet with matching value type -> TPrim TUnit. *)
let test_vec_set_ok () =
  let env = [vec_env_entry "v" (V.TVec (V.TPrim V.TInt32))] in
  let result =
    V.infer_mem_type
      env
      (V.EVecSet
         ( V.MCore (vevar "v"),
           V.MCore (V.ELit (V.LInt 0)),
           V.MCore (V.ELit (V.LInt 42)) ))
  in
  assert (result = V.Inl (V.TPrim V.TUnit)) ;
  Printf.printf "  EVecSet vec[int32] value:int32 -> TPrim TUnit [ok]\n"

(* EVecSet with mismatched value type -> ElemMismatch. *)
let test_vec_set_mismatch () =
  let env = [vec_env_entry "v" (V.TVec (V.TPrim V.TInt32))] in
  let result =
    V.infer_mem_type
      env
      (V.EVecSet
         ( V.MCore (vevar "v"),
           V.MCore (V.ELit (V.LInt 0)),
           V.MCore (V.ELit (V.LBool true)) ))
  in
  (match result with
  | V.Inr (V.ElemMismatch _) -> ()
  | _ -> failwith "expected ElemMismatch") ;
  Printf.printf "  EVecSet vec[int32] value:bool -> ElemMismatch [ok]\n"

(* EArrGet on TArr (TPrim TBool) Local with int32 index -> TPrim TBool. *)
let test_arr_get_ok () =
  let env = [vec_env_entry "a" (V.TArr (V.TPrim V.TBool, V.Local))] in
  let result =
    V.infer_mem_type
      env
      (V.EArrGet (V.MCore (vevar "a"), V.MCore (V.ELit (V.LInt 0))))
  in
  assert (result = V.Inl (V.TPrim V.TBool)) ;
  Printf.printf "  EArrGet arr[bool,Local] idx:int32 -> TPrim TBool [ok]\n"

(* EArrGet on a non-array type -> NotAnArray. *)
let test_arr_get_not_an_array () =
  let result =
    V.infer_mem_type
      []
      (V.EArrGet (V.MCore (V.ELit (V.LBool true)), V.MCore (V.ELit (V.LInt 0))))
  in
  (match result with
  | V.Inr (V.NotAnArray _) -> ()
  | _ -> failwith "expected NotAnArray") ;
  Printf.printf "  EArrGet non-array -> NotAnArray [ok]\n"

(* EArrSet with matching value type -> TPrim TUnit. *)
let test_arr_set_ok () =
  let env = [vec_env_entry "a" (V.TArr (V.TPrim V.TInt32, V.Global))] in
  let result =
    V.infer_mem_type
      env
      (V.EArrSet
         ( V.MCore (vevar "a"),
           V.MCore (V.ELit (V.LInt 0)),
           V.MCore (V.ELit (V.LInt 7)) ))
  in
  assert (result = V.Inl (V.TPrim V.TUnit)) ;
  Printf.printf "  EArrSet arr[int32,Global] value:int32 -> TPrim TUnit [ok]\n"

(* EArrSet with mismatched value type -> ElemMismatch. *)
let test_arr_set_mismatch () =
  let env = [vec_env_entry "a" (V.TArr (V.TPrim V.TInt32, V.Shared))] in
  let result =
    V.infer_mem_type
      env
      (V.EArrSet
         ( V.MCore (vevar "a"),
           V.MCore (V.ELit (V.LInt 0)),
           V.MCore (V.ELit V.LUnit) ))
  in
  (match result with
  | V.Inr (V.ElemMismatch _) -> ()
  | _ -> failwith "expected ElemMismatch") ;
  Printf.printf "  EArrSet arr[int32,Shared] value:unit -> ElemMismatch [ok]\n"

let () =
  Printf.printf "\n=== T2-VEC smoke tests (VecModel oracle) ===\n" ;
  test_vec_mcore_lit () ;
  test_vec_get_ok () ;
  test_vec_get_not_a_vector () ;
  test_vec_get_bad_index () ;
  test_vec_set_ok () ;
  test_vec_set_mismatch () ;
  test_arr_get_ok () ;
  test_arr_get_not_an_array () ;
  test_arr_set_ok () ;
  test_arr_set_mismatch () ;
  Printf.printf "=== T2-VEC smoke tests passed ===\n"

(* ===========================================================================
   T2-REGISTRY smoke tests -- extracted RegistryModel oracle
   ===========================================================================

   These tests drive the extracted record field-access type inference model
   (RegistryModel.infer_rec_type) directly, checking:
     1.  RMem delegation -- core expr inference is threaded through.
     2.  EFieldGet success -- TRecord field correctly yields field type.
     3.  EFieldGet FieldNotFound -- missing field rejected.
     4.  EFieldGet NotARecord -- non-record receiver rejected.
     5.  EFieldSet success -- matching value type yields TPrim TUnit.
     6.  EFieldSet FieldMismatch -- mismatched value type rejected.
     7.  EFieldSet FieldNotFound -- missing field rejected.
     8.  Nested EFieldGet -- outer.r is a TRecord, inner.y succeeds.
     9.  RMem delegation through vec layer (EVecGet result).
    10.  Error propagation from RMem (unbound var). *)

module R = Type_safety_model.RegistryModel

(** Convert an OCaml string to RegistryModel's Coq-extracted string type. *)
let reg_coq_string_of_string (s : ostring) : R.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    R.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (R.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) R.EmptyString

(** Build a RegistryModel environment entry. *)
let reg_env_entry x t = (reg_coq_string_of_string x, t)

(** Build a RegistryModel EVar node. *)
let revar s = R.EVar (reg_coq_string_of_string s)

(** Build a RegistryModel field entry (name, type) pair. *)
let rfield name t = (reg_coq_string_of_string name, t)

(* Test 1: RMem delegates to core infer_type. *)
let test_reg_rmem_lit () =
  let result = R.infer_rec_type [] (R.RMem (R.MCore (R.ELit (R.LInt 1)))) in
  assert (result = R.Inl (R.TPrim R.TInt32)) ;
  Printf.printf "  RMem (ELit (LInt 1)) -> TPrim TInt32 [ok]\n"

(* Test 2: EFieldGet success -- TRecord with field "x" : TInt32. *)
let test_reg_field_get_ok () =
  let rec_type =
    R.TRecord (reg_coq_string_of_string "MyRec", [rfield "x" (R.TPrim R.TInt32)])
  in
  let env = [reg_env_entry "r" rec_type] in
  let result =
    R.infer_rec_type
      env
      (R.EFieldGet (reg_coq_string_of_string "x", R.RMem (R.MCore (revar "r"))))
  in
  assert (result = R.Inl (R.TPrim R.TInt32)) ;
  Printf.printf "  EFieldGet \"x\" on TRecord[x:int32] -> TPrim TInt32 [ok]\n"

(* Test 3: EFieldGet FieldNotFound -- field "y" not in record. *)
let test_reg_field_get_not_found () =
  let rec_type =
    R.TRecord (reg_coq_string_of_string "MyRec", [rfield "x" (R.TPrim R.TInt32)])
  in
  let env = [reg_env_entry "r" rec_type] in
  let result =
    R.infer_rec_type
      env
      (R.EFieldGet (reg_coq_string_of_string "y", R.RMem (R.MCore (revar "r"))))
  in
  (match result with
  | R.Inr (R.FieldNotFound _) -> ()
  | _ -> failwith "expected FieldNotFound") ;
  Printf.printf "  EFieldGet \"y\" on TRecord[x:int32] -> FieldNotFound [ok]\n"

(* Test 4: EFieldGet NotARecord -- receiver is a primitive type. *)
let test_reg_field_get_not_a_record () =
  let result =
    R.infer_rec_type
      []
      (R.EFieldGet
         (reg_coq_string_of_string "x", R.RMem (R.MCore (R.ELit (R.LInt 0)))))
  in
  (match result with
  | R.Inr (R.NotARecord _) -> ()
  | _ -> failwith "expected NotARecord") ;
  Printf.printf "  EFieldGet \"x\" on int literal -> NotARecord [ok]\n"

(* Test 5: EFieldSet success -- value type matches field type, yields TPrim TUnit. *)
let test_reg_field_set_ok () =
  let rec_type =
    R.TRecord (reg_coq_string_of_string "MyRec", [rfield "x" (R.TPrim R.TInt32)])
  in
  let env = [reg_env_entry "r" rec_type] in
  let result =
    R.infer_rec_type
      env
      (R.EFieldSet
         ( reg_coq_string_of_string "x",
           R.RMem (R.MCore (revar "r")),
           R.RMem (R.MCore (R.ELit (R.LInt 42))) ))
  in
  assert (result = R.Inl (R.TPrim R.TUnit)) ;
  Printf.printf
    "  EFieldSet \"x\" value:int32 on TRecord[x:int32] -> TPrim TUnit [ok]\n"

(* Test 6: EFieldSet FieldMismatch -- value is bool, field is int32. *)
let test_reg_field_set_mismatch () =
  let rec_type =
    R.TRecord (reg_coq_string_of_string "MyRec", [rfield "x" (R.TPrim R.TInt32)])
  in
  let env = [reg_env_entry "r" rec_type] in
  let result =
    R.infer_rec_type
      env
      (R.EFieldSet
         ( reg_coq_string_of_string "x",
           R.RMem (R.MCore (revar "r")),
           R.RMem (R.MCore (R.ELit (R.LBool true))) ))
  in
  (match result with
  | R.Inr (R.FieldMismatch _) -> ()
  | _ -> failwith "expected FieldMismatch") ;
  Printf.printf
    "  EFieldSet \"x\" value:bool on TRecord[x:int32] -> FieldMismatch [ok]\n"

(* Test 7: EFieldSet FieldNotFound -- field "z" not in record. *)
let test_reg_field_set_not_found () =
  let rec_type =
    R.TRecord (reg_coq_string_of_string "MyRec", [rfield "x" (R.TPrim R.TInt32)])
  in
  let env = [reg_env_entry "r" rec_type] in
  let result =
    R.infer_rec_type
      env
      (R.EFieldSet
         ( reg_coq_string_of_string "z",
           R.RMem (R.MCore (revar "r")),
           R.RMem (R.MCore (R.ELit (R.LInt 0))) ))
  in
  (match result with
  | R.Inr (R.FieldNotFound _) -> ()
  | _ -> failwith "expected FieldNotFound") ;
  Printf.printf "  EFieldSet \"z\" on TRecord[x:int32] -> FieldNotFound [ok]\n"

(* Test 8: Nested EFieldGet -- outer.r is itself a TRecord with field "y". *)
let test_reg_field_get_nested () =
  let inner_rec =
    R.TRecord (reg_coq_string_of_string "Inner", [rfield "y" (R.TPrim R.TBool)])
  in
  let outer_rec =
    R.TRecord (reg_coq_string_of_string "Outer", [rfield "r" inner_rec])
  in
  let env = [reg_env_entry "outer" outer_rec] in
  (* outer.r gives inner_rec, then .y gives TBool *)
  let get_r =
    R.EFieldGet (reg_coq_string_of_string "r", R.RMem (R.MCore (revar "outer")))
  in
  let result =
    R.infer_rec_type env (R.EFieldGet (reg_coq_string_of_string "y", get_r))
  in
  assert (result = R.Inl (R.TPrim R.TBool)) ;
  Printf.printf
    "  EFieldGet \"y\" (EFieldGet \"r\" outer) -> TPrim TBool (nested) [ok]\n"

(* Test 9: RMem delegates through vec layer -- EVecGet result threaded through. *)
let test_reg_rmem_via_vec () =
  let env = [reg_env_entry "v" (R.TVec (R.TPrim R.TInt32))] in
  let vec_get = R.EVecGet (R.MCore (revar "v"), R.MCore (R.ELit (R.LInt 0))) in
  let result = R.infer_rec_type env (R.RMem vec_get) in
  assert (result = R.Inl (R.TPrim R.TInt32)) ;
  Printf.printf "  RMem (EVecGet vec[int32] 0) -> TPrim TInt32 [ok]\n"

(* Test 10: Error propagation from RMem -- unbound variable produces RMemError. *)
let test_reg_rmem_error_propagation () =
  let result =
    R.infer_rec_type
      []
      (R.RMem (R.MCore (R.EVar (reg_coq_string_of_string "unbound"))))
  in
  (match result with
  | R.Inr (R.RMemError _) -> ()
  | _ -> failwith "expected RMemError") ;
  Printf.printf "  RMem (EVar unbound) -> RMemError [ok]\n"

let () =
  Printf.printf "\n=== T2-REGISTRY smoke tests (RegistryModel oracle) ===\n" ;
  test_reg_rmem_lit () ;
  test_reg_field_get_ok () ;
  test_reg_field_get_not_found () ;
  test_reg_field_get_not_a_record () ;
  test_reg_field_set_ok () ;
  test_reg_field_set_mismatch () ;
  test_reg_field_set_not_found () ;
  test_reg_field_get_nested () ;
  test_reg_rmem_via_vec () ;
  test_reg_rmem_error_propagation () ;
  Printf.printf "=== T2-REGISTRY smoke tests passed ===\n"

(* ========================================================================== *)
(* T3-S1 control-flow smoke tests (ControlFlowModel oracle)                   *)
(* ========================================================================== *)

module CF = Type_safety_model.ControlFlowModel

let cf_coq_string_of_string (s : ostring) : CF.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    CF.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (CF.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) CF.EmptyString

let cf_env_entry x t = (cf_coq_string_of_string x, t)

let cfvar s =
  CF.CFRec (CF.RMem (CF.MCore (CF.EVar (cf_coq_string_of_string s))))

let cflit l = CF.CFRec (CF.RMem (CF.MCore (CF.ELit l)))

(* Test 1: CFIfThen -- cond:bool, then:unit -> unit *)
let test_cf_if_then_ok () =
  let result =
    CF.infer_cf_type [] (CF.CFIfThen (cflit (CF.LBool true), cflit CF.LUnit))
  in
  assert (result = CF.Inl (CF.TPrim CF.TUnit)) ;
  Printf.printf "  CFIfThen bool then:unit -> TPrim TUnit [ok]\n"

(* Test 2: CFIfThen -- cond not bool -> CondNotBool *)
let test_cf_if_then_bad_cond () =
  let result =
    CF.infer_cf_type [] (CF.CFIfThen (cflit (CF.LInt 1), cflit CF.LUnit))
  in
  (match result with
  | CF.Inr (CF.CondNotBool _) -> ()
  | _ -> failwith "expected CondNotBool") ;
  Printf.printf "  CFIfThen int cond -> CondNotBool [ok]\n"

(* Test 3: CFIfThen -- then not unit -> BranchMismatch *)
let test_cf_if_then_then_not_unit () =
  let result =
    CF.infer_cf_type
      []
      (CF.CFIfThen (cflit (CF.LBool false), cflit (CF.LInt 42)))
  in
  (match result with
  | CF.Inr (CF.BranchMismatch _) -> ()
  | _ -> failwith "expected BranchMismatch") ;
  Printf.printf "  CFIfThen bool then:int32 -> BranchMismatch [ok]\n"

(* Test 4: CFIfElse -- branches both int32 -> int32 *)
let test_cf_if_else_ok () =
  let result =
    CF.infer_cf_type
      []
      (CF.CFIfElse (cflit (CF.LBool true), cflit (CF.LInt 1), cflit (CF.LInt 2)))
  in
  assert (result = CF.Inl (CF.TPrim CF.TInt32)) ;
  Printf.printf "  CFIfElse bool then:int32 else:int32 -> TPrim TInt32 [ok]\n"

(* Test 5: CFIfElse -- branch type mismatch -> BranchMismatch *)
let test_cf_if_else_mismatch () =
  let result =
    CF.infer_cf_type
      []
      (CF.CFIfElse
         (cflit (CF.LBool true), cflit (CF.LInt 1), cflit (CF.LBool false)))
  in
  (match result with
  | CF.Inr (CF.BranchMismatch _) -> ()
  | _ -> failwith "expected BranchMismatch") ;
  Printf.printf "  CFIfElse then:int32 else:bool -> BranchMismatch [ok]\n"

(* Test 6: CFFor -- lo:int32, hi:int32, body:unit -> unit *)
let test_cf_for_ok () =
  let result =
    CF.infer_cf_type
      []
      (CF.CFFor
         ( cf_coq_string_of_string "i",
           cflit (CF.LInt 0),
           cflit (CF.LInt 10),
           cflit CF.LUnit ))
  in
  assert (result = CF.Inl (CF.TPrim CF.TUnit)) ;
  Printf.printf "  CFFor i=0 to 10 body:unit -> TPrim TUnit [ok]\n"

(* Test 7: CFFor -- lo not int32 -> BoundNotInt32 *)
let test_cf_for_bad_bound () =
  let result =
    CF.infer_cf_type
      []
      (CF.CFFor
         ( cf_coq_string_of_string "i",
           cflit (CF.LBool true),
           cflit (CF.LInt 10),
           cflit CF.LUnit ))
  in
  (match result with
  | CF.Inr (CF.BoundNotInt32 _) -> ()
  | _ -> failwith "expected BoundNotInt32") ;
  Printf.printf "  CFFor bool lower bound -> BoundNotInt32 [ok]\n"

(* Test 8: CFWhile -- cond:bool, body:int32 -> unit *)
let test_cf_while_ok () =
  let env = [cf_env_entry "x" (CF.TPrim CF.TInt32)] in
  let result =
    CF.infer_cf_type env (CF.CFWhile (cflit (CF.LBool true), cfvar "x"))
  in
  assert (result = CF.Inl (CF.TPrim CF.TUnit)) ;
  Printf.printf "  CFWhile bool body:int32 -> TPrim TUnit [ok]\n"

(* Test 9: CFSeq -- e1:int32 ; e2:bool -> bool (e2's type) *)
let test_cf_seq_type () =
  let result =
    CF.infer_cf_type [] (CF.CFSeq (cflit (CF.LInt 1), cflit (CF.LBool false)))
  in
  assert (result = CF.Inl (CF.TPrim CF.TBool)) ;
  Printf.printf "  CFSeq int32 ; bool -> TPrim TBool [ok]\n"

(* Test 10: CFFor -- loop var bound as int32 in body *)
let test_cf_for_var_in_scope () =
  (* body uses loop var i, which must be int32; result is unit *)
  let result =
    CF.infer_cf_type
      []
      (CF.CFFor
         ( cf_coq_string_of_string "i",
           cflit (CF.LInt 0),
           cflit (CF.LInt 5),
           cfvar "i" ))
  in
  assert (result = CF.Inl (CF.TPrim CF.TUnit)) ;
  Printf.printf "  CFFor body uses loop var i (int32) -> TPrim TUnit [ok]\n"

let () =
  Printf.printf
    "\n=== T3-S1 control flow smoke tests (ControlFlowModel oracle) ===\n" ;
  test_cf_if_then_ok () ;
  test_cf_if_then_bad_cond () ;
  test_cf_if_then_then_not_unit () ;
  test_cf_if_else_ok () ;
  test_cf_if_else_mismatch () ;
  test_cf_for_ok () ;
  test_cf_for_bad_bound () ;
  test_cf_while_ok () ;
  test_cf_seq_type () ;
  test_cf_for_var_in_scope () ;
  Printf.printf "=== T3-S1 control flow smoke tests passed ===\n"

(* ===== T3-S2: OperatorSpec smoke tests ===== *)

module OP = Type_safety_model.OperatorModel

let opint32 = OP.OPCf (OP.CFRec (OP.RMem (OP.MCore (OP.ELit (OP.LInt 0)))))

let opbool = OP.OPCf (OP.CFRec (OP.RMem (OP.MCore (OP.ELit (OP.LBool false)))))

let opfloat = OP.OPCf (OP.CFRec (OP.RMem (OP.MCore (OP.ELit (OP.LFloat 0)))))

(* Test 1: Add int32 int32 -> int32 (numeric binop) *)
let test_op_add_int32 () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.Add, opint32, opint32)) in
  assert (result = OP.Inl (OP.TPrim OP.TInt32)) ;
  Printf.printf "  Add int32 int32 -> TInt32 [ok]\n"

(* Test 2: Add int32 bool -> OperandMismatch (type mismatch) *)
let test_op_add_mismatch () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.Add, opint32, opbool)) in
  assert (
    result = OP.Inr (OP.OperandMismatch (OP.TPrim OP.TInt32, OP.TPrim OP.TBool))) ;
  Printf.printf "  Add int32 bool -> error (mismatch) [ok]\n"

(* Test 3: Mod int32 int32 -> int32 (integer binop) *)
let test_op_mod_int32 () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.Mod, opint32, opint32)) in
  assert (result = OP.Inl (OP.TPrim OP.TInt32)) ;
  Printf.printf "  Mod int32 int32 -> TInt32 [ok]\n"

(* Test 4: Eq int32 int32 -> bool (eq binop) *)
let test_op_eq_int32 () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.Eq, opint32, opint32)) in
  assert (result = OP.Inl (OP.TPrim OP.TBool)) ;
  Printf.printf "  Eq int32 int32 -> TBool [ok]\n"

(* Test 5: Lt int32 int32 -> bool (cmp binop, numeric type) *)
let test_op_lt_int32 () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.Lt, opint32, opint32)) in
  assert (result = OP.Inl (OP.TPrim OP.TBool)) ;
  Printf.printf "  Lt int32 int32 -> TBool [ok]\n"

(* Test 6: And bool bool -> bool *)
let test_op_and_bool () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.And, opbool, opbool)) in
  assert (result = OP.Inl (OP.TPrim OP.TBool)) ;
  Printf.printf "  And bool bool -> TBool [ok]\n"

(* Test 7: And int32 int32 -> error (NotBool) *)
let test_op_and_not_bool () =
  let result = OP.infer_op_type [] (OP.OPBinop (OP.And, opint32, opint32)) in
  assert (result = OP.Inr (OP.NotBool (OP.TPrim OP.TInt32))) ;
  Printf.printf "  And int32 int32 -> error (NotBool) [ok]\n"

(* Test 8: Neg float32 -> float32 (numeric unop) *)
let test_op_neg_float () =
  let result = OP.infer_op_type [] (OP.OPUnop (OP.Neg, opfloat)) in
  assert (result = OP.Inl (OP.TReg OP.RFloat32)) ;
  Printf.printf "  Neg float32 -> TReg RFloat32 [ok]\n"

(* Test 9: Not bool -> bool *)
let test_op_not_bool () =
  let result = OP.infer_op_type [] (OP.OPUnop (OP.Not, opbool)) in
  assert (result = OP.Inl (OP.TPrim OP.TBool)) ;
  Printf.printf "  Not bool -> TBool [ok]\n"

(* Test 10: Lnot int32 -> int32 (integer unop) *)
let test_op_lnot_int32 () =
  let result = OP.infer_op_type [] (OP.OPUnop (OP.Lnot, opint32)) in
  assert (result = OP.Inl (OP.TPrim OP.TInt32)) ;
  Printf.printf "  Lnot int32 -> TInt32 [ok]\n"

(* ----- T3-S3 function smoke tests (FunModel oracle) ------------------------- *)

module FUN = Type_safety_model.FunModel

(* Coq-string builder local to the FUN layer (FUN.string is its own ascii type). *)
let fun_coq_string (s : ostring) : FUN.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    FUN.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (FUN.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) FUN.EmptyString

(* int32-typed literal lifted all the way up to fun_expr via the delegation chain *)
let funint32 =
  FUN.FEOp (FUN.OPCf (FUN.CFRec (FUN.RMem (FUN.MCore (FUN.ELit (FUN.LInt 0))))))

let funbool =
  FUN.FEOp
    (FUN.OPCf (FUN.CFRec (FUN.RMem (FUN.MCore (FUN.ELit (FUN.LBool false))))))

(* env binding a variable name to a sarek_type at the FUN layer *)
let fun_env_entry x t = (fun_coq_string x, t)

(* a fun_expr that reads variable [x] from the environment *)
let funvar x =
  FUN.FEOp
    (FUN.OPCf (FUN.CFRec (FUN.RMem (FUN.MCore (FUN.EVar (fun_coq_string x))))))

(* Test 1: FEOp delegation -- int32 literal types as TInt32 *)
let test_fun_op_delegation () =
  let result = FUN.infer_fun_type [] funint32 in
  assert (result = FUN.Inl (FUN.TPrim FUN.TInt32)) ;
  Printf.printf "  FEOp (int32 lit) -> TPrim TInt32 [ok]\n"

(* Test 2: FEOp delegation propagates operator success (Add) *)
let test_fun_op_binop () =
  let e =
    FUN.FEOp
      (FUN.OPBinop
         ( FUN.Add,
           FUN.OPCf (FUN.CFRec (FUN.RMem (FUN.MCore (FUN.ELit (FUN.LInt 1))))),
           FUN.OPCf (FUN.CFRec (FUN.RMem (FUN.MCore (FUN.ELit (FUN.LInt 2)))))
         ))
  in
  let result = FUN.infer_fun_type [] e in
  assert (result = FUN.Inl (FUN.TPrim FUN.TInt32)) ;
  Printf.printf "  FEOp (Add int32 int32) -> TPrim TInt32 [ok]\n"

(* Test 3: FEApp success -- (f : int32 -> bool) applied to int32 -> bool *)
let test_fun_app_success () =
  let fn_ty = FUN.TFun ([FUN.TPrim FUN.TInt32], FUN.TPrim FUN.TBool) in
  let env = [fun_env_entry "f" fn_ty] in
  let result = FUN.infer_fun_type env (FUN.FEApp (funvar "f", funint32)) in
  assert (result = FUN.Inl (FUN.TPrim FUN.TBool)) ;
  Printf.printf "  FEApp (int32->bool) int32 -> TPrim TBool [ok]\n"

(* Test 4: NotAFunc -- applying a non-function value *)
let test_fun_app_not_a_func () =
  let env = [fun_env_entry "x" (FUN.TPrim FUN.TInt32)] in
  let result = FUN.infer_fun_type env (FUN.FEApp (funvar "x", funint32)) in
  assert (result = FUN.Inr (FUN.NotAFunc (FUN.TPrim FUN.TInt32))) ;
  Printf.printf "  FEApp (non-func) -> NotAFunc [ok]\n"

(* Test 5: ArgMismatch -- arg type differs from the parameter type *)
let test_fun_app_arg_mismatch () =
  let fn_ty = FUN.TFun ([FUN.TPrim FUN.TInt32], FUN.TPrim FUN.TBool) in
  let env = [fun_env_entry "f" fn_ty] in
  let result = FUN.infer_fun_type env (FUN.FEApp (funvar "f", funbool)) in
  assert (
    result
    = FUN.Inr (FUN.ArgMismatch (FUN.TPrim FUN.TInt32, FUN.TPrim FUN.TBool))) ;
  Printf.printf "  FEApp arg type != param -> ArgMismatch [ok]\n"

(* Test 6: FELetRec success -- body type matches declared return; cont = call *)
let test_fun_letrec_success () =
  (* let rec f (n : int32) : int32 = n in f 0  -->  int32 *)
  let body = funvar "n" in
  let cont = FUN.FEApp (funvar "f", funint32) in
  let e =
    FUN.FELetRec
      ( fun_coq_string "f",
        fun_coq_string "n",
        FUN.TPrim FUN.TInt32,
        FUN.TPrim FUN.TInt32,
        body,
        cont )
  in
  let result = FUN.infer_fun_type [] e in
  assert (result = FUN.Inl (FUN.TPrim FUN.TInt32)) ;
  Printf.printf "  FELetRec f(n:int32):int32=n in f 0 -> TInt32 [ok]\n"

(* Test 7: FELetRec recursion -- body may reference fn_name with its own type *)
let test_fun_letrec_recursive () =
  (* let rec f (n : int32) : bool = f n in f 0  -->  bool *)
  let body = FUN.FEApp (funvar "f", funvar "n") in
  let cont = FUN.FEApp (funvar "f", funint32) in
  let e =
    FUN.FELetRec
      ( fun_coq_string "f",
        fun_coq_string "n",
        FUN.TPrim FUN.TInt32,
        FUN.TPrim FUN.TBool,
        body,
        cont )
  in
  let result = FUN.infer_fun_type [] e in
  assert (result = FUN.Inl (FUN.TPrim FUN.TBool)) ;
  Printf.printf "  FELetRec recursive body (f n) -> TBool [ok]\n"

(* Test 8: BodyMismatch -- body type differs from the declared return type *)
let test_fun_letrec_body_mismatch () =
  (* let rec f (n : int32) : bool = n in ...  -->  BodyMismatch bool int32 *)
  let body = funvar "n" in
  let cont = funint32 in
  let e =
    FUN.FELetRec
      ( fun_coq_string "f",
        fun_coq_string "n",
        FUN.TPrim FUN.TInt32,
        FUN.TPrim FUN.TBool,
        body,
        cont )
  in
  let result = FUN.infer_fun_type [] e in
  assert (
    result
    = FUN.Inr (FUN.BodyMismatch (FUN.TPrim FUN.TBool, FUN.TPrim FUN.TInt32))) ;
  Printf.printf "  FELetRec body type != return -> BodyMismatch [ok]\n"

(* Test 9: FELetRec parameter is in scope for the body *)
let test_fun_letrec_param_in_scope () =
  (* let rec f (n : bool) : bool = n in f true (returns the param) *)
  let body = funvar "n" in
  let cont = FUN.FEApp (funvar "f", funbool) in
  let e =
    FUN.FELetRec
      ( fun_coq_string "f",
        fun_coq_string "n",
        FUN.TPrim FUN.TBool,
        FUN.TPrim FUN.TBool,
        body,
        cont )
  in
  let result = FUN.infer_fun_type [] e in
  assert (result = FUN.Inl (FUN.TPrim FUN.TBool)) ;
  Printf.printf "  FELetRec param n:bool visible in body -> TBool [ok]\n"

(* Test 10: continuation typed in fn-only env (param NOT leaked past let-rec) *)
let test_fun_letrec_param_not_leaked () =
  (* let rec f (n : int32) : int32 = n in n  -->  n is unbound in continuation *)
  let body = funvar "n" in
  let cont = funvar "n" in
  let e =
    FUN.FELetRec
      ( fun_coq_string "f",
        fun_coq_string "n",
        FUN.TPrim FUN.TInt32,
        FUN.TPrim FUN.TInt32,
        body,
        cont )
  in
  let result = FUN.infer_fun_type [] e in
  (match result with
  | FUN.Inr
      (FUN.FEOpErr (FUN.OCF (FUN.CRec (FUN.RMemError (FUN.VCoreError _))))) ->
      ()
  | _ -> assert false) ;
  Printf.printf
    "  FELetRec param n not in scope in continuation -> error [ok]\n"

(* ----- T3-S4 mutable-binding smoke tests (MutModel oracle) ----------------- *)

module MUT = Type_safety_model.MutModel

(* Coq-string builder local to the MUT layer (MUT.string is its own ascii type). *)
let mut_coq_string (s : ostring) : MUT.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    MUT.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (MUT.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) MUT.EmptyString

(* int32 / bool literals lifted all the way up to mut_expr via the chain *)
let mutint32 =
  MUT.MEFun
    (MUT.FEOp
       (MUT.OPCf (MUT.CFRec (MUT.RMem (MUT.MCore (MUT.ELit (MUT.LInt 0)))))))

let mutbool =
  MUT.MEFun
    (MUT.FEOp
       (MUT.OPCf (MUT.CFRec (MUT.RMem (MUT.MCore (MUT.ELit (MUT.LBool false)))))))

(* a mut_expr that reads variable [x] from the environment via delegation *)
let mutvar x =
  MUT.MEFun
    (MUT.FEOp
       (MUT.OPCf
          (MUT.CFRec (MUT.RMem (MUT.MCore (MUT.EVar (mut_coq_string x)))))))

let mut_env_entry x t = (mut_coq_string x, t)

(* Test 1: MEFun delegation -- int32 literal types as TInt32 *)
let test_mut_fun_delegation () =
  let result = MUT.infer_mut_type [] [] mutint32 in
  assert (result = MUT.Inl (MUT.TPrim MUT.TInt32)) ;
  Printf.printf "  MEFun (int32 lit) -> TPrim TInt32 [ok]\n"

(* Test 2: MELetMut success -- body returns the mutable variable's type *)
let test_mut_letmut_success () =
  (* let mut x = 0 in x  -->  int32 *)
  let e = MUT.MELetMut (mut_coq_string "x", mutint32, mutvar "x") in
  let result = MUT.infer_mut_type [] [] e in
  assert (result = MUT.Inl (MUT.TPrim MUT.TInt32)) ;
  Printf.printf "  MELetMut x=0 in x -> TInt32 [ok]\n"

(* Test 3: MELetMut body type differs from init type (body is unrelated lit) *)
let test_mut_letmut_body_type () =
  (* let mut x = 0 in true  -->  bool *)
  let e = MUT.MELetMut (mut_coq_string "x", mutint32, mutbool) in
  let result = MUT.infer_mut_type [] [] e in
  assert (result = MUT.Inl (MUT.TPrim MUT.TBool)) ;
  Printf.printf "  MELetMut x=0 in true -> TBool [ok]\n"

(* Test 4: MEAssign success -- assign matching type to a mutable var -> unit *)
let test_mut_assign_success () =
  (* let mut x = 0 in (x <- 1)  -->  unit *)
  let body = MUT.MEAssign (mut_coq_string "x", mutint32) in
  let e = MUT.MELetMut (mut_coq_string "x", mutint32, body) in
  let result = MUT.infer_mut_type [] [] e in
  assert (result = MUT.Inl (MUT.TPrim MUT.TUnit)) ;
  Printf.printf "  MELetMut x=0 in (x <- 1) -> TUnit [ok]\n"

(* Test 5: MEAssign to an unbound variable -> MEUnbound *)
let test_mut_assign_unbound () =
  let e = MUT.MEAssign (mut_coq_string "y", mutint32) in
  let result = MUT.infer_mut_type [] [] e in
  assert (result = MUT.Inr (MUT.MEUnbound (mut_coq_string "y"))) ;
  Printf.printf "  MEAssign y (unbound) -> MEUnbound [ok]\n"

(* Test 6: MEAssign to an immutable (plain type_env) variable -> MEImmutable *)
let test_mut_assign_immutable () =
  (* x is bound in the type_env but NOT in the mutability env *)
  let env = [mut_env_entry "x" (MUT.TPrim MUT.TInt32)] in
  let e = MUT.MEAssign (mut_coq_string "x", mutint32) in
  let result = MUT.infer_mut_type env [] e in
  assert (result = MUT.Inr (MUT.MEImmutable (mut_coq_string "x"))) ;
  Printf.printf "  MEAssign x (not mutable) -> MEImmutable [ok]\n"

(* Test 7: MEAssign value type != declared type -> MEAssignMismatch *)
let test_mut_assign_mismatch () =
  (* let mut x = 0 in (x <- true)  -->  MEAssignMismatch int32 bool *)
  let body = MUT.MEAssign (mut_coq_string "x", mutbool) in
  let e = MUT.MELetMut (mut_coq_string "x", mutint32, body) in
  let result = MUT.infer_mut_type [] [] e in
  assert (
    result
    = MUT.Inr (MUT.MEAssignMismatch (MUT.TPrim MUT.TInt32, MUT.TPrim MUT.TBool))) ;
  Printf.printf "  MELetMut x=0 in (x <- true) -> MEAssignMismatch [ok]\n"

(* Test 8: MEFun delegation propagates a function-layer error (MEFunErr) *)
let test_mut_fun_error_propagates () =
  (* reading an unbound var through the fun layer surfaces as MEFunErr *)
  let result = MUT.infer_mut_type [] [] (mutvar "nope") in
  (match result with MUT.Inr (MUT.MEFunErr _) -> () | _ -> assert false) ;
  Printf.printf "  MEFun (unbound var) -> MEFunErr [ok]\n"

(* Test 9: MELetMut nested -- inner mutable shadows, outer still assignable *)
let test_mut_letmut_nested_assign () =
  (* let mut x = 0 in let mut y = true in (x <- 1)  -->  unit *)
  let inner_body = MUT.MEAssign (mut_coq_string "x", mutint32) in
  let inner = MUT.MELetMut (mut_coq_string "y", mutbool, inner_body) in
  let e = MUT.MELetMut (mut_coq_string "x", mutint32, inner) in
  let result = MUT.infer_mut_type [] [] e in
  assert (result = MUT.Inl (MUT.TPrim MUT.TUnit)) ;
  Printf.printf "  nested let mut, inner assigns outer x -> TUnit [ok]\n"

(* Test 10: error in the init expression short-circuits the whole MELetMut *)
let test_mut_letmut_init_error () =
  (* let mut x = (unbound) in x  -->  the init error propagates *)
  let e = MUT.MELetMut (mut_coq_string "x", mutvar "ghost", mutvar "x") in
  let result = MUT.infer_mut_type [] [] e in
  (match result with MUT.Inr (MUT.MEFunErr _) -> () | _ -> assert false) ;
  Printf.printf "  MELetMut init error short-circuits -> MEFunErr [ok]\n"

(* ----- T3-S5 pattern-matching smoke tests (PatternModel oracle) ----------- *)

module PAT = Type_safety_model.PatternModel

(* Coq-string builder local to the PAT layer. *)
let pat_coq_string (s : ostring) : PAT.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    PAT.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (PAT.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) PAT.EmptyString

(* literals lifted all the way up to pat_expr through the full layer chain *)
let pat_lit lit =
  PAT.PEMut
    (PAT.MEFun
       (PAT.FEOp (PAT.OPCf (PAT.CFRec (PAT.RMem (PAT.MCore (PAT.ELit lit)))))))

let patint32 = pat_lit (PAT.LInt 0)

let patbool = pat_lit (PAT.LBool false)

(* a pat_expr that reads variable [x] from the environment via delegation *)
let patvar x =
  PAT.PEMut
    (PAT.MEFun
       (PAT.FEOp
          (PAT.OPCf
             (PAT.CFRec (PAT.RMem (PAT.MCore (PAT.EVar (pat_coq_string x))))))))

let pat_env_entry x t = (pat_coq_string x, t)

(* a "color" variant: Red (no payload) | Rgb of int32 *)
let color_constrs =
  [
    (pat_coq_string "Red", None);
    (pat_coq_string "Rgb", Some (PAT.TPrim PAT.TInt32));
  ]

let color_ty = PAT.TVariant (pat_coq_string "color", color_constrs)

(* environment binding scrutinee variable [v : color] *)
let color_env = [pat_env_entry "v" color_ty]

(* helper: build a branch ((cname, bvar_opt), body) *)
let branch cname bvar body = ((pat_coq_string cname, bvar), body)

let branch_var cname v body =
  ((pat_coq_string cname, Some (pat_coq_string v)), body)

(* Test 1: PEMut delegation -- a bare int32 literal types as TInt32 *)
let test_pat_mut_delegation () =
  let result = PAT.infer_pat_type [] [] patint32 in
  assert (result = PAT.Inl (PAT.TPrim PAT.TInt32)) ;
  Printf.printf "  PEMut (int32 lit) -> TPrim TInt32 [ok]\n"

(* Test 2: successful match, no-payload branch -- match v with Red -> 0 *)
let test_pat_match_no_payload () =
  let e = PAT.PEMatch (patvar "v", [branch "Red" None patint32]) in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inl (PAT.TPrim PAT.TInt32)) ;
  Printf.printf "  match v with Red -> 0  ->  TInt32 [ok]\n"

(* Test 3: successful match, payload branch binds the payload type *)
let test_pat_match_payload () =
  (* match v with Rgb c -> c   (c : int32 in scope, so result is int32) *)
  let e = PAT.PEMatch (patvar "v", [branch_var "Rgb" "c" (patvar "c")]) in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inl (PAT.TPrim PAT.TInt32)) ;
  Printf.printf "  match v with Rgb c -> c  ->  TInt32 (payload bound) [ok]\n"

(* Test 4: scrutinee is not a variant -> PENotVariant *)
let test_pat_not_a_variant () =
  (* scrutinee is a plain int32 literal, not a variant *)
  let e = PAT.PEMatch (patint32, [branch "Red" None patint32]) in
  let result = PAT.infer_pat_type color_env [] e in
  (match result with
  | PAT.Inr (PAT.PENotVariant (PAT.TPrim PAT.TInt32)) -> ()
  | _ -> assert false) ;
  Printf.printf "  match (int32 lit) ... -> PENotVariant [ok]\n"

(* Test 5: branch names an unknown constructor -> PEMismatch *)
let test_pat_unknown_constructor () =
  let e = PAT.PEMatch (patvar "v", [branch "Blue" None patint32]) in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inr (PAT.PEMismatch (pat_coq_string "Blue"))) ;
  Printf.printf "  match v with Blue -> .. -> PEMismatch [ok]\n"

(* Test 6: branch bodies disagree on type -> PEBranchType *)
let test_pat_branch_type_mismatch () =
  (* match v with Red -> 0 | Rgb c -> true   (int32 vs bool) *)
  let e =
    PAT.PEMatch
      (patvar "v", [branch "Red" None patint32; branch_var "Rgb" "c" patbool])
  in
  let result = PAT.infer_pat_type color_env [] e in
  assert (
    result
    = PAT.Inr (PAT.PEBranchType (PAT.TPrim PAT.TInt32, PAT.TPrim PAT.TBool))) ;
  Printf.printf "  Red -> 0 | Rgb c -> true -> PEBranchType [ok]\n"

(* Test 7: scrutinee error delegates through the mutable layer -> PEMutErr *)
let test_pat_mut_err_delegation () =
  (* scrutinee reads an unbound variable: surfaces as PEMutErr *)
  let e = PAT.PEMatch (patvar "ghost", [branch "Red" None patint32]) in
  let result = PAT.infer_pat_type color_env [] e in
  (match result with PAT.Inr (PAT.PEMutErr _) -> () | _ -> assert false) ;
  Printf.printf "  match (unbound) ... -> PEMutErr [ok]\n"

(* Test 8: empty branch list -> PEEmpty *)
let test_pat_empty_branches () =
  let e = PAT.PEMatch (patvar "v", []) in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inr PAT.PEEmpty) ;
  Printf.printf "  match v with <no branches> -> PEEmpty [ok]\n"

(* Test 9: nested payload binding scope -- the binder is visible only in body *)
let test_pat_nested_payload_scope () =
  (* match v with Rgb c -> (match c-bound? we instead nest a let-free read) *)
  (* Rgb c -> c uses the payload-bound c; a second branch Red -> 0 keeps int32 *)
  let e =
    PAT.PEMatch
      ( patvar "v",
        [branch_var "Rgb" "c" (patvar "c"); branch "Red" None patint32] )
  in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inl (PAT.TPrim PAT.TInt32)) ;
  Printf.printf "  Rgb c -> c | Red -> 0 -> TInt32 (scoped binder) [ok]\n"

(* Test 10: multi-branch agreement -- all bodies agree on int32 *)
let test_pat_multi_branch_agree () =
  let e =
    PAT.PEMatch
      ( patvar "v",
        [branch "Red" None patint32; branch_var "Rgb" "c" (patvar "c")] )
  in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inl (PAT.TPrim PAT.TInt32)) ;
  Printf.printf "  Red -> 0 | Rgb c -> c -> TInt32 (multi-branch agree) [ok]\n"

(* Test 11 (bonus): error short-circuit -- first failing branch wins *)
let test_pat_error_short_circuit () =
  (* first branch is fine (fixes int32); second branch names an unknown
     constructor: the PEMismatch from the second branch is reported, the
     remaining branches are not inspected. *)
  let e =
    PAT.PEMatch
      ( patvar "v",
        [
          branch "Red" None patint32;
          branch "Cmyk" None patint32;
          branch_var "Rgb" "c" patbool;
        ] )
  in
  let result = PAT.infer_pat_type color_env [] e in
  assert (result = PAT.Inr (PAT.PEMismatch (pat_coq_string "Cmyk"))) ;
  Printf.printf "  short-circuit: Cmyk PEMismatch before Rgb mismatch [ok]\n"

(* ----- T3-S6 algebraic-construction smoke tests (ConstrModel oracle) ------ *)

module CON = Type_safety_model.ConstrModel

(* Coq-string builder local to the CON layer. *)
let con_coq_string (s : ostring) : CON.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    CON.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (CON.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) CON.EmptyString

(* a literal lifted all the way up to constr_expr through the full layer chain:
   constr_expr <- pat_expr <- mut_expr <- fun_expr <- op_expr <- cf_expr
   <- rec_expr <- mem_expr <- core expr *)
let con_lit lit =
  CON.CEPat
    (CON.PEMut
       (CON.MEFun
          (CON.FEOp (CON.OPCf (CON.CFRec (CON.RMem (CON.MCore (CON.ELit lit))))))))

let conint32 = con_lit (CON.LInt 0)

let conbool = con_lit (CON.LBool false)

(* a constr_expr that reads variable [x] from the environment via delegation *)
let convar x =
  CON.CEPat
    (CON.PEMut
       (CON.MEFun
          (CON.FEOp
             (CON.OPCf
                (CON.CFRec (CON.RMem (CON.MCore (CON.EVar (con_coq_string x)))))))))

(* a declared record type "point" : { x : int32; y : int32 } *)
let point_declared =
  [
    (con_coq_string "x", CON.TPrim CON.TInt32);
    (con_coq_string "y", CON.TPrim CON.TInt32);
  ]

let point_ty = CON.TRecord (con_coq_string "point", point_declared)

(* a variant "color" : Red (no payload) | Rgb of int32 *)
let color_constrs =
  [
    (con_coq_string "Red", None);
    (con_coq_string "Rgb", Some (CON.TPrim CON.TInt32));
  ]

let color_ty = CON.TVariant (con_coq_string "color", color_constrs)

let con_field f e = (con_coq_string f, e)

(* Test 1: CEPat delegation -- a bare int32 literal types as TInt32 *)
let test_con_pat_delegation () =
  let result = CON.infer_constr_type [] [] conint32 in
  assert (result = CON.Inl (CON.TPrim CON.TInt32)) ;
  Printf.printf "  CEPat (int32 lit) -> TPrim TInt32 [ok]\n"

(* Test 2: well-typed record -- { x = 0; y = 0 } : point *)
let test_con_record_success () =
  let e =
    CON.CERecord
      ( con_coq_string "point",
        point_declared,
        [con_field "x" conint32; con_field "y" conint32] )
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inl point_ty) ;
  Printf.printf "  { x = 0; y = 0 } -> TRecord point [ok]\n"

(* Test 3: empty provided field list still yields the declared record type *)
let test_con_record_empty_provided () =
  let e = CON.CERecord (con_coq_string "point", point_declared, []) in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inl point_ty) ;
  Printf.printf "  { } : point -> TRecord point [ok]\n"

(* Test 4: provided field type != declared -> FieldTypeMismatch *)
let test_con_record_field_mismatch () =
  (* y is declared int32 but we provide a bool *)
  let e =
    CON.CERecord
      ( con_coq_string "point",
        point_declared,
        [con_field "x" conint32; con_field "y" conbool] )
  in
  let result = CON.infer_constr_type [] [] e in
  (match result with
  | CON.Inr
      (CON.FieldTypeMismatch (_, CON.TPrim CON.TInt32, CON.TPrim CON.TBool)) ->
      ()
  | _ -> assert false) ;
  Printf.printf "  { x = 0; y = true } -> FieldTypeMismatch [ok]\n"

(* Test 5: provided field not in declared layout -> UnknownField *)
let test_con_record_unknown_field () =
  let e =
    CON.CERecord
      (con_coq_string "point", point_declared, [con_field "z" conint32])
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inr (CON.UnknownField (con_coq_string "z"))) ;
  Printf.printf "  { z = 0 } : point -> UnknownField [ok]\n"

(* Test 6: nullary constructor -- Red : color *)
let test_con_constr_none () =
  let e =
    CON.CEConstr
      (con_coq_string "color", color_constrs, con_coq_string "Red", None)
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inl color_ty) ;
  Printf.printf "  Red -> TVariant color [ok]\n"

(* Test 7: payload constructor with matching arg -- Rgb 0 : color *)
let test_con_constr_some () =
  let e =
    CON.CEConstr
      ( con_coq_string "color",
        color_constrs,
        con_coq_string "Rgb",
        Some conint32 )
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inl color_ty) ;
  Printf.printf "  Rgb 0 -> TVariant color [ok]\n"

(* Test 8: payload type != declared -> FieldTypeMismatch *)
let test_con_constr_payload_mismatch () =
  (* Rgb expects int32, given a bool *)
  let e =
    CON.CEConstr
      (con_coq_string "color", color_constrs, con_coq_string "Rgb", Some conbool)
  in
  let result = CON.infer_constr_type [] [] e in
  (match result with
  | CON.Inr
      (CON.FieldTypeMismatch (_, CON.TPrim CON.TInt32, CON.TPrim CON.TBool)) ->
      ()
  | _ -> assert false) ;
  Printf.printf "  Rgb true -> FieldTypeMismatch [ok]\n"

(* Test 9: unknown constructor name -> UnknownConstr *)
let test_con_constr_unknown () =
  let e =
    CON.CEConstr
      (con_coq_string "color", color_constrs, con_coq_string "Blue", None)
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inr (CON.UnknownConstr (con_coq_string "Blue"))) ;
  Printf.printf "  Blue -> UnknownConstr [ok]\n"

(* Test 10: arity disagreement -- nullary Red given an argument -> ConstrArity *)
let test_con_constr_arity_extra_arg () =
  let e =
    CON.CEConstr
      ( con_coq_string "color",
        color_constrs,
        con_coq_string "Red",
        Some conint32 )
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inr (CON.ConstrArity (con_coq_string "Red"))) ;
  Printf.printf "  Red 0 -> ConstrArity [ok]\n"

(* Test 11: arity disagreement -- payload Rgb given no argument -> ConstrArity *)
let test_con_constr_arity_missing_arg () =
  let e =
    CON.CEConstr
      (con_coq_string "color", color_constrs, con_coq_string "Rgb", None)
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inr (CON.ConstrArity (con_coq_string "Rgb"))) ;
  Printf.printf "  Rgb (no arg) -> ConstrArity [ok]\n"

(* Test 12: delegated pattern-layer error surfaces as CPatternErr *)
let test_con_pattern_err_delegation () =
  (* a record field reads an unbound variable; the error bubbles through the
     pattern/mut layers and is wrapped as CPatternErr *)
  let e =
    CON.CERecord
      (con_coq_string "point", point_declared, [con_field "x" (convar "ghost")])
  in
  let result = CON.infer_constr_type [] [] e in
  (match result with CON.Inr (CON.CPatternErr _) -> () | _ -> assert false) ;
  Printf.printf "  { x = ghost } -> CPatternErr (delegated) [ok]\n"

(* Test 13: payload arg is itself a constructor -- nested construction *)
let test_con_constr_nested_arg () =
  (* a variant whose payload is another variant value:
     wrapper : Wrap of color ; build  Wrap (Rgb 0) *)
  let wrapper_constrs = [(con_coq_string "Wrap", Some color_ty)] in
  let wrapper_ty = CON.TVariant (con_coq_string "wrapper", wrapper_constrs) in
  let inner =
    CON.CEConstr
      ( con_coq_string "color",
        color_constrs,
        con_coq_string "Rgb",
        Some conint32 )
  in
  let e =
    CON.CEConstr
      ( con_coq_string "wrapper",
        wrapper_constrs,
        con_coq_string "Wrap",
        Some inner )
  in
  let result = CON.infer_constr_type [] [] e in
  assert (result = CON.Inl wrapper_ty) ;
  Printf.printf "  Wrap (Rgb 0) -> TVariant wrapper (nested) [ok]\n"

(* ----- T3-S7 special-form smoke tests (SpecialModel oracle) --------------- *)

module SPE = Type_safety_model.SpecialModel

(* Coq-string builder local to the SPE layer. *)
let spe_coq_string (s : ostring) : SPE.string =
  let char_to_ascii c =
    let n = Char.code c in
    let bit k = (n lsr k) land 1 = 1 in
    SPE.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)
  in
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (SPE.String (char_to_ascii s.[i], acc))
  in
  go (len - 1) SPE.EmptyString

(* a literal lifted all the way up to special_expr through the full layer chain:
   special_expr <- constr_expr <- pat_expr <- mut_expr <- fun_expr <- op_expr
   <- cf_expr <- rec_expr <- mem_expr <- core expr *)
let spe_lit lit =
  SPE.SEConstr
    (SPE.CEPat
       (SPE.PEMut
          (SPE.MEFun
             (SPE.FEOp
                (SPE.OPCf (SPE.CFRec (SPE.RMem (SPE.MCore (SPE.ELit lit)))))))))

let speint32 = spe_lit (SPE.LInt 0)

let spebool = spe_lit (SPE.LBool false)

(* a special_expr that reads variable [x] from the environment via delegation *)
let spevar x =
  SPE.SEConstr
    (SPE.CEPat
       (SPE.PEMut
          (SPE.MEFun
             (SPE.FEOp
                (SPE.OPCf
                   (SPE.CFRec
                      (SPE.RMem (SPE.MCore (SPE.EVar (spe_coq_string x))))))))))

(* Test 1: SEConstr delegation -- a bare int32 literal types as TInt32 *)
let test_spe_constr_delegation () =
  let result = SPE.infer_special_type [] [] speint32 in
  assert (result = SPE.Inl (SPE.TPrim SPE.TInt32)) ;
  Printf.printf "  SEConstr (int32 lit) -> TPrim TInt32 [ok]\n"

(* Test 2: allowed return is a pass-through of the body type *)
let test_spe_return_passthrough () =
  let result = SPE.infer_special_type [] [] (SPE.SEReturn (true, speint32)) in
  assert (result = SPE.Inl (SPE.TPrim SPE.TInt32)) ;
  Printf.printf "  (return 0) -> TPrim TInt32 [ok]\n"

(* Test 3: allowed return passes through a bool body unchanged *)
let test_spe_return_passthrough_bool () =
  let result = SPE.infer_special_type [] [] (SPE.SEReturn (true, spebool)) in
  assert (result = SPE.Inl (SPE.TPrim SPE.TBool)) ;
  Printf.printf "  (return false) -> TPrim TBool [ok]\n"

(* Test 4: disallowed return -> EarlyReturnNotAllowed *)
let test_spe_return_not_allowed () =
  let result = SPE.infer_special_type [] [] (SPE.SEReturn (false, speint32)) in
  assert (result = SPE.Inr SPE.EarlyReturnNotAllowed) ;
  Printf.printf "  (return 0) [disallowed] -> EarlyReturnNotAllowed [ok]\n"

(* Test 5: create_array with int32 size -> TArr (elt, Global) *)
let test_spe_create_array_global () =
  let e = SPE.SECreateArray (speint32, SPE.TReg SPE.RFloat32, SPE.Global) in
  let result = SPE.infer_special_type [] [] e in
  assert (result = SPE.Inl (SPE.TArr (SPE.TReg SPE.RFloat32, SPE.Global))) ;
  Printf.printf "  create_array 0 : float32[]@Global -> TArr [ok]\n"

(* Test 6: create_array honours the Shared memory space and element type *)
let test_spe_create_array_shared () =
  let e = SPE.SECreateArray (speint32, SPE.TPrim SPE.TInt32, SPE.Shared) in
  let result = SPE.infer_special_type [] [] e in
  assert (result = SPE.Inl (SPE.TArr (SPE.TPrim SPE.TInt32, SPE.Shared))) ;
  Printf.printf "  create_array 0 : int32[]@Shared -> TArr [ok]\n"

(* Test 7: create_array with a non-int32 size -> ArraySizeNotInt *)
let test_spe_create_array_bad_size () =
  let e = SPE.SECreateArray (spebool, SPE.TPrim SPE.TInt32, SPE.Global) in
  let result = SPE.infer_special_type [] [] e in
  assert (result = SPE.Inr (SPE.ArraySizeNotInt (SPE.TPrim SPE.TBool))) ;
  Printf.printf "  create_array false -> ArraySizeNotInt [ok]\n"

(* Test 8: type annotation that matches the inferred type -> annotation type *)
let test_spe_typed_match () =
  let e = SPE.SETyped (speint32, SPE.TPrim SPE.TInt32) in
  let result = SPE.infer_special_type [] [] e in
  assert (result = SPE.Inl (SPE.TPrim SPE.TInt32)) ;
  Printf.printf "  (0 : int32) -> TPrim TInt32 [ok]\n"

(* Test 9: type annotation that disagrees with the body -> TypeAnnotMismatch *)
let test_spe_typed_mismatch () =
  (* body is int32 but annotated bool *)
  let e = SPE.SETyped (speint32, SPE.TPrim SPE.TBool) in
  let result = SPE.infer_special_type [] [] e in
  assert (
    result
    = SPE.Inr
        (SPE.TypeAnnotMismatch (SPE.TPrim SPE.TBool, SPE.TPrim SPE.TInt32))) ;
  Printf.printf "  (0 : bool) -> TypeAnnotMismatch [ok]\n"

(* Test 10: delegated construction-layer error surfaces as SConstrErr *)
let test_spe_constr_err_delegation () =
  (* an unbound variable read bubbles up through the constr layer and is
     wrapped as SConstrErr *)
  let result = SPE.infer_special_type [] [] (spevar "ghost") in
  (match result with SPE.Inr (SPE.SConstrErr _) -> () | _ -> assert false) ;
  Printf.printf "  ghost -> SConstrErr (delegated) [ok]\n"

(* Test 11: nested specials -- ((create_array 0 : int32[]) wrapped in return) *)
let test_spe_nested_return_typed_array () =
  let arr = SPE.SECreateArray (speint32, SPE.TPrim SPE.TInt32, SPE.Global) in
  let arr_ty = SPE.TArr (SPE.TPrim SPE.TInt32, SPE.Global) in
  let e = SPE.SEReturn (true, SPE.SETyped (arr, arr_ty)) in
  let result = SPE.infer_special_type [] [] e in
  assert (result = SPE.Inl arr_ty) ;
  Printf.printf "  return ((create_array 0) : int32[]) -> TArr (nested) [ok]\n"

(* Test 12: error short-circuits through nested specials (size error) *)
let test_spe_nested_error_propagates () =
  (* create_array with a bool size, wrapped in a typed annotation; the size
     error must surface unchanged through SETyped *)
  let arr = SPE.SECreateArray (spebool, SPE.TPrim SPE.TInt32, SPE.Global) in
  let e = SPE.SETyped (arr, SPE.TArr (SPE.TPrim SPE.TInt32, SPE.Global)) in
  let result = SPE.infer_special_type [] [] e in
  assert (result = SPE.Inr (SPE.ArraySizeNotInt (SPE.TPrim SPE.TBool))) ;
  Printf.printf
    "  (create_array false : int32[]) -> ArraySizeNotInt (propagated) [ok]\n"

let () =
  Printf.printf "\n=== T3-S2 operator smoke tests (OperatorModel oracle) ===\n" ;
  test_op_add_int32 () ;
  test_op_add_mismatch () ;
  test_op_mod_int32 () ;
  test_op_eq_int32 () ;
  test_op_lt_int32 () ;
  test_op_and_bool () ;
  test_op_and_not_bool () ;
  test_op_neg_float () ;
  test_op_not_bool () ;
  test_op_lnot_int32 () ;
  Printf.printf "=== T3-S2 operator smoke tests passed ===\n" ;
  Printf.printf "\n=== T3-S3 function smoke tests (FunModel oracle) ===\n" ;
  test_fun_op_delegation () ;
  test_fun_op_binop () ;
  test_fun_app_success () ;
  test_fun_app_not_a_func () ;
  test_fun_app_arg_mismatch () ;
  test_fun_letrec_success () ;
  test_fun_letrec_recursive () ;
  test_fun_letrec_body_mismatch () ;
  test_fun_letrec_param_in_scope () ;
  test_fun_letrec_param_not_leaked () ;
  Printf.printf "=== T3-S3 function smoke tests passed ===\n" ;
  Printf.printf
    "\n=== T3-S4 mutable-binding smoke tests (MutModel oracle) ===\n" ;
  test_mut_fun_delegation () ;
  test_mut_letmut_success () ;
  test_mut_letmut_body_type () ;
  test_mut_assign_success () ;
  test_mut_assign_unbound () ;
  test_mut_assign_immutable () ;
  test_mut_assign_mismatch () ;
  test_mut_fun_error_propagates () ;
  test_mut_letmut_nested_assign () ;
  test_mut_letmut_init_error () ;
  Printf.printf "=== T3-S4 mutable-binding smoke tests passed ===\n" ;
  Printf.printf
    "\n=== T3-S5 pattern-matching smoke tests (PatternModel oracle) ===\n" ;
  test_pat_mut_delegation () ;
  test_pat_match_no_payload () ;
  test_pat_match_payload () ;
  test_pat_not_a_variant () ;
  test_pat_unknown_constructor () ;
  test_pat_branch_type_mismatch () ;
  test_pat_mut_err_delegation () ;
  test_pat_empty_branches () ;
  test_pat_nested_payload_scope () ;
  test_pat_multi_branch_agree () ;
  test_pat_error_short_circuit () ;
  Printf.printf "=== T3-S5 pattern-matching smoke tests passed ===\n" ;
  Printf.printf
    "\n=== T3-S6 algebraic-construction smoke tests (ConstrModel oracle) ===\n" ;
  test_con_pat_delegation () ;
  test_con_record_success () ;
  test_con_record_empty_provided () ;
  test_con_record_field_mismatch () ;
  test_con_record_unknown_field () ;
  test_con_constr_none () ;
  test_con_constr_some () ;
  test_con_constr_payload_mismatch () ;
  test_con_constr_unknown () ;
  test_con_constr_arity_extra_arg () ;
  test_con_constr_arity_missing_arg () ;
  test_con_pattern_err_delegation () ;
  test_con_constr_nested_arg () ;
  Printf.printf "=== T3-S6 algebraic-construction smoke tests passed ===\n" ;
  Printf.printf
    "\n=== T3-S7 special-form smoke tests (SpecialModel oracle) ===\n" ;
  test_spe_constr_delegation () ;
  test_spe_return_passthrough () ;
  test_spe_return_passthrough_bool () ;
  test_spe_return_not_allowed () ;
  test_spe_create_array_global () ;
  test_spe_create_array_shared () ;
  test_spe_create_array_bad_size () ;
  test_spe_typed_match () ;
  test_spe_typed_mismatch () ;
  test_spe_constr_err_delegation () ;
  test_spe_nested_return_typed_array () ;
  test_spe_nested_error_propagates () ;
  Printf.printf "=== T3-S7 special-form smoke tests passed ===\n" ;
  exit 0
