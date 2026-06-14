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
              (Gen.list_size (Gen.int_range 0 3)
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
  let result = M.infer_type [] (M.ETuple [M.ELit (M.LInt 1); M.ELit (M.LBool true)]) in
  assert (result = M.Inl (M.TTuple [M.TPrim M.TInt32; M.TPrim M.TBool])) ;
  Printf.printf "  ETuple [1; true] -> TTuple [TInt32; TBool] [ok]\n"

let test_tuple_nested () =
  let inner = M.ETuple [M.ELit (M.LInt 0); M.ELit M.LUnit] in
  let result = M.infer_type [] (M.ETuple [inner; M.ELit (M.LBool false)]) in
  assert (result = M.Inl (M.TTuple [M.TTuple [M.TPrim M.TInt32; M.TPrim M.TUnit]; M.TPrim M.TBool])) ;
  Printf.printf "  ETuple [ETuple; false] -> TTuple [TTuple; TBool] [ok]\n"

let test_tuple_error_propagates () =
  let result = M.infer_type [] (M.ETuple [M.ELit (M.LInt 1); M.EVar (coq_string_of_string "x")]) in
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
  exit passed
