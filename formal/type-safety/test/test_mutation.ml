(******************************************************************************)
(* test_mutation.ml
 *
 * Non-vacuousness mutation tests for the extracted TypeSafetyModel.
 *
 * Two targeted mutants of TypeSafetyModel.infer_type are exercised:
 *
 *   M1 (type-erasure):    always returns Inl (TPrim TUnit) regardless of input.
 *                         Caught by: a literal float should give TReg RFloat32,
 *                         not TUnit.
 *
 *   M2 (variable-blind):  EVar always fails (Inr UnboundVar) regardless of env.
 *                         Caught by: a let-bound variable should type-check, but
 *                         M2 fails when the body reads it via EVar.
 *
 * Each mutant is tested with assert_mutation_caught, which exits 1 if the
 * mutation is not caught by the corresponding QCheck2 property.
 ******************************************************************************)

module M = Type_safety_model.TypeSafetyModel

(* ----- Coq string helper ---------------------------------------------------- *)

let coq_char c =
  let n = Char.code c in
  let bit k = (n lsr k) land 1 = 1 in
  M.Ascii (bit 0, bit 1, bit 2, bit 3, bit 4, bit 5, bit 6, bit 7)

let coq_str s =
  let len = String.length s in
  let rec go i acc =
    if i < 0 then acc else go (i - 1) (M.String (coq_char s.[i], acc))
  in
  go (len - 1) M.EmptyString

(* ----- M1: type-erasure mutant ---------------------------------------------- *)

let infer_type_m1 (_env : M.type_env) (_e : M.expr) : M.infer_result =
  M.Inl (M.TPrim M.TUnit)

(* M1 is caught by: a float literal should give TReg RFloat32, not TUnit. *)
let float_literal_correct_with infer_f =
  QCheck2.Test.make
    ~name:"float_literal_not_unit"
    ~count:200
    QCheck2.Gen.nat
    (fun n -> infer_f [] (M.ELit (M.LFloat n)) <> M.Inl (M.TPrim M.TUnit))

(* ----- M2: variable-blind mutant -------------------------------------------- *)

let rec infer_type_m2 env e =
  match e with
  | M.ELit l -> M.infer_type env (M.ELit l)
  | M.EVar x -> M.Inr (M.UnboundVar x)
  | M.ELet (x, e1, e2) -> (
      match infer_type_m2 env e1 with
      | M.Inl t1 -> infer_type_m2 ((x, t1) :: env) e2
      | M.Inr err -> M.Inr err)
  | M.ETuple es -> (
      let rec go = function
        | [] -> M.Inl []
        | h :: rest -> (
            match infer_type_m2 env h with
            | M.Inl t -> (
                match go rest with
                | M.Inl ts -> M.Inl (t :: ts)
                | M.Inr err -> M.Inr err)
            | M.Inr err -> M.Inr err)
      in
      match go es with
      | M.Inl ts -> M.Inl (M.TTuple ts)
      | M.Inr err -> M.Inr err)

(* M2 is caught by: a let-bound variable is readable in the body under the
   real inferrer, but M2 returns Inr (UnboundVar "x") for every EVar. *)
let let_bound_var_with infer_f =
  QCheck2.Test.make
    ~name:"let_bound_var_readable"
    ~count:1
    (QCheck2.Gen.return ())
    (fun () ->
      let x = coq_str "x" in
      let e = M.ELet (x, M.ELit (M.LInt 0), M.EVar x) in
      infer_f [] e = M.Inl (M.TPrim M.TInt32))

(* ----- Harness -------------------------------------------------------------- *)

let assert_mutation_caught label test =
  let caught = ref false in
  (try QCheck2.Test.check_exn ~rand:(Random.State.make [|42|]) test
   with QCheck2.Test.Test_fail _ | QCheck2.Test.Test_error _ -> caught := true) ;
  if not !caught then begin
    Printf.eprintf "MUTATION NOT CAUGHT: %s\n%!" label ;
    exit 1
  end
  else Printf.printf "MUTATION CAUGHT: %s\n%!" label

let () =
  assert_mutation_caught
    "M1 (type-erasure) vs float_literal_correct"
    (float_literal_correct_with infer_type_m1) ;
  assert_mutation_caught
    "M2 (variable-blind) vs let_bound_var_readable"
    (let_bound_var_with infer_type_m2)
