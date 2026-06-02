(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Test - IR comparison utilities
 *
 * This module provides utilities for comparing Kirc_Ast.k_ext values
 * to verify that old and new implementations produce identical IR.
 ******************************************************************************)

open Kirc_Ast

(** Difference found between two IR trees *)
type ir_diff =
  | NodeMismatch of {path : string; expected : string; got : string}
  | StructureMismatch of {path : string; desc : string}
  | CountMismatch of {path : string; expected : int; got : int}

(** Compare two k_ext values for structural equality *)
let rec ir_equal (ir1 : k_ext) (ir2 : k_ext) : bool =
  match (ir1, ir2) with
  | Kern (p1, b1), Kern (p2, b2) -> ir_equal p1 p2 && ir_equal b1 b2
  | Block b1, Block b2 -> ir_equal b1 b2
  | Params p1, Params p2 -> ir_equal p1 p2
  | Plus (a1, b1), Plus (a2, b2)
  | Plusf (a1, b1), Plusf (a2, b2)
  | Min (a1, b1), Min (a2, b2)
  | Minf (a1, b1), Minf (a2, b2)
  | Mul (a1, b1), Mul (a2, b2)
  | Mulf (a1, b1), Mulf (a2, b2)
  | Div (a1, b1), Div (a2, b2)
  | Divf (a1, b1), Divf (a2, b2)
  | Mod (a1, b1), Mod (a2, b2)
  | Or (a1, b1), Or (a2, b2)
  | And (a1, b1), And (a2, b2)
  | EqBool (a1, b1), EqBool (a2, b2)
  | LtBool (a1, b1), LtBool (a2, b2)
  | GtBool (a1, b1), GtBool (a2, b2)
  | LtEBool (a1, b1), LtEBool (a2, b2)
  | GtEBool (a1, b1), GtEBool (a2, b2)
  | Seq (a1, b1), Seq (a2, b2)
  | Set (a1, b1), Set (a2, b2)
  | SetV (a1, b1), SetV (a2, b2)
  | Acc (a1, b1), Acc (a2, b2)
  | Local (a1, b1), Local (a2, b2)
  | IntVecAcc (a1, b1), IntVecAcc (a2, b2)
  | While (a1, b1), While (a2, b2)
  | Concat (a1, b1), Concat (a2, b2)
  | RecSet (a1, b1), RecSet (a2, b2) ->
      ir_equal a1 a2 && ir_equal b1 b2
  | Not a1, Not a2 | Return a1, Return a2 | Decl a1, Decl a2 -> ir_equal a1 a2
  | Ife (c1, t1, e1), Ife (c2, t2, e2)
  | SetLocalVar (c1, t1, e1), SetLocalVar (c2, t2, e2) ->
      ir_equal c1 c2 && ir_equal t1 t2 && ir_equal e1 e2
  | DoLoop (v1, l1, h1, b1), DoLoop (v2, l2, h2, b2) ->
      ir_equal v1 v2 && ir_equal l1 l2 && ir_equal h1 h2 && ir_equal b1 b2
  | If (c1, t1), If (c2, t2) -> ir_equal c1 c2 && ir_equal t1 t2
  | Id s1, Id s2 -> s1 = s2
  | IdName s1, IdName s2 -> s1 = s2
  | IntVar (i1, s1, _m1), IntVar (i2, s2, _m2) -> i1 = i2 && s1 = s2
  | FloatVar (i1, s1, _m1), FloatVar (i2, s2, _m2) -> i1 = i2 && s1 = s2
  | DoubleVar (i1, s1, _m1), DoubleVar (i2, s2, _m2) -> i1 = i2 && s1 = s2
  | BoolVar (i1, s1, _m1), BoolVar (i2, s2, _m2) -> i1 = i2 && s1 = s2
  | UnitVar (i1, s1, _m1), UnitVar (i2, s2, _m2) -> i1 = i2 && s1 = s2
  | VecVar (t1, i1, s1), VecVar (t2, i2, s2) ->
      ir_equal t1 t2 && i1 = i2 && s1 = s2
  | Int i1, Int i2 -> i1 = i2
  | Float f1, Float f2 -> f1 = f2
  | Double d1, Double d2 -> d1 = d2
  | IntId (s1, i1), IntId (s2, i2) -> s1 = s2 && i1 = i2
  | Intrinsics (c1, o1), Intrinsics (c2, o2) -> c1 = c2 && o1 = o2
  | Empty, Empty -> true
  | Unit, Unit -> true
  | Arr (n1, s1, t1, m1), Arr (n2, s2, t2, m2) ->
      n1 = n2 && ir_equal s1 s2 && t1 = t2 && m1 = m2
  | Custom (t1, i1, s1), Custom (t2, i2, s2) -> t1 = t2 && i1 = i2 && s1 = s2
  | CustomVar (t1, c1, s1), CustomVar (t2, c2, s2) ->
      t1 = t2 && c1 = c2 && s1 = s2
  | RecGet (r1, f1), RecGet (r2, f2) -> ir_equal r1 r2 && f1 = f2
  | Record (n1, fs1), Record (n2, fs2) ->
      n1 = n2
      && List.length fs1 = List.length fs2
      && List.for_all2 ir_equal fs1 fs2
  | Constr (t1, c1, as1), Constr (t2, c2, as2) ->
      t1 = t2 && c1 = c2
      && List.length as1 = List.length as2
      && List.for_all2 ir_equal as1 as2
  | Match (t1, s1, cs1), Match (t2, s2, cs2) ->
      t1 = t2 && ir_equal s1 s2
      && Array.length cs1 = Array.length cs2
      && Array.for_all2 case_equal cs1 cs2
  | App (f1, as1), App (f2, as2) ->
      ir_equal f1 f2
      && Array.length as1 = Array.length as2
      && Array.for_all2 ir_equal as1 as2
  | GlobalFun (b1, r1, n1), GlobalFun (b2, r2, n2) ->
      ir_equal b1 b2 && r1 = r2 && n1 = n2
  | Map (f1, a1, b1), Map (f2, a2, b2) ->
      ir_equal f1 f2 && ir_equal a1 a2 && ir_equal b1 b2
  | Pragma (os1, b1), Pragma (os2, b2) -> os1 = os2 && ir_equal b1 b2
  | CastDoubleVar (i1, s1), CastDoubleVar (i2, s2) -> i1 = i2 && s1 = s2
  | EqCustom (n1, a1, b1), EqCustom (n2, a2, b2) ->
      n1 = n2 && ir_equal a1 a2 && ir_equal b1 b2
  | GInt _, GInt _ -> true (* Cannot compare closures *)
  | GFloat _, GFloat _ -> true
  | GFloat64 _, GFloat64 _ -> true
  | Native _, Native _ -> true
  | GIntVar s1, GIntVar s2 -> s1 = s2
  | GFloatVar s1, GFloatVar s2 -> s1 = s2
  | GFloat64Var s1, GFloat64Var s2 -> s1 = s2
  | NativeWithFallback _, NativeWithFallback _ ->
      (* Can't compare functions, just check both are NativeWithFallback *)
      true
  | _, _ -> false

and case_equal (i1, o1, b1) (i2, o2, b2) = i1 = i2 && o1 = o2 && ir_equal b1 b2

(** Get constructor name for diff reporting *)
let constructor_name : k_ext -> string = function
  | Kern _ -> "Kern"
  | Block _ -> "Block"
  | Params _ -> "Params"
  | Plus _ -> "Plus"
  | Plusf _ -> "Plusf"
  | Min _ -> "Min"
  | Minf _ -> "Minf"
  | Mul _ -> "Mul"
  | Mulf _ -> "Mulf"
  | Div _ -> "Div"
  | Divf _ -> "Divf"
  | Mod _ -> "Mod"
  | Id _ -> "Id"
  | IdName _ -> "IdName"
  | GlobalFun _ -> "GlobalFun"
  | IntVar _ -> "IntVar"
  | FloatVar _ -> "FloatVar"
  | UnitVar _ -> "UnitVar"
  | CastDoubleVar _ -> "CastDoubleVar"
  | DoubleVar _ -> "DoubleVar"
  | BoolVar _ -> "BoolVar"
  | Arr _ -> "Arr"
  | VecVar _ -> "VecVar"
  | Concat _ -> "Concat"
  | Constr _ -> "Constr"
  | Record _ -> "Record"
  | RecGet _ -> "RecGet"
  | RecSet _ -> "RecSet"
  | Empty -> "Empty"
  | Seq _ -> "Seq"
  | Return _ -> "Return"
  | Set _ -> "Set"
  | Decl _ -> "Decl"
  | SetV _ -> "SetV"
  | SetLocalVar _ -> "SetLocalVar"
  | Intrinsics _ -> "Intrinsics"
  | IntId _ -> "IntId"
  | Int _ -> "Int"
  | Float _ -> "Float"
  | Double _ -> "Double"
  | Custom _ -> "Custom"
  | CustomVar _ -> "CustomVar"
  | IntVecAcc _ -> "IntVecAcc"
  | Local _ -> "Local"
  | Acc _ -> "Acc"
  | Ife _ -> "Ife"
  | If _ -> "If"
  | Match _ -> "Match"
  | Or _ -> "Or"
  | And _ -> "And"
  | Not _ -> "Not"
  | EqCustom _ -> "EqCustom"
  | EqBool _ -> "EqBool"
  | LtBool _ -> "LtBool"
  | GtBool _ -> "GtBool"
  | LtEBool _ -> "LtEBool"
  | GtEBool _ -> "GtEBool"
  | DoLoop _ -> "DoLoop"
  | While _ -> "While"
  | App _ -> "App"
  | GInt _ -> "GInt"
  | GFloat _ -> "GFloat"
  | GFloat64 _ -> "GFloat64"
  | Native _ -> "Native"
  | GIntVar _ -> "GIntVar"
  | GFloatVar _ -> "GFloatVar"
  | GFloat64Var _ -> "GFloat64Var"
  | NativeWithFallback _ -> "NativeWithFallback"
  | IntrinsicRef _ -> "IntrinsicRef"
  | Pragma _ -> "Pragma"
  | Map _ -> "Map"
  | Unit -> "Unit"

(** Produce a detailed diff between two IR trees *)
let rec diff_ir ?(path = "") (expected : k_ext) (got : k_ext) : ir_diff list =
  if ir_equal expected got then []
  else
    let cn_exp = constructor_name expected in
    let cn_got = constructor_name got in
    if cn_exp <> cn_got then
      [NodeMismatch {path; expected = cn_exp; got = cn_got}]
    else
      (* Same constructor, different content - recurse *)
      match (expected, got) with
      | Kern (p1, b1), Kern (p2, b2) ->
          diff_ir ~path:(path ^ "/params") p1 p2
          @ diff_ir ~path:(path ^ "/body") b1 b2
      | Seq (a1, b1), Seq (a2, b2) ->
          diff_ir ~path:(path ^ "/seq[0]") a1 a2
          @ diff_ir ~path:(path ^ "/seq[1]") b1 b2
      | Plus (a1, b1), Plus (a2, b2) | Plusf (a1, b1), Plusf (a2, b2) ->
          diff_ir ~path:(path ^ "/left") a1 a2
          @ diff_ir ~path:(path ^ "/right") b1 b2
      | IntVar (i1, s1, _m1), IntVar (i2, s2, _m2) when i1 <> i2 || s1 <> s2 ->
          [
            NodeMismatch
              {
                path;
                expected = Printf.sprintf "IntVar(%d, %S)" i1 s1;
                got = Printf.sprintf "IntVar(%d, %S)" i2 s2;
              };
          ]
      | _ -> [StructureMismatch {path; desc = "Content differs"}]

(** Pretty print a diff *)
let pp_diff fmt = function
  | NodeMismatch {path; expected; got} ->
      Format.fprintf fmt "At %s: expected %s, got %s" path expected got
  | StructureMismatch {path; desc} -> Format.fprintf fmt "At %s: %s" path desc
  | CountMismatch {path; expected; got} ->
      Format.fprintf fmt "At %s: expected %d items, got %d" path expected got

(** Pretty print a list of diffs *)
let pp_diffs fmt diffs =
  List.iter (fun d -> Format.fprintf fmt "  %a@." pp_diff d) diffs

(** Summary for test output *)
let diff_summary (expected : k_ext) (got : k_ext) : string =
  let diffs = diff_ir expected got in
  if diffs = [] then "IR trees are identical"
  else
    Format.asprintf
      "Found %d differences:@.%a"
      (List.length diffs)
      pp_diffs
      diffs
