(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types
module A = Sarek_ir_analysis
module Names = Set.Make (String)

let is_vector (v : var) = match v.var_type with TVec _ -> true | _ -> false

let folder ~fe ~fs : Names.t A.folder =
  {
    A.fe;
    ft = (fun acc _ -> acc);
    fs;
    fnative = (fun acc -> acc);
    visit_lvalue = true;
  }

(* Every variable name mentioned in [e]: an indirect store's base expression may
   reach a parameter through arbitrary arithmetic, so all of them count. *)
let vars_of_expr e =
  let collect acc = function EVar v -> Names.add v.var_name acc | _ -> acc in
  A.expr_fold (folder ~fe:collect ~fs:(fun acc _ -> acc)) Names.empty e

let rec lvalue_targets = function
  | LVar _ -> Names.empty
  | LArrayElem (name, _) -> Names.singleton name
  | LArrayElemExpr (base, _) -> vars_of_expr base
  | LRecordField (inner, _) -> lvalue_targets inner

(* Vector-typed variables among the arguments of a call. *)
let vector_args args =
  List.fold_left
    (fun acc -> function
      | EVar v when is_vector v -> Names.add v.var_name acc | _ -> acc)
    Names.empty
    args

let written_by_expr acc = function
  | EIntrinsic (_, _, args) -> Names.union acc (vector_args args)
  | EApp (fn, args) -> Names.union acc (vector_args (fn :: args))
  | _ -> acc

let written_by_stmt acc = function
  | SAssign (lv, _) -> Names.union acc (lvalue_targets lv)
  | SCoopmat (CM_store {dst; _}) -> Names.add dst acc
  | _ -> acc

let written_names (k : kernel) =
  let f = folder ~fe:written_by_expr ~fs:written_by_stmt in
  let acc = A.stmt_fold f Names.empty k.kern_body in
  List.fold_left (A.helper_fold f) acc k.kern_funcs

let has_native (k : kernel) =
  A.kernel_fold (A.exists_folder ~leaf:(fun _ -> false) ~native:true ()) false k

let params (k : kernel) : string list =
  if has_native k then []
  else
    let written = written_names k in
    List.filter_map
      (function
        | DParam (v, Some {arr_memspace = Global; _})
          when is_vector v && not (Names.mem v.var_name written) ->
            Some v.var_name
        | DParam _ | DLocal _ | DShared _ -> None)
      k.kern_params
