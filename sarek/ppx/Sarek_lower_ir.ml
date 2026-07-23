(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module lowers the typed AST directly to Sarek_ir_ppx, bypassing
 * the legacy Kirc_Ast. This provides a cleaner IR for the V2 execution path.
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
module Ir = Sarek_ir_ppx

(** Mangle type names for C compatibility *)
let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

(** Convert Sarek_types.typ to Sarek_ir_ppx.elttype *)
let rec elttype_of_typ (ty : typ) : Ir.elttype =
  match repr ty with
  | TPrim TInt32 -> Ir.TInt32
  | TPrim TBool -> Ir.TBool
  | TPrim TUnit -> Ir.TUnit
  | TReg Int64 -> Ir.TInt64
  | TReg Int -> Ir.TInt32 (* int maps to int32 on GPU *)
  | TReg Float32 -> Ir.TFloat32
  | TReg Float64 -> Ir.TFloat64
  | TReg Char -> Ir.TInt32 (* char represented as int32 *)
  | TReg (Custom _) -> Ir.TInt32 (* Custom types handled separately *)
  | TVec elem_ty -> Ir.TVec (elttype_of_typ elem_ty)
  | TArr (elem_ty, mem) ->
      let ir_mem =
        match mem with
        | Sarek_types.Global -> Ir.Global
        | Sarek_types.Shared -> Ir.Shared
        | Sarek_types.Local -> Ir.Local
      in
      Ir.TArray (elttype_of_typ elem_ty, ir_mem)
  | TRecord (name, fields) ->
      Ir.TRecord (name, List.map (fun (n, ty) -> (n, elttype_of_typ ty)) fields)
  | TVariant (name, constrs) ->
      Ir.TVariant
        ( name,
          List.map
            (fun (n, ty_opt) -> (n, constr_payload_elttypes ty_opt))
            constrs )
  (* NOTE: TTuple/TFun are NOT rejected here even though this is where the
     original bug report pointed. This function converts the type of *any*
     typed value reachable during lowering, including local/helper function
     bindings inside a kernel body (e.g. `let make_p x y z = ... in ...`,
     see test_visibility_private.ml, test_transpose.ml, bench_nbody.ml) -
     those legitimately have TFun type and must keep lowering (their
     "elttype" is a don't-care placeholder, never actually used as data).
     The real defect is kernel *parameters* of tuple/function type, which
     are rejected explicitly in lower_param below, at the point where a
     type is about to become a formal parameter. *)
  | TTuple _ -> Ir.TInt32
  | TFun _ -> Ir.TInt32
  | TVar _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Kernel parameter type is a type variable — polymorphic kernels are \
         not supported. Annotate the parameter with a concrete type (e.g. \
         float32, int32)."

(** Flatten a variant constructor's payload type into its IR field list. A
    multi-argument constructor ([MkPair of float32 * float32]) carries a tuple
    payload type; it is FLATTENED to one IR field per component
    ([TFloat32; TFloat32]) so it lines up field-for-field with the multi-binder
    pattern ([PConstr ("MkPair", ["x"; "y"])]) and the flat [_0/_1] tagged-union
    payload every source generator already emits. A single, non-tuple payload
    stays one field. Only scalar-primitive tuple components flatten; any other
    tuple keeps the (placeholder) single-field mapping. *)
and constr_payload_elttypes (ty_opt : typ option) : Ir.elttype list =
  match ty_opt with
  | None -> []
  | Some ty -> (
      match repr ty with
      | TTuple tys when List.for_all is_scalar_prim_typ tys ->
          List.map elttype_of_typ tys
      | _ -> [elttype_of_typ ty])

(** Scalar-primitive predicate on a source [typ], matching the component set
    accepted for flattened tuple payloads (mirrors [prim_component_elttype]). *)
and is_scalar_prim_typ (t : typ) : bool =
  match repr t with
  | TPrim TInt32 | TPrim TBool | TReg Int64 | TReg Float32 | TReg Float64 ->
      true
  | _ -> false

(* L13: tuple-typed vector elements. A tuple used as a vector element type is
   lowered to a synthesized packed record with positional fields [_0], [_1],
   ..., reusing the record/aggregate codegen path unchanged (the layout is
   computed by Sarek_ir_layout, byte-for-byte identically on host and device).
   Only scalar-primitive components are supported in this tier; nested tuples,
   records, vectors or functions as components are rejected. *)
let elttype_prim_tag (e : Ir.elttype) : string option =
  match e with
  | Ir.TInt32 -> Some "int32"
  | Ir.TInt64 -> Some "int64"
  | Ir.TFloat32 -> Some "float32"
  | Ir.TFloat64 -> Some "float64"
  | Ir.TBool -> Some "bool"
  | _ -> None

(** Scalar-primitive check on the SOURCE type of a tuple component. This must
    match on [Sarek_types.typ] directly, NOT via [elttype_of_typ]: that
    conversion maps [TTuple]/[TFun] to a placeholder [Ir.TInt32] (see its NOTE),
    so routing the primitivity check through it would silently accept nested
    tuples or function components as int32 fields and synthesize a wrong layout.
    Returns the component's IR type iff it is a supported scalar. *)
let prim_component_elttype (t : typ) : Ir.elttype option =
  match repr t with
  | TPrim TInt32 -> Some Ir.TInt32
  | TPrim TBool -> Some Ir.TBool
  | TReg Int64 -> Some Ir.TInt64
  | TReg Float32 -> Some Ir.TFloat32
  | TReg Float64 -> Some Ir.TFloat64
  | _ -> None

let tuple_field_name i = Printf.sprintf "_%d" i

(** Mangled nominal name of the record synthesized for a tuple shape, e.g.
    [_tup_float32_int32] for [float32 * int32]. Kept in sync with the host-side
    [Sarek_tuple_vec] builder so both agree on the element identity. *)
let tuple_record_name (comps : Ir.elttype list) : string =
  "_tup"
  ^ String.concat
      ""
      (List.map
         (fun e ->
           match elttype_prim_tag e with Some t -> "_" ^ t | None -> "_x")
         comps)

(** Synthesized record fields (positional [_0.._n]) for a tuple's component
    types. Raises if any component is not a scalar primitive. *)
let tuple_record_fields (tys : typ list) : string * (string * Ir.elttype) list =
  let comps =
    List.map
      (fun t ->
        match prim_component_elttype t with
        | Some e -> e
        | None ->
            Ppxlib.Location.raise_errorf
              ~loc:Ppxlib.Location.none
              "Tuple-typed vector elements support only scalar components \
               (float32/float64/int32/int64/bool); nested tuples, records, \
               vectors or functions inside a vector-element tuple are not \
               supported.")
      tys
  in
  let name = tuple_record_name comps in
  (name, List.mapi (fun i e -> (tuple_field_name i, e)) comps)

let tuple_tmp_counter = ref 0

(** Is [t] a tuple whose components are all scalar primitives (the shape we
    synthesize a record for)? *)
let is_primitive_tuple (t : typ) : bool =
  match repr t with
  | TTuple tys -> List.for_all (fun t -> prim_component_elttype t <> None) tys
  | _ -> false

(** Element IR type of a vector whose element is [t]; a tuple element becomes
    its synthesized record, everything else is the ordinary mapping. *)
let vector_elem_elttype (t : typ) : Ir.elttype =
  match repr t with
  | TTuple tys ->
      let name, fields = tuple_record_fields tys in
      Ir.TRecord (name, fields)
  | _ -> elttype_of_typ t

(** Convert Sarek_types.memspace to Sarek_ir_ppx.memspace *)
let memspace_of_memspace (mem : Sarek_types.memspace) : Ir.memspace =
  match mem with
  | Sarek_types.Global -> Ir.Global
  | Sarek_types.Shared -> Ir.Shared
  | Sarek_types.Local -> Ir.Local

(** Get C type string for a typ *)
let rec c_type_of_typ ty =
  match repr ty with
  | TPrim TInt32 -> "int"
  | TPrim TBool -> "int"
  | TPrim TUnit -> "void"
  | TReg Int -> "int"
  | TReg Int64 -> "long"
  | TReg Float32 -> "float"
  | TReg Float64 -> "double"
  | TReg Char -> "char"
  | TReg (Custom name) -> mangle_type_name name
  | TRecord (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVariant (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVec t -> c_type_of_typ t ^ " *"
  | TArr (t, _) -> c_type_of_typ t ^ " *"
  | _ -> "int"

(** Generate C struct definition and builder for record types *)
let record_constructor_strings name (fields : (string * typ) list) =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let struct_fields =
    List.map
      (fun (fname, fty) -> "  " ^ c_type_of_typ fty ^ " " ^ fname ^ ";")
      fields
  in
  let struct_def =
    "struct " ^ struct_name ^ " {\n" ^ String.concat "\n" struct_fields ^ "\n};"
  in
  let params =
    String.concat
      ", "
      (List.map (fun (fname, fty) -> c_type_of_typ fty ^ " " ^ fname) fields)
  in
  let assigns =
    String.concat
      "\n"
      (List.map
         (fun (fname, _) -> "  res." ^ fname ^ " = " ^ fname ^ ";")
         fields)
  in
  let builder =
    "struct " ^ struct_name ^ " build_" ^ struct_name ^ "(" ^ params ^ ") {\n"
    ^ "  struct " ^ struct_name ^ " res;\n" ^ assigns ^ "\n  return res;\n}"
  in
  [struct_def; builder]

(** Generate C struct definitions and builders for variant types *)
let variant_constructor_strings name constrs =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let constr_structs =
    List.map
      (fun (cname, carg) ->
        let field =
          match carg with
          | None -> "  int " ^ name ^ "_sarek_" ^ cname ^ "_t;"
          | Some ty ->
              "  " ^ c_type_of_typ ty ^ " " ^ name ^ "_sarek_" ^ cname ^ "_t;"
        in
        "struct " ^ name ^ "_sarek_" ^ cname ^ " {\n" ^ field ^ "\n};")
      constrs
  in
  let union_fields =
    List.map
      (fun (cname, _) ->
        "  struct " ^ name ^ "_sarek_" ^ cname ^ " " ^ name ^ "_sarek_" ^ cname
        ^ ";")
      constrs
  in
  let union_def =
    "union " ^ name ^ "_sarek_union {\n"
    ^ String.concat "\n" union_fields
    ^ "\n};"
  in
  let main_struct =
    "struct " ^ struct_name ^ " {\n" ^ "  int " ^ name ^ "_sarek_tag;\n"
    ^ "  union " ^ name ^ "_sarek_union " ^ name ^ "_sarek_union;\n" ^ "};"
  in
  let builders =
    List.mapi
      (fun idx (cname, carg) ->
        let params, assign =
          match carg with
          | None -> ("", "  /* no payload */")
          | Some ty ->
              let pname = "v" in
              ( c_type_of_typ ty ^ " " ^ pname,
                "  res." ^ name ^ "_sarek_union." ^ name ^ "_sarek_" ^ cname
                ^ "." ^ name ^ "_sarek_" ^ cname ^ "_t = " ^ pname ^ ";" )
        in
        "struct " ^ struct_name ^ " build_" ^ name ^ "_" ^ cname ^ "(" ^ params
        ^ ") {\n" ^ "  struct " ^ struct_name ^ " res;\n" ^ "  res." ^ name
        ^ "_sarek_tag = " ^ string_of_int idx ^ ";\n" ^ assign ^ "\n"
        ^ "  return res;\n}")
      constrs
  in
  constr_structs @ (union_def :: main_struct :: builders)

(* Debug counters for IR lowering *)
let ir_lower_expr_count = ref 0

let ir_lower_stmt_count = ref 0

(** Lowering state *)
type state = {
  mutable next_var_id : int;
  fun_map : (string, tparam list * texpr) Hashtbl.t;
  lowering_stack : (string, unit) Hashtbl.t;
  lowered_funs : (string, Ir.helper_func) Hashtbl.t;
      (** Lowered helper functions: name -> helper_func *)
  mutable lowered_funs_order : string list;
      (** Order in which functions were lowered (for dependency ordering) *)
  types : (string, (string * Ir.elttype) list) Hashtbl.t;
      (** Collected record types: type_name -> [(field_name, field_type); ...]
      *)
  variants : (string, (string * Ir.elttype list) list) Hashtbl.t;
      (** Collected variant types: type_name ->
          [(constructor_name, payload_types); ...] *)
}

let create_state fun_map =
  {
    next_var_id = 0;
    fun_map;
    lowering_stack = Hashtbl.create 8;
    lowered_funs = Hashtbl.create 8;
    lowered_funs_order = [];
    types = Hashtbl.create 8;
    variants = Hashtbl.create 8;
  }

let fresh_id state =
  let id = state.next_var_id in
  state.next_var_id <- id + 1 ;
  id

(** Convert Sarek_ast.binop to Sarek_ir_ppx.binop *)
let ir_binop (op : binop) (_ty : typ) : Ir.binop =
  match op with
  | Add -> Ir.Add
  | Sub -> Ir.Sub
  | Mul -> Ir.Mul
  | Div -> Ir.Div
  | Mod -> Ir.Mod
  | Eq -> Ir.Eq
  | Ne -> Ir.Ne
  | Lt -> Ir.Lt
  | Le -> Ir.Le
  | Gt -> Ir.Gt
  | Ge -> Ir.Ge
  | And -> Ir.And
  | Or -> Ir.Or
  | Lsl -> Ir.Shl
  | Lsr ->
      (* Never reached: TEBinop (Lsr, _, _) is intercepted in lower_expr
         and rewritten via lower_lsr into a logical-shift expression tree,
         because Ir.Shr is arithmetic on every backend (see
         briefs/fix-critical-semantics-evidence.md, G phase 1). Kept here
         so this match stays a total, honest structural map of
         Sarek_ast.binop. *)
      Ir.Shr
  | Asr -> Ir.Shr (* arithmetic shift; Ir.Shr is arithmetic on every backend *)
  | Land -> Ir.BitAnd
  | Lor -> Ir.BitOr
  | Lxor -> Ir.BitXor

(** Convert Sarek_ast.unop to Sarek_ir_ppx.unop *)
let ir_unop (op : unop) : Ir.unop =
  match op with Neg -> Ir.Neg | Not -> Ir.Not | Lnot -> Ir.BitNot

(** Convert memspace *)
let lower_memspace = function
  | Local -> Ir.Local
  | Shared -> Ir.Shared
  | Global -> Ir.Global

(** Create a var from typed var info *)
let make_var name id ty mutable_ : Ir.var =
  {
    var_name = name;
    var_id = id;
    var_type = elttype_of_typ ty;
    var_mutable = mutable_;
  }

(** Lower a declaration *)
let lower_decl ~mutable_ id name ty : Ir.decl =
  let v = make_var name id ty mutable_ in
  Ir.DLocal (v, None)

(** Transform a statement to ensure it returns a value. This adds SReturn to
    leaf statements without re-traversing the original AST. *)
let rec make_returning stmt =
  match stmt with
  | Ir.SReturn _ -> stmt (* Already returns *)
  | Ir.SExpr e -> Ir.SReturn e (* Expression -> return it *)
  | Ir.SIf (c, t, Some e) ->
      Ir.SIf (c, make_returning t, Some (make_returning e))
  | Ir.SIf (c, t, None) ->
      Ir.SIf (c, make_returning t, Some (Ir.SReturn (Ir.EConst Ir.CUnit)))
  | Ir.SMatch (e, cases) ->
      Ir.SMatch (e, List.map (fun (p, b) -> (p, make_returning b)) cases)
  | Ir.SSeq stmts -> (
      match List.rev stmts with
      | [] -> Ir.SReturn (Ir.EConst Ir.CUnit)
      | last :: rest -> Ir.SSeq (List.rev (make_returning last :: rest)))
  | Ir.SLet (v, e, body) -> Ir.SLet (v, e, make_returning body)
  | Ir.SLetMut (v, e, body) -> Ir.SLetMut (v, e, make_returning body)
  | Ir.SPragma (opts, body) -> Ir.SPragma (opts, make_returning body)
  | Ir.SFor _ | Ir.SWhile _ | Ir.SAssign _ | Ir.SBarrier | Ir.SWarpBarrier
  | Ir.SMemFence | Ir.SEmpty | Ir.SNative _ ->
      (* These are side-effect statements; return unit after *)
      Ir.SSeq [stmt; Ir.SReturn (Ir.EConst Ir.CUnit)]
  | Ir.SBlock body -> Ir.SBlock (make_returning body)

(** [true] iff [e] is a syntactically trivial IR expression ([EVar] or
    [EConst]). Trivial expressions have no side effects, so duplicating them in
    the tree built by {!lower_lsr} is semantically inert. *)
let is_trivial_ir_expr (e : Ir.expr) : bool =
  match e with Ir.EVar _ | Ir.EConst _ -> true | _ -> false

(** Lower [a lsr b] (logical/unsigned right shift) to an IR expression tree
    built only from existing IR nodes.

    Ir.Shr is emitted as an *arithmetic* (sign-extending) shift by every
    consumer (CUDA/OpenCL/Metal/GLSL/WGSL emit plain [>>] on a signed C/GLSL int
    type; PTX and the interpreter use [shr.s32]/[Int32.shift_right] - see G
    phase 1 in briefs/fix-critical-semantics-evidence.md). There is no IR node
    for a logical shift and none may be added (formal/codegen-ptx models [Shr]
    itself), so [lsr] is expressed via the classic arithmetic-shift identity,
    width-aware via [width_bits]:

    {[
      lshr (a, n)
      =
      if n = 0 then a
      else
        ashr (a, n) lxor (ashr (a, width - 1) lsl ((width - n) land (width - 1)))
    ]}

    [ashr(a, width-1)] is all-1s when [a] is negative and all-0s otherwise;
    shifted left by [(width - n) land (width - 1)] (equal to [width - n] for
    every [n] in [1..width-1]) it isolates exactly the [n] sign-extended bits
    that [ashr(a, n)] filled in, and XOR-ing them off recovers the zero-filled
    logical shift.

    {b Why the [land (width - 1)] mask, not just an [n = 0] guard.} PTX's [selp]
    and WGSL's [select()] (see [Sarek_ir_ptx_expr.ml] and [Sarek_ir_wgsl.ml])
    evaluate BOTH branches of the resulting [EIf] before selecting one - the
    [EIf] only picks which *value* is used, it does not skip *computing* the
    other branch's subexpressions on those backends. So even though the
    then-branch ([a_ir]) is selected when [n = 0], the else-branch's internal
    [Sub(width_bits, b_ir)] is still evaluated, and without masking it would be
    exactly [width_bits] (shift-by-32/64), which is undefined/rejected on some
    backends. Masking with [width_bits - 1] keeps that shift count in
    [0..width_bits-1] for every [n], including [n = 0] (where it reduces to a
    well-defined shift-by-0, unused anyway since the [EIf] selects [a_ir]),
    while leaving the result unchanged for [n] in [1..width_bits-1] (masking a
    value already in range is a no-op). Shift amounts with [n < 0] or
    [n >= width] are unspecified, matching the pre-existing behaviour of
    [Shl]/[Shr] on out-of-range counts.

    {b Duplication / side-effect safety.} [a_ir] and [b_ir] each appear three
    times in the tree above (in [sign_fill], in [arith_shift]/[top_bits], and in
    the final [EIf]'s branches/condition). [Sarek_ir_ppx.expr] is documented as
    "pure, no side effects" and has no let-binding form (only [Ir.stmt]'s
    [SLet]/[SLetMut] bind values, and those wrap a *statement* continuation, not
    an expression one) - so there is no way to evaluate a subexpression once and
    reuse the result across multiple expression positions. Hoisting via a
    synthetic [EApp] call (which would single-evaluate its arguments) is not
    viable either: PTX - the backend this rewrite specifically targets - does
    not implement device function calls ([Sarek_ir_ptx_expr.ml] rejects [EApp]
    outright). Consequently, if [a_ir] or [b_ir] is *not* trivial (e.g. it
    embeds an [EIntrinsic] atomic call), this tree would silently evaluate that
    operand's side effect multiple times. Rather than accept that, this function
    restricts the rewrite to trivial operands ([EVar]/[EConst], see
    {!is_trivial_ir_expr}) and raises a located PPX error for every other case,
    directing the user to hoist the operand into a `let` before the shift. This
    tree is NOT safe for arbitrary operands - only for trivial ones. *)
let lower_lsr ~(loc : Ppxlib.Location.t) (a_ir : Ir.expr) (b_ir : Ir.expr)
    (ty : Ir.elttype) : Ir.expr =
  if not (is_trivial_ir_expr a_ir && is_trivial_ir_expr b_ir) then
    Ppxlib.Location.raise_errorf
      ~loc
      "lsr: this logical-shift operand is not a plain variable or constant. \
       Sarek_ir_ppx.expr has no let-binding form, so the lsr expansion would \
       evaluate this operand multiple times, silently duplicating any side \
       effect it contains (e.g. an atomic intrinsic call). Bind it to a local \
       variable first and retry, e.g. `let x = <expr> in x lsr n`." ;
  let width_bits = match ty with Ir.TInt64 -> 64 | _ -> 32 in
  let const n =
    match ty with
    | Ir.TInt64 -> Ir.EConst (Ir.CInt64 (Int64.of_int n))
    | _ -> Ir.EConst (Ir.CInt32 (Int32.of_int n))
  in
  let sign_fill = Ir.EBinop (Ir.Shr, a_ir, const (width_bits - 1)) in
  let shift_count =
    Ir.EBinop
      ( Ir.BitAnd,
        Ir.EBinop (Ir.Sub, const width_bits, b_ir),
        const (width_bits - 1) )
  in
  let top_bits = Ir.EBinop (Ir.Shl, sign_fill, shift_count) in
  let arith_shift = Ir.EBinop (Ir.Shr, a_ir, b_ir) in
  let logical_shift = Ir.EBinop (Ir.BitXor, arith_shift, top_bits) in
  Ir.EIf (Ir.EBinop (Ir.Eq, b_ir, const 0), a_ir, logical_shift)

(** Convert a typed expression to IR expression *)
let rec lower_expr (state : state) (te : texpr) : Ir.expr =
  incr ir_lower_expr_count ;
  (* Log progress every 10000 calls *)
  if !ir_lower_expr_count mod 10000 = 0 then
    Sarek_debug.log_to_file
      (Printf.sprintf "    [IR] lower_expr progress: %d" !ir_lower_expr_count) ;
  match te.te with
  | TEUnit -> Ir.EConst Ir.CUnit
  | TEBool b -> Ir.EConst (Ir.CBool b)
  | TEInt i -> Ir.EConst (Ir.CInt32 (Int32.of_int i))
  | TEInt32 i -> Ir.EConst (Ir.CInt32 i)
  | TEInt64 i -> Ir.EConst (Ir.CInt64 i)
  | TEFloat f -> Ir.EConst (Ir.CFloat32 f)
  | TEDouble f -> Ir.EConst (Ir.CFloat64 f)
  | TEVar (name, id) -> (
      match repr te.ty with
      | TFun _ ->
          (* Function reference - just use the name *)
          let v = make_var name id te.ty false in
          Ir.EVar v
      | _ ->
          let v = make_var name id te.ty false in
          Ir.EVar v)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> Ir.EArrayReadExpr (lower_expr state vec, lower_expr state idx))
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> Ir.EArrayReadExpr (lower_expr state arr, lower_expr state idx))
  | TEFieldGet (r, field, _) -> Ir.ERecordField (lower_expr state r, field)
  | TEBinop (Lsr, a, b) ->
      lower_lsr
        ~loc:(Sarek_ast.loc_to_ppxlib te.te_loc)
        (lower_expr state a)
        (lower_expr state b)
        (elttype_of_typ te.ty)
  | TEBinop (op, a, b) ->
      Ir.EBinop (ir_binop op te.ty, lower_expr state a, lower_expr state b)
  | TEUnop (op, a) -> Ir.EUnop (ir_unop op, lower_expr state a)
  | TEApp (fn, args) -> (
      let args_ir = List.map (lower_expr state) args in
      match fn.te with
      | TEVar (name, _) when Hashtbl.mem state.fun_map name ->
          (* Module-level function call *)
          if Hashtbl.mem state.lowered_funs name then
            (* Already lowered - use cached *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else if Hashtbl.mem state.lowering_stack name then
            (* Recursive call - emit by name *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else
            (* First time - lower the function *)
            let params, body = Hashtbl.find state.fun_map name in
            let ret_ty = repr body.ty in
            Hashtbl.add state.lowering_stack name () ;
            let fun_body_ir = lower_stmt state body in
            Hashtbl.remove state.lowering_stack name ;
            (* Use make_returning to add return statements without re-traversing *)
            let fun_body_ir = make_returning fun_body_ir in
            (* Convert tparam list to var list *)
            let hf_params =
              List.mapi
                (fun i (p : tparam) ->
                  make_var p.tparam_name i p.tparam_type false)
                params
            in
            let helper_func : Ir.helper_func =
              {
                hf_name = name;
                hf_params;
                hf_ret_type = elttype_of_typ ret_ty;
                hf_body = fun_body_ir;
              }
            in
            Hashtbl.add state.lowered_funs name helper_func ;
            state.lowered_funs_order <- name :: state.lowered_funs_order ;
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
      | _ -> Ir.EApp (lower_expr state fn, args_ir))
  | TERecord (name, fields) ->
      (* Register the record type if not already registered *)
      if not (Hashtbl.mem state.types name) then begin
        let field_types =
          List.map (fun (n, e) -> (n, elttype_of_typ e.ty)) fields
        in
        Hashtbl.add state.types name field_types
      end ;
      Ir.ERecord (name, List.map (fun (n, e) -> (n, lower_expr state e)) fields)
  | TEConstr (ty_name, constr, arg) ->
      (* Register the variant type if not already registered.
         Get constructors from the expression's type (which has full variant info) *)
      if not (Hashtbl.mem state.variants ty_name) then begin
        match repr te.ty with
        | TVariant (_, constrs) ->
            let constr_types =
              List.map
                (fun (cname, ty_opt) -> (cname, constr_payload_elttypes ty_opt))
                constrs
            in
            Hashtbl.add state.variants ty_name constr_types
        | _ -> ()
      end ;
      (* Flatten a multi-argument constructor's tuple payload into one IR
         argument per component, matching the flattened field registration
         above and the multi-binder pattern side. A literal tuple whose
         components are all scalar primitives is the shape the typer produces
         for [MkPair (a, b)]; everything else stays a single argument. *)
      let args =
        match arg with
        | None -> []
        | Some {te = TETuple comps; ty = tup_ty; _}
          when match repr tup_ty with
               | TTuple tys -> List.for_all is_scalar_prim_typ tys
               | _ -> false ->
            List.map (lower_expr state) comps
        | Some e -> [lower_expr state e]
      in
      Ir.EVariant (ty_name, constr, args)
  | TETuple exprs -> (
      (* L13: a tuple literal whose components are scalar primitives is lowered
         to the same synthesized record ([_0], [_1], ...) used for tuple vector
         elements. This carries the nominal type name so struct-based backends
         (OpenCL/GLSL/CUDA/Metal) emit a typed compound literal rather than a
         bare, type-less brace initializer, and it matches what a tuple vector
         slot stores. Non-primitive tuples keep the generic [Ir.ETuple]. *)
      match repr te.ty with
      | TTuple tys
        when List.for_all (fun t -> prim_component_elttype t <> None) tys ->
          let comps =
            List.map
              (fun t ->
                match prim_component_elttype t with
                | Some e -> e
                | None -> assert false (* guarded above *))
              tys
          in
          let name = tuple_record_name comps in
          if not (Hashtbl.mem state.types name) then
            Hashtbl.add
              state.types
              name
              (List.mapi (fun i e -> (tuple_field_name i, e)) comps) ;
          Ir.ERecord
            ( name,
              List.mapi
                (fun i e -> (tuple_field_name i, lower_expr state e))
                exprs )
      | _ -> Ir.ETuple (List.map (lower_expr state) exprs))
  | TEGlobalRef (name, ty) ->
      let v = make_var name 0 ty false in
      Ir.EVar v
  | TEIntrinsicConst ref -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) -> Ir.EIntrinsic (path, name, [])
      | Sarek_env.CorePrimitiveRef name -> Ir.EIntrinsic ([], name, []))
  | TEIntrinsicFun (ref, _conv, args) -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) ->
          Ir.EIntrinsic (path, name, List.map (lower_expr state) args)
      | Sarek_env.CorePrimitiveRef name ->
          Ir.EIntrinsic ([], name, List.map (lower_expr state) args))
  (* If-then-else as expression (returns a value) *)
  | TEIf (cond, then_, Some else_) ->
      Ir.EIf
        (lower_expr state cond, lower_expr state then_, lower_expr state else_)
  | TEIf (cond, then_, None) ->
      (* No else branch - only valid for unit-returning expressions *)
      Ir.EIf (lower_expr state cond, lower_expr state then_, Ir.EConst Ir.CUnit)
  (* Match as expression *)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_expr state body))
          cases
      in
      Ir.EMatch (lower_expr state e, ir_cases)
  (* These expression forms require statement context - should be caught by typer *)
  | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ | TELet _ | TELetRec _
  | TELetMut _ | TEFor _ | TEWhile _ | TESeq _ | TEReturn _ | TECreateArray _
  | TENative _ | TEPragma _ | TELetShared _ | TESuperstep _ | TEOpen _ ->
      failwith
        "Internal error: lower_expr called with statement-only expression. \
         This should have been caught by the type checker."

(** Convert a typed expression to IR statement *)
and lower_stmt (state : state) (te : texpr) : Ir.stmt =
  incr ir_lower_stmt_count ;
  match te.te with
  | TEUnit -> Ir.SEmpty
  | TESeq [] -> Ir.SEmpty
  | TESeq [e] -> lower_stmt state e
  | TESeq es -> Ir.SSeq (List.map (lower_stmt state) es)
  | TEVecSet (vec, idx, value) -> (
      match vec.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ ->
          (* Complex base expression - use LArrayElemExpr *)
          Ir.SAssign
            ( Ir.LArrayElemExpr (lower_expr state vec, lower_expr state idx),
              lower_expr state value ))
  | TEArrSet (arr, idx, value) -> (
      match arr.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ ->
          (* Complex base expression - use LArrayElemExpr *)
          Ir.SAssign
            ( Ir.LArrayElemExpr (lower_expr state arr, lower_expr state idx),
              lower_expr state value ))
  | TEFieldSet (r, field, _, value) ->
      let lv = lower_lvalue state r field in
      Ir.SAssign (lv, lower_expr state value)
  | TEAssign (name, id, value) ->
      let v = make_var name id value.ty true in
      Ir.SAssign (Ir.LVar v, lower_expr state value)
  | TELet (name, id, value, body) -> (
      match value.te with
      (* Special case: create_array - need proper array declaration *)
      | TECreateArray (size, elem_ty, mem) ->
          let size_ir = lower_expr state size in
          let v = make_var name id (TArr (elem_ty, mem)) false in
          let body_ir = lower_stmt state body in
          Ir.SLet
            ( v,
              Ir.EArrayCreate
                (elttype_of_typ elem_ty, size_ir, memspace_of_memspace mem),
              body_ir )
      (* Normal let binding *)
      | _ ->
          let v = make_var name id value.ty false in
          Ir.SLet (v, lower_expr state value, lower_stmt state body))
  | TELetMut (name, id, value, body) ->
      let v = make_var name id value.ty true in
      Ir.SLetMut (v, lower_expr state value, lower_stmt state body)
  | TELetRec (name, _id, params, fn_body, cont) ->
      (* Register function in fun_map for later inlining when called *)
      Hashtbl.add state.fun_map name (params, fn_body) ;
      lower_stmt state cont
  | TEIf (cond, then_, else_opt) ->
      Ir.SIf
        ( lower_expr state cond,
          lower_stmt state then_,
          Option.map (lower_stmt state) else_opt )
  | TEFor (var, id, lo, hi, dir, body) ->
      let v = make_var var id (TPrim TInt32) true in
      let ir_dir = match dir with Upto -> Ir.Upto | Downto -> Ir.Downto in
      Ir.SFor
        ( v,
          lower_expr state lo,
          lower_expr state hi,
          ir_dir,
          lower_stmt state body )
  | TEWhile (cond, body) ->
      Ir.SWhile (lower_expr state cond, lower_stmt state body)
  | TEMatch (e, [({tpat = TPTuple pats; _}, body)])
    when is_primitive_tuple e.ty
         && List.for_all
              (fun p ->
                match p.tpat with TPVar _ | TPAny -> true | _ -> false)
              pats ->
      (* L13: a single-arm tuple-pattern match is not a variant dispatch; the
         C-like backends (OpenCL/GLSL/CUDA/Metal) would otherwise emit a bogus
         [switch (x.tag)]. Rewrite it to a record destructure: bind the
         scrutinee once, then bind each component to its [_i] field. This works
         uniformly on every backend (field access is already supported). *)
      let rec_elt = vector_elem_elttype e.ty in
      incr tuple_tmp_counter ;
      let tmp_name = Printf.sprintf "__sarek_tup_%d" !tuple_tmp_counter in
      let tmp_var : Ir.var =
        {
          var_name = tmp_name;
          var_id = - !tuple_tmp_counter;
          var_type = rec_elt;
          var_mutable = false;
        }
      in
      let body_ir = lower_stmt state body in
      let bound =
        List.fold_right
          (fun (i, p) acc ->
            match p.tpat with
            | TPVar (name, id) ->
                let field =
                  Ir.ERecordField (Ir.EVar tmp_var, tuple_field_name i)
                in
                Ir.SLet (make_var name id p.tpat_ty false, field, acc)
            | _ -> acc)
          (List.mapi (fun i p -> (i, p)) pats)
          body_ir
      in
      Ir.SLet (tmp_var, lower_expr state e, bound)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_stmt state body))
          cases
      in
      Ir.SMatch (lower_expr state e, ir_cases)
  | TEReturn e -> Ir.SReturn (lower_expr state e)
  | TEPragma (opts, body) -> Ir.SPragma (opts, lower_stmt state body)
  | TELetShared (name, id, elem_ty, size_opt, body) ->
      (* Shared memory declaration: __shared__ type name[size]; or __local type name[size]; *)
      let size_ir =
        match size_opt with
        | Some size -> lower_expr state size
        | None -> Ir.EIntrinsic (["Sarek_stdlib"; "Gpu"], "block_dim_x", [])
      in
      let v = make_var name id (TArr (elem_ty, Sarek_types.Shared)) false in
      let elem_ir = elttype_of_typ elem_ty in
      (* Use EArrayCreate with Shared memspace - codegen will emit proper declaration *)
      Ir.SLet
        (v, Ir.EArrayCreate (elem_ir, size_ir, Ir.Shared), lower_stmt state body)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      (* Wrap step_body in SBlock to create C scope for variable isolation *)
      Ir.SSeq
        [
          Ir.SBlock (lower_stmt state step_body);
          Ir.SBarrier;
          lower_stmt state cont;
        ]
  | TEOpen (_path, body) -> lower_stmt state body
  | TECreateArray (_size, _elem_ty, _mem) ->
      (* Standalone array creation - just emit unit *)
      Ir.SExpr (Ir.EConst Ir.CUnit)
  | TENative {gpu; ocaml} -> Ir.SNative {gpu; ocaml}
  (* Pure expressions as statements *)
  | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEVecGet _ | TEArrGet _ | TEFieldGet _ | TEBinop _ | TEUnop _
  | TEApp _ | TERecord _ | TEConstr _ | TETuple _ | TEGlobalRef _
  | TEIntrinsicConst _ | TEIntrinsicFun _ ->
      Ir.SExpr (lower_expr state te)

and lower_lvalue (state : state) (r : texpr) (field : string) : Ir.lvalue =
  match r.te with
  | TEVar (name, id) ->
      let v = make_var name id r.ty false in
      Ir.LRecordField (Ir.LVar v, field)
  | TEFieldGet (base, inner_field, _) ->
      Ir.LRecordField (lower_lvalue state base inner_field, field)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) ->
          Ir.LRecordField (Ir.LArrayElem (name, lower_expr state idx), field)
      | _ ->
          Ir.LRecordField
            ( Ir.LArrayElemExpr (lower_expr state vec, lower_expr state idx),
              field ))
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) ->
          Ir.LRecordField (Ir.LArrayElem (name, lower_expr state idx), field)
      | _ ->
          Ir.LRecordField
            ( Ir.LArrayElemExpr (lower_expr state arr, lower_expr state idx),
              field ))
  | _ ->
      failwith
        "Internal error: lower_lvalue called with non-lvalue expression. This \
         should have been caught by the type checker."

and lower_pattern (pat : tpattern) : Ir.pattern =
  match pat.tpat with
  | TPAny -> Ir.PWild
  | TPVar (name, _) -> Ir.PConstr ("", [name])
  | TPConstr (_ty_name, constr, arg_pat) ->
      let vars =
        match arg_pat with None -> [] | Some p -> extract_pattern_vars p
      in
      Ir.PConstr (constr, vars)
  | TPTuple pats ->
      Ir.PConstr ("tuple", List.concat_map extract_pattern_vars pats)

and extract_pattern_vars (pat : tpattern) : string list =
  match pat.tpat with
  | TPAny -> ["_"]
  | TPVar (name, _) -> [name]
  | TPConstr (_, _, Some p) -> extract_pattern_vars p
  | TPConstr (_, _, None) -> []
  | TPTuple pats -> List.concat_map extract_pattern_vars pats

(** Convert a kernel parameter to IR declaration *)
let lower_param (p : tparam) : Ir.decl =
  (match repr p.tparam_type with
  | TTuple _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Tuple-typed kernel parameters are not supported; pass components as \
         separate parameters."
  | TFun _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Function-typed kernel parameters are not supported."
  | TArr (t, _) -> (
      (* An array of tuples or functions would silently collapse to TInt32 in
         elttype_of_typ (wrong stride, garbage layout) — reject at the
         formal-parameter boundary. (L13: local-array-of-tuple aggregate
         support is a follow-up; only global vector elements are lowered.) *)
      match repr t with
      | TTuple _ ->
          Ppxlib.Location.raise_errorf
            ~loc:Ppxlib.Location.none
            "Arrays of tuples are not supported as kernel parameters; declare \
             a record type with [@@sarek.type] instead, or use a vector."
      | TFun _ ->
          Ppxlib.Location.raise_errorf
            ~loc:Ppxlib.Location.none
            "Arrays of functions are not supported as kernel parameters."
      | _ -> ())
  | TVec t -> (
      match repr t with
      | TFun _ ->
          Ppxlib.Location.raise_errorf
            ~loc:Ppxlib.Location.none
            "Vectors of functions are not supported as kernel parameters."
      | TTuple tys ->
          (* L13: a vector of tuples is lowered to a vector of the synthesized
             packed record; validate the components eagerly for a clear error. *)
          ignore (tuple_record_fields tys)
      | _ -> ())
  | _ -> ()) ;
  let elt =
    match repr p.tparam_type with
    | TVec t -> Ir.TVec (vector_elem_elttype t)
    | _ -> elttype_of_typ p.tparam_type
  in
  let v =
    {
      Ir.var_name = p.tparam_name;
      var_id = p.tparam_id;
      var_type = elt;
      var_mutable = false;
    }
  in
  if p.tparam_is_vec then
    let elem_ty =
      match repr p.tparam_type with TVec t -> vector_elem_elttype t | _ -> elt
    in
    Ir.DParam (v, Some {arr_elttype = elem_ty; arr_memspace = Ir.Global})
  else Ir.DParam (v, None)

(** Lower a complete kernel *)
let lower_kernel (kernel : tkernel) : Ir.kernel * string list =
  (* Reset and log counters *)
  ir_lower_expr_count := 0 ;
  ir_lower_stmt_count := 0 ;
  let kern_name = Option.value kernel.tkern_name ~default:"anon" in
  Sarek_debug.log_to_file
    (Printf.sprintf "  [IR] lower_kernel start: %s" kern_name) ;
  (* Build fun_map from module items (local + registry) *)
  let fun_map = Hashtbl.create 8 in
  let all_mod_items = kernel.tkern_module_items in
  List.iter
    (function
      | TMFun (name, _, params, body) ->
          Hashtbl.replace fun_map name (params, body)
      | _ -> ())
    all_mod_items ;
  let state = create_state fun_map in

  (* Register record types from parameter types (especially vector element types) *)
  let rec register_types_from_typ ty =
    match repr ty with
    | TRecord (name, fields) ->
        if not (Hashtbl.mem state.types name) then begin
          let field_types =
            List.map (fun (n, t) -> (n, elttype_of_typ t)) fields
          in
          Hashtbl.add state.types name field_types
        end ;
        List.iter (fun (_, t) -> register_types_from_typ t) fields
    | TVec elem_ty -> register_types_from_typ elem_ty
    | TArr (elem_ty, _) -> register_types_from_typ elem_ty
    | TTuple tys ->
        (* L13: register the synthesized record for a tuple aggregate so the
           codegen types table knows its field layout, mirroring records.
           Primitivity is checked on the SOURCE types (prim_component_elttype),
           never via elttype_of_typ, whose TTuple/TFun placeholder would let a
           nested tuple register as an int32 field. *)
        (match
           List.map prim_component_elttype tys |> fun opts ->
           if List.for_all Option.is_some opts then
             Some (List.map Option.get opts)
           else None
         with
        | Some comps ->
            let name = tuple_record_name comps in
            if not (Hashtbl.mem state.types name) then
              Hashtbl.add
                state.types
                name
                (List.mapi (fun i e -> (tuple_field_name i, e)) comps)
        | None -> ()) ;
        List.iter register_types_from_typ tys
    | _ -> ()
  in
  List.iter
    (fun (p : tparam) -> register_types_from_typ p.tparam_type)
    kernel.tkern_params ;

  (* Lower module-level constants *)
  let module_items_ir =
    List.fold_right
      (fun item acc ->
        match item with
        | TMConst (name, id, ty, expr) ->
            let v = make_var name id ty false in
            Ir.SSeq [Ir.SLet (v, lower_expr state expr, Ir.SEmpty); acc]
        | TMFun _ -> acc)
      all_mod_items
      Ir.SEmpty
  in

  (* Generate type constructors *)
  let constructors =
    List.concat
      (List.map
         (function
           | TTypeRecord {tdecl_name; tdecl_fields; _} ->
               (* Strip mutability flag from fields *)
               let fields = List.map (fun (n, ty, _) -> (n, ty)) tdecl_fields in
               record_constructor_strings tdecl_name fields
           | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
               variant_constructor_strings tdecl_name tdecl_constructors)
         kernel.tkern_type_decls)
  in

  (* Lower body *)
  let body_ir = lower_stmt state kernel.tkern_body in
  Sarek_debug.log_to_file
    (Printf.sprintf
       "  [IR] lower_kernel done: %s (expr=%d, stmt=%d)"
       kern_name
       !ir_lower_expr_count
       !ir_lower_stmt_count) ;
  let full_body =
    match module_items_ir with
    | Ir.SEmpty -> body_ir
    | _ -> Ir.SSeq [module_items_ir; body_ir]
  in

  (* Collect types from state *)
  let types_list =
    Hashtbl.fold (fun name fields acc -> (name, fields) :: acc) state.types []
  in
  (* Collect variant types from state *)
  let variants_list =
    Hashtbl.fold
      (fun name constrs acc -> (name, constrs) :: acc)
      state.variants
      []
  in
  (* Collect helper functions from state, in dependency order *)
  let funcs_list =
    List.rev state.lowered_funs_order
    |> List.filter_map (fun name -> Hashtbl.find_opt state.lowered_funs name)
  in
  ( {
      Ir.kern_name = Option.value kernel.tkern_name ~default:"sarek_kern";
      (* "kernel" is reserved in OpenCL *)
      kern_params = List.map lower_param kernel.tkern_params;
      kern_locals = [];
      kern_body = full_body;
      kern_types = types_list;
      kern_variants = variants_list;
      kern_funcs = funcs_list;
      kern_native_fn = None;
      (* Native fn is added during quoting *)
    },
    constructors )

(** Get the return value declaration for a kernel *)
let lower_return_value (kernel : tkernel) : Ir.decl option =
  match repr kernel.tkern_return_type with
  | TPrim TUnit -> None
  | ty ->
      let v = make_var "result" 0 ty true in
      Some (Ir.DLocal (v, None))
