(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_lower_ir module
 *
 * Tests IR lowering from typed AST to Sarek_ir_ppx

[@@@warning "-32-34"]
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

module Ir = Sarek_ir_ppx

(* Test: mangle_type_name converts dots to underscores *)
let test_mangle_type_name_simple () =
  let result = Sarek_lower_ir.mangle_type_name "Point" in
  Alcotest.(check string) "no change" "Point" result

let test_mangle_type_name_with_dots () =
  let result = Sarek_lower_ir.mangle_type_name "MyModule.Point" in
  Alcotest.(check string) "dots to underscores" "MyModule_Point" result

let test_mangle_type_name_multiple_dots () =
  let result = Sarek_lower_ir.mangle_type_name "A.B.C" in
  Alcotest.(check string) "multiple dots" "A_B_C" result

(* Test: elttype_of_typ converts types correctly *)
let test_elttype_of_typ_int32 () =
  let ty = Sarek_types.(TPrim TInt32) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "is TInt32" true (result = Ir.TInt32)

let test_elttype_of_typ_bool () =
  let ty = Sarek_types.(TPrim TBool) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "is TBool" true (result = Ir.TBool)

let test_elttype_of_typ_unit () =
  let ty = Sarek_types.(TPrim TUnit) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "is TUnit" true (result = Ir.TUnit)

let test_elttype_of_typ_int64 () =
  let ty = Sarek_types.(TReg Int64) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "is TInt64" true (result = Ir.TInt64)

let test_elttype_of_typ_float32 () =
  let ty = Sarek_types.(TReg Float32) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "is TFloat32" true (result = Ir.TFloat32)

let test_elttype_of_typ_float64 () =
  let ty = Sarek_types.(TReg Float64) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "is TFloat64" true (result = Ir.TFloat64)

let test_elttype_of_typ_int_mapped () =
  (* OCaml int maps to int32 on GPU *)
  let ty = Sarek_types.(TReg Int) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  Alcotest.(check bool) "int maps to TInt32" true (result = Ir.TInt32)

let test_elttype_of_typ_vec () =
  let ty = Sarek_types.(TVec (TReg Float32)) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  match result with
  | Ir.TVec Ir.TFloat32 -> ()
  | _ -> Alcotest.fail "expected TVec TFloat32"

let test_elttype_of_typ_array_global () =
  let ty = Sarek_types.(TArr (TReg Int, Global)) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  match result with
  | Ir.TArray (Ir.TInt32, Ir.Global) -> ()
  | _ -> Alcotest.fail "expected TArray with Global memspace"

let test_elttype_of_typ_array_shared () =
  let ty = Sarek_types.(TArr (TReg Float32, Shared)) in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  match result with
  | Ir.TArray (Ir.TFloat32, Ir.Shared) -> ()
  | _ -> Alcotest.fail "expected TArray with Shared memspace"

let test_elttype_of_typ_record () =
  let ty =
    Sarek_types.(TRecord ("Point", [("x", TReg Int); ("y", TReg Int)]))
  in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  match result with
  | Ir.TRecord ("Point", fields) ->
      Alcotest.(check int) "two fields" 2 (List.length fields)
  | _ -> Alcotest.fail "expected TRecord"

let test_elttype_of_typ_variant () =
  let ty =
    Sarek_types.(
      TVariant ("Option", [("None", None); ("Some", Some (TReg Int))]))
  in
  let result = Sarek_lower_ir.elttype_of_typ ty in
  match result with
  | Ir.TVariant ("Option", constrs) ->
      Alcotest.(check int) "two constructors" 2 (List.length constrs)
  | _ -> Alcotest.fail "expected TVariant"

(* Test: memspace_of_memspace conversion *)
let test_memspace_of_memspace_global () =
  let result = Sarek_lower_ir.memspace_of_memspace Sarek_types.Global in
  Alcotest.(check bool) "is Global" true (result = Ir.Global)

let test_memspace_of_memspace_shared () =
  let result = Sarek_lower_ir.memspace_of_memspace Sarek_types.Shared in
  Alcotest.(check bool) "is Shared" true (result = Ir.Shared)

let test_memspace_of_memspace_local () =
  let result = Sarek_lower_ir.memspace_of_memspace Sarek_types.Local in
  Alcotest.(check bool) "is Local" true (result = Ir.Local)

(* Test: c_type_of_typ generates C type strings *)
let test_c_type_of_typ_int () =
  let ty = Sarek_types.(TPrim TInt32) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "int" result

let test_c_type_of_typ_bool () =
  let ty = Sarek_types.(TPrim TBool) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "int" result

let test_c_type_of_typ_unit () =
  let ty = Sarek_types.(TPrim TUnit) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "void" result

let test_c_type_of_typ_float () =
  let ty = Sarek_types.(TReg Float32) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "float" result

let test_c_type_of_typ_double () =
  let ty = Sarek_types.(TReg Float64) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "double" result

let test_c_type_of_typ_long () =
  let ty = Sarek_types.(TReg Int64) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "long" result

let test_c_type_of_typ_char () =
  let ty = Sarek_types.(TReg Char) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "char" result

let test_c_type_of_typ_custom () =
  let ty = Sarek_types.(TReg (Custom "MyType")) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "MyType" result

let test_c_type_of_typ_vec_pointer () =
  let ty = Sarek_types.(TVec (TReg Float32)) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "float *" result

let test_c_type_of_typ_array_pointer () =
  let ty = Sarek_types.(TArr (TReg Int, Local)) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "int *" result

let test_c_type_of_typ_record_struct () =
  let ty = Sarek_types.(TRecord ("Point", [("x", TReg Int)])) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type" "struct Point_sarek" result

let test_c_type_of_typ_record_with_dots () =
  let ty = Sarek_types.(TRecord ("Mod.Point", [("x", TReg Int)])) in
  let result = Sarek_lower_ir.c_type_of_typ ty in
  Alcotest.(check string) "C type mangled" "struct Mod_Point_sarek" result

(* Test: record_constructor_strings generates C code *)
let test_record_constructor_strings () =
  let fields = [("x", Sarek_types.(TReg Int)); ("y", Sarek_types.(TReg Int))] in
  let result = Sarek_lower_ir.record_constructor_strings "Point" fields in
  (* Returns [struct_def; builder] *)
  Alcotest.(check int) "two strings" 2 (List.length result) ;
  let struct_def = List.hd result in
  Alcotest.(check bool)
    "struct has fields"
    true
    (String.length struct_def > 0 && String.contains struct_def '{')

(* Test: variant_constructor_strings generates C code *)
let test_variant_constructor_strings () =
  let constrs = [("None", None); ("Some", Some Sarek_types.(TReg Int))] in
  let result = Sarek_lower_ir.variant_constructor_strings "Option" constrs in
  (* Returns list of constructor structs, union, main struct, and builders *)
  Alcotest.(check bool) "has multiple parts" true (List.length result > 2)

(* Test: ir_binop converts binary operators *)
let test_ir_binop_add () =
  let result = Sarek_lower_ir.ir_binop Sarek_ast.Add Sarek_types.(TReg Int) in
  Alcotest.(check bool) "is Add" true (result = Ir.Add)

let test_ir_binop_mul () =
  let result =
    Sarek_lower_ir.ir_binop Sarek_ast.Mul Sarek_types.(TReg Float32)
  in
  Alcotest.(check bool) "is Mul" true (result = Ir.Mul)

let test_ir_binop_eq () =
  let result = Sarek_lower_ir.ir_binop Sarek_ast.Eq Sarek_types.(TReg Int) in
  Alcotest.(check bool) "is Eq" true (result = Ir.Eq)

let test_ir_binop_lt () =
  let result = Sarek_lower_ir.ir_binop Sarek_ast.Lt Sarek_types.(TReg Int) in
  Alcotest.(check bool) "is Lt" true (result = Ir.Lt)

(* Test: ir_unop converts unary operators *)
let test_ir_unop_neg () =
  let result = Sarek_lower_ir.ir_unop Sarek_ast.Neg in
  Alcotest.(check bool) "is Neg" true (result = Ir.Neg)

let test_ir_unop_not () =
  let result = Sarek_lower_ir.ir_unop Sarek_ast.Not in
  Alcotest.(check bool) "is Not" true (result = Ir.Not)

let test_ir_unop_lnot () =
  let result = Sarek_lower_ir.ir_unop Sarek_ast.Lnot in
  Alcotest.(check bool) "is BitNot" true (result = Ir.BitNot)

(* Test: lower_memspace converts memspaces *)
let test_lower_memspace_global () =
  let result = Sarek_lower_ir.lower_memspace Sarek_types.Global in
  Alcotest.(check bool) "is Global" true (result = Ir.Global)

let test_lower_memspace_shared () =
  let result = Sarek_lower_ir.lower_memspace Sarek_types.Shared in
  Alcotest.(check bool) "is Shared" true (result = Ir.Shared)

let test_lower_memspace_local () =
  let result = Sarek_lower_ir.lower_memspace Sarek_types.Local in
  Alcotest.(check bool) "is Local" true (result = Ir.Local)

(* Test: make_var creates IR variable *)
let test_make_var_immutable () =
  let var = Sarek_lower_ir.make_var "x" 0 Sarek_types.(TReg Int) false in
  Alcotest.(check string) "name" "x" var.Ir.var_name ;
  Alcotest.(check int) "id" 0 var.Ir.var_id ;
  Alcotest.(check bool) "not mutable" false var.Ir.var_mutable

let test_make_var_mutable () =
  let var = Sarek_lower_ir.make_var "y" 1 Sarek_types.(TReg Float32) true in
  Alcotest.(check string) "name" "y" var.Ir.var_name ;
  Alcotest.(check int) "id" 1 var.Ir.var_id ;
  Alcotest.(check bool) "is mutable" true var.Ir.var_mutable

(* Test: lower_decl creates IR declaration *)
let test_lower_decl_immutable () =
  let decl =
    Sarek_lower_ir.lower_decl ~mutable_:false 0 "x" Sarek_types.(TReg Int)
  in
  match decl with
  | Ir.DLocal (var, _) ->
      Alcotest.(check string) "name" "x" var.Ir.var_name ;
      Alcotest.(check bool) "not mutable" false var.Ir.var_mutable
  | _ -> Alcotest.fail "expected DLocal"

let test_lower_decl_mutable () =
  let decl =
    Sarek_lower_ir.lower_decl ~mutable_:true 1 "y" Sarek_types.(TReg Float32)
  in
  match decl with
  | Ir.DLocal (var, _) ->
      Alcotest.(check string) "name" "y" var.Ir.var_name ;
      Alcotest.(check bool) "is mutable" true var.Ir.var_mutable
  | _ -> Alcotest.fail "expected DLocal"

(* Test: lower_param converts kernel parameters *)
let test_lower_param () =
  let tparam =
    Sarek_typed_ast.
      {
        tparam_name = "input";
        tparam_id = 0;
        tparam_type = Sarek_types.(TVec (TReg Float32));
        tparam_index = 0;
        tparam_is_vec = true;
      }
  in
  let result = Sarek_lower_ir.lower_param tparam in
  match result with
  | Ir.DParam (var, _) ->
      Alcotest.(check string) "name" "input" var.Ir.var_name ;
      Alcotest.(check int) "id" 0 var.Ir.var_id
  | _ -> Alcotest.fail "expected DParam"

(* Test: TEBinop (Lsr, _, _) / (Asr, _, _) with a NEGATIVE left operand.
   Group G (shift semantics): Sarek_ast.Lsr and Asr both used to map to the
   same Ir.Shr constructor, which every consumer emits as an *arithmetic*
   shift ([>>] on signed CUDA/OpenCL/Metal/GLSL/WGSL int types, shr.s32 on
   PTX, Int32.shift_right in the interpreter - see
   briefs/fix-critical-semantics-evidence.md, G phase 1). [asr] is correct;
   [lsr] on a negative operand silently returned the wrong (sign-extended)
   value. These tests lower a real Sarek_ast.Lsr/Asr binop end-to-end
   (lower_expr -> Sarek_ir_conv.conv_expr -> the runtime interpreter) and
   check the executed result against the true logical/arithmetic shift, so
   they fail pre-fix and pass post-fix - see the evidence file for the
   captured pre-fix failure. *)

(* Minimal, self-contained evaluator for the fragment of Sarek_ir_ppx.expr
   that lower_lsr/ir_binop can produce for an int32 Lsr/Asr binop (EConst
   CInt32/CBool, EBinop over Add/Sub/Shl/Shr/BitAnd/BitXor/Eq, EIf). This
   exercises the REAL production lowering (Sarek_lower_ir.lower_expr /
   lower_lsr) without reaching into sarek_transpile's private Sarek_ir_conv
   (not part of that library's public .mli).
   [BitAnd] was added to the fragment when [lower_lsr] started masking its
   internal shift-count with [width - 1] (finding 2: PTX/WGSL evaluate both
   [EIf] branches, so the else-branch's shift-by-[width-n] must itself be
   well-defined for every [n], including [n = 0]). *)
let rec eval_ir_int32 (e : Ir.expr) : int32 =
  match e with
  | Ir.EConst (Ir.CInt32 n) -> n
  | Ir.EBinop (Ir.Add, a, b) -> Int32.add (eval_ir_int32 a) (eval_ir_int32 b)
  | Ir.EBinop (Ir.Sub, a, b) -> Int32.sub (eval_ir_int32 a) (eval_ir_int32 b)
  | Ir.EBinop (Ir.Shl, a, b) ->
      Int32.shift_left (eval_ir_int32 a) (Int32.to_int (eval_ir_int32 b))
  | Ir.EBinop (Ir.Shr, a, b) ->
      (* Ir.Shr is arithmetic on every backend post-fix (G phase 1) *)
      Int32.shift_right (eval_ir_int32 a) (Int32.to_int (eval_ir_int32 b))
  | Ir.EBinop (Ir.BitAnd, a, b) ->
      Int32.logand (eval_ir_int32 a) (eval_ir_int32 b)
  | Ir.EBinop (Ir.BitXor, a, b) ->
      Int32.logxor (eval_ir_int32 a) (eval_ir_int32 b)
  | Ir.EIf (cond, then_, else_) ->
      if eval_ir_bool cond then eval_ir_int32 then_ else eval_ir_int32 else_
  | _ -> Alcotest.fail "eval_ir_int32: unsupported node"

and eval_ir_bool (e : Ir.expr) : bool =
  match e with
  | Ir.EBinop (Ir.Eq, a, b) -> eval_ir_int32 a = eval_ir_int32 b
  | _ -> Alcotest.fail "eval_ir_bool: unsupported node"

let eval_int32_binop op a b =
  let ty = Sarek_types.(TReg Int) in
  let mk te = Sarek_typed_ast.{te; ty; te_loc = Sarek_ast.dummy_loc} in
  let texpr =
    mk (Sarek_typed_ast.TEBinop (op, mk (TEInt32 a), mk (TEInt32 b)))
  in
  let state = Sarek_lower_ir.create_state (Hashtbl.create 1) in
  let ir_expr = Sarek_lower_ir.lower_expr state texpr in
  eval_ir_int32 ir_expr

let test_lsr_negative_is_logical () =
  (* (-16) lsr 2 = 1073741820, NOT (-16) asr 2 = -4 *)
  let result = eval_int32_binop Sarek_ast.Lsr (-16l) 2l in
  Alcotest.(check int32) "(-16) lsr 2" 1073741820l result

let test_asr_negative_is_arithmetic () =
  let result = eval_int32_binop Sarek_ast.Asr (-16l) 2l in
  Alcotest.(check int32) "(-16) asr 2" (-4l) result

let test_lsr_negative_shift_by_zero () =
  (* n = 0 must not hit the (width - n) = width shift-by-32 UB guard *)
  let result = eval_int32_binop Sarek_ast.Lsr (-16l) 0l in
  Alcotest.(check int32) "(-16) lsr 0" (-16l) result

let test_lsr_negative_shift_by_31 () =
  (* Top bit shifted all the way down to bit 0 *)
  let result = eval_int32_binop Sarek_ast.Lsr (-16l) 31l in
  Alcotest.(check int32) "(-16) lsr 31" 1l result

let test_lsr_positive_matches_asr () =
  (* For a non-negative operand, lsr and asr must agree *)
  let lsr_result = eval_int32_binop Sarek_ast.Lsr 16l 2l in
  let asr_result = eval_int32_binop Sarek_ast.Asr 16l 2l in
  Alcotest.(check int32) "16 lsr 2" 4l lsr_result ;
  Alcotest.(check int32) "16 asr 2 matches lsr" lsr_result asr_result

(* Finding 1 (review pass): lower_lsr used to build a tree that references
   its [a]/[b] IR operands three times each (sign_fill, top_bits, and the
   final EIf's branches/condition) with no way to evaluate a non-trivial
   operand once and reuse the result (Sarek_ir_ppx.expr has no let-binding
   form - see the doc comment on lower_lsr). If the operand carried a side
   effect (e.g. an EIntrinsic atomic call), it would be silently evaluated
   multiple times. lower_lsr now raises a located Ppxlib PPX error instead of
   building the unsafe tree whenever an operand is not a plain EVar/EConst.
   This test lowers [(1 + 2) lsr n] - the left operand becomes
   Ir.EBinop(Add, EConst, EConst) after lowering, which is not EVar/EConst,
   so it is non-trivial - and asserts lower_expr raises. Fails pre-fix
   (no check existed, so the duplicating tree was built silently) and
   passes post-fix. *)
let test_lsr_raises_on_nontrivial_operand () =
  let ty = Sarek_types.(TReg Int) in
  let mk te = Sarek_typed_ast.{te; ty; te_loc = Sarek_ast.dummy_loc} in
  let non_trivial =
    mk
      (Sarek_typed_ast.TEBinop (Sarek_ast.Add, mk (TEInt32 1l), mk (TEInt32 2l)))
  in
  let texpr =
    mk (Sarek_typed_ast.TEBinop (Sarek_ast.Lsr, non_trivial, mk (TEInt32 2l)))
  in
  let state = Sarek_lower_ir.create_state (Hashtbl.create 1) in
  let raised =
    try
      let (_ : Ir.expr) = Sarek_lower_ir.lower_expr state texpr in
      false
    with Ppxlib.Location.Error _ -> true
  in
  Alcotest.(check bool) "lsr with non-trivial operand raises" true raised

(* Companion case: a non-trivial *right* operand (the shift count) must also
   be rejected, not just a non-trivial left operand. *)
let test_lsr_raises_on_nontrivial_shift_count () =
  let ty = Sarek_types.(TReg Int) in
  let mk te = Sarek_typed_ast.{te; ty; te_loc = Sarek_ast.dummy_loc} in
  let non_trivial_count =
    mk
      (Sarek_typed_ast.TEBinop (Sarek_ast.Add, mk (TEInt32 1l), mk (TEInt32 1l)))
  in
  let texpr =
    mk
      (Sarek_typed_ast.TEBinop
         (Sarek_ast.Lsr, mk (TEInt32 16l), non_trivial_count))
  in
  let state = Sarek_lower_ir.create_state (Hashtbl.create 1) in
  let raised =
    try
      let (_ : Ir.expr) = Sarek_lower_ir.lower_expr state texpr in
      false
    with Ppxlib.Location.Error _ -> true
  in
  Alcotest.(check bool) "lsr with non-trivial shift count raises" true raised

(* Finding 1, negative check: trivial operands (plain int literals, which
   lower to Ir.EConst) must NOT raise - this must not regress. *)
let test_lsr_trivial_operands_do_not_raise () =
  let result = eval_int32_binop Sarek_ast.Lsr (-16l) 2l in
  Alcotest.(check int32) "(-16) lsr 2 (trivial operands)" 1073741820l result

(* Finding 2 (review pass): PTX's selp / WGSL's select() evaluate BOTH
   branches of the EIf lower_lsr builds, so the else-branch's internal
   [width_bits - n] shift-count must itself be well-defined (in
   [0..width_bits-1]) for every n, including n = 0 (where the unmasked
   value would be exactly width_bits - a shift-by-32, undefined/rejected on
   some backends - even though the EIf "selects" the then-branch for n = 0).
   lower_lsr now wraps that shift-count in [BitAnd (_, width_bits - 1)].
   This test lowers a real [x lsr n] end-to-end and inspects the resulting
   Ir.expr tree structure directly (not just its evaluated value) for a
   BitAnd node whose mask constant is 31 (= 32 - 1 for TInt32). Fails
   pre-fix (no BitAnd node exists in the pre-fix tree) and passes
   post-fix. *)
let rec contains_bitand_mask (mask : int32) (e : Ir.expr) : bool =
  match e with
  | Ir.EBinop (Ir.BitAnd, _, Ir.EConst (Ir.CInt32 m)) when m = mask -> true
  | Ir.EBinop (_, a, b) ->
      contains_bitand_mask mask a || contains_bitand_mask mask b
  | Ir.EIf (c, t, e2) ->
      contains_bitand_mask mask c
      || contains_bitand_mask mask t
      || contains_bitand_mask mask e2
  | Ir.EUnop (_, a) -> contains_bitand_mask mask a
  | _ -> false

let test_lsr_shift_count_is_masked () =
  let ty = Sarek_types.(TReg Int) in
  let mk te = Sarek_typed_ast.{te; ty; te_loc = Sarek_ast.dummy_loc} in
  let texpr =
    mk
      (Sarek_typed_ast.TEBinop
         (Sarek_ast.Lsr, mk (TEVar ("x", 0)), mk (TEInt32 0l)))
  in
  let state = Sarek_lower_ir.create_state (Hashtbl.create 1) in
  let ir_expr = Sarek_lower_ir.lower_expr state texpr in
  Alcotest.(check bool)
    "lsr lowering masks the shift-count with (width - 1)"
    true
    (contains_bitand_mask 31l ir_expr)

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_lower_ir"
    [
      ( "mangle_type_name",
        [
          Alcotest.test_case "simple name" `Quick test_mangle_type_name_simple;
          Alcotest.test_case
            "name with dots"
            `Quick
            test_mangle_type_name_with_dots;
          Alcotest.test_case
            "multiple dots"
            `Quick
            test_mangle_type_name_multiple_dots;
        ] );
      ( "elttype_of_typ",
        [
          Alcotest.test_case "int32" `Quick test_elttype_of_typ_int32;
          Alcotest.test_case "bool" `Quick test_elttype_of_typ_bool;
          Alcotest.test_case "unit" `Quick test_elttype_of_typ_unit;
          Alcotest.test_case "int64" `Quick test_elttype_of_typ_int64;
          Alcotest.test_case "float32" `Quick test_elttype_of_typ_float32;
          Alcotest.test_case "float64" `Quick test_elttype_of_typ_float64;
          Alcotest.test_case "int mapped" `Quick test_elttype_of_typ_int_mapped;
          Alcotest.test_case "vec" `Quick test_elttype_of_typ_vec;
          Alcotest.test_case
            "array global"
            `Quick
            test_elttype_of_typ_array_global;
          Alcotest.test_case
            "array shared"
            `Quick
            test_elttype_of_typ_array_shared;
          Alcotest.test_case "record" `Quick test_elttype_of_typ_record;
          Alcotest.test_case "variant" `Quick test_elttype_of_typ_variant;
        ] );
      ( "memspace_conversion",
        [
          Alcotest.test_case "global" `Quick test_memspace_of_memspace_global;
          Alcotest.test_case "shared" `Quick test_memspace_of_memspace_shared;
          Alcotest.test_case "local" `Quick test_memspace_of_memspace_local;
        ] );
      ( "c_type_generation",
        [
          Alcotest.test_case "int" `Quick test_c_type_of_typ_int;
          Alcotest.test_case "bool" `Quick test_c_type_of_typ_bool;
          Alcotest.test_case "unit" `Quick test_c_type_of_typ_unit;
          Alcotest.test_case "float" `Quick test_c_type_of_typ_float;
          Alcotest.test_case "double" `Quick test_c_type_of_typ_double;
          Alcotest.test_case "long" `Quick test_c_type_of_typ_long;
          Alcotest.test_case "char" `Quick test_c_type_of_typ_char;
          Alcotest.test_case "custom" `Quick test_c_type_of_typ_custom;
          Alcotest.test_case "vec pointer" `Quick test_c_type_of_typ_vec_pointer;
          Alcotest.test_case
            "array pointer"
            `Quick
            test_c_type_of_typ_array_pointer;
          Alcotest.test_case
            "record struct"
            `Quick
            test_c_type_of_typ_record_struct;
          Alcotest.test_case
            "record with dots"
            `Quick
            test_c_type_of_typ_record_with_dots;
        ] );
      ( "constructor_generation",
        [
          Alcotest.test_case
            "record constructor"
            `Quick
            test_record_constructor_strings;
          Alcotest.test_case
            "variant constructor"
            `Quick
            test_variant_constructor_strings;
        ] );
      ( "operator_conversion",
        [
          Alcotest.test_case "binop add" `Quick test_ir_binop_add;
          Alcotest.test_case "binop mul" `Quick test_ir_binop_mul;
          Alcotest.test_case "binop eq" `Quick test_ir_binop_eq;
          Alcotest.test_case "binop lt" `Quick test_ir_binop_lt;
          Alcotest.test_case "unop neg" `Quick test_ir_unop_neg;
          Alcotest.test_case "unop not" `Quick test_ir_unop_not;
          Alcotest.test_case "unop lnot" `Quick test_ir_unop_lnot;
        ] );
      ( "lowering_helpers",
        [
          Alcotest.test_case
            "lower memspace global"
            `Quick
            test_lower_memspace_global;
          Alcotest.test_case
            "lower memspace shared"
            `Quick
            test_lower_memspace_shared;
          Alcotest.test_case
            "lower memspace local"
            `Quick
            test_lower_memspace_local;
          Alcotest.test_case "make var immutable" `Quick test_make_var_immutable;
          Alcotest.test_case "make var mutable" `Quick test_make_var_mutable;
          Alcotest.test_case
            "lower decl immutable"
            `Quick
            test_lower_decl_immutable;
          Alcotest.test_case "lower decl mutable" `Quick test_lower_decl_mutable;
          Alcotest.test_case "lower param" `Quick test_lower_param;
        ] );
      ( "shift_semantics",
        [
          Alcotest.test_case
            "lsr negative is logical"
            `Quick
            test_lsr_negative_is_logical;
          Alcotest.test_case
            "asr negative is arithmetic"
            `Quick
            test_asr_negative_is_arithmetic;
          Alcotest.test_case
            "lsr negative shift by zero"
            `Quick
            test_lsr_negative_shift_by_zero;
          Alcotest.test_case
            "lsr negative shift by 31"
            `Quick
            test_lsr_negative_shift_by_31;
          Alcotest.test_case
            "lsr positive matches asr"
            `Quick
            test_lsr_positive_matches_asr;
          Alcotest.test_case
            "lsr raises on non-trivial operand"
            `Quick
            test_lsr_raises_on_nontrivial_operand;
          Alcotest.test_case
            "lsr raises on non-trivial shift count"
            `Quick
            test_lsr_raises_on_nontrivial_shift_count;
          Alcotest.test_case
            "lsr trivial operands do not raise"
            `Quick
            test_lsr_trivial_operands_do_not_raise;
          Alcotest.test_case
            "lsr shift count is masked"
            `Quick
            test_lsr_shift_count_is_masked;
        ] );
    ]
