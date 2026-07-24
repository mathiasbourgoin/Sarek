(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_analysis - Analysis functions for GPU kernel IR *)

open Sarek_ir_types

(** Check if an element type is or contains float64 *)
let rec elttype_uses_float64 = function
  | TFloat64 -> true
  | TRecord (_, fields) ->
      List.exists (fun (_, t) -> elttype_uses_float64 t) fields
  | TVariant (_, constrs) ->
      List.exists
        (fun (_, args) -> List.exists elttype_uses_float64 args)
        constrs
  | TArray (elt, _) | TVec elt -> elttype_uses_float64 elt
  | TInt32 | TInt64 | TFloat32 | TBool | TUnit -> false

(** Check if a constant is float64 *)
let const_uses_float64 = function CFloat64 _ -> true | _ -> false

(** Check if an expression uses float64 *)
let rec expr_uses_float64 = function
  | EConst c -> const_uses_float64 c
  | EVar v -> elttype_uses_float64 v.var_type
  | EBinop (_, e1, e2) -> expr_uses_float64 e1 || expr_uses_float64 e2
  | EUnop (_, e) -> expr_uses_float64 e
  | EArrayRead (_, idx) -> expr_uses_float64 idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_float64 base || expr_uses_float64 idx
  | ERecordField (e, _) -> expr_uses_float64 e
  | EIntrinsic (_, _, args) -> List.exists expr_uses_float64 args
  | ECast (ty, e) -> elttype_uses_float64 ty || expr_uses_float64 e
  | ETuple exprs -> List.exists expr_uses_float64 exprs
  | EApp (fn, args) ->
      expr_uses_float64 fn || List.exists expr_uses_float64 args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_float64 e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_float64 args
  | EArrayLen _ -> false
  | EArrayCreate (ty, size, _) ->
      elttype_uses_float64 ty || expr_uses_float64 size
  | EIf (cond, then_, else_) ->
      expr_uses_float64 cond || expr_uses_float64 then_
      || expr_uses_float64 else_
  | EMatch (scrutinee, cases) ->
      expr_uses_float64 scrutinee
      || List.exists (fun (_, e) -> expr_uses_float64 e) cases

(** Check if a statement uses float64 *)
let rec stmt_uses_float64 = function
  | SAssign (_, e) -> expr_uses_float64 e
  | SSeq stmts -> List.exists stmt_uses_float64 stmts
  | SIf (cond, then_, else_) ->
      expr_uses_float64 cond || stmt_uses_float64 then_
      || Option.fold ~none:false ~some:stmt_uses_float64 else_
  | SWhile (cond, body) -> expr_uses_float64 cond || stmt_uses_float64 body
  | SFor (v, lo, hi, _, body) ->
      elttype_uses_float64 v.var_type
      || expr_uses_float64 lo || expr_uses_float64 hi || stmt_uses_float64 body
  | SMatch (scrutinee, cases) ->
      expr_uses_float64 scrutinee
      || List.exists (fun (_, s) -> stmt_uses_float64 s) cases
  | SReturn e | SExpr e -> expr_uses_float64 e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (v, e, body) | SLetMut (v, e, body) ->
      elttype_uses_float64 v.var_type
      || expr_uses_float64 e || stmt_uses_float64 body
  | SPragma (_, body) | SBlock body -> stmt_uses_float64 body
  (* Deliberately asymmetric vs. stmt_uses_atomics's SNative arm: native-block float64 usage is a separate, not-yet-decided question - see KB / review notes; do not change this arm as part of the atomics fix. *)
  | SNative _ -> false

(** Check if a declaration uses float64 *)
let decl_uses_float64 = function
  | DParam (v, arr_info) ->
      elttype_uses_float64 v.var_type
      || Option.fold
           ~none:false
           ~some:(fun ai -> elttype_uses_float64 ai.arr_elttype)
           arr_info
  | DLocal (v, init) ->
      elttype_uses_float64 v.var_type
      || Option.fold ~none:false ~some:expr_uses_float64 init
  | DShared (_, ty, size) ->
      elttype_uses_float64 ty
      || Option.fold ~none:false ~some:expr_uses_float64 size

(** Check if a helper function uses float64 *)
let helper_uses_float64 hf =
  elttype_uses_float64 hf.hf_ret_type
  || List.exists (fun v -> elttype_uses_float64 v.var_type) hf.hf_params
  || stmt_uses_float64 hf.hf_body

(** Check if a kernel uses float64 anywhere *)
let kernel_uses_float64 k =
  List.exists decl_uses_float64 k.kern_params
  || List.exists decl_uses_float64 k.kern_locals
  || stmt_uses_float64 k.kern_body
  || List.exists helper_uses_float64 k.kern_funcs
  || List.exists
       (fun (_, fields) ->
         List.exists (fun (_, t) -> elttype_uses_float64 t) fields)
       k.kern_types
  || List.exists
       (fun (_, constrs) ->
         List.exists
           (fun (_, args) -> List.exists elttype_uses_float64 args)
           constrs)
       k.kern_variants

(** {1 Atomic-operation detection}

    Atomic intrinsics have no dedicated IR constructor: the PPX lowers every
    atomic primitive (see the [category = "atomic"] entries registered in
    [sarek/ppx/Sarek_core_primitives.ml], and the [%sarek_intrinsic] atomics in
    [sarek/Sarek_stdlib/Gpu.ml]) to a plain [EIntrinsic (path, name, args)]
    node, e.g. ["atomic_add_int32"], ["atomic_cas_int32"],
    ["atomic_add_global_int32"], ... All such names share the ["atomic_"] prefix
    by registration convention.

    REGISTRATION POINT: this is the single source of truth for recognizing an
    atomic intrinsic from IR. If a future atomic primitive is registered under a
    name that does not start with ["atomic_"], update [is_atomic_intrinsic_name]
    below (and consider exporting the name list from Sarek_core_primitives.ml
    instead of relying on the prefix convention). Do not duplicate this check
    elsewhere. *)
let is_atomic_intrinsic_name name =
  let prefix = "atomic_" in
  String.length name >= String.length prefix
  && String.sub name 0 (String.length prefix) = prefix

(** Check if an expression contains an atomic intrinsic call *)
let rec expr_uses_atomics = function
  | EIntrinsic (_, name, args) ->
      is_atomic_intrinsic_name name || List.exists expr_uses_atomics args
  | EConst _ | EVar _ -> false
  | EBinop (_, e1, e2) -> expr_uses_atomics e1 || expr_uses_atomics e2
  | EUnop (_, e) -> expr_uses_atomics e
  | EArrayRead (_, idx) -> expr_uses_atomics idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_atomics base || expr_uses_atomics idx
  | ERecordField (e, _) -> expr_uses_atomics e
  | ECast (_, e) -> expr_uses_atomics e
  | ETuple exprs -> List.exists expr_uses_atomics exprs
  | EApp (fn, args) ->
      expr_uses_atomics fn || List.exists expr_uses_atomics args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_atomics e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_atomics args
  | EArrayLen _ -> false
  | EArrayCreate (_, size, _) -> expr_uses_atomics size
  | EIf (cond, then_, else_) ->
      expr_uses_atomics cond || expr_uses_atomics then_
      || expr_uses_atomics else_
  | EMatch (scrutinee, cases) ->
      expr_uses_atomics scrutinee
      || List.exists (fun (_, e) -> expr_uses_atomics e) cases

(** Check if an lvalue contains an atomic intrinsic call (in its index/base
    expression). LVar has no sub-expression; LRecordField recurses into the
    inner lvalue. *)
let rec lvalue_uses_atomics = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_atomics idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_atomics base || expr_uses_atomics idx
  | LRecordField (lv, _) -> lvalue_uses_atomics lv

(** Check if a statement contains an atomic intrinsic call *)
let rec stmt_uses_atomics = function
  | SAssign (lv, e) -> lvalue_uses_atomics lv || expr_uses_atomics e
  | SSeq stmts -> List.exists stmt_uses_atomics stmts
  | SIf (cond, then_, else_) ->
      expr_uses_atomics cond || stmt_uses_atomics then_
      || Option.fold ~none:false ~some:stmt_uses_atomics else_
  | SWhile (cond, body) -> expr_uses_atomics cond || stmt_uses_atomics body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_atomics lo || expr_uses_atomics hi || stmt_uses_atomics body
  | SMatch (scrutinee, cases) ->
      expr_uses_atomics scrutinee
      || List.exists (fun (_, s) -> stmt_uses_atomics s) cases
  | SReturn e | SExpr e -> expr_uses_atomics e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_atomics e || stmt_uses_atomics body
  | SPragma (_, body) | SBlock body -> stmt_uses_atomics body
  | SNative _ ->
      (* Conservative: inline native GPU code is opaque; fusion must not
         assume it is atomic-free. *)
      true

(** Check if a declaration contains an atomic intrinsic call (in its
    initializer/size expression, if any) *)
let decl_uses_atomics = function
  | DParam _ -> false
  | DLocal (_, init) -> Option.fold ~none:false ~some:expr_uses_atomics init
  | DShared (_, _, size) -> Option.fold ~none:false ~some:expr_uses_atomics size

(** Check if a helper function contains an atomic intrinsic call *)
let helper_uses_atomics hf = stmt_uses_atomics hf.hf_body

(** Check if a kernel uses atomic operations anywhere: params/locals
    initializers, body, and helper functions called from the kernel. Helper
    bodies must be walked explicitly — a body-only check would miss atomics
    hidden inside a called helper function. *)
let kernel_uses_atomics k =
  List.exists decl_uses_atomics k.kern_params
  || List.exists decl_uses_atomics k.kern_locals
  || stmt_uses_atomics k.kern_body
  || List.exists helper_uses_atomics k.kern_funcs

(** {1 Integer-remainder detection}

    [EBinop (Mod, _, _)] is always integer remainder — float [mod] is lowered to
    the [fmod]/[mod] intrinsic (an [EIntrinsic]), never to [Ir.Mod]. Backends
    that cannot lower [%] directly (e.g. GLSL, whose [%] is undefined for
    negative operands) use this to decide whether to emit a remainder helper. *)
let rec expr_uses_int_mod = function
  | EBinop (Mod, _, _) -> true
  | EConst _ | EVar _ -> false
  | EBinop (_, e1, e2) -> expr_uses_int_mod e1 || expr_uses_int_mod e2
  | EUnop (_, e) -> expr_uses_int_mod e
  | EArrayRead (_, idx) -> expr_uses_int_mod idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_int_mod base || expr_uses_int_mod idx
  | ERecordField (e, _) -> expr_uses_int_mod e
  | EIntrinsic (_, _, args) -> List.exists expr_uses_int_mod args
  | ECast (_, e) -> expr_uses_int_mod e
  | ETuple exprs -> List.exists expr_uses_int_mod exprs
  | EApp (fn, args) ->
      expr_uses_int_mod fn || List.exists expr_uses_int_mod args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_int_mod e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_int_mod args
  | EArrayLen _ -> false
  | EArrayCreate (_, size, _) -> expr_uses_int_mod size
  | EIf (cond, then_, else_) ->
      expr_uses_int_mod cond || expr_uses_int_mod then_
      || expr_uses_int_mod else_
  | EMatch (scrutinee, cases) ->
      expr_uses_int_mod scrutinee
      || List.exists (fun (_, e) -> expr_uses_int_mod e) cases

let rec lvalue_uses_int_mod = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_int_mod idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_int_mod base || expr_uses_int_mod idx
  (* Recurse into the nested lvalue: its array index may carry a [mod], e.g.
     [arr.(j mod n).field <- v]. A non-recursive arm would miss it and skip
     emitting the [sarek_smod] helper the emitted index references. Mirrors
     [lvalue_uses_atomics]. *)
  | LRecordField (lv, _) -> lvalue_uses_int_mod lv

let rec stmt_uses_int_mod = function
  | SAssign (lv, e) -> lvalue_uses_int_mod lv || expr_uses_int_mod e
  | SSeq stmts -> List.exists stmt_uses_int_mod stmts
  | SIf (cond, then_, else_) ->
      expr_uses_int_mod cond || stmt_uses_int_mod then_
      || Option.fold ~none:false ~some:stmt_uses_int_mod else_
  | SWhile (cond, body) -> expr_uses_int_mod cond || stmt_uses_int_mod body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_int_mod lo || expr_uses_int_mod hi || stmt_uses_int_mod body
  | SMatch (scrutinee, cases) ->
      expr_uses_int_mod scrutinee
      || List.exists (fun (_, s) -> stmt_uses_int_mod s) cases
  | SReturn e | SExpr e -> expr_uses_int_mod e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_int_mod e || stmt_uses_int_mod body
  | SPragma (_, body) | SBlock body -> stmt_uses_int_mod body
  (* Inline native GPU code is opaque text; assume it may contain a remainder
     so a helper it references is still emitted. *)
  | SNative _ -> true

let decl_uses_int_mod = function
  | DParam _ -> false
  | DLocal (_, init) -> Option.fold ~none:false ~some:expr_uses_int_mod init
  | DShared (_, _, size) -> Option.fold ~none:false ~some:expr_uses_int_mod size

let helper_uses_int_mod hf = stmt_uses_int_mod hf.hf_body

(** Check if a kernel uses integer remainder anywhere: locals initializers,
    body, and helper functions. Helper bodies are walked explicitly (a helper
    may use [mod] even when the top-level body does not). *)
let kernel_uses_int_mod k =
  List.exists decl_uses_int_mod k.kern_params
  || List.exists decl_uses_int_mod k.kern_locals
  || stmt_uses_int_mod k.kern_body
  || List.exists helper_uses_int_mod k.kern_funcs

(** {1 copysign detection}

    [copysign] is not a dedicated IR node (unlike [Mod]); it is an ordinary
    [EIntrinsic (path, "copysign", [x; y])] emitted for [Float32.copysign] and
    [Float64.copysign]. GLSL has no [copysign] builtin under any name, and
    [abs(x)*sign(y)] is wrong for [y=0] (GLSL [sign(0)=0] zeroes the result,
    whereas C [copysign(x, ±0) = ±|x|]) and for the [x=0]/NaN sign-transfer edge
    cases. The GLSL backend therefore lowers it to a bit-level [sarek_copysign]
    helper emitted in the preamble; this predicate decides whether that helper
    is emitted. Mirrors [kernel_uses_int_mod]. *)
let is_copysign_intrinsic_name name = String.equal name "copysign"

let rec expr_uses_copysign = function
  | EIntrinsic (_, name, args) ->
      is_copysign_intrinsic_name name || List.exists expr_uses_copysign args
  | EConst _ | EVar _ -> false
  | EBinop (_, e1, e2) -> expr_uses_copysign e1 || expr_uses_copysign e2
  | EUnop (_, e) -> expr_uses_copysign e
  | EArrayRead (_, idx) -> expr_uses_copysign idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_copysign base || expr_uses_copysign idx
  | ERecordField (e, _) -> expr_uses_copysign e
  | ECast (_, e) -> expr_uses_copysign e
  | ETuple exprs -> List.exists expr_uses_copysign exprs
  | EApp (fn, args) ->
      expr_uses_copysign fn || List.exists expr_uses_copysign args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_copysign e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_copysign args
  | EArrayLen _ -> false
  | EArrayCreate (_, size, _) -> expr_uses_copysign size
  | EIf (cond, then_, else_) ->
      expr_uses_copysign cond || expr_uses_copysign then_
      || expr_uses_copysign else_
  | EMatch (scrutinee, cases) ->
      expr_uses_copysign scrutinee
      || List.exists (fun (_, e) -> expr_uses_copysign e) cases

(** Recurse into the nested lvalue: its array index may carry a [copysign]
    result cast to an int index, e.g.
    [arr.(int_of_float (copysign ...)).field <- v]. A non-recursive
    [LRecordField] arm would miss it and skip emitting the [sarek_copysign]
    helper the emitted index references. Mirrors [lvalue_uses_int_mod] — the
    round-3 LRecordField lesson. *)
let rec lvalue_uses_copysign = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_copysign idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_copysign base || expr_uses_copysign idx
  | LRecordField (lv, _) -> lvalue_uses_copysign lv

let rec stmt_uses_copysign = function
  | SAssign (lv, e) -> lvalue_uses_copysign lv || expr_uses_copysign e
  | SSeq stmts -> List.exists stmt_uses_copysign stmts
  | SIf (cond, then_, else_) ->
      expr_uses_copysign cond || stmt_uses_copysign then_
      || Option.fold ~none:false ~some:stmt_uses_copysign else_
  | SWhile (cond, body) -> expr_uses_copysign cond || stmt_uses_copysign body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_copysign lo || expr_uses_copysign hi || stmt_uses_copysign body
  | SMatch (scrutinee, cases) ->
      expr_uses_copysign scrutinee
      || List.exists (fun (_, s) -> stmt_uses_copysign s) cases
  | SReturn e | SExpr e -> expr_uses_copysign e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_copysign e || stmt_uses_copysign body
  | SPragma (_, body) | SBlock body -> stmt_uses_copysign body
  (* Inline native GPU code is opaque text; assume it may reference a copysign
     helper so the helper it references is still emitted. Mirrors
     [stmt_uses_int_mod]. *)
  | SNative _ -> true

let decl_uses_copysign = function
  | DParam _ -> false
  | DLocal (_, init) -> Option.fold ~none:false ~some:expr_uses_copysign init
  | DShared (_, _, size) ->
      Option.fold ~none:false ~some:expr_uses_copysign size

let helper_uses_copysign hf = stmt_uses_copysign hf.hf_body

(** Check if a kernel uses [copysign] anywhere: locals initializers, body, and
    helper functions. Helper bodies are walked explicitly (a helper may use
    [copysign] even when the top-level body does not). *)
let kernel_uses_copysign k =
  List.exists decl_uses_copysign k.kern_params
  || List.exists decl_uses_copysign k.kern_locals
  || stmt_uses_copysign k.kern_body
  || List.exists helper_uses_copysign k.kern_funcs

(** {1 Float64 intrinsic detection}

    Collects the names of every path-qualified Float64 math intrinsic invoked
    anywhere in a kernel — an [EIntrinsic (path, name, _)] whose [path] carries
    a ["Float64"] component (matching the four registry-exposing paths
    [["Float64"]], [["Math"; "Float64"]] and their [Sarek_stdlib_meta] twins,
    exactly the test the GLSL polyfill already uses).

    A backend with no native f64 transcendental (GLSL core has no double
    overload for sin/cos/exp/log/pow/… — see [Sarek_ir_glsl]) uses this to
    decide which software helper family ([Sarek_ir_softmath]) to emit per
    kernel. Names are returned deduplicated; the caller filters to the subset it
    routes to helpers and maps the composed cases (exp2/log2/cbrt). Helper
    bodies and locals are walked explicitly, mirroring [kernel_uses_copysign].
*)
let path_is_float64 path = List.mem "Float64" path

let rec expr_float64_intrinsics acc = function
  | EIntrinsic (path, name, args) ->
      let acc = if path_is_float64 path then name :: acc else acc in
      List.fold_left expr_float64_intrinsics acc args
  | EConst _ | EVar _ | EArrayLen _ -> acc
  | EBinop (_, e1, e2) ->
      expr_float64_intrinsics (expr_float64_intrinsics acc e1) e2
  | EUnop (_, e) | ERecordField (e, _) | ECast (_, e) ->
      expr_float64_intrinsics acc e
  | EArrayRead (_, idx) -> expr_float64_intrinsics acc idx
  | EArrayReadExpr (base, idx) ->
      expr_float64_intrinsics (expr_float64_intrinsics acc base) idx
  | ETuple exprs | EVariant (_, _, exprs) ->
      List.fold_left expr_float64_intrinsics acc exprs
  | EApp (fn, args) ->
      List.fold_left
        expr_float64_intrinsics
        (expr_float64_intrinsics acc fn)
        args
  | ERecord (_, fields) ->
      List.fold_left (fun a (_, e) -> expr_float64_intrinsics a e) acc fields
  | EArrayCreate (_, size, _) -> expr_float64_intrinsics acc size
  | EIf (cond, then_, else_) ->
      expr_float64_intrinsics
        (expr_float64_intrinsics (expr_float64_intrinsics acc cond) then_)
        else_
  | EMatch (scrutinee, cases) ->
      List.fold_left
        (fun a (_, e) -> expr_float64_intrinsics a e)
        (expr_float64_intrinsics acc scrutinee)
        cases

let rec lvalue_float64_intrinsics acc = function
  | LVar _ -> acc
  | LArrayElem (_, idx) -> expr_float64_intrinsics acc idx
  | LArrayElemExpr (base, idx) ->
      expr_float64_intrinsics (expr_float64_intrinsics acc base) idx
  | LRecordField (lv, _) -> lvalue_float64_intrinsics acc lv

let rec stmt_float64_intrinsics acc = function
  | SAssign (lv, e) ->
      expr_float64_intrinsics (lvalue_float64_intrinsics acc lv) e
  | SSeq stmts -> List.fold_left stmt_float64_intrinsics acc stmts
  | SIf (cond, then_, else_) ->
      let acc = expr_float64_intrinsics acc cond in
      let acc = stmt_float64_intrinsics acc then_ in
      Option.fold ~none:acc ~some:(stmt_float64_intrinsics acc) else_
  | SWhile (cond, body) ->
      stmt_float64_intrinsics (expr_float64_intrinsics acc cond) body
  | SFor (_, lo, hi, _, body) ->
      stmt_float64_intrinsics
        (expr_float64_intrinsics (expr_float64_intrinsics acc lo) hi)
        body
  | SMatch (scrutinee, cases) ->
      List.fold_left
        (fun a (_, s) -> stmt_float64_intrinsics a s)
        (expr_float64_intrinsics acc scrutinee)
        cases
  | SReturn e | SExpr e -> expr_float64_intrinsics acc e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence | SNative _ -> acc
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      stmt_float64_intrinsics (expr_float64_intrinsics acc e) body
  | SPragma (_, body) | SBlock body -> stmt_float64_intrinsics acc body

let decl_float64_intrinsics acc = function
  | DParam _ -> acc
  | DLocal (_, init) ->
      Option.fold ~none:acc ~some:(expr_float64_intrinsics acc) init
  | DShared (_, _, size) ->
      Option.fold ~none:acc ~some:(expr_float64_intrinsics acc) size

let helper_float64_intrinsics acc hf = stmt_float64_intrinsics acc hf.hf_body

(** Deduplicated names of the Float64 math intrinsics a kernel invokes. *)
let kernel_float64_intrinsics k =
  let acc = List.fold_left decl_float64_intrinsics [] k.kern_params in
  let acc = List.fold_left decl_float64_intrinsics acc k.kern_locals in
  let acc = stmt_float64_intrinsics acc k.kern_body in
  let acc = List.fold_left helper_float64_intrinsics acc k.kern_funcs in
  List.sort_uniq compare acc

(** {1 Non-finite Float64 constant detection}

    A [CFloat64] whose value is ±inf or NaN cannot be spelled as a GLSL literal
    (GLSL has no inf/nan literal), so a backend targeting GLSL reconstructs it
    from its bit pattern via [int64BitsToDouble] — which needs
    [GL_ARB_gpu_shader_int64]. Such a constant can occur independently of any
    transcendental (e.g. a user-written [Float64.infinity]), so the int64
    extension must be gated on this too, not only on the software helper family.
    Mirrors [kernel_uses_copysign]. *)
let const_is_nonfinite_float64 = function
  | CFloat64 f -> not (Float.is_finite f)
  | _ -> false

let rec expr_uses_nonfinite_f64 = function
  | EConst c -> const_is_nonfinite_float64 c
  | EVar _ | EArrayLen _ -> false
  | EBinop (_, e1, e2) | EArrayReadExpr (e1, e2) ->
      expr_uses_nonfinite_f64 e1 || expr_uses_nonfinite_f64 e2
  | EUnop (_, e) | ERecordField (e, _) | ECast (_, e) | EArrayRead (_, e) ->
      expr_uses_nonfinite_f64 e
  | EIntrinsic (_, _, args) | ETuple args | EVariant (_, _, args) ->
      List.exists expr_uses_nonfinite_f64 args
  | EApp (fn, args) ->
      expr_uses_nonfinite_f64 fn || List.exists expr_uses_nonfinite_f64 args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_nonfinite_f64 e) fields
  | EArrayCreate (_, size, _) -> expr_uses_nonfinite_f64 size
  | EIf (c, t, e) ->
      expr_uses_nonfinite_f64 c || expr_uses_nonfinite_f64 t
      || expr_uses_nonfinite_f64 e
  | EMatch (s, cases) ->
      expr_uses_nonfinite_f64 s
      || List.exists (fun (_, e) -> expr_uses_nonfinite_f64 e) cases

let rec lvalue_uses_nonfinite_f64 = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_nonfinite_f64 idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_nonfinite_f64 base || expr_uses_nonfinite_f64 idx
  | LRecordField (lv, _) -> lvalue_uses_nonfinite_f64 lv

let rec stmt_uses_nonfinite_f64 = function
  | SAssign (lv, e) -> lvalue_uses_nonfinite_f64 lv || expr_uses_nonfinite_f64 e
  | SSeq stmts -> List.exists stmt_uses_nonfinite_f64 stmts
  | SIf (cond, then_, else_) ->
      expr_uses_nonfinite_f64 cond
      || stmt_uses_nonfinite_f64 then_
      || Option.fold ~none:false ~some:stmt_uses_nonfinite_f64 else_
  | SWhile (cond, body) ->
      expr_uses_nonfinite_f64 cond || stmt_uses_nonfinite_f64 body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_nonfinite_f64 lo || expr_uses_nonfinite_f64 hi
      || stmt_uses_nonfinite_f64 body
  | SMatch (scrutinee, cases) ->
      expr_uses_nonfinite_f64 scrutinee
      || List.exists (fun (_, s) -> stmt_uses_nonfinite_f64 s) cases
  | SReturn e | SExpr e -> expr_uses_nonfinite_f64 e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence | SNative _ -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_nonfinite_f64 e || stmt_uses_nonfinite_f64 body
  | SPragma (_, body) | SBlock body -> stmt_uses_nonfinite_f64 body

let decl_uses_nonfinite_f64 = function
  | DParam _ -> false
  | DLocal (_, init) ->
      Option.fold ~none:false ~some:expr_uses_nonfinite_f64 init
  | DShared (_, _, size) ->
      Option.fold ~none:false ~some:expr_uses_nonfinite_f64 size

let helper_uses_nonfinite_f64 hf = stmt_uses_nonfinite_f64 hf.hf_body

(** Whether the kernel contains a non-finite Float64 constant anywhere. *)
let kernel_uses_nonfinite_float64 k =
  List.exists decl_uses_nonfinite_f64 k.kern_params
  || List.exists decl_uses_nonfinite_f64 k.kern_locals
  || stmt_uses_nonfinite_f64 k.kern_body
  || List.exists helper_uses_nonfinite_f64 k.kern_funcs

(** {1 Generic intrinsic-usage detection}

    Whether a kernel calls a named [EIntrinsic] anywhere. Generalizes the
    bespoke [kernel_uses_copysign] / [kernel_uses_int_mod] walkers for backends
    that must conditionally emit a helper for one intrinsic (e.g. the GLSL
    [sarek_fmod] helper for [Float32.fmod]/[Float64.fmod], which GLSL has no
    builtin for). Matches on the intrinsic [name] only, ignoring the module
    path, so both the [Float32] and [Float64] spellings are detected. Inline
    native GPU code ([SNative]) is opaque text and is conservatively assumed to
    reference the intrinsic, mirroring the copysign/int_mod detectors. *)
let rec expr_uses_intrinsic name = function
  | EIntrinsic (_, n, args) ->
      String.equal n name || List.exists (expr_uses_intrinsic name) args
  | EBinop (_, e1, e2) ->
      expr_uses_intrinsic name e1 || expr_uses_intrinsic name e2
  | EUnop (_, e) -> expr_uses_intrinsic name e
  | EArrayRead (_, idx) -> expr_uses_intrinsic name idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_intrinsic name base || expr_uses_intrinsic name idx
  | ERecordField (e, _) -> expr_uses_intrinsic name e
  | ECast (_, e) -> expr_uses_intrinsic name e
  | ETuple exprs -> List.exists (expr_uses_intrinsic name) exprs
  | EApp (fn, args) ->
      expr_uses_intrinsic name fn || List.exists (expr_uses_intrinsic name) args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_intrinsic name e) fields
  | EVariant (_, _, args) -> List.exists (expr_uses_intrinsic name) args
  | EArrayLen _ -> false
  | EArrayCreate (_, size, _) -> expr_uses_intrinsic name size
  | EIf (cond, then_, else_) ->
      expr_uses_intrinsic name cond
      || expr_uses_intrinsic name then_
      || expr_uses_intrinsic name else_
  | EMatch (scrutinee, cases) ->
      expr_uses_intrinsic name scrutinee
      || List.exists (fun (_, e) -> expr_uses_intrinsic name e) cases
  | EConst _ | EVar _ -> false

let rec lvalue_uses_intrinsic name = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_intrinsic name idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_intrinsic name base || expr_uses_intrinsic name idx
  | LRecordField (lv, _) -> lvalue_uses_intrinsic name lv

let rec stmt_uses_intrinsic name = function
  | SAssign (lv, e) ->
      lvalue_uses_intrinsic name lv || expr_uses_intrinsic name e
  | SSeq stmts -> List.exists (stmt_uses_intrinsic name) stmts
  | SIf (cond, then_, else_) ->
      expr_uses_intrinsic name cond
      || stmt_uses_intrinsic name then_
      || Option.fold ~none:false ~some:(stmt_uses_intrinsic name) else_
  | SWhile (cond, body) ->
      expr_uses_intrinsic name cond || stmt_uses_intrinsic name body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_intrinsic name lo
      || expr_uses_intrinsic name hi
      || stmt_uses_intrinsic name body
  | SMatch (scrutinee, cases) ->
      expr_uses_intrinsic name scrutinee
      || List.exists (fun (_, s) -> stmt_uses_intrinsic name s) cases
  | SReturn e | SExpr e -> expr_uses_intrinsic name e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_intrinsic name e || stmt_uses_intrinsic name body
  | SPragma (_, body) | SBlock body -> stmt_uses_intrinsic name body
  | SNative _ -> true

let decl_uses_intrinsic name = function
  | DParam _ -> false
  | DLocal (_, init) ->
      Option.fold ~none:false ~some:(expr_uses_intrinsic name) init
  | DShared (_, _, size) ->
      Option.fold ~none:false ~some:(expr_uses_intrinsic name) size

let kernel_uses_intrinsic name k =
  List.exists (decl_uses_intrinsic name) k.kern_params
  || List.exists (decl_uses_intrinsic name) k.kern_locals
  || stmt_uses_intrinsic name k.kern_body
  || List.exists
       (fun (hf : helper_func) -> stmt_uses_intrinsic name hf.hf_body)
       k.kern_funcs
