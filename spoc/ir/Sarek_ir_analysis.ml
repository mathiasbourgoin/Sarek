(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_analysis - Analysis functions for GPU kernel IR *)

open Sarek_ir_types

(** {1 Generic IR traversal}

    Every "does a kernel use feature X?" detector — and the Float64-intrinsic
    name collector — shares one traversal skeleton over the IR
    (expr/lvalue/stmt/decl/helper/kernel). Historically each family duplicated
    that skeleton, so adding an IR node meant editing ~7 copies and any omission
    silently under-reported a feature.

    A single polymorphic fold now carries the skeleton; each family supplies
    only its per-node behaviour through a {!folder} record. This is also the
    requirement-extraction primitive a future capability/affinity model reuses,
    hence the fully general ['a] accumulator rather than a fixed boolean.

    The four hooks capture every axis along which the old families differed:
    - [fe]: combine the accumulator with a single expression node. This is the
      only place a family's "leaf" fires; the traversal always recurses into the
      node's sub-expressions afterwards. A rich leaf may inspect an expression's
      embedded {e types} here (e.g. [ECast]/[EArrayCreate] element types, an
      [EVar]'s [var_type]) — Float64 detection does exactly this, so the
      traversal never forces a lowest-common-denominator leaf.
    - [ft]: combine the accumulator with an element type occurring at a binder
      or declaration ([SFor]/[SLet]/[SLetMut] binder, [DParam]/[DShared] types,
      helper return/param types, kernel record/variant field types). Families
      that do not inspect types leave this at the identity, which makes those
      positions contribute nothing — reproducing the old "types ignored"
      behaviour exactly.
    - [fnative]: combine the accumulator at an [SNative] node. Inline native GPU
      code is opaque text, so its polarity is asymmetric across families
      (atomics/int_mod/copysign/generic-intrinsic conservatively assume the
      feature is present; float64/nonfinite/collector treat it as absent). This
      is carried explicitly and never flattened away.
    - [visit_lvalue]: whether [SAssign] descends into its l-value. The float64
      detector deliberately ignores assignment l-values; the others recurse (an
      index expression can hide the feature). *)
type 'a folder = {
  fe : 'a -> expr -> 'a;
  ft : 'a -> elttype -> 'a;
  fnative : 'a -> 'a;
  visit_lvalue : bool;
}

let rec expr_fold f acc e =
  let acc = f.fe acc e in
  match e with
  | EConst _ | EVar _ | EArrayLen _ -> acc
  | EBinop (_, e1, e2) | EArrayReadExpr (e1, e2) ->
      expr_fold f (expr_fold f acc e1) e2
  | EUnop (_, e) | ERecordField (e, _) | ECast (_, e) | EArrayRead (_, e) ->
      expr_fold f acc e
  | EIntrinsic (_, _, args) | ETuple args | EVariant (_, _, args) ->
      List.fold_left (expr_fold f) acc args
  | EApp (fn, args) -> List.fold_left (expr_fold f) (expr_fold f acc fn) args
  | ERecord (_, fields) ->
      List.fold_left (fun a (_, e) -> expr_fold f a e) acc fields
  | EArrayCreate (_, size, _) -> expr_fold f acc size
  | EIf (cond, then_, else_) ->
      expr_fold f (expr_fold f (expr_fold f acc cond) then_) else_
  | EMatch (scrutinee, cases) ->
      List.fold_left
        (fun a (_, e) -> expr_fold f a e)
        (expr_fold f acc scrutinee)
        cases

let rec lvalue_fold f acc = function
  | LVar _ -> acc
  | LArrayElem (_, idx) -> expr_fold f acc idx
  | LArrayElemExpr (base, idx) -> expr_fold f (expr_fold f acc base) idx
  | LRecordField (lv, _) -> lvalue_fold f acc lv

let rec stmt_fold f acc = function
  | SAssign (lv, e) ->
      let acc = if f.visit_lvalue then lvalue_fold f acc lv else acc in
      expr_fold f acc e
  | SSeq stmts -> List.fold_left (stmt_fold f) acc stmts
  | SIf (cond, then_, else_) ->
      let acc = expr_fold f acc cond in
      let acc = stmt_fold f acc then_ in
      Option.fold ~none:acc ~some:(stmt_fold f acc) else_
  | SWhile (cond, body) -> stmt_fold f (expr_fold f acc cond) body
  | SFor (v, lo, hi, _, body) ->
      let acc = f.ft acc v.var_type in
      stmt_fold f (expr_fold f (expr_fold f acc lo) hi) body
  | SMatch (scrutinee, cases) ->
      List.fold_left
        (fun a (_, s) -> stmt_fold f a s)
        (expr_fold f acc scrutinee)
        cases
  | SReturn e | SExpr e -> expr_fold f acc e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> acc
  | SLet (v, e, body) | SLetMut (v, e, body) ->
      let acc = f.ft acc v.var_type in
      stmt_fold f (expr_fold f acc e) body
  | SPragma (_, body) | SBlock body -> stmt_fold f acc body
  | SNative _ -> f.fnative acc

let decl_fold f acc = function
  | DParam (v, arr_info) ->
      let acc = f.ft acc v.var_type in
      Option.fold ~none:acc ~some:(fun ai -> f.ft acc ai.arr_elttype) arr_info
  | DLocal (v, init) ->
      let acc = f.ft acc v.var_type in
      Option.fold ~none:acc ~some:(expr_fold f acc) init
  | DShared (_, ty, size) ->
      let acc = f.ft acc ty in
      Option.fold ~none:acc ~some:(expr_fold f acc) size

let helper_fold f acc hf =
  let acc = f.ft acc hf.hf_ret_type in
  let acc = List.fold_left (fun a v -> f.ft a v.var_type) acc hf.hf_params in
  stmt_fold f acc hf.hf_body

(** Fold the whole kernel: params, locals, body, helper functions, and record
    /variant field types. The type positions ([ft]) contribute nothing for
    detectors that do not inspect types, so families that historically skipped
    [kern_types]/[kern_variants] are unaffected by visiting them here. *)
let kernel_fold f acc k =
  let acc = List.fold_left (decl_fold f) acc k.kern_params in
  let acc = List.fold_left (decl_fold f) acc k.kern_locals in
  let acc = stmt_fold f acc k.kern_body in
  let acc = List.fold_left (helper_fold f) acc k.kern_funcs in
  let acc =
    List.fold_left
      (fun a (_, fields) -> List.fold_left (fun a (_, t) -> f.ft a t) a fields)
      acc
      k.kern_types
  in
  List.fold_left
    (fun a (_, constrs) ->
      List.fold_left (fun a (_, args) -> List.fold_left f.ft a args) a constrs)
    acc
    k.kern_variants

(** A boolean [folder] for an "exists" detector. [leaf] fires per expression
    node (and may inspect embedded types); [type_leaf] fires per binder/decl
    type; [native] is the [SNative] verdict; [visit_lvalue] controls whether
    [SAssign] descends into its l-value. *)
let exists_folder ~leaf ?(type_leaf = fun _ -> false) ~native
    ?(visit_lvalue = true) () =
  {
    fe = (fun acc e -> acc || leaf e);
    ft = (fun acc t -> acc || type_leaf t);
    fnative = (if native then fun _ -> true else fun acc -> acc);
    visit_lvalue;
  }

(** {1 Numeric-width feature detection}

    ONE parameterised detector family, not one family per width. Adding a width
    (bf16 is next) is a constructor in {!feature}, an arm in {!elttype_uses},
    and a line in {!folder} — not a fresh copy of a rich leaf, a folder and five
    wrappers. The previous shape had float64 and float16 as two structurally
    identical families whose own docstring said so.

    The family has a {e rich} leaf: it inspects element types, not just
    constructors, at every binder, declaration, cast and array construction,
    plus record/variant field types at the kernel level. Two properties are
    shared by every width and are deliberate:

    - [SNative] is treated as feature-free. Inline native GPU text is opaque;
      whether a native block may hand-write a wide type is a separate,
      not-yet-decided question (see KB / review notes), and a native block that
      does so is responsible for its own feature declaration. This is asymmetric
      vs. the atomics detector below, which is conservative there.
    - Assignment l-values are not descended into.

    The one per-width asymmetry is CONSTANTS: float64 has [CFloat64] literals,
    float16 has no literal and hence no [CFloat16] constant (see
    {!Sarek_ir_types.elttype}). An f16 value always enters through
    [ECast (TFloat16, _)] or an f16-typed binder/parameter, both of which the
    leaf sees. {!const_uses} expresses that directly rather than by omitting an
    arm.

    Consumers: [kernel_uses Float64] drives the OpenCL/GLSL fp64
    pragma/extension, and [kernel_uses Float16] drives both the CUDA/HIP
    conditional [#include <cuda_fp16.h>] and the slice-2 rejection gate at every
    backend's [generate] entry (see {!Sarek_ir_codegen.reject_feature}).

    {!kernel_requirements} is the set-valued form: it is what a future
    [Kernel.requirements] capability field reduces to, and it lives in the right
    layer already ([spoc/ir], no backend dependencies). *)

type feature = Float64 | Float16 | Int64

let all_features = [Float64; Float16; Int64]

let feature_name = function
  | Float64 -> "float64"
  | Float16 -> "float16"
  | Int64 -> "int64"

(** Does element type [t] mention the width [f], transitively through records,
    variants, arrays and vectors? *)
let rec elttype_uses (f : feature) = function
  | TFloat64 -> f = Float64
  | TFloat16 -> f = Float16
  | TInt64 -> f = Int64
  | TRecord (_, fields) -> List.exists (fun (_, t) -> elttype_uses f t) fields
  | TVariant (_, constrs) ->
      List.exists (fun (_, args) -> List.exists (elttype_uses f) args) constrs
  | TArray (elt, _) | TVec elt -> elttype_uses f elt
  | TInt32 | TFloat32 | TBool | TUnit -> false

(** Is constant [c] a literal of width [f]? [Float64] and [Int64] each have one
    ([CFloat64], [CInt64]); f16 has no literal form, so this is [false] for
    [Float16] by construction rather than by a missing case. *)
let const_uses (f : feature) c =
  match (f, c) with
  | Float64, CFloat64 _ -> true
  | Int64, CInt64 _ -> true
  | _ -> false

let feature_leaf f = function
  | EConst c -> const_uses f c
  | EVar v -> elttype_uses f v.var_type
  | ECast (ty, _) | EArrayCreate (ty, _, _) -> elttype_uses f ty
  | _ -> false

let folder_of f =
  exists_folder
    ~leaf:(feature_leaf f)
    ~type_leaf:(elttype_uses f)
    ~native:false
    ~visit_lvalue:false
    ()

(* Built once per width, so the detectors stay allocation-free at call sites
   exactly as the former top-level [float64_folder] / [float16_folder] were. *)
let float64_folder = folder_of Float64

let float16_folder = folder_of Float16

let int64_folder = folder_of Int64

let folder = function
  | Float64 -> float64_folder
  | Float16 -> float16_folder
  | Int64 -> int64_folder

(** Does expression [e] use width [f]? *)
let expr_uses f e = expr_fold (folder f) false e

(** Does statement [s] use width [f]? *)
let stmt_uses f s = stmt_fold (folder f) false s

(** Does declaration [d] use width [f]? *)
let decl_uses f d = decl_fold (folder f) false d

(** Does helper function [hf] use width [f]? *)
let helper_uses f hf = helper_fold (folder f) false hf

(** Does kernel [k] use width [f] anywhere — params, locals, body, helper params
    and return types, and record/variant field types? *)
let kernel_uses f k = kernel_fold (folder f) false k

(** The set of numeric-width features kernel [k] requires. The set-valued form
    of {!kernel_uses}; a future [Kernel.requirements] is this. *)
let kernel_requirements k = List.filter (fun f -> kernel_uses f k) all_features

(** {2 Per-width aliases}

    Thin, so existing call sites and the analysis test suite do not churn. New
    code should prefer [kernel_uses Float16] over the alias. *)

let elttype_uses_float64 t = elttype_uses Float64 t

let const_uses_float64 c = const_uses Float64 c

let expr_uses_float64 e = expr_uses Float64 e

let stmt_uses_float64 s = stmt_uses Float64 s

let decl_uses_float64 d = decl_uses Float64 d

let helper_uses_float64 hf = helper_uses Float64 hf

let kernel_uses_float64 k = kernel_uses Float64 k

let elttype_uses_float16 t = elttype_uses Float16 t

let expr_uses_float16 e = expr_uses Float16 e

let stmt_uses_float16 s = stmt_uses Float16 s

let decl_uses_float16 d = decl_uses Float16 d

let helper_uses_float16 hf = helper_uses Float16 hf

let kernel_uses_float16 k = kernel_uses Float16 k

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
    elsewhere.

    Inline native GPU code ([SNative]) is opaque; fusion must not assume it is
    atomic-free, so the detector is conservative there. *)
let is_atomic_intrinsic_name name =
  let prefix = "atomic_" in
  String.length name >= String.length prefix
  && String.sub name 0 (String.length prefix) = prefix

let atomics_leaf = function
  | EIntrinsic (_, name, _) -> is_atomic_intrinsic_name name
  | _ -> false

let atomics_folder = exists_folder ~leaf:atomics_leaf ~native:true ()

(** Check if an expression contains an atomic intrinsic call *)
let expr_uses_atomics e = expr_fold atomics_folder false e

(** Check if an l-value contains an atomic intrinsic call (in its index/base
    expression). [LVar] has no sub-expression; [LRecordField] recurses into the
    inner l-value. *)
let lvalue_uses_atomics lv = lvalue_fold atomics_folder false lv

(** Check if a statement contains an atomic intrinsic call *)
let stmt_uses_atomics s = stmt_fold atomics_folder false s

(** Check if a declaration contains an atomic intrinsic call *)
let decl_uses_atomics d = decl_fold atomics_folder false d

(** Check if a helper function contains an atomic intrinsic call *)
let helper_uses_atomics hf = helper_fold atomics_folder false hf

(** Check if a kernel uses atomic operations anywhere: params/locals
    initializers, body, and helper functions called from the kernel. Helper
    bodies are walked explicitly — a body-only check would miss atomics hidden
    inside a called helper function. *)
let kernel_uses_atomics k = kernel_fold atomics_folder false k

(** {1 Integer-remainder detection}

    [EBinop (Mod, _, _)] is always integer remainder — float [mod] is lowered to
    the [fmod]/[mod] intrinsic (an [EIntrinsic]), never to [Ir.Mod]. Backends
    that cannot lower [%] directly (e.g. GLSL, whose [%] is undefined for
    negative operands) use this to decide whether to emit a remainder helper.

    L-values are recursed (an array index may carry a [mod], e.g.
    [arr.(j mod n).field <- v]); [SNative] is conservatively assumed to contain
    a remainder so any helper it references is still emitted. *)
let int_mod_leaf = function EBinop (Mod, _, _) -> true | _ -> false

let int_mod_folder = exists_folder ~leaf:int_mod_leaf ~native:true ()

let expr_uses_int_mod e = expr_fold int_mod_folder false e

let lvalue_uses_int_mod lv = lvalue_fold int_mod_folder false lv

let stmt_uses_int_mod s = stmt_fold int_mod_folder false s

let decl_uses_int_mod d = decl_fold int_mod_folder false d

let helper_uses_int_mod hf = helper_fold int_mod_folder false hf

(** Check if a kernel uses integer remainder anywhere: locals initializers,
    body, and helper functions. *)
let kernel_uses_int_mod k = kernel_fold int_mod_folder false k

(** {1 copysign detection}

    [copysign] is not a dedicated IR node (unlike [Mod]); it is an ordinary
    [EIntrinsic (path, "copysign", [x; y])] emitted for [Float32.copysign] and
    [Float64.copysign]. GLSL has no [copysign] builtin under any name, and
    [abs(x)*sign(y)] is wrong for [y=0] (GLSL [sign(0)=0] zeroes the result,
    whereas C [copysign(x, ±0) = ±|x|]) and for the [x=0]/NaN sign-transfer edge
    cases. The GLSL backend therefore lowers it to a bit-level [sarek_copysign]
    helper emitted in the preamble; this predicate decides whether that helper
    is emitted. L-values are recursed (the round-3 LRecordField lesson) and
    [SNative] is conservatively assumed to reference the helper. *)
let is_copysign_intrinsic_name name = String.equal name "copysign"

let copysign_leaf = function
  | EIntrinsic (_, name, _) -> is_copysign_intrinsic_name name
  | _ -> false

let copysign_folder = exists_folder ~leaf:copysign_leaf ~native:true ()

let expr_uses_copysign e = expr_fold copysign_folder false e

let lvalue_uses_copysign lv = lvalue_fold copysign_folder false lv

let stmt_uses_copysign s = stmt_fold copysign_folder false s

let decl_uses_copysign d = decl_fold copysign_folder false d

let helper_uses_copysign hf = helper_fold copysign_folder false hf

(** Check if a kernel uses [copysign] anywhere: locals initializers, body, and
    helper functions. *)
let kernel_uses_copysign k = kernel_fold copysign_folder false k

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
    routes to helpers and maps the composed cases (exp2/log2/cbrt). This is a
    {e collector} rather than a boolean detector, so it uses the generic fold
    with a [string list] accumulator: it ignores types ([ft] = identity) and
    treats [SNative] as contributing nothing. *)
let path_is_float64 path = List.mem "Float64" path

(** Deduplicated names of the Float64 math intrinsics a kernel invokes. *)
let kernel_float64_intrinsics k =
  let folder =
    {
      fe =
        (fun acc -> function
          | EIntrinsic (path, name, _) when path_is_float64 path -> name :: acc
          | _ -> acc);
      ft = (fun acc _ -> acc);
      fnative = (fun acc -> acc);
      visit_lvalue = true;
    }
  in
  List.sort_uniq compare (kernel_fold folder [] k)

(** {1 Non-finite Float64 constant detection}

    A [CFloat64] whose value is ±inf or NaN cannot be spelled as a GLSL literal
    (GLSL has no inf/nan literal), so a backend targeting GLSL reconstructs it
    from its bit pattern via [int64BitsToDouble] — which needs
    [GL_ARB_gpu_shader_int64]. Such a constant can occur independently of any
    transcendental (e.g. a user-written [Float64.infinity]), so the int64
    extension must be gated on this too, not only on the software helper family.
    [SNative] is treated as non-finite-free (native code carries its own
    literals). *)
let const_is_nonfinite_float64 = function
  | CFloat64 f -> not (Float.is_finite f)
  | _ -> false

let nonfinite_f64_leaf = function
  | EConst c -> const_is_nonfinite_float64 c
  | _ -> false

let nonfinite_f64_folder =
  exists_folder ~leaf:nonfinite_f64_leaf ~native:false ()

(** Whether the kernel contains a non-finite Float64 constant anywhere. *)
let kernel_uses_nonfinite_float64 k = kernel_fold nonfinite_f64_folder false k

(** {1 Generic intrinsic-usage detection}

    Whether a kernel calls a named [EIntrinsic] anywhere. Generalizes the
    bespoke [kernel_uses_copysign] / [kernel_uses_int_mod] walkers for backends
    that must conditionally emit a helper for one intrinsic (e.g. the GLSL
    [sarek_fmod] helper for [Float32.fmod]/[Float64.fmod], which GLSL has no
    builtin for). Matches on the intrinsic [name] only, ignoring the module
    path, so both the [Float32] and [Float64] spellings are detected. Inline
    native GPU code ([SNative]) is opaque text and is conservatively assumed to
    reference the intrinsic, mirroring the copysign/int_mod detectors. *)
let kernel_uses_intrinsic name k =
  let leaf = function
    | EIntrinsic (_, n, _) -> String.equal n name
    | _ -> false
  in
  kernel_fold (exists_folder ~leaf ~native:true ()) false k
