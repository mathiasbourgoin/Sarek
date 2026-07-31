(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module parses OCaml AST (from ppxlib) into Sarek_ast.
 ******************************************************************************)

open Ppxlib
open Sarek_parse_helpers

(** Re-export helpers used by external callers *)
exception Parse_error_exn = Sarek_parse_helpers.Parse_error_exn

let extract_name_from_pattern = Sarek_parse_helpers.extract_name_from_pattern

let extract_param_from_pattern = Sarek_parse_helpers.extract_param_from_pattern

let extract_type_from_pattern = Sarek_parse_helpers.extract_type_from_pattern

let collect_fun_params = Sarek_parse_helpers.collect_fun_params

let pattern_of_param = Sarek_parse_helpers.pattern_of_param

let parse_binop = Sarek_parse_helpers.parse_binop

let parse_unop = Sarek_parse_helpers.parse_unop

let parse_type = Sarek_parse_helpers.parse_type

(** The refusal for a [when] guard on a match case (backlog-191).

    Before this refusal existed, [parse_expression]'s [Pexp_match] arm read
    [pc_lhs] and [pc_rhs] and never looked at [pc_guard]. The guard was dropped
    in the PARSER — upstream of type-checking, of every lowering pass and of
    backend selection — so the arm became unconditional in the one AST all
    backends are generated from. The kernel compiled and computed a different
    function than its source says.

    How silent that was depends on the arm shape, and the two halves pull
    against each other. A wrong-ANSWER repro needs two arms on the SAME
    constructor, one guarded and one not — the guard is then the only thing
    choosing between them, so dropping it changes the result. But dropping it
    also leaves two syntactically identical arms, which is exactly OCaml's
    warning 11 [redundant-case]; under the [(:standard -w -32-33-34-69)] flags
    the e2e suites use, 11 is left on as an error. So for the shape that
    demonstrates a wrong answer there WAS a diagnostic, just one pointing at a
    redundant case rather than at a dropped guard. The genuinely silent shapes
    are the ones where no two arms share a constructor: the guard is dropped
    with nothing to make the arms look redundant, and the arm becomes
    unconditional with no warning at all. Either way the refusal is the fix —
    this is not a bug you can rely on warning 11 to catch.

    Why this refuses rather than lowering the guard. Three things would have to
    change together, and none of them is local:

    - [Sarek_ir_ppx]'s [EMatch] and [SMatch] hold [(pattern * _) list]. There is
      no guard slot to lower INTO, so a guard needs a new field on the match
      node — and not on one type. There are two IRs: [Sarek_ir_ppx] (the PPX
      compile-time IR, in sarek_frontend) and [Sarek_ir_types] ([spoc/ir], the
      runtime types the code generators read), bridged by [Sarek_ir_conv]; the
      device emitters consume the SECOND, and [Sarek_ast] carries a third
      like-named [EMatch] on the surface AST. The field has to be threaded
      through each representation and through the conversion between them, and
      the set of places that read those constructors is large and open — see the
      paragraph below the bullets.
    - Every C-family emitter builds a match as a nested tag ternary whose LAST
      arm is emitted unconditionally, and the PTX emitter branches to the last
      arm unconditionally. A guarded arm has to be able to FALL THROUGH to the
      next one, which that shape cannot express.
    - Exhaustiveness ([Sarek_ir_ptx_expr.check_match_exhaustive]) reasons on
      constructor coverage alone. A guarded arm covers its constructor
      syntactically but not semantically, so once guards exist a match can fail
      at run time — and the device backends have no trap to fail INTO (the
      interpreter's [Pattern_match_failure] has no GPU counterpart).

    On the first bullet's blast radius: it is not a closed list, and this
    comment does not try to close it. [git grep -l "EMatch\|SMatch"] reports 37
    non-test source files, spread over [sarek/ppx], [sarek/codegen],
    [sarek/interp], [sarek/transpile], [sarek/sarek] and [spoc/ir], plus the
    negative and golden tests and the [formal/type-safety] Rocq models with
    their extracted [PatternModel]/[ConstrModel]. Among them: at least the six
    device emitters ([Sarek_ir_cuda], [Sarek_ir_opencl], [Sarek_ir_metal],
    [Sarek_ir_ptx] with its [_expr]/[_stmt] halves, [Sarek_ir_glsl],
    [Sarek_ir_wgsl]) and roughly a dozen IR passes and traversals (tag erasure,
    vector inlining, softmath, monomorphisation, defunctionalisation, the three
    tailrec passes, convergence analysis, fusion, [Sarek_lower_ir],
    [Sarek_ir_conv], and [spoc/ir]'s analysis/codegen/pp), alongside the native
    and interpreter evaluators. Re-run the grep for the current set rather than
    trusting an enumeration: an earlier round of this comment stated it as a
    closed "every consumer" list that named four emitters and omitted, among
    others, [Sarek_ir_glsl] and [Sarek_ir_wgsl].

    So this is "not yet supported", not "cannot be supported": the guarded-arm
    subset that is always followed by an unguarded arm for the same constructor
    is implementable. It is a feature with an IR change in it, not a fix, and
    until it is built a refusal is the only honest answer. *)
let when_guard_msg =
  "`when` guards on match cases are not supported in kernels (not yet \
   implemented). Sarek's match IR has no guard slot, so this guard would be \
   discarded and the arm would match unconditionally — the kernel would \
   compile and run, computing something other than what this source says. \
   Rewrite the guard as an `if` inside the arm body: `| C x when cond -> a` \
   becomes `| C x -> if cond then a else b`, where `b` is what the arm below \
   would have computed for the same constructor."

(** [{ r with f = e }] — functional record update.

    [Pexp_record]'s second component is the [with] base. It was read as [_base]
    and never used, so [{ r with x = 1.0 }] parsed to exactly the same [ERecord]
    as [{ x = 1.0 }] and the base was gone (backlog-192).

    It was NOT silent, and an earlier revision of this comment and of
    record_update_msg said it was. Measured on this tree by removing this
    refusal and building sarek/tests/negative/test_record_update.ml: the kernel
    fails to compile with OCaml's own

    Error: Some record fields are undefined: y

    pinned to the user's [{ p with x = 1.0 }] expression. The mechanism is that
    the PPX re-emits the record literal into the generated native fallback with
    the original location, so OCaml sees a literal missing a field and rejects
    it. Any [with] that actually omits a field omits it there too, so this is
    not a property of that one case — but "silently wrong device code", the
    [when] guard's failure mode, is not what this was. What it was is a
    diagnostic that names a missing field and never mentions that the [with] was
    discarded, which is why the refusal is still worth having.

    The refusal is also not "cannot be supported": there is no
    copy-then-overwrite form in the record lowering, and adding one is a
    feature. *)
let record_update_msg =
  "functional record update (`{ r with f = e }`) is not supported in kernels \
   (not yet implemented). Sarek's record literal is lowered to a struct \
   initialiser with no copy-then-overwrite form, so the base record is dropped \
   and only the fields you name are set — which the compiler then reports as \
   an undefined field somewhere else, without saying that the `with` was the \
   cause. Name every field, reading the ones you are not changing from the \
   base: `{ f = e; g = r.g }`."

(** A record field named through a functor application. The field longident used
    to fall back to the literal name ["field"] for anything that was not
    [Lident]/[Ldot] — a field name the record almost certainly does not have, so
    the drop surfaced (if at all) as an unrelated unknown-field error.

    UNREACHABLE from OCaml source: the grammar's field position is
    [mk_longident(mod_longident, LIDENT)] and [mod_longident] has no
    functor-application production, so [r.F(X).f] is a syntax error — checked
    with [ocamlc -stop-after parsing] on [let f r = r.F(X).lbl], which reports
    "Syntax error". The arm exists so that no [Longident] shape can reach a
    fabricated field name if that ever changes. *)
let record_field_path_msg =
  "this record field cannot be named through a functor application (`F(X).f`) \
   in a kernel: fields are resolved by NAME against the record type. Write the \
   bare field name."

(** A labelled or optional argument at a CALL SITE. [Pexp_apply]'s argument list
    is [(arg_label * expression) list] and the label was read as [_], so
    [f ~b:2 ~a:1] was lowered as the positional [f 2 1] — silently passing the
    arguments in written order rather than in declared order. A labelled
    PARAMETER is already refused by [collect_fun_params], so no kernel-visible
    function can legitimately be called this way. *)
let labelled_argument_msg =
  "a labelled or optional argument (`f ~x:e`, `f ?x:e`) is not supported in a \
   kernel call: arguments are passed positionally, so the label would be \
   dropped and the arguments would be passed in the order they are WRITTEN, \
   not the order they are declared. Kernel functions cannot declare labelled \
   parameters either. Pass the arguments positionally."

(** A functor application in the module path of [let open ... in].

    UNREACHABLE from OCaml source: [let open F(X) in e] parses to [Pexp_open]
    over a [Pmod_apply], not over a [Pmod_ident] carrying a [Lapply] longident
    (checked with [ocamlc -stop-after parsing]), and [Pmod_apply] is refused by
    [parse_expression]'s final arm through [Sarek_unsupported]. The arm is kept
    so the [Longident] match stays total: the reachable defect it replaces was
    the [| _ -> []] beside it, which mapped every path DEEPER than [M.N] to the
    empty path — and an empty path is not inert, [Sarek_native_gen_expr]'s
    [TEOpen] arm reaches [failwith "empty module path in TEOpen"]. Measured on
    this tree before the fix: a kernel containing [let open A.B.C in] failed
    with "Sarek internal error: Failure(\"empty module path in TEOpen\")". *)
let functor_open_msg =
  "a functor application in `let open ... in` (`let open F(X) in e`) is not \
   supported in a kernel: the open is re-emitted as a module path in generated \
   native code, and a functor application is not a path. Alias it outside the \
   kernel (`module M = F(X)`) and write `let open M in`."

(** A return-type annotation on the kernel function itself. There is no slot for
    it in [Sarek_ast.kernel] — [kern_params] and [kern_body] are the whole
    signature — so it was read by nobody. *)
let kernel_return_type_msg =
  "a return-type annotation on a kernel (`[%kernel fun (a : int32 vector) : \
   unit -> ...]`) is not supported: a kernel returns nothing to the host, so \
   there is no return type to check the annotation against and it would be \
   discarded. Remove it; annotate the parameters instead."

(** Both payload readers used to handle an unannotated module constant
    inconsistently: the [let module] fold DROPPED the binding (so a kernel using
    it failed with an unbound variable pointing at the USE), while the top-level
    fold refused it with a message that was false for the [let x : t = e]
    spelling, whose annotation neither of them looked at. *)
let module_const_needs_type_msg =
  "a constant declared for a kernel needs a type annotation (`let x : float32 \
   = 1.0`): Sarek has no expression-level inference for a module constant, and \
   without a type there is nothing to generate a device declaration from. \
   Annotate it, or move it inside the kernel body where inference runs."

(** Apply a declared result type to a function body as a constraint.

    [Sarek_ast.MFun] has no type field, so a module-item helper's declared
    result type had nowhere to go and was dropped. [ETyped] carries it without
    changing any representation: [Sarek_typer]'s [ETyped] arm unifies and
    returns the INNER typed expression with the substituted type, so no node
    reaches the IR and no lowering pass sees anything new.

    HOW MUCH THIS CONSTRAINS, exactly. For a MONOMORPHIC annotation, the
    declared type. For one containing a type VARIABLE it is weaker than the
    source says, and that is a property of this route rather than of the
    annotation: [Sarek_typer]'s [ETyped] arm converts with
    [type_of_type_expr_env], which allocates a FRESH type-variable context,
    while an [MFun]'s parameters are converted in the context built for the
    item. So in [let f (x : 'a) : 'a = ...] the two ['a]s become different
    variables, and the annotation constrains the SHAPE of the result rather than
    its identity with the parameter. The [ELetRec] route (a kernel-BODY helper)
    does not share the limitation: it converts the result type in the same
    [tvar_ctx] as the parameters.

    Not refused, because sarek/tests/e2e/test_module_poly.ml legitimately writes
    [let[@sarek.module] identity (x : 'a) : 'a = x]. Recorded here and in
    kb/sarek/ppx/parser.md instead. Found by the cross-runtime review. *)
let constrain_body (ty : Sarek_ast.type_expr option) (body : Sarek_ast.expr) :
    Sarek_ast.expr =
  match ty with
  | None -> body
  | Some t ->
      {
        Sarek_ast.e = Sarek_ast.ETyped (body, t);
        Sarek_ast.expr_loc = body.Sarek_ast.expr_loc;
      }

(** The declared RESULT type of a [let] binding with [nparams] parameters.

    Three spellings reach here and only one of them had a reader before
    backlog-192: [let (x : t) = e] (in the pattern, read), [let x : t = e] (in
    [pvb_constraint] — dropped), and [let f x : t = e] (in [Pexp_function]'s
    constraint slot — dropped). The second is the spelling nearly everything in
    this tree uses.

    A whole-binding annotation on a FUNCTION is refused rather than peeled. An
    earlier revision of this function peeled one arrow per parameter and used
    the result — which silently discarded every DOMAIN the user had written:
    [let (f : float32 -> int32) = fun (x : int32) -> x] was accepted as an
    [int32 -> int32] helper, the declared [float32] read by nobody. That is the
    same defect class the sweep exists to close, introduced by the sweep's own
    first draft (found by the cross-runtime review). Refusing costs nothing,
    because a kernel function's parameters must ALREADY carry their own
    annotations — [extract_param_from_pattern] raises "Kernel parameters must
    have type annotations" otherwise — so the domain half of a whole-binding
    arrow is always redundant with them, and never checked against them. *)
let binding_result_type (vb : value_binding) (nparams : int) :
    Sarek_ast.type_expr option =
  match (binding_type vb, nparams) with
  | None, 0 -> None
  | None, _ -> fun_return_type vb.pvb_expr
  | Some t, 0 -> Some t
  | Some _, _ ->
      raise
        (Parse_error_exn
           ( "a whole-binding type annotation on a function (`let (f : a -> b) \
              = fun x -> ...`, `let f : a -> b = fun x -> ...`) is not \
              supported in a kernel: the parameters carry their own \
              annotations, which a kernel requires anyway, and nothing checks \
              the domain half of this one against them — it would be \
              discarded. Annotate the result instead: `let f (x : a) : b = \
              ...`.",
             vb.pvb_pat.ppat_loc ))

(** Parse let%shared: let%shared name : type [= size] in body Syntax: let%shared
    tile : float32 array in body let%shared tile : float32 array = 64 in body *)
let rec parse_let_shared parse_expr (expr : expression) : Sarek_ast.expr_desc =
  match expr.pexp_desc with
  (* Pattern: let name : type = size in body
     Note: when size is (), we treat it as no size specified *)
  | Pexp_let
      ( Nonrecursive,
        [
          {
            pvb_pat = {ppat_desc = Ppat_constraint (name_pat, elem_type); _};
            pvb_expr = size_expr;
            _;
          };
        ],
        body_expr ) ->
      let name =
        match name_pat.ppat_desc with
        | Ppat_var {txt; _} -> txt
        | _ ->
            raise
              (Parse_error_exn ("Expected variable name", name_pat.ppat_loc))
      in
      let elem_ty = parse_type elem_type in
      (* Check if size is unit - if so, no size specified *)
      let size =
        match size_expr.pexp_desc with
        | Pexp_construct ({txt = Lident "()"; _}, None) -> None
        | _ -> Some (parse_expr size_expr)
      in
      let body = parse_expr body_expr in
      Sarek_ast.ELetShared (name, elem_ty, size, body)
  (* Shorthand: name : type in body (no explicit let) *)
  | Pexp_constraint
      ({pexp_desc = Pexp_sequence (name_expr, body_expr); _}, elem_type) -> (
      match name_expr.pexp_desc with
      | Pexp_ident {txt = Lident name; _} ->
          let elem_ty = parse_type elem_type in
          let body = parse_expr body_expr in
          Sarek_ast.ELetShared (name, elem_ty, None, body)
      | _ ->
          raise
            (Parse_error_exn
               ("Expected identifier for shared array name", expr.pexp_loc)))
  | _ ->
      raise
        (Parse_error_exn
           ("Expected 'let%shared name : type [= size] in body'", expr.pexp_loc))

(** Parse let%superstep: let%superstep [~divergent] name = body in cont Syntax:
    let%superstep load = tile.(i) <- v in cont let%superstep ~divergent final =
    ... in cont *)
and parse_superstep parse_expr (expr : expression) : Sarek_ast.expr_desc =
  match expr.pexp_desc with
  (* Pattern: let name = body in cont *)
  | Pexp_let
      ( Nonrecursive,
        [{pvb_pat; pvb_expr = step_body; pvb_attributes; _}],
        cont_expr ) ->
      let name =
        match extract_name_from_pattern pvb_pat with
        | Some n -> n
        | None ->
            raise
              (Parse_error_exn ("Expected superstep name", pvb_pat.ppat_loc))
      in
      (* Check for the divergent attribute. Any OTHER attribute used to be
         skipped in silence, so a misspelling ([@divergnt]) read as "not
         divergent" and the convergence checker was applied to a step the user
         had declared divergent (backlog-192). *)
      let divergent =
        List.exists
          (fun (attr : attribute) -> attr.attr_name.txt = "divergent")
          pvb_attributes
      in
      List.iter
        (fun (attr : attribute) ->
          if attr.attr_name.txt <> "divergent" then
            raise
              (Parse_error_exn
                 ( Printf.sprintf
                     "the attribute `%s` on a superstep binding is not \
                      interpreted: `divergent` is the only one a superstep \
                      reads, and an attribute nothing reads is silently \
                      discarded. Remove it, or write `divergent` if that is \
                      what you meant."
                     attr.attr_name.txt,
                   attr.attr_loc )))
        pvb_attributes ;
      let body = parse_expr step_body in
      let cont = parse_expr cont_expr in
      Sarek_ast.ESuperstep (name, divergent, body, cont)
  | _ ->
      raise
        (Parse_error_exn
           ("Expected 'let%superstep name = body in cont'", expr.pexp_loc))

(** Parse an expression *)
and parse_expression (expr : expression) : Sarek_ast.expr =
  let loc = loc_of_ppxlib expr.pexp_loc in
  let e =
    match expr.pexp_desc with
    (* Unit *)
    | Pexp_construct ({txt = Lident "()"; _}, None) -> Sarek_ast.EUnit
    (* Boolean literals *)
    | Pexp_construct ({txt = Lident "true"; _}, None) -> Sarek_ast.EBool true
    | Pexp_construct ({txt = Lident "false"; _}, None) -> Sarek_ast.EBool false
    (* Integer literals *)
    | Pexp_constant (Pconst_integer (s, Some 'l')) ->
        Sarek_ast.EInt32 (Int32.of_string s)
    | Pexp_constant (Pconst_integer (s, Some 'L')) ->
        Sarek_ast.EInt64 (Int64.of_string s)
    | Pexp_constant (Pconst_integer (s, None)) ->
        Sarek_ast.EInt (int_of_string s)
    (* Float literals. The OCaml float-literal suffix selects the width:
       'g'/'G' -> EDouble (float64), bare/any other suffix -> EFloat (float32).
       Keeping bare literals at float32 preserves the GPU-default width and full
       backward compatibility; [1.0G] is the explicit way to write an f64 literal. *)
    | Pexp_constant (Pconst_float (s, (Some 'g' | Some 'G'))) ->
        Sarek_ast.EDouble (float_of_string s)
    | Pexp_constant (Pconst_float (s, _)) ->
        Sarek_ast.EFloat (float_of_string s)
    (* Variables *)
    | Pexp_ident {txt = Lident name; _} -> Sarek_ast.EVar name
    (* Module-qualified identifiers: Module.name -> "Module.name"
       This preserves the qualified name for cross-module function lookup.
       The typer will look up "Module.name" in the environment/registry. *)
    | Pexp_ident {txt = Ldot (Lident modname, name); _} ->
        Sarek_ast.EVar (modname ^ "." ^ name)
    (* Vector/array access: e.(i) or e.[i] *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident "Array.get"; _}; _},
          [(Nolabel, arr); (Nolabel, idx)] ) ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Ldot (Lident "Array", "get"); _}; _},
          [(Nolabel, arr); (Nolabel, idx)] ) ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    (* Vector/array set: e.(i) <- x *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Ldot (Lident "Array", "set"); _}; _},
          [(Nolabel, arr); (Nolabel, idx); (Nolabel, value)] ) ->
        Sarek_ast.EArrSet
          (parse_expression arr, parse_expression idx, parse_expression value)
    (* Custom indexing: v.%[i] -> EArrGet *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident ".%[]"; _}; _},
          [(Nolabel, arr); (Nolabel, idx)] ) ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    (* Custom indexing: v.%[i] <- x -> EArrSet *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident ".%[]<-"; _}; _},
          [(Nolabel, arr); (Nolabel, idx); (Nolabel, value)] ) ->
        Sarek_ast.EArrSet
          (parse_expression arr, parse_expression idx, parse_expression value)
    (* Mutable assignment: x := v *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident ":="; _}; _},
          [(Nolabel, lhs); (Nolabel, rhs)] ) ->
        parse_assign_form lhs rhs
    (* `a.(i)` needs no arm of its own: OCaml desugars it to `Array.get a i`,
       which the two Array.get arms above match. The arm that used to sit here
       was guarded by [is_array_access], a function whose body was the literal
       [false] with a comment calling itself "a simplified check" — so it never
       fired, and it advertised a syntax route that did not exist. Removed with
       the function (backlog-192). *)
    (* Pragma - pragma ["opt1"; "opt2"] body - must come before binary ops *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident "pragma"; _}; _},
          [(Nolabel, opts_expr); (Nolabel, body)] ) ->
        parse_pragma_form opts_expr body
    (* create_array size memspace - special form for local/shared arrays
       Must come before binary operators since it has 2 arguments *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident "create_array"; _}; _},
          [(Nolabel, size_expr); (Nolabel, mem_expr)] ) ->
        parse_create_array_form size_expr mem_expr
    (* Binary operators - exclude create_array which is handled above *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident op; _}; _},
          [(Nolabel, e1); (Nolabel, e2)] )
      when op <> "create_array" ->
        parse_binop_or_app_form expr op e1 e2
    (* Unary operators *)
    | Pexp_apply
        ({pexp_desc = Pexp_ident {txt = Lident op; _}; _}, [(Nolabel, e)])
      when parse_unop op <> None -> (
        match parse_unop op with
        | Some unop -> Sarek_ast.EUnop (unop, parse_expression e)
        | None ->
            (* Should be unreachable due to when guard, but handle gracefully *)
            raise
              (Parse_error_exn
                 ( "Internal error: unary operator check inconsistency",
                   expr.pexp_loc )))
    (* Function application *)
    | Pexp_apply (fn, args) ->
        let fn_expr = parse_expression fn in
        let arg_exprs =
          List.map
            (fun (label, e) ->
              (* The label used to be read as [_] and dropped, turning [f ~b:2
                 ~a:1] into the positional [f 2 1] (backlog-192). *)
              (match label with
              | Nolabel -> ()
              | Labelled _ | Optional _ ->
                  raise (Parse_error_exn (labelled_argument_msg, e.pexp_loc))) ;
              parse_expression e)
            args
        in
        Sarek_ast.EApp (fn_expr, arg_exprs)
    (* Let binding *)
    | Pexp_let (Nonrecursive, [vb], body) -> parse_let_form vb body
    (* If-then-else *)
    | Pexp_ifthenelse (cond, then_e, else_opt) ->
        Sarek_ast.EIf
          ( parse_expression cond,
            parse_expression then_e,
            Option.map parse_expression else_opt )
    (* For loop *)
    | Pexp_for ({ppat_desc = Ppat_var {txt = var; _}; _}, lo, hi, dir, body) ->
        let d =
          match dir with Upto -> Sarek_ast.Upto | Downto -> Sarek_ast.Downto
        in
        Sarek_ast.EFor
          ( var,
            parse_expression lo,
            parse_expression hi,
            d,
            parse_expression body )
    (* While loop *)
    | Pexp_while (cond, body) ->
        Sarek_ast.EWhile (parse_expression cond, parse_expression body)
    (* Sequence *)
    | Pexp_sequence (e1, e2) ->
        Sarek_ast.ESeq (parse_expression e1, parse_expression e2)
    (* Match *)
    | Pexp_match (scrutinee, cases) ->
        let parsed_cases =
          List.map
            (fun case ->
              (* A [when] guard has no representation anywhere downstream:
                 [Sarek_ir_ppx]'s [EMatch]/[SMatch] carry [(pattern * _) list]
                 with no guard slot, so the guard used to be dropped HERE and
                 the arm became unconditional. Refuse instead of silently
                 changing the program's meaning (backlog-191). *)
              (match case.pc_guard with
              | Some guard ->
                  raise (Parse_error_exn (when_guard_msg, guard.pexp_loc))
              | None -> ()) ;
              let pat = parse_pattern case.pc_lhs in
              let body = parse_expression case.pc_rhs in
              (pat, body))
            cases
        in
        Sarek_ast.EMatch (parse_expression scrutinee, parsed_cases)
    (* Record construction *)
    | Pexp_record (fields, base) ->
        (* The [with] base used to be read as [_base] and dropped, so
           [{ r with x = e }] became [{ x = e }] and every unnamed field was left
           uninitialised on the device (backlog-192). *)
        (match base with
        | Some b -> raise (Parse_error_exn (record_update_msg, b.pexp_loc))
        | None -> ()) ;
        let parsed_fields =
          List.map
            (fun ({txt; loc = fld_loc}, e) ->
              let name =
                match txt with
                | Lident n -> n
                | Ldot (_, n) -> n
                | Lapply _ ->
                    raise (Parse_error_exn (record_field_path_msg, fld_loc))
              in
              (name, parse_expression e))
            fields
        in
        Sarek_ast.ERecord (None, parsed_fields)
    (* Field access *)
    | Pexp_field (record, {txt = Lident field; _}) ->
        Sarek_ast.EFieldGet (parse_expression record, field)
    (* Field set (via setfield) *)
    | Pexp_setfield (record, {txt = Lident field; _}, value) ->
        Sarek_ast.EFieldSet
          (parse_expression record, field, parse_expression value)
    (* Constructor application *)
    | Pexp_construct ({txt = Lident name; _}, arg_opt) ->
        Sarek_ast.EConstr (name, Option.map parse_expression arg_opt)
    (* Tuple *)
    | Pexp_tuple es -> Sarek_ast.ETuple (List.map parse_expression es)
    (* Type annotation *)
    | Pexp_constraint (e, ty) ->
        Sarek_ast.ETyped (parse_expression e, parse_type ty)
    (* Open expression *)
    | Pexp_open
        ({popen_expr = {pmod_desc = Pmod_ident {txt; loc = mod_loc}; _}; _}, e)
      ->
        (* Depths beyond [M.N] used to flatten to the EMPTY path, which
           Sarek_native_gen_expr turns into `failwith "empty module path in
           TEOpen"` rather than a diagnostic, and which Sarek_env.open_module
           treats as a silent no-op (backlog-192). Flatten at any depth; refuse
           only what is not a path. [popen_override] is deliberately not read:
           `open!` differs from `open` only in whether OCaml warns about
           shadowing, which is settled before the kernel is parsed. *)
        let rec flatten = function
          | Lident n -> [n]
          | Ldot (li, n) -> flatten li @ [n]
          | Lapply _ -> raise (Parse_error_exn (functor_open_msg, mod_loc))
        in
        Sarek_ast.EOpen (flatten txt, parse_expression e)
    (* Lambda - for local functions in kernels *)
    | _ when is_function_expression expr ->
        raise
          (Parse_error_exn
             ( "Standalone lambda expressions are not supported in kernels; \
                use let-bound functions",
               expr.pexp_loc ))
    (* Extension point: [%global name] - reference to OCaml value *)
    | Pexp_extension
        ( {txt = "global"; _},
          PStr
            [
              {
                pstr_desc =
                  Pstr_eval
                    ({pexp_desc = Pexp_ident {txt = Lident name; _}; _}, _);
                _;
              };
            ] ) ->
        Sarek_ast.EGlobalRef name
    (* Extension point: [%native gpu_fun, ocaml_expr]
       Inline device code with OCaml fallback for interpreter/native runtimes.
       gpu_fun: (fun dev -> "cuda/opencl code")
       ocaml_expr: OCaml expression to execute on interpreter/native *)
    | Pexp_extension
        ({txt = "native"; _}, PStr [{pstr_desc = Pstr_eval (inner_expr, _); _}])
      -> (
        (* Parse tuple (gpu_fun, ocaml_expr) *)
        match inner_expr.pexp_desc with
        | Pexp_tuple [gpu; ocaml] -> Sarek_ast.ENative {gpu; ocaml}
        | _ ->
            raise
              (Parse_error_exn
                 ( "[%native] requires a tuple: (fun dev -> ..., ocaml_fallback)",
                   expr.pexp_loc )))
    (* Extension: let%shared name : type [= size] in body *)
    | Pexp_extension
        ({txt = "shared"; _}, PStr [{pstr_desc = Pstr_eval (inner_expr, _); _}])
      ->
        parse_let_shared parse_expression inner_expr
    (* Extension: let%superstep [~divergent] name = body in cont *)
    | Pexp_extension
        ( {txt = "superstep"; _},
          PStr [{pstr_desc = Pstr_eval (inner_expr, _); _}] ) ->
        parse_superstep parse_expression inner_expr
    (* The refusal names the construct. It used to say "Unsupported expression"
       for every one of the ~20 expression forms that reach here, which told the
       user nothing about which part of their kernel was the problem
       (backlog-192). Sarek_unsupported's table has no wildcard arm, so a new
       ppxlib constructor stops the build instead of reaching a user as
       silence. *)
    | d ->
        raise
          (Parse_error_exn
             (Sarek_unsupported.expression_refusal d, expr.pexp_loc))
  in
  {Sarek_ast.e; Sarek_ast.expr_loc = loc}

(** Extract body for mutable assignment: x := v *)
and parse_assign_form (lhs : expression) (rhs : expression) :
    Sarek_ast.expr_desc =
  match lhs.pexp_desc with
  | Pexp_ident {txt = Lident name; _} ->
      Sarek_ast.EAssign (name, parse_expression rhs)
  | _ ->
      raise
        (Parse_error_exn
           ("Expected variable on left-hand side of :=", lhs.pexp_loc))

(** Extract body for pragma form: pragma ["opt1"; "opt2"] body *)
and parse_pragma_form (opts_expr : expression) (body : expression) :
    Sarek_ast.expr_desc =
  let rec collect_strings acc expr =
    match expr.pexp_desc with
    | Pexp_construct ({txt = Lident "[]"; _}, None) -> List.rev acc
    | Pexp_construct
        ({txt = Lident "::"; _}, Some {pexp_desc = Pexp_tuple [hd; tl]; _}) -> (
        match hd.pexp_desc with
        | Pexp_constant (Pconst_string (s, _, _)) ->
            collect_strings (s :: acc) tl
        | _ ->
            raise
              (Parse_error_exn ("pragma options must be strings", hd.pexp_loc)))
    | _ ->
        raise
          (Parse_error_exn
             ("pragma expects a list of strings", opts_expr.pexp_loc))
  in
  let opts = collect_strings [] opts_expr in
  Sarek_ast.EPragma (opts, parse_expression body)

(** Extract body for create_array form: create_array size memspace *)
and parse_create_array_form (size_expr : expression) (mem_expr : expression) :
    Sarek_ast.expr_desc =
  let size = parse_expression size_expr in
  let mem =
    match mem_expr.pexp_desc with
    | Pexp_construct ({txt = Lident "Local"; _}, None) -> Sarek_ast.Local
    | Pexp_construct ({txt = Lident "Shared"; _}, None) -> Sarek_ast.Shared
    | Pexp_construct ({txt = Lident "Global"; _}, None) -> Sarek_ast.Global
    | _ ->
        raise
          (Parse_error_exn
             ( "create_array expects Local, Shared, or Global as memspace",
               mem_expr.pexp_loc ))
  in
  (* Type comes from let binding annotation, use type variable for inference *)
  Sarek_ast.ECreateArray (size, Sarek_ast.TEVar "_infer", mem)

(** Extract body for binary operator or function application arm *)
and parse_binop_or_app_form (expr : expression) (op : string) (e1 : expression)
    (e2 : expression) : Sarek_ast.expr_desc =
  match parse_binop op with
  | Some binop ->
      Sarek_ast.EBinop (binop, parse_expression e1, parse_expression e2)
  | None ->
      (* Regular function application with infix *)
      Sarek_ast.EApp
        ( parse_expression
            {
              expr with
              pexp_desc = Pexp_ident {txt = Lident op; loc = expr.pexp_loc};
            },
          [parse_expression e1; parse_expression e2] )

(** Extract body for let binding arm *)
and parse_let_form (vb : value_binding) (body : expression) :
    Sarek_ast.expr_desc =
  let pvb_pat = vb.pvb_pat and pvb_expr = vb.pvb_expr in
  (* A tuple-pattern let ([let (a, b) = e in body]) is not a named binding;
     desugar it to the single-arm tuple [match] the lowering already handles as
     a positional-record destructure ([let ..], line ~800 of
     Sarek_lower_ir.ml). This is the [let]-pattern half of local tuple support;
     [match] was already covered directly. Constraints on the pattern (e.g.
     [let ((a, b) : t) = e]) unwrap to the inner tuple. *)
  let rec is_tuple_pattern (p : pattern) =
    match p.ppat_desc with
    | Ppat_tuple _ -> true
    | Ppat_constraint (p, _) -> is_tuple_pattern p
    | _ -> false
  in
  if is_tuple_pattern pvb_pat then begin
    (* A tuple-pattern binding has nowhere to put a type: [EMatch] carries none.
       Reading the annotation and discarding it is the defect this sweep is
       about, so it is refused instead. *)
    (match binding_type vb with
    | Some _ ->
        raise
          (Parse_error_exn
             ( "a type annotation on a tuple-pattern `let` (`let (a, b) : t = \
                e`) is not supported in a kernel: the binding is lowered to a \
                single-arm `match`, which carries no type, so the annotation \
                would be discarded. Annotate the bound expression instead \
                (`let (a, b) = (e : t)`).",
               pvb_pat.ppat_loc ))
    | None -> ()) ;
    Sarek_ast.EMatch
      ( parse_expression pvb_expr,
        [(parse_pattern pvb_pat, parse_expression body)] )
  end
  else parse_named_let_form vb body

and parse_named_let_form (vb : value_binding) (body : expression) :
    Sarek_ast.expr_desc =
  let pvb_pat = vb.pvb_pat and pvb_expr = vb.pvb_expr in
  let name =
    match extract_name_from_pattern pvb_pat with
    | Some n -> n
    | None ->
        raise (Parse_error_exn ("Expected variable pattern", pvb_pat.ppat_loc))
  in
  (* Detect function definitions and emit ELetRec for local functions *)
  let fun_params, fun_body = collect_fun_params pvb_expr in
  (* [ELetRec]'s type slot is the RESULT type (Sarek_typer's [ret_ty_opt]), and
     [binding_result_type] is what makes all three annotation spellings reach it.
     The old code read only the pattern one and put it in UNPEELED, so the one
     spelling that did reach the slot ([let (f : a -> b) = fun x -> ...]) unified
     an arrow against the body's type. *)
  let ty = binding_result_type vb (List.length fun_params) in
  if fun_params <> [] then
    match fun_body with
    | Some (Fun_body body_expr) ->
        let parsed_params =
          List.map
            (fun p -> extract_param_from_pattern (pattern_of_param p))
            fun_params
        in
        let fn_body = parse_expression body_expr in
        Sarek_ast.ELetRec
          (name, parsed_params, ty, fn_body, parse_expression body)
    | Some (Fun_cases _) ->
        raise
          (Parse_error_exn
             ( "Pattern-matching functions not supported in let bindings",
               pvb_expr.pexp_loc ))
    | None ->
        raise (Parse_error_exn ("Expected function body", pvb_expr.pexp_loc))
  else
    let mut_expr =
      match pvb_expr.pexp_desc with
      | Pexp_apply
          ( {pexp_desc = Pexp_ident {txt = Lident "mut"; _}; _},
            [(Nolabel, inner)] ) ->
          Some inner
      | _ -> None
    in
    let is_mutable = Option.is_some mut_expr in
    let value_expr =
      match mut_expr with Some inner -> inner | None -> pvb_expr
    in
    if is_mutable then
      Sarek_ast.ELetMut
        (name, ty, parse_expression value_expr, parse_expression body)
    else
      Sarek_ast.ELet
        (name, ty, parse_expression value_expr, parse_expression body)

(** Every field of a kernel-payload [type_declaration] that [parse_payload] does
    not read, refused rather than dropped.

    A payload type declaration is consumed by the PPX and never reaches OCaml,
    so a field nobody reads here is a field nobody reads at all. The two
    attributes Sarek itself declares are accepted and ignored on purpose: inside
    a payload every type declaration is already a Sarek type, so [@@sarek.type]
    there is redundant rather than dropped. *)
let check_payload_type_decl (td : type_declaration) : unit =
  let loc = td.ptype_loc in
  if td.ptype_params <> [] then
    raise
      (Parse_error_exn
         ( "a parameterised type (`type 'a t = ...`) is not supported in a \
            kernel: a device struct has one fixed layout, and a type parameter \
            would give it one per instantiation. Declare the type at each \
            element type you need.",
           loc )) ;
  if td.ptype_cstrs <> [] then
    raise
      (Parse_error_exn
         ( "a type constraint (`constraint 'a = t`) is not supported in a \
            kernel type declaration: it constrains a type parameter, and \
            kernel types have none.",
           loc )) ;
  (match td.ptype_private with
  | Private ->
      raise
        (Parse_error_exn
           ( "a private type (`type t = private ...`) is not supported in a \
              kernel: the generated device code and the generated accessors \
              read the representation directly, so privacy could not be \
              enforced and would be silently ignored.",
             loc ))
  | Public -> ()) ;
  (match (td.ptype_kind, td.ptype_manifest) with
  | Ptype_abstract, Some _ ->
      raise
        (Parse_error_exn
           ( "a type alias (`type t = u`) is not supported in a kernel: types \
              are resolved by name and no alias is followed, so uses of `t` \
              would not find `u`. Use `u` directly.",
             loc ))
  | Ptype_abstract, None ->
      raise
        (Parse_error_exn
           ( "an abstract type (`type t`) is not supported in a kernel: it has \
              no representation to generate a device struct from.",
             loc ))
  | Ptype_open, _ ->
      raise
        (Parse_error_exn
           ( "an extensible variant (`type t = ..`) is not supported in a \
              kernel: a variant's device representation is fixed by its \
              declaration order, which a later extension would change.",
             loc ))
  | (Ptype_record _ | Ptype_variant _), Some _ ->
      raise
        (Parse_error_exn
           ( "a type declaration with both a manifest and a representation \
              (`type t = u = { ... }`) is not supported in a kernel: only the \
              representation is read, so the manifest would be discarded.",
             loc ))
  | (Ptype_record _ | Ptype_variant _), None -> ()) ;
  List.iter
    (fun (attr : attribute) ->
      match attr.attr_name.txt with
      | "sarek.type" | "sarek.type_private" -> ()
      | other ->
          raise
            (Parse_error_exn
               ( Printf.sprintf
                   "the attribute `%s` on a type declaration inside a kernel \
                    is not interpreted by anything: a kernel payload is \
                    consumed by the PPX and never reaches OCaml, so the \
                    attribute would be silently discarded. Remove it, or \
                    declare the type outside the kernel where OCaml can see \
                    it."
                   other,
                 attr.attr_loc )))
    td.ptype_attributes

(** Parse a function expression into a kernel *)
let parse_kernel_function (expr : expression) : Sarek_ast.kernel =
  let loc = loc_of_ppxlib expr.pexp_loc in
  let params, body = collect_fun_params expr in
  (* [Sarek_ast.kernel] has no return-type field, so a return annotation on the
     kernel function had nowhere to go and was dropped (backlog-192). *)
  (match fun_return_type expr with
  | Some _ -> raise (Parse_error_exn (kernel_return_type_msg, expr.pexp_loc))
  | None -> ()) ;
  match body with
  | Some (Fun_cases _) ->
      raise
        (Parse_error_exn
           ("Pattern-matching functions not supported as kernels", expr.pexp_loc))
  | Some (Fun_body body_expr) ->
      if params = [] then
        raise
          (Parse_error_exn
             ("Kernel must have at least one parameter", expr.pexp_loc)) ;
      let parsed_params =
        List.map
          (fun p -> extract_param_from_pattern (pattern_of_param p))
          params
      in
      let body = parse_expression body_expr in
      {
        Sarek_ast.kern_name = None;
        kern_types = [];
        kern_module_items = [];
        kern_external_item_count = 0;
        kern_params = parsed_params;
        kern_body = body;
        kern_loc = loc;
      }
  | None -> raise (Parse_error_exn ("Kernel must be a function", expr.pexp_loc))

(** Parse from ppxlib payload *)
let parse_payload (payload : expression) : Sarek_ast.kernel =
  let parse_module_items_from_structure items =
    List.fold_left
      (fun (types_acc, mods_acc) (item : structure_item) ->
        match item.pstr_desc with
        | Pstr_type (rec_flag, decls) ->
            (* [Pstr_type]'s rec_flag WAS dropped, and an earlier revision of
               this comment excused it by saying a kernel type cannot refer to
               another kernel type in its fields. That is false — nested records
               are supported and exercised (sarek/tests/e2e/test_nested_types.ml)
               — so `nonrec` is meaningful here and was being ignored: Sarek
               resolves a field's type by NAME, which under `type nonrec t` is
               the wrong binding. Refused rather than re-justified (found by the
               cross-runtime review). *)
            (match rec_flag with
            | Recursive -> ()
            | Nonrecursive ->
                raise
                  (Parse_error_exn
                     ( "`type nonrec` is not supported in a kernel: Sarek \
                        resolves a field's type by name against the types \
                        declared for this kernel, and has no way to mean \"the \
                        one from the enclosing scope instead\". Drop `nonrec`, \
                        or rename the type.",
                       (List.hd decls).ptype_loc ))) ;
            let tdecls =
              List.map
                (fun (td : type_declaration) ->
                  let loc = td.ptype_loc in
                  (* Fields of [type_declaration] that had no reader at all: type
                     parameters, constraints, privacy, the manifest and the
                     attributes were all dropped in silence (backlog-192). *)
                  check_payload_type_decl td ;
                  match td.ptype_kind with
                  | Ptype_record labels ->
                      Sarek_ast.Type_record
                        {
                          tdecl_name = td.ptype_name.txt;
                          tdecl_module = None;
                          tdecl_fields = parse_record_fields labels;
                          tdecl_loc = loc_of_ppxlib loc;
                        }
                  | Ptype_variant constrs ->
                      Sarek_ast.Type_variant
                        {
                          tdecl_name = td.ptype_name.txt;
                          tdecl_module = None;
                          tdecl_constructors =
                            parse_variant_constructors constrs;
                          tdecl_loc = loc_of_ppxlib loc;
                        }
                  | _ ->
                      raise
                        (Parse_error_exn
                           ( "Unsupported type declaration in kernel payload",
                             loc )))
                decls
            in
            (List.rev_append tdecls types_acc, mods_acc)
        | Pstr_value (rec_flag, vbs) ->
            let is_recursive = rec_flag = Recursive in
            let mods =
              List.fold_left
                (fun acc vb ->
                  let name =
                    match extract_name_from_pattern vb.pvb_pat with
                    | Some n -> n
                    | None ->
                        raise
                          (Parse_error_exn
                             ("Expected variable pattern", vb.pvb_pat.ppat_loc))
                  in
                  let params, body =
                    match collect_fun_params vb.pvb_expr with
                    | params, Some (Fun_body fn_body) when params <> [] ->
                        (params, fn_body)
                    | _, Some (Fun_cases _) ->
                        raise
                          (Parse_error_exn
                             ( "Pattern-matching functions not supported in \
                                module items",
                               vb.pvb_expr.pexp_loc ))
                    | _ -> ([], vb.pvb_expr)
                  in
                  let ty = binding_result_type vb (List.length params) in
                  if params <> [] then
                    let parsed_params =
                      List.map
                        (fun p ->
                          extract_param_from_pattern (pattern_of_param p))
                        params
                    in
                    let fn_body = parse_expression body in
                    Sarek_ast.MFun
                      ( name,
                        is_recursive,
                        parsed_params,
                        constrain_body ty fn_body )
                    :: acc
                  else
                    let value = parse_expression vb.pvb_expr in
                    match ty with
                    | Some t -> Sarek_ast.MConst (name, t, value) :: acc
                    | None ->
                        (* The binding used to be DROPPED here — an unannotated
                           module constant was simply not registered, so a kernel
                           referring to it failed with an unbound variable
                           pointing at the USE (backlog-192). The top-level
                           payload path already refused this; the two now
                           agree. *)
                        raise
                          (Parse_error_exn
                             (module_const_needs_type_msg, vb.pvb_pat.ppat_loc)))
                mods_acc
                vbs
            in
            (types_acc, mods)
        (* Everything else used to return the accumulator UNCHANGED, so a
           declaration written inside a kernel module was silently absent from
           the kernel environment (backlog-192). *)
        | d ->
            raise
              (Parse_error_exn
                 (Sarek_unsupported.structure_item_refusal d, item.pstr_loc)))
      ([], [])
      items
  in

  let rec collect_mods types_acc mods_acc e =
    match e.pexp_desc with
    (* [_name] is deliberately not read: [tdecl_module] stays [None] for a
       payload-local module, which is what makes its types resolvable by their
       BARE name inside this kernel ([Sarek_typer] qualifies the registered name
       when [tdecl_module] is [Some]). Writing the name in would make `M.t`
       required and `t` unresolvable, changing every payload that declares one. *)
    | Pexp_letmodule ({txt = Some _name; _}, mod_expr, body) ->
        let inner_types, inner_mods =
          match mod_expr.pmod_desc with
          | Pmod_structure items -> parse_module_items_from_structure items
          (* Every other module expression used to contribute ([], []) — a
             payload's `let module M = N in` or `let module M = F(X) in` brought
             in NOTHING, silently (backlog-192). *)
          | _ ->
              raise
                (Parse_error_exn
                   ( "only a literal structure is supported here (`let module \
                      M = struct ... end in`): a kernel payload's types and \
                      helpers are read out of that structure, and there is \
                      nothing to read out of a module path, a functor \
                      application or a functor. Write the declarations out.",
                     mod_expr.pmod_loc ))
        in
        collect_mods
          (List.rev_append inner_types types_acc)
          (List.rev_append inner_mods mods_acc)
          body
    | Pexp_letmodule ({txt = None; loc = anon_loc; _}, _, _) ->
        raise
          (Parse_error_exn
             ( "an anonymous module (`let module _ = struct ... end in`) is \
                not supported at the top of a kernel payload: it used to fall \
                through as if it were the kernel function itself, which fails \
                later with the unrelated \"Kernel must be a function\". Give \
                it a name.",
               anon_loc ))
    | Pexp_open (_, body) ->
        (* Skip past 'let open M in' and continue collecting. The opened path is
           deliberately dropped rather than turned into an EOpen: names in a
           kernel payload are resolved against Sarek's own environment, which the
           open does not populate. Inside the kernel BODY the same construct IS
           kept, because the native backend re-emits it. *)
        collect_mods types_acc mods_acc body
    | Pexp_let (rec_flag, [vb], body) ->
        (* Capture top-level let as module const/fun *)
        let pvb_pat = vb.pvb_pat and pvb_expr = vb.pvb_expr in
        let is_recursive = rec_flag = Recursive in
        let name =
          match extract_name_from_pattern pvb_pat with
          | Some n -> n
          | None ->
              raise
                (Parse_error_exn ("Expected variable pattern", pvb_pat.ppat_loc))
        in
        let module_items =
          match collect_fun_params pvb_expr with
          | params, Some (Fun_body fn_body) when params <> [] ->
              let ty = binding_result_type vb (List.length params) in
              let parsed_params =
                List.map
                  (fun p -> extract_param_from_pattern (pattern_of_param p))
                  params
              in
              let fn_body = parse_expression fn_body in
              Sarek_ast.MFun
                (name, is_recursive, parsed_params, constrain_body ty fn_body)
              :: mods_acc
          | _, Some (Fun_cases _) ->
              raise
                (Parse_error_exn
                   ( "Pattern-matching functions not supported in module items",
                     pvb_expr.pexp_loc ))
          | _ ->
              let value = parse_expression pvb_expr in
              let ty =
                match binding_result_type vb 0 with
                | Some t -> t
                | None ->
                    raise
                      (Parse_error_exn
                         (module_const_needs_type_msg, pvb_pat.ppat_loc))
              in
              Sarek_ast.MConst (name, ty, value) :: mods_acc
        in
        collect_mods types_acc module_items body
    | Pexp_let (_, (_ :: _ :: _ as vbs), _) ->
        raise
          (Parse_error_exn
             ( "simultaneous bindings (`let a = ... and b = ... in`) are not \
                supported at the top of a kernel payload: the whole `let` used \
                to fall through as if it were the kernel function — not one of \
                the bindings was read — and failed later with the unrelated \
                \"Kernel must be a function\". Write them as separate `let`s.",
               (List.nth vbs 1).pvb_loc ))
    | _ -> (List.rev types_acc, List.rev mods_acc, e)
  in
  let type_decls, module_items, core = collect_mods [] [] payload in
  let kern = parse_kernel_function core in
  {kern with kern_types = type_decls; kern_module_items = module_items}
