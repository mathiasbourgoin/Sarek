(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Refusal text for the OCaml constructs the Sarek kernel parser does not
    implement (backlog-192).

    WHY THIS MODULE EXISTS. [Sarek_parse] reads the ppxlib Parsetree and builds
    [Sarek_ast]. Where the surface language offers something the kernel subset
    does not implement, the parser used in several places to drop it and carry
    on — the [when]-guard drop fixed in backlog-191 is the measured instance:
    [pc_guard] was never read, so the guard vanished upstream of the typer, of
    every lowering pass and of backend selection, and the kernel compiled and
    computed a different function than its source said.

    A dropped construct is worse than a refused one in a specific way: the user
    gets no diagnostic at all, and the wrongness surfaces (if it surfaces) as a
    wrong ANSWER on a device, arbitrarily far from its cause. So the accepted
    subset has to be a boundary the parser states out loud, not a filter it
    applies quietly.

    WHY THE TABLES BELOW HAVE NO WILDCARD ARM. Each [*_refusal] function matches
    every arm of its Parsetree variant explicitly, so the OCaml compiler is what
    notices when ppxlib grows a constructor: the build stops until somebody
    writes down what the new arm is, instead of it reaching a user as silence. A
    [| _ ->] here would hand that back.

    What makes the build actually stop is [-warn-error +8] in [sarek/ppx/dune],
    on this library. Measured 2026-07-31 by deleting the [Pexp_try] arm below:
    dune's dev [:standard] already turns warning 8 into an error (exit 1), but
    [--profile=release] left it a plain warning and the build SUCCEEDED at exit
    0 until that flag was added; with it, release fails at exit 1 too and the
    unmutated tree builds at 0 in both profiles. [-w +8], which was the flag
    this comment used to credit, changes nothing in either profile — [:standard]
    supplies the dev error and supplies nothing in release. The guarantee is
    scoped to [sarek_frontend], the library this module is in: none of the PPX's
    other libraries carries the flag, and one test directory
    (sarek/tests/codegen_golden) already carried it before this change.

    WHAT THESE FUNCTIONS ARE NOT. They are not a claim that everything they name
    is refused THROUGH THEM. Several arms below are partially implemented — the
    parser handles a shape and falls through to its catch-all for the rest (an
    identifier is read at depth one, [M.x], and no deeper; a [for] binder is
    read when it is a variable, and not when it is [_]). For those the text
    states the subset that IS accepted, because "not supported" alone would be
    false. Which arms are in that state is recorded per-site in
    kb/sarek/ppx/parser.md, not here. *)

open Ppxlib

(** The four extension points [Sarek_parse.parse_expression] interprets inside a
    kernel body. Written without brackets on purpose: this string is embedded in
    refusal text, and a bracketed sigil spelling in message text is a claim
    about a declared construct that scripts/check-ppx-construct-names.sh checks
    against the PPX's own name table. These four are matched by hand on the AST
    rather than declared through [Extension.declare], so no declaration context
    exists to check a sigil count against. *)
let kernel_extensions =
  "the ones a kernel body accepts are `global`, `native`, `shared` (as \
   `let%shared`) and `superstep` (as `let%superstep`)"

(** Refusal text for an expression the kernel subset does not accept.

    Reached from [Sarek_parse.parse_expression]'s final arm, so it is asked
    about BOTH constructs that are wholly unimplemented and shapes of
    partially-implemented constructs that the earlier arms did not match. *)
let expression_refusal (d : expression_desc) : string =
  match d with
  | Pexp_ident _ ->
      "this identifier path is too deep for a kernel: only `x` and `M.x` are \
       resolved, and a functor application in a path (`F(X).x`) is never \
       resolved. Alias the module outside the kernel (`module M = A.B`) and \
       write `M.x`, or pass the value in as a kernel parameter."
  | Pexp_constant _ ->
      "this literal has no kernel representation. The literals a kernel body \
       accepts are `1` (int), `1l` (int32), `1L` (int64), `1.0` (float32) and \
       `1.0G` (float64). Character and string literals, and integer literals \
       with any other suffix, have no device representation at all."
  | Pexp_let _ ->
      "only a single-binding, non-recursive `let x = e in body` is supported \
       in a kernel. Write `let rec` helpers as separate bindings (a \
       kernel-local `let f x = ...` is already compiled as a recursive \
       function), and split `let a = ... and b = ... in` into nested `let`s — \
       simultaneous bindings would need a temporaries pass the lowering does \
       not have."
  | Pexp_function _ ->
      "UNREACHABLE: a `fun` expression is caught by parse_expression's `_ when \
       is_function_expression expr` arm, which refuses it with its own \
       let-bound-function advice. Note that this one is a GUARDED arm, not a \
       constructor match, so the unreachability rests on that guard holding \
       for every Pexp_function. This arm exists to keep the table \
       wildcard-free."
  | Pexp_apply _ ->
      "UNREACHABLE: an application is parsed by an arm of its own that matches \
       the constructor totally, and a sub-node the parser cannot build raises \
       THAT sub-node's refusal rather than falling through to here. This arm \
       exists to keep the table wildcard-free; seeing this text in a real \
       diagnostic means the parser lost an arm."
  | Pexp_match _ ->
      "UNREACHABLE: a `match` is parsed by an arm of its own that matches the \
       constructor totally, and a sub-node the parser cannot build raises THAT \
       sub-node's refusal rather than falling through to here. This arm exists \
       to keep the table wildcard-free; seeing this text in a real diagnostic \
       means the parser lost an arm."
  | Pexp_try _ ->
      "`try ... with` is not supported in kernels: there are no exceptions on \
       a device, and no unwinding mechanism to lower a handler onto. Return a \
       sentinel value and test it in the caller."
  | Pexp_tuple _ ->
      "UNREACHABLE: a tuple is parsed by an arm of its own that matches the \
       constructor totally, and a sub-node the parser cannot build raises THAT \
       sub-node's refusal rather than falling through to here. This arm exists \
       to keep the table wildcard-free; seeing this text in a real diagnostic \
       means the parser lost an arm."
  | Pexp_construct _ ->
      "this constructor is not usable in a kernel: only an unqualified \
       constructor of a type declared with [@@sarek.type] is resolved, so `M.C \
       x` must be written `C x` with the type in scope. `()`, `true` and \
       `false` are the only built-in constructors."
  | Pexp_variant _ ->
      "polymorphic variants (`` `Tag ``) are not supported in kernels: they \
       carry no declared tag numbering, and the device representation of a \
       variant is its declaration order. Declare a normal variant type with \
       [@@sarek.type] instead."
  | Pexp_record _ ->
      "UNREACHABLE: a record literal is parsed by an arm of its own that \
       matches the constructor totally, and a sub-node the parser cannot build \
       raises THAT sub-node's refusal rather than falling through to here. \
       This arm exists to keep the table wildcard-free; seeing this text in a \
       real diagnostic means the parser lost an arm."
  | Pexp_field _ ->
      "a qualified field access (`r.M.f`) is not supported in a kernel: field \
       names are resolved against the record type, not against a module path. \
       Write `r.f`."
  | Pexp_setfield _ ->
      "a qualified field assignment (`r.M.f <- e`) is not supported in a \
       kernel: field names are resolved against the record type, not against a \
       module path. Write `r.f <- e`."
  | Pexp_array _ ->
      "an array literal (`[| e1; e2 |]`) is not supported in a kernel: a \
       device array has to be allocated with a memory space, so there is no \
       literal form. Use `create_array n Local` (or `Shared`, or `Global`) and \
       assign the elements."
  | Pexp_ifthenelse _ ->
      "UNREACHABLE: an `if` is parsed by an arm of its own that matches the \
       constructor totally, and a sub-node the parser cannot build raises THAT \
       sub-node's refusal rather than falling through to here. This arm exists \
       to keep the table wildcard-free; seeing this text in a real diagnostic \
       means the parser lost an arm."
  | Pexp_sequence _ ->
      "UNREACHABLE: a sequence is parsed by an arm of its own that matches the \
       constructor totally, and a sub-node the parser cannot build raises THAT \
       sub-node's refusal rather than falling through to here. This arm exists \
       to keep the table wildcard-free; seeing this text in a real diagnostic \
       means the parser lost an arm."
  | Pexp_while _ ->
      "UNREACHABLE: a `while` is parsed by an arm of its own that matches the \
       constructor totally, and a sub-node the parser cannot build raises THAT \
       sub-node's refusal rather than falling through to here. This arm exists \
       to keep the table wildcard-free; seeing this text in a real diagnostic \
       means the parser lost an arm."
  | Pexp_for _ ->
      "a `for` loop must bind a variable: `for i = lo to hi do ... done`. A \
       wildcard binder (`for _ = ...`) is not supported, because the lowered \
       loop emits its induction variable by name."
  | Pexp_constraint _ ->
      "UNREACHABLE: an annotated expression is parsed by an arm of its own \
       that matches the constructor totally, and a sub-node the parser cannot \
       build raises THAT sub-node's refusal rather than falling through to \
       here. This arm exists to keep the table wildcard-free; seeing this text \
       in a real diagnostic means the parser lost an arm."
  | Pexp_coerce _ ->
      "a coercion (`(e :> t)`) is not supported in a kernel: there is no \
       subtyping in the kernel type system, so a coercion has no meaning to \
       lower. Use a type annotation (`(e : t)`) if you meant to fix a type."
  | Pexp_send _ ->
      "a method call (`obj # m`) is not supported in a kernel: objects have no \
       device representation."
  | Pexp_new _ ->
      "object instantiation (`new c`) is not supported in a kernel: objects \
       have no device representation."
  | Pexp_setinstvar _ ->
      "an instance-variable assignment (`x <- e`) is not supported in a \
       kernel: objects have no device representation. A mutable kernel-local \
       is `let x = mut e in ...`, assigned with `x := e`."
  | Pexp_override _ ->
      "an object override (`{< x = e >}`) is not supported in a kernel: \
       objects have no device representation."
  | Pexp_letmodule _ ->
      "`let module M = struct ... end in` is only read at the TOP of a \
       `[%kernel]` payload, where it declares the kernel's types and helpers. \
       It cannot appear inside a kernel body, because a module is not a value \
       the device code has anywhere to put."
  | Pexp_letexception _ ->
      "`let exception E in` is not supported in kernels: there are no \
       exceptions on a device."
  | Pexp_assert _ ->
      "`assert` is not supported in a kernel: a failing assertion raises, and \
       there are no exceptions on a device. Compute the condition and take a \
       branch, or check it on the host before launching."
  | Pexp_lazy _ ->
      "`lazy` is not supported in a kernel: a thunk is a heap-allocated \
       closure, and there is no device heap to allocate it in."
  | Pexp_poly _ ->
      "an explicitly polymorphic expression is not supported in a kernel: \
       every kernel-local function is monomorphised at its call sites, so it \
       cannot carry a quantified type."
  | Pexp_object _ ->
      "`object ... end` is not supported in a kernel: objects have no device \
       representation."
  | Pexp_newtype _ ->
      "a locally abstract type (`fun (type a) -> ...`) is not supported in a \
       kernel: the type would have to survive monomorphisation, and the kernel \
       type system has no abstract types."
  | Pexp_pack _ ->
      "a first-class module (`(module M)`) is not supported in a kernel: a \
       module is not a value the device code has anywhere to put."
  | Pexp_open _ ->
      "`let open M in e` is accepted in a kernel for any plain module path, at \
       any depth. What is not resolved is an open of something that is not a \
       path: a functor application (`let open F(X) in`) or a structure (`let \
       open struct ... end in`). Alias it outside the kernel (`module M = \
       F(X)`) and open the alias."
  | Pexp_letop _ ->
      "a binding operator (`let* x = ...`, `let+ x = ...`) is not supported in \
       a kernel: it desugars to an application of a user-defined `let*`, which \
       is a higher-order function the defunctionaliser cannot see through. Use \
       a plain `let`."
  | Pexp_extension (name, _) ->
      Printf.sprintf
        "the extension point `%s` is not one Sarek interprets in a kernel body \
         (%s). An extension the parser does not know is not applied and not \
         reported by anything else, so it is refused here."
        name.txt
        kernel_extensions
  | Pexp_unreachable ->
      "`.` (the unreachable-case marker) is not supported in a kernel: it \
       lowers to a run-time failure, and a device has nothing to fail into."

(** Refusal text for a pattern the kernel subset does not accept. Reached from
    [Sarek_parse_helpers.parse_pattern]'s final arm. *)
let pattern_refusal (d : pattern_desc) : string =
  match d with
  | Ppat_any | Ppat_var _ | Ppat_tuple _ ->
      "UNREACHABLE: a wildcard, variable or tuple pattern is parsed by an arm \
       of its own that matches the constructor totally, and a sub-node the \
       parser cannot build raises THAT sub-node's refusal rather than falling \
       through to here. This arm exists to keep the table wildcard-free; \
       seeing this text in a real diagnostic means the parser lost an arm."
  | Ppat_alias _ ->
      "an `as` alias in a pattern (`p as x`) is not supported in a kernel: the \
       lowering binds either the whole scrutinee or its fields, not both. Bind \
       the scrutinee with a `let` before the `match` and use its name."
  | Ppat_constant _ ->
      "matching on a literal is not supported in a kernel: a `match` lowers to \
       a test on a variant TAG, and a literal has no tag. Use `if e = k then \
       ... else ...`."
  | Ppat_interval _ ->
      "a range pattern (`'a' .. 'z'`) is not supported in a kernel: a `match` \
       lowers to a test on a variant tag, and a range has no tag. Use a \
       comparison in an `if`."
  | Ppat_construct _ ->
      "only an unqualified constructor pattern of a type declared with \
       [@@sarek.type] is supported in a kernel, so `M.C x` must be written `C \
       x` with the type in scope."
  | Ppat_variant _ ->
      "a polymorphic-variant pattern (`` `Tag ``) is not supported in kernels: \
       the device representation of a variant is its declaration order, and a \
       polymorphic variant declares none. Declare a normal variant type with \
       [@@sarek.type] instead."
  | Ppat_record _ ->
      "a record pattern (`{ f = p }`) is not supported in a kernel: only a \
       variable or a constructor pattern is matched against. Bind the record \
       and read its fields (`let r = ... in r.f`)."
  | Ppat_array _ ->
      "an array pattern (`[| p1; p2 |]`) is not supported in a kernel: a \
       device array has no length known to the match, so there is nothing to \
       test."
  | Ppat_or _ ->
      "an or-pattern (`p1 | p2`) is not supported in a kernel: each arm lowers \
       to one tag test with one binding set, and an or-pattern has several. \
       Write one arm per constructor."
  | Ppat_constraint _ ->
      "UNREACHABLE: an annotated pattern is parsed by an arm of its own that \
       matches the constructor totally, and a sub-node the parser cannot build \
       raises THAT sub-node's refusal rather than falling through to here. \
       This arm exists to keep the table wildcard-free; seeing this text in a \
       real diagnostic means the parser lost an arm."
  | Ppat_type _ -> "a type-directed pattern (`#t`) is not supported in kernels."
  | Ppat_lazy _ ->
      "a `lazy` pattern is not supported in a kernel: there are no thunks on a \
       device."
  | Ppat_unpack _ ->
      "a module unpacking pattern (`(module M)`) is not supported in a kernel: \
       a module is not a value the device code has anywhere to put."
  | Ppat_exception _ ->
      "an `exception` pattern is not supported in kernels: there are no \
       exceptions on a device."
  | Ppat_extension (name, _) ->
      Printf.sprintf
        "the extension point `%s` is not interpreted in a kernel pattern: no \
         extension point is. An extension the parser does not know is not \
         applied and not reported by anything else, so it is refused here."
        name.txt
  | Ppat_open _ ->
      "a pattern-scoped open (`M.(p)`) is not supported in kernels. Put the \
       `open` outside the `match`."

(** Refusal text for a core_type the kernel subset does not accept. Reached from
    [Sarek_parse_helpers.parse_type]'s final arm, which until backlog-192
    returned the type constructor named ["unknown"] instead — a name the typer
    resolves to an EMPTY RECORD type, so an unsupported annotation became a
    phantom type rather than an error. *)
let core_type_refusal (d : core_type_desc) : string =
  match d with
  | Ptyp_any ->
      "a wildcard type annotation (`_`) cannot be used in a kernel: every \
       kernel type must be known at PPX time, because the device code is \
       generated from it. Write the type, or leave the annotation off entirely \
       and let inference run."
  | Ptyp_var _ | Ptyp_arrow _ | Ptyp_tuple _ | Ptyp_constr _ ->
      "UNREACHABLE: a type variable, arrow, tuple or constructor is parsed by \
       an arm of its own that matches the constructor totally, and a sub-node \
       the parser cannot build raises THAT sub-node's refusal rather than \
       falling through to here. This arm exists to keep the table \
       wildcard-free; seeing this text in a real diagnostic means the parser \
       lost an arm."
  | Ptyp_object _ | Ptyp_class _ ->
      "an object or class type is not supported in a kernel: objects have no \
       device representation."
  | Ptyp_alias _ ->
      "a type alias binder (`t as 'a`) is not supported in a kernel \
       annotation: the kernel type system has no recursive types for it to \
       name. Write the type without the alias."
  | Ptyp_variant _ ->
      "a polymorphic-variant type (`` [> `A ] ``) is not supported in kernels: \
       the device representation of a variant is its declaration order, and a \
       polymorphic variant declares none. Declare a normal variant type with \
       [@@sarek.type] instead."
  | Ptyp_poly _ ->
      "UNREACHABLE, and not unsupported either: `parse_type` looks THROUGH a \
       quantifier on purpose (the quantified variables are the ones `Ptyp_var` \
       already carries by name), so a quantified type is accepted and a bad \
       type under it raises that type's own refusal. This arm keeps the table \
       wildcard-free."
  | Ptyp_package _ ->
      "a first-class module type (`(module S)`) is not supported in a kernel: \
       a module is not a value the device code has anywhere to put."
  | Ptyp_open _ ->
      "a type-scoped open (`M.(t)`) is not supported in a kernel annotation. \
       Put the `open` outside, or write the type path in full."
  | Ptyp_extension (name, _) ->
      Printf.sprintf
        "the extension point `%s` is not interpreted in a kernel type \
         annotation: no extension point is. An extension the parser does not \
         know is not applied and not reported by anything else, so it is \
         refused here."
        name.txt

(** Refusal text for a structure item inside a kernel payload's
    [let module M = struct ... end]. Reached from [Sarek_parse.parse_payload],
    whose module-item fold until backlog-192 returned its accumulator unchanged
    for everything it did not recognise — so a helper the user wrote in that
    module simply was not there, and the kernel failed later with an unbound
    name, or (worse) resolved a same-named binding from somewhere else. *)
let structure_item_refusal (d : structure_item_desc) : string =
  match d with
  | Pstr_type _ | Pstr_value _ ->
      "UNREACHABLE: a type or value declaration is parsed by an arm of its own \
       that matches the constructor totally, and a sub-node the parser cannot \
       build raises THAT sub-node's refusal rather than falling through to \
       here. This arm exists to keep the table wildcard-free; seeing this text \
       in a real diagnostic means the parser lost an arm."
  | Pstr_eval _ ->
      "a bare expression in a kernel module has no effect and is not evaluated \
       anywhere: a kernel module contributes type declarations, helper \
       functions and typed constants. Bind it (`let x : t = e`) or delete it."
  | Pstr_primitive _ ->
      "an `external` declaration is not supported in a kernel module: there is \
       no device linker to bind it. Inline device code with the `native` \
       extension point instead."
  | Pstr_typext _ ->
      "extending a variant type (`type t += C`) is not supported in a kernel \
       module: a variant's device representation is fixed by its declaration \
       order, which an extension would change after the fact."
  | Pstr_exception _ ->
      "an exception declaration is not supported in a kernel module: there are \
       no exceptions on a device."
  | Pstr_module _ | Pstr_recmodule _ ->
      "a nested module is not supported in a kernel module: the payload reader \
       is one level deep, so its contents would be silently absent. Move the \
       declarations up one level."
  | Pstr_modtype _ ->
      "a module-type declaration is not supported in a kernel module: it \
       contributes no type, helper or constant the kernel can use."
  | Pstr_open _ ->
      "an `open` inside a kernel module is not supported: names in a kernel \
       module are resolved unqualified against the kernel's own environment, \
       not through an open. Write the referenced names in full."
  | Pstr_class _ | Pstr_class_type _ ->
      "a class declaration is not supported in a kernel module: objects have \
       no device representation."
  | Pstr_include _ ->
      "`include` is not supported in a kernel module: nothing copies the \
       included signature's items into the kernel environment, so they would \
       be silently absent."
  | Pstr_attribute _ ->
      "a floating attribute is not supported in a kernel module: no attribute \
       is interpreted at this position, so it would be silently discarded."
  | Pstr_extension ((name, _), _) ->
      Printf.sprintf
        "the extension point `%s` is not interpreted inside a kernel module \
         (%s, and none of them is a module item). An extension the parser does \
         not know is not applied and not reported by anything else, so it is \
         refused here."
        name.txt
        kernel_extensions
