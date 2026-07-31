(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module defines error types and error reporting for the Sarek compiler.
 * Errors include source locations for better diagnostics.
 ******************************************************************************)

open Sarek_ast
open Sarek_types

(** Error types *)
type error =
  | Unbound_variable of string * loc
  | Unbound_constructor of string * loc
  | Unbound_field of string * loc
  | Unbound_type of string * loc
  | Type_mismatch of {expected : typ; got : typ; loc : loc}
  | Cannot_unify of typ * typ * loc
  | Not_a_function of typ * loc
  | Wrong_arity of {expected : int; got : int; loc : loc}
  | Not_a_vector of typ * loc
  | Not_an_array of typ * loc
  | Not_a_record of typ * loc
  | Field_not_found of string * typ * loc
  | Immutable_variable of string * loc
  | Recursive_type of typ * loc
  | Unsupported_expression of string * loc
  | Parse_error of string * loc
  | Invalid_kernel of string * loc
  | Duplicate_field of string * loc
  | Missing_type_annotation of string * loc
  | Invalid_intrinsic of string * loc
  | Barrier_in_diverged_flow of loc
  | Warp_collective_in_diverged_flow of string * loc
  | Reserved_keyword of string * loc
  | Reserved_prefix of string * loc
  (* Lowering errors - occur during typed AST → IR transformation *)
  | Unsupported_type_in_registration of string * loc
  | Unsupported_constructor_form of loc
  | Unsupported_registration_form of loc
  | Unsupported_tuple_in_variant of loc
  | Unsupported_function_in_variant of loc
  | Unknown_variant_type of string * loc
  | Expression_needs_statement_context of string * loc
  | Invalid_lvalue of loc
  | Function_value_escapes of string * typ * loc
  | Instantiation_mismatch of {callee : string; t1 : typ; t2 : typ; loc : loc}
      (** A call site could not be typed against the callee's type. Reported
          instead of a bare [Cannot_unify] when the callee is NAMED, because the
          bare form ("Cannot unify types: float32 and float64") gives the user
          two type names and no indication of which function they came from —
          the reported failure mode of a polymorphic [@sarek.module] helper
          instantiated at a non-default element type (#97). *)
  | Aggregate_equality_operand of string * typ * loc
      (** backlog-194: a value with no comparable device representation reached
          [=] or [<>] — a tuple, record or variant (a struct: no backend lowers
          the comparison field-wise, and the C-family emitters print [a == b],
          which those compilers reject), or a function (inlined at its call
          sites and never emitted, so [f == g] named identifiers that do not
          exist in the generated source). The set is
          {!Sarek_types.is_uncomparable_operand_typ}; the [typ] is the operand
          type, so the message can name what was compared. Pointer-shaped
          operands (a vector, a local array) are NOT this error: [src = dst]
          emits [(src == dst)], which clang -x cl and glslangValidator both
          accept. *)
  | Float16_operand of string * loc
      (** An f16 value reached an operator. f16 is a storage-only type, so this
          is always a user error with a specific remedy; reporting it as
          [Type_mismatch {expected = int32}] (what [check_numeric]'s
          fall-through produced) told the user nothing. The string names the
          operator. *)

(** Get the location from an error *)
let error_loc = function
  | Unbound_variable (_, loc) -> loc
  | Unbound_constructor (_, loc) -> loc
  | Unbound_field (_, loc) -> loc
  | Unbound_type (_, loc) -> loc
  | Type_mismatch {loc; _} -> loc
  | Cannot_unify (_, _, loc) -> loc
  | Not_a_function (_, loc) -> loc
  | Wrong_arity {loc; _} -> loc
  | Not_a_vector (_, loc) -> loc
  | Not_an_array (_, loc) -> loc
  | Not_a_record (_, loc) -> loc
  | Field_not_found (_, _, loc) -> loc
  | Immutable_variable (_, loc) -> loc
  | Recursive_type (_, loc) -> loc
  | Unsupported_expression (_, loc) -> loc
  | Parse_error (_, loc) -> loc
  | Invalid_kernel (_, loc) -> loc
  | Duplicate_field (_, loc) -> loc
  | Missing_type_annotation (_, loc) -> loc
  | Invalid_intrinsic (_, loc) -> loc
  | Barrier_in_diverged_flow loc -> loc
  | Warp_collective_in_diverged_flow (_, loc) -> loc
  | Reserved_keyword (_, loc) -> loc
  | Reserved_prefix (_, loc) -> loc
  | Unsupported_type_in_registration (_, loc) -> loc
  | Unsupported_constructor_form loc -> loc
  | Unsupported_registration_form loc -> loc
  | Unsupported_tuple_in_variant loc -> loc
  | Unsupported_function_in_variant loc -> loc
  | Unknown_variant_type (_, loc) -> loc
  | Expression_needs_statement_context (_, loc) -> loc
  | Invalid_lvalue loc -> loc
  | Function_value_escapes (_, _, loc) -> loc
  | Instantiation_mismatch {loc; _} -> loc
  | Aggregate_equality_operand (_, _, loc) -> loc
  | Float16_operand (_, loc) -> loc

(** Human-readable operator name for diagnostics, e.g. ['=' -> "'='"]. Lives
    here rather than in {!Sarek_typer} because two modules render an operator
    into a diagnostic — the typer's [infer_binop] and the aggregate-equality
    backstop in {!Sarek_lower_ir} (backlog-194) — and a per-module [match] over
    [binop] is two spellings of one name with nothing comparing them. *)
let binop_display_name = function
  | Add -> "'+' / '+.'"
  | Sub -> "'-' / '-.'"
  | Mul -> "'*' / '*.'"
  | Div -> "'/' / '/.'"
  | Mod -> "'mod'"
  | And -> "'&&'"
  | Or -> "'||'"
  | Eq -> "'='"
  | Ne -> "'<>'"
  | Lt -> "'<'"
  | Le -> "'<='"
  | Gt -> "'>'"
  | Ge -> "'>='"
  | Land -> "'land'"
  | Lor -> "'lor'"
  | Lxor -> "'lxor'"
  | Lsl -> "'lsl'"
  | Lsr -> "'lsr'"
  | Asr -> "'asr'"

(** backlog-194. The body of the aggregate-equality refusal, WITHOUT the leading
    operator name. It is a function rather than inline text in {!pp_error}
    because two sites emit this refusal — the typer's, through
    [Aggregate_equality_operand], and the post-monomorphisation backstop in
    {!Sarek_lower_ir}, which raises a located Ppxlib error directly. An earlier
    revision had the second site build an [Aggregate_equality_operand] with a
    FAKE location purely to borrow this printer; sharing the body instead is
    what lets the two agree on the wording without either lying about a loc.

    What the text does and does not claim. It says no backend lowers the
    comparison field-wise, which is true of all seven. It does NOT enumerate
    what each backend prints: that was a standing claim about six modules with
    nothing tying it to them, and it is the per-backend evidence, dated, that
    belongs in the negative-test headers instead.

    The remedy is given in SOURCE terms. It previously advised [a._0 = b._0] for
    a tuple, which is not writable: [_0] is the synthesized field name of the
    internal [_tup_*] record, and a field access on a tuple-typed value is
    refused by the typer with [Not_a_record]. Destructuring is the spelling that
    works. *)
let aggregate_equality_body (ty : typ) : string =
  match repr ty with
  | TFun _ ->
      (* A separate sentence, because the struct wording would be false here.
         The struct members fail for want of a field-wise lowering; a function
         operand fails earlier and harder — the emitted `f == g` names
         identifiers that were inlined away and never declared, so clang -x cl
         reports "use of undeclared identifier". There is also no remedy of the
         "compare the parts" kind to offer. *)
      Format.asprintf
        "cannot compare two function values in a kernel (here: %a). A \
         kernel-local function is inlined at its call sites and is never \
         emitted as a device object, so there is nothing to compare: the \
         generated code named two identifiers that do not exist in it. Compare \
         the RESULTS of applying them, or use a variant tag to select \
         behaviour."
        pp_typ
        ty
  | _ ->
      Format.asprintf
        "cannot compare two values of type %a in a kernel: no backend lowers \
         that comparison to a field-wise one, so it cannot be emitted. Compare \
         the components explicitly — destructure a tuple first (`let a0, a1 = \
         a in let b0, b1 = b in a0 = b0 && a1 = b1`), use the fields of a \
         record (`a.x = b.x && a.y = b.y`), and match on both sides for a \
         variant."
        pp_typ
        ty

(** Pretty print an error *)
let pp_error fmt = function
  | Unbound_variable (name, _) -> Format.fprintf fmt "Unbound variable: %s" name
  | Unbound_constructor (name, _) ->
      Format.fprintf fmt "Unbound constructor: %s" name
  | Unbound_field (name, _) ->
      Format.fprintf fmt "Unbound record field: %s" name
  | Unbound_type (name, _) -> Format.fprintf fmt "Unbound type: %s" name
  | Type_mismatch {expected; got; _} ->
      Format.fprintf
        fmt
        "Type mismatch: expected %a, got %a"
        pp_typ
        expected
        pp_typ
        got
  | Cannot_unify (t1, t2, _) ->
      Format.fprintf fmt "Cannot unify types: %a and %a" pp_typ t1 pp_typ t2
  | Not_a_function (t, _) ->
      Format.fprintf fmt "Expected a function type, got %a" pp_typ t
  | Wrong_arity {expected; got; _} ->
      Format.fprintf
        fmt
        "Wrong number of arguments: expected %d, got %d"
        expected
        got
  | Not_a_vector (t, _) ->
      Format.fprintf fmt "Expected a vector type, got %a" pp_typ t
  | Not_an_array (t, _) ->
      Format.fprintf fmt "Expected an array type, got %a" pp_typ t
  | Not_a_record (t, _) ->
      Format.fprintf fmt "Expected a record type, got %a" pp_typ t
  | Field_not_found (name, t, _) ->
      Format.fprintf fmt "Field %s not found in type %a" name pp_typ t
  | Immutable_variable (name, _) ->
      Format.fprintf fmt "Variable %s is not mutable" name
  | Recursive_type (t, _) ->
      Format.fprintf fmt "Recursive type detected: %a" pp_typ t
  | Unsupported_expression (desc, _) ->
      Format.fprintf
        fmt
        "Unsupported expression: %s (tip: mutable locals in kernels must use \
         \"let fx = mut ...\"; refs/OCaml stdlib mutables are not supported)"
        desc
  | Parse_error (msg, _) -> Format.fprintf fmt "Parse error: %s" msg
  | Invalid_kernel (msg, _) -> Format.fprintf fmt "Invalid kernel: %s" msg
  | Duplicate_field (name, _) -> Format.fprintf fmt "Duplicate field: %s" name
  | Missing_type_annotation (name, _) ->
      Format.fprintf fmt "Missing type annotation for parameter: %s" name
  | Invalid_intrinsic (name, _) ->
      Format.fprintf fmt "Invalid intrinsic: %s" name
  | Barrier_in_diverged_flow _ ->
      Format.fprintf
        fmt
        "Barrier called in diverged control flow. All threads in a workgroup \
         must reach the barrier together"
  | Warp_collective_in_diverged_flow (name, _) ->
      Format.fprintf
        fmt
        "Warp collective '%s' called in diverged control flow. All threads in \
         a warp must participate together"
        name
  | Reserved_keyword (name, _) ->
      Format.fprintf
        fmt
        "'%s' is a reserved C/CUDA/OpenCL keyword and cannot be used as a \
         function or variable name"
        name
  | Reserved_prefix (name, _) ->
      Format.fprintf
        fmt
        "identifiers beginning with 'sarek_' are reserved by the Sarek code \
         generator; rename '%s'"
        name
  | Unsupported_type_in_registration (type_desc, _) ->
      Format.fprintf
        fmt
        "Unsupported type in [@@@@sarek.type] registration: %s"
        type_desc
  | Unsupported_constructor_form _ ->
      Format.fprintf
        fmt
        "Only zero or single-argument constructors are supported in variant \
         types"
  | Unsupported_registration_form _ ->
      Format.fprintf
        fmt
        "Only record and variant types can be registered with [@@@@sarek.type]"
  | Unsupported_tuple_in_variant _ ->
      Format.fprintf
        fmt
        "Tuple types are not yet supported in variant constructor arguments"
  | Unsupported_function_in_variant _ ->
      Format.fprintf
        fmt
        "Function types are not supported in variant constructor arguments"
  | Unknown_variant_type (name, _) ->
      Format.fprintf
        fmt
        "Unknown type in variant constructor: %s. Custom types must be \
         registered with [@@@@sarek.type]"
        name
  | Expression_needs_statement_context (expr_type, _) ->
      Format.fprintf
        fmt
        "Expression of type %s can only appear in statement context (e.g., let \
         binding, sequence)"
        expr_type
  | Invalid_lvalue _ ->
      Format.fprintf
        fmt
        "Invalid left-hand side of assignment. Expected variable, field \
         access, or array element"
  | Function_value_escapes (msg, t, _) ->
      Format.fprintf
        fmt
        "Function value escapes: %s. Offending type: %a. Function values in \
         kernels must be bound to a `let` and applied directly (they have no \
         runtime representation)."
        msg
        pp_typ
        t
  | Instantiation_mismatch {callee; t1; t2; _} ->
      Format.fprintf
        fmt
        "'%s' cannot be used at this call site: %a and %a do not match.@ A \
         [@sarek.module] helper is monomorphised per call site, so a \
         polymorphic helper can only be instantiated at the element types its \
         BODY admits — a body using a float32-only operation (the \
         Sarek_stdlib.Std math intrinsics are float32) cannot be instantiated \
         at float64. Give the helper a concrete type, or use the \
         element-type-specific module (e.g. Sarek_float64)."
        callee
        pp_typ
        t1
        pp_typ
        t2
  | Aggregate_equality_operand (what, ty, _) ->
      Format.fprintf fmt "%s %s" what (aggregate_equality_body ty)
  | Float16_operand (what, _) ->
      Format.fprintf
        fmt
        "float16 is a storage-only type and has no arithmetic: %s cannot be \
         applied to a float16 value. Widen to float32 first with \
         float32_of_float16, and narrow the result back with \
         float16_of_float32 (e.g. `float16_of_float32 (float32_of_float16 \
         a.(i) +. float32_of_float16 b.(i))`)."
        what

(** Convert error to string *)
let error_to_string e = Format.asprintf "%a" pp_error e

(** Print error with location *)
let pp_error_with_loc fmt e =
  let loc = error_loc e in
  Format.fprintf
    fmt
    "%s:%d:%d: %a"
    loc.loc_file
    loc.loc_line
    loc.loc_col
    pp_error
    e

(** Result type for error accumulation *)
type 'a result = ('a, error list) Result.t

(** Monadic operations for error handling *)
let ( let* ) = Result.bind

let ok x = Ok x

let error e = Error [e]

let errors es = Error es

let map_result f = function Ok x -> Ok (f x) | Error e -> Error e

let combine_results results =
  let rec aux acc = function
    | [] -> Ok (List.rev acc)
    | Ok x :: rest -> aux (x :: acc) rest
    | Error es :: rest ->
        (* Accumulate errors from remaining results *)
        let more_errors =
          List.filter_map (function Ok _ -> None | Error es -> Some es) rest
        in
        Error (es @ List.concat more_errors)
  in
  aux [] results

(** Report error to ppxlib *)
let report_error e =
  let loc = loc_to_ppxlib (error_loc e) in
  Ppxlib.Location.raise_errorf ~loc "%a" pp_error e

(** Report multiple errors *)
let report_errors = function
  | [] -> ()
  | e :: _ -> report_error e (* Report first error *)

(** Raise error as OCaml exception for use in PPX *)
exception Sarek_error of error

let raise_error e = raise (Sarek_error e)
