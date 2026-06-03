(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Pure OCaml-kernel-source to GPU-source transpiler.

    [Sarek_transpile] converts an OCaml kernel source string through the full
    Sarek frontend (parse → type → convergence → mono → tailrec → lower) and
    emits GPU source code for the requested backend.

    The library is FFI-free: it does not depend on [spoc_core] or [ctypes] and
    therefore links cleanly as bytecode and (with polyfills) under js_of_ocaml.
*)

(** GPU backend selector. *)
type backend = CUDA | OpenCL | Metal | GLSL | WGSL

(** Structured error type. Every frontend failure is converted to one of these
    variants; no exception escapes {!of_source}. *)
type error =
  | Parse_error of string * Sarek_ast.loc
      (** OCaml parser or Sarek parse error with message and location. *)
  | Type_error of Sarek_error.error list
      (** Type-inference or constraint-solving failure. *)
  | Convergence_error of Sarek_error.error list
      (** Barrier-safety analysis failure. *)
  | Unsupported_native of Sarek_ast.loc
      (** Kernel contains [[%native]] which cannot be transpiled purely. *)
  | Internal_error of string
      (** Unexpected exception — indicates a bug, not a user error. *)

(** [string_of_error e] returns a human-readable representation of [e]. Intended
    for debugging and test output. *)
val string_of_error : error -> string

(** [of_source backend src] parses [src] as an OCaml kernel expression and runs
    the full frontend pipeline:
    + OCaml parse → ppxlib expression
    + Sarek parse → [Sarek_ast.kernel]
    + [[%native]] rejection
    + Type inference ([Sarek_typer.infer_kernel])
    + Convergence check ([Sarek_convergence.check_kernel])
    + Monomorphisation, tail-recursion transform, IR lowering
    + Code generation via [sarek_codegen]

    Returns [Ok gpu_source] on success, or [Error e] with a structured
    description of the first failure encountered.

    All frontend exceptions are caught and converted to [error] values. *)
val of_source : backend -> string -> (string, error) result
