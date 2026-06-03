(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_transpile - Pure OCaml-kernel-source -> GPU-source transpiler
 *
 * Ties sarek_frontend + sarek_codegen + sarek_stdlib_meta together into a
 * single [of_source : backend -> string -> (string, error) result] function.
 * No FFI, no ctypes, no spoc_core.  Links as pure bytecode and (with
 * polyfills) as js_of_ocaml.
 ******************************************************************************)

(** GPU backend selector *)
type backend = CUDA | OpenCL | Metal | GLSL

(** Structured error: every frontend failure is converted here — no exception
    escapes [of_source]. *)
type error =
  | Parse_error of string * Sarek_ast.loc
      (** OCaml parser or Sarek parse error *)
  | Type_error of Sarek_error.error list
      (** Typer / constraint-solving error *)
  | Convergence_error of Sarek_error.error list  (** Barrier-safety error *)
  | Unsupported_native of Sarek_ast.loc
      (** Kernel contains [%native] which cannot be transpiled purely *)
  | Internal_error of string  (** Unexpected exception (should not occur) *)

(** Convert [error] to a human-readable string. *)
let string_of_error = function
  | Parse_error (msg, loc) ->
      Printf.sprintf
        "parse error at %s:%d: %s"
        loc.Sarek_ast.loc_file
        loc.Sarek_ast.loc_line
        msg
  | Type_error errs ->
      Printf.sprintf
        "type error: %s"
        (String.concat "; " (List.map Sarek_error.error_to_string errs))
  | Convergence_error errs ->
      Printf.sprintf
        "convergence error: %s"
        (String.concat "; " (List.map Sarek_error.error_to_string errs))
  | Unsupported_native loc ->
      Printf.sprintf
        "[%%native] at %s:%d cannot be transpiled purely"
        loc.Sarek_ast.loc_file
        loc.Sarek_ast.loc_line
  | Internal_error msg -> Printf.sprintf "internal error: %s" msg

(******************************************************************************)
(* [%native] detection - walk the kernel body looking for ENative nodes.     *)
(******************************************************************************)

let rec check_expr_no_native (e : Sarek_ast.expr) : (unit, Sarek_ast.loc) result
    =
  match e.Sarek_ast.e with
  | Sarek_ast.ENative _ -> Error e.Sarek_ast.expr_loc
  (* Leaf nodes *)
  | Sarek_ast.EUnit | Sarek_ast.EBool _ | Sarek_ast.EInt _ | Sarek_ast.EInt32 _
  | Sarek_ast.EInt64 _ | Sarek_ast.EFloat _ | Sarek_ast.EDouble _
  | Sarek_ast.EVar _ | Sarek_ast.EGlobalRef _
  | Sarek_ast.EConstr (_, None) ->
      Ok ()
  (* One child *)
  | Sarek_ast.EReturn sub
  | Sarek_ast.EAssign (_, sub)
  | Sarek_ast.EFieldGet (sub, _)
  | Sarek_ast.EUnop (_, sub)
  | Sarek_ast.EConstr (_, Some sub)
  | Sarek_ast.ETyped (sub, _)
  | Sarek_ast.EOpen (_, sub)
  | Sarek_ast.EPragma (_, sub)
  | Sarek_ast.ECreateArray (sub, _, _) ->
      check_expr_no_native sub
  (* Two children *)
  | Sarek_ast.EBinop (_, a, b)
  | Sarek_ast.EVecGet (a, b)
  | Sarek_ast.EArrGet (a, b)
  | Sarek_ast.ESeq (a, b)
  | Sarek_ast.EWhile (a, b)
  | Sarek_ast.ELet (_, _, a, b)
  | Sarek_ast.ELetMut (_, _, a, b)
  | Sarek_ast.ESuperstep (_, _, a, b)
  | Sarek_ast.ELetRec (_, _, _, a, b)
  | Sarek_ast.EFieldSet (a, _, b) -> (
      match check_expr_no_native a with
      | Error _ as err -> err
      | Ok () -> check_expr_no_native b)
  (* Three children *)
  | Sarek_ast.EVecSet (a, b, c)
  | Sarek_ast.EArrSet (a, b, c)
  | Sarek_ast.EFor (_, a, b, _, c) -> (
      match check_expr_no_native a with
      | Error _ as err -> err
      | Ok () -> (
          match check_expr_no_native b with
          | Error _ as err -> err
          | Ok () -> check_expr_no_native c))
  (* EIf: condition * then * else option *)
  | Sarek_ast.EIf (c, t, e_opt) -> (
      match check_expr_no_native c with
      | Error _ as err -> err
      | Ok () -> (
          match check_expr_no_native t with
          | Error _ as err -> err
          | Ok () -> (
              match e_opt with
              | None -> Ok ()
              | Some e -> check_expr_no_native e)))
  (* EApp: function * args list *)
  | Sarek_ast.EApp (f, args) -> (
      match check_expr_no_native f with
      | Error _ as err -> err
      | Ok () -> check_exprs_no_native args)
  (* ERecord: module_name option * (field * expr) list *)
  | Sarek_ast.ERecord (_, fields) -> check_exprs_no_native (List.map snd fields)
  | Sarek_ast.ETuple es -> check_exprs_no_native es
  (* EMatch: discriminant * (pattern * body) list *)
  | Sarek_ast.EMatch (e, cases) -> (
      match check_expr_no_native e with
      | Error _ as err -> err
      | Ok () ->
          List.fold_left
            (fun acc (_, body) ->
              match acc with
              | Error _ as err -> err
              | Ok () -> check_expr_no_native body)
            (Ok ())
            cases)
  (* ELetShared: string * type_expr * size option * body *)
  | Sarek_ast.ELetShared (_, _, sz_opt, body) -> (
      match sz_opt with
      | None -> check_expr_no_native body
      | Some sz -> (
          match check_expr_no_native sz with
          | Error _ as err -> err
          | Ok () -> check_expr_no_native body))

and check_exprs_no_native exprs =
  List.fold_left
    (fun acc e ->
      match acc with Error _ as err -> err | Ok () -> check_expr_no_native e)
    (Ok ())
    exprs

let check_kernel_no_native (kernel : Sarek_ast.kernel) :
    (unit, Sarek_ast.loc) result =
  let check_module_item = function
    | Sarek_ast.MConst (_, _, e) -> check_expr_no_native e
    | Sarek_ast.MFun (_, _, _, body) -> check_expr_no_native body
  in
  match
    List.fold_left
      (fun acc item ->
        match acc with Error _ as err -> err | Ok () -> check_module_item item)
      (Ok ())
      kernel.Sarek_ast.kern_module_items
  with
  | Error _ as err -> err
  | Ok () -> check_expr_no_native kernel.Sarek_ast.kern_body

(******************************************************************************)
(* String -> ppxlib expression bridge                                         *)
(******************************************************************************)

(** Parse an OCaml expression string into a ppxlib expression. Uses
    [Ppxlib.Parse.expression] which yields a ppxlib-typed AST directly. *)
let expr_of_string src =
  let lexbuf = Lexing.from_string src in
  lexbuf.Lexing.lex_curr_p <-
    {lexbuf.Lexing.lex_curr_p with Lexing.pos_fname = "<transpile>"} ;
  Ppxlib.Parse.expression lexbuf

(******************************************************************************)
(* Backend dispatch                                                           *)
(******************************************************************************)

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "<transpile>";
      loc_line = 1;
      loc_col = 0;
      loc_end_line = 1;
      loc_end_col = 0;
    }

open Sarek_codegen

(** Set [current_framework] in each codegen module so the pure registry picks
    the right device names (e.g. sinf vs sin on CUDA). *)
let set_framework backend =
  let name =
    match backend with
    | CUDA -> "CUDA"
    | OpenCL -> "OpenCL"
    | Metal -> "Metal"
    | GLSL -> "GLSL"
  in
  Sarek_ir_cuda.current_framework := Some name ;
  Sarek_ir_opencl.current_framework := Some name ;
  Sarek_ir_metal.current_framework := Some name

let emit_backend backend (k : Sarek_ir_ppx.kernel) =
  let k_types = Sarek_ir_conv.conv_kernel k in
  match backend with
  | CUDA -> Sarek_ir_cuda.generate k_types
  | OpenCL -> Sarek_ir_opencl.generate k_types
  | Metal -> Sarek_ir_metal.generate k_types
  | GLSL -> Sarek_ir_glsl.generate k_types

(******************************************************************************)
(* Main pipeline                                                              *)
(******************************************************************************)

let loc_of_lexing (p : Lexing.position) : Sarek_ast.loc =
  Sarek_ast.
    {
      loc_file = p.pos_fname;
      loc_line = p.pos_lnum;
      loc_col = p.pos_cnum - p.pos_bol;
      loc_end_line = p.pos_lnum;
      loc_end_col = p.pos_cnum - p.pos_bol;
    }

(** [of_source backend src] parses [src] as an OCaml kernel expression, runs the
    full frontend pipeline parse -> [%native] check -> type -> convergence ->
    mono -> tailrec -> lower -> codegen and returns the GPU source for
    [backend].

    All frontend errors are converted to [error]; no exception escapes. *)
let of_source (backend : backend) (src : string) : (string, error) result =
  (* Ensure stdlib_meta intrinsics are registered before running the pipeline.
     This is idempotent - multiple calls are safe. *)
  Sarek_stdlib_meta.force_init () ;
  set_framework backend ;
  (* Step 1: OCaml parser -> ppxlib expression *)
  match
    try Ok (expr_of_string src) with
    | Syntaxerr.Error se ->
        let loc = Syntaxerr.location_of_error se in
        Error
          (Parse_error
             ( Printexc.to_string (Syntaxerr.Error se),
               loc_of_lexing loc.loc_start ))
    | exn -> Error (Parse_error (Printexc.to_string exn, dummy_loc))
  with
  | Error _ as err -> err
  | Ok ppxlib_expr -> (
      (* Step 2: Sarek parse -> Sarek_ast.kernel *)
      match
        try Ok (Sarek_parse.parse_payload ppxlib_expr) with
        | Sarek_parse_helpers.Parse_error_exn (msg, loc) ->
            Error (Parse_error (msg, Sarek_ast.loc_of_ppxlib loc))
        | exn -> Error (Internal_error (Printexc.to_string exn))
      with
      | Error _ as err -> err
      | Ok kernel -> (
          (* Step 3: reject [%native] *)
          match check_kernel_no_native kernel with
          | Error loc -> Error (Unsupported_native loc)
          | Ok () -> (
              (* Step 4: type inference *)
              let env = Sarek_env.(empty |> with_stdlib) in
              match Sarek_typer.infer_kernel env kernel with
              | Error errs -> Error (Type_error errs)
              | Ok tkernel -> (
                  (* Step 5: convergence *)
                  match Sarek_convergence.check_kernel tkernel with
                  | Error errs -> Error (Convergence_error errs)
                  | Ok () -> (
                      (* Steps 6-7: mono -> tailrec -> lower -> emit.
                         Guarded so any internal raise maps to Internal_error
                         rather than escaping of_source (per the .mli contract). *)
                      try
                        let mono = Sarek_mono.monomorphize tkernel in
                        let tr = Sarek_tailrec.transform_kernel mono in
                        let ir_kernel, _warnings =
                          Sarek_lower_ir.lower_kernel tr
                        in
                        Ok (emit_backend backend ir_kernel)
                      with exn ->
                        Error (Internal_error (Printexc.to_string exn)))))))
