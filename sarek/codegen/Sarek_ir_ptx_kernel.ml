(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX kernel emitter: parameter/local declaration, register-block output, PTX
    file header, and top-level generate entry points. *)

open Sarek_ir_types
open Sarek_ir_ptx_types
open Sarek_ir_ptx_expr
open Sarek_ir_ptx_stmt

(** {1 Parameter and local emitters} *)

(** Emit ld.param instructions for each kernel parameter, binding registers into
    [env]. Returns the formatted .param declaration block string to be embedded
    in the .entry header. *)
let emit_params buf alloc (env : env) (params : decl list) : string =
  let param_decls = Buffer.create 256 in
  let first = ref true in
  List.iter
    (fun decl ->
      match decl with
      | DParam (v, arr_info_opt) -> (
          if not !first then Buffer.add_string param_decls ",\n" ;
          first := false ;
          match v.var_type with
          | TVec _ | TArray _ ->
              (* The launch convention (Execute.expand_to_run_source_args)
                 passes every vector as a (ptr, length) pair, mirroring the
                 CUDA C signature "T* x, int sarek_x_length". Declare both
                 params here even when the body never reads the length —
                 otherwise every following parameter is read from the wrong
                 kernelParams slot. *)
              let len_name = length_param_name v.var_name in
              Buffer.add_string
                param_decls
                (Printf.sprintf
                   "    .param .u64 param_%s,\n    .param .u32 param_%s"
                   v.var_name
                   len_name) ;
              let r = new_u64 alloc in
              env_bind env v.var_name r ;
              (match arr_info_opt with
              | Some info ->
                  (* Aggregate element types are validated at param time so a
                     rejected layout (misaligned leaf, nested variant, …)
                     fails with a precise error naming the parameter, instead
                     of surfacing later at a ld/st site (FR-030). *)
                  (match info.arr_elttype with
                  | TRecord _ | TVariant _ -> (
                      match Sarek_ir_layout.elttype_layout info.arr_elttype with
                      | Ok _ -> ()
                      | Error err ->
                          fail
                            (Printf.sprintf
                               "PTX codegen: vector parameter '%s': %s"
                               v.var_name
                               (Sarek_ir_layout.layout_error_message err)))
                  | _ -> ()) ;
                  Hashtbl.replace
                    alloc.arr_elt_types
                    v.var_name
                    info.arr_elttype
              | None ->
                  fail
                    (Printf.sprintf
                       "DParam '%s': TVec/TArray parameter missing array \
                        element-type info"
                       v.var_name)) ;
              emit buf "ld.param.u64 %s, [param_%s];" r v.var_name ;
              let r_len = new_u32 alloc in
              env_bind env len_name r_len ;
              emit buf "ld.param.u32 %s, [param_%s];" r_len len_name
          | TInt32 | TBool ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u32 param_%s" v.var_name) ;
              let r = new_u32 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u32 %s, [param_%s];" r v.var_name
          | TInt64 ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u64 param_%s" v.var_name) ;
              let r = new_u64 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u64 %s, [param_%s];" r v.var_name
          | TFloat32 ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .f32 param_%s" v.var_name) ;
              let r = new_f32 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.f32 %s, [param_%s];" r v.var_name
          | TFloat64 ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .f64 param_%s" v.var_name) ;
              let r = new_f64 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.f64 %s, [param_%s];" r v.var_name
          | TUnit ->
              Buffer.add_string
                param_decls
                (Printf.sprintf "    .param .u32 param_%s" v.var_name) ;
              let r = new_u32 alloc in
              env_bind env v.var_name r ;
              emit buf "ld.param.u32 %s, [param_%s];" r v.var_name
          | TRecord (tname, _) | TVariant (tname, _) ->
              (* C-17 / FR-030: by-value aggregate params have no host
                 marshalling; a TVec of the same type IS accepted (EC-11). *)
              fail
                (Printf.sprintf
                   "PTX codegen: kernel parameter '%s' is a bare \
                    record/variant of type '%s' passed by value; pass fields \
                    as separate scalar params or use a 1-element '%s' vector"
                   v.var_name
                   tname
                   tname))
      | DLocal _ | DShared _ -> ())
    params ;
  Buffer.contents param_decls

(* ptx_align_of_elttype / ptx_btype_of_elttype now live in
   Sarek_ir_ptx_types, shared with the SLet shared-array path in
   Sarek_ir_ptx_stmt. *)

let emit_locals buf shared_buf alloc (env : env) (locals : decl list) : unit =
  List.iter
    (fun decl ->
      match decl with
      | DLocal (v, init_opt) -> (
          let r = new_reg_for_type alloc v.var_type in
          env_bind env v.var_name r ;
          match init_opt with
          | None -> ()
          | Some e ->
              let r_init = emit_expr buf alloc env e in
              let mov_op =
                match v.var_type with
                | TFloat32 -> "mov.f32"
                | TFloat64 -> "mov.f64"
                | TInt64 -> "mov.u64"
                | _ -> "mov.u32"
              in
              emit buf "%s %s, %s;" mov_op r r_init)
      | DShared (name, elt, size_opt) ->
          let n =
            match size_opt with
            | None ->
                unsupported
                  (Printf.sprintf
                     "DShared '%s': dynamic shared memory (size=None) not yet \
                      supported"
                     name)
            | Some (EConst (CInt32 n)) when Int32.compare n 0l > 0 ->
                Int32.to_int n
            | Some (EConst (CInt32 _)) ->
                fail
                  (Printf.sprintf
                     "PTX codegen: DShared '%s': size must be positive"
                     name)
            | Some _ ->
                unsupported
                  (Printf.sprintf
                     "DShared '%s': non-literal size not supported"
                     name)
          in
          let align = ptx_align_of_elttype elt in
          let btype = ptx_btype_of_elttype elt in
          Buffer.add_string
            shared_buf
            (Printf.sprintf
               "    .shared .align %d .%s %s[%d];\n"
               align
               btype
               name
               n) ;
          let r = new_u32 alloc in
          env_bind env name r ;
          emit buf "mov.u32 %s, %s;" r name ;
          Hashtbl.replace alloc.arr_memspaces name () ;
          Hashtbl.replace alloc.arr_elt_types name elt
      | DParam _ -> ())
    locals

(** {1 Register block declaration} *)

(** Emit .reg declarations based on the allocator high-water marks. Must be
    called AFTER all emit_* calls complete. *)
let emit_reg_decls buf alloc =
  if alloc.u32 > 0 then Printf.bprintf buf "    .reg .u32 %%r<%d>;\n" alloc.u32 ;
  if alloc.u64 > 0 then Printf.bprintf buf "    .reg .u64 %%rd<%d>;\n" alloc.u64 ;
  if alloc.f32 > 0 then Printf.bprintf buf "    .reg .f32 %%f<%d>;\n" alloc.f32 ;
  if alloc.f64 > 0 then Printf.bprintf buf "    .reg .f64 %%fd<%d>;\n" alloc.f64 ;
  if alloc.pred > 0 then
    Printf.bprintf buf "    .reg .pred %%p<%d>;\n" alloc.pred

(** {1 Top-level kernel generator} *)

(** Generate the PTX file header.
    @param sm_target
      SM architecture string, e.g. ["sm_86"] for Ampere or ["sm_61"] for Pascal.
      Defaults to ["sm_86"] (RTX 30xx / A100+).
    @param ptx_version PTX language version. Defaults to ["8.0"] (CUDA 11.8+).
*)
let make_ptx_header ?(sm_target = "sm_86") ?(ptx_version = "8.0") () =
  Printf.sprintf
    ".version %s\n.target %s\n.address_size 64\n"
    ptx_version
    sm_target

(** Generate PTX for a single kernel. Three-phase: (1) emit body to count
    registers, (2) build header with correct register counts, (3) concatenate.
    @param sm_target Override the default [sm_86] target for older hardware. *)
let generate ?(sm_target = "sm_86") (k : kernel) : string =
  let alloc = make_alloc () in
  List.iter
    (fun hf ->
      (* The __sarek_ prefix is reserved for emitter-internal helpers (e.g.
         the f64 softmath family); a user helper with such a name would be
         silently clobbered by the on-demand registration in emit_intrinsic. *)
      if String.length hf.hf_name >= 8 && String.sub hf.hf_name 0 8 = "__sarek_"
      then
        unsupported
          ("helper '" ^ hf.hf_name
         ^ "': the '__sarek_' name prefix is reserved for emitter-internal \
            helpers; rename the function") ;
      Hashtbl.replace alloc.funcs hf.hf_name hf)
    k.kern_funcs ;
  List.iter
    (fun (name, ctors) -> Hashtbl.replace alloc.variant_decls name ctors)
    k.kern_variants ;
  let env = make_env () in
  let body_buf = Buffer.create 2048 in
  let shared_buf = Buffer.create 256 in
  let param_str = emit_params body_buf alloc env k.kern_params in
  emit_locals body_buf shared_buf alloc env k.kern_locals ;
  emit_stmt body_buf alloc env k.kern_body ;
  Buffer.add_string body_buf "    ret;\n" ;
  let out = Buffer.create 4096 in
  Buffer.add_string out (make_ptx_header ~sm_target ()) ;
  Buffer.add_char out '\n' ;
  Printf.bprintf out ".entry %s(\n" k.kern_name ;
  Buffer.add_string out param_str ;
  Buffer.add_string out "\n)\n{\n" ;
  emit_reg_decls out alloc ;
  Buffer.add_buffer out shared_buf ;
  (* Shared arrays declared mid-body (SLet-bound let%shared). *)
  Buffer.add_buffer out alloc.shared_decls ;
  Buffer.add_char out '\n' ;
  Buffer.add_buffer out body_buf ;
  Buffer.add_string out "}\n" ;
  Buffer.contents out

(** Same interface as [Sarek_ir_cuda.generate_with_types]. Record and variant
    type definitions are not representable as PTX struct types; this is a
    documented design gap in ptx-spike-findings.md. *)
let generate_with_types ~types:_ (k : kernel) : string = generate k
