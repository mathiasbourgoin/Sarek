(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_codegen - Shared code-generation helpers for GPU backends
 *
 * Hosts logic that was previously duplicated verbatim across the CUDA, OpenCL,
 * Metal, and Vulkan IR generators: type-name mangling and variant type emission.
 ******************************************************************************)

(** Mangle OCaml type name to valid C/GLSL identifier (e.g., "Module.point" ->
    "Module_point") *)
let mangle_name name = String.map (fun c -> if c = '.' then '_' else c) name

(** Emit a C/MSL tagged-union variant type (enum + typedef struct + union +
    inline constructors). [type_of_elttype] and [constructor_prefix] are the
    only backend-specific inputs. *)
let gen_variant_def ~type_of_elttype ~constructor_prefix buf (name, constrs) =
  let mangled = mangle_name name in
  (* Enum for tags - use simple names for switch case labels *)
  Buffer.add_string buf "enum { " ;
  List.iteri
    (fun i (cname, _) ->
      if i > 0 then Buffer.add_string buf ", " ;
      Buffer.add_string buf cname ;
      Buffer.add_string buf " = " ;
      Buffer.add_string buf (string_of_int i))
    constrs ;
  Buffer.add_string buf " };\n" ;
  (* Struct with tag and union *)
  Buffer.add_string buf "typedef struct {\n  int tag;\n" ;
  (* Generate union if any constructor has payload *)
  let has_payload = List.exists (fun (_, args) -> args <> []) constrs in
  if has_payload then begin
    Buffer.add_string buf "  union {\n" ;
    List.iter
      (fun (cname, args) ->
        match args with
        | [] -> () (* No payload for this constructor *)
        | [ty] ->
            Buffer.add_string buf "    " ;
            Buffer.add_string buf (type_of_elttype ty) ;
            Buffer.add_string buf (" " ^ cname ^ "_v;\n")
        | _ ->
            (* Multiple args - generate struct *)
            Buffer.add_string buf "    struct { " ;
            List.iteri
              (fun i ty ->
                if i > 0 then Buffer.add_string buf " " ;
                Buffer.add_string buf (type_of_elttype ty) ;
                Buffer.add_string buf (Printf.sprintf " _%d;" i))
              args ;
            Buffer.add_string buf (" } " ^ cname ^ "_v;\n"))
      constrs ;
    Buffer.add_string buf "  } data;\n"
  end ;
  Buffer.add_string buf ("} " ^ mangled ^ ";\n\n") ;
  (* Constructor functions *)
  List.iteri
    (fun _i (cname, args) ->
      Buffer.add_string
        buf
        (constructor_prefix ^ " " ^ mangled ^ " make_" ^ mangled ^ "_" ^ cname
       ^ "(") ;
      (match args with
      | [] -> ()
      | [ty] ->
          Buffer.add_string buf (type_of_elttype ty) ;
          Buffer.add_string buf " v"
      | _ ->
          List.iteri
            (fun j ty ->
              if j > 0 then Buffer.add_string buf ", " ;
              Buffer.add_string buf (type_of_elttype ty) ;
              Buffer.add_string buf (Printf.sprintf " v%d" j))
            args) ;
      Buffer.add_string buf (") {\n  " ^ mangled ^ " r;\n") ;
      Buffer.add_string buf ("  r.tag = " ^ cname ^ ";\n") ;
      (match args with
      | [] -> ()
      | [_] -> Buffer.add_string buf ("  r.data." ^ cname ^ "_v = v;\n")
      | _ ->
          List.iteri
            (fun j _ ->
              Buffer.add_string
                buf
                (Printf.sprintf "  r.data.%s_v._%d = v%d;\n" cname j j))
            args) ;
      Buffer.add_string buf "  return r;\n}\n\n")
    constrs

(** Emit a GLSL variant type. GLSL lacks enum/typedef/union, so tags are
    const-int declarations, the type is a bare struct with flat payload fields,
    and constructors have no qualifier prefix. *)
let gen_variant_def_glsl ~type_of_elttype buf (name, constrs) =
  let mangled = mangle_name name in
  (* Enum constants *)
  List.iteri
    (fun i (cname, _) ->
      Buffer.add_string buf (Printf.sprintf "const int %s = %d;\n" cname i))
    constrs ;
  Buffer.add_char buf '\n' ;
  (* Struct with tag and union-like data *)
  Buffer.add_string buf (Printf.sprintf "struct %s {\n  int tag;\n" mangled) ;
  let has_payload = List.exists (fun (_, args) -> args <> []) constrs in
  if has_payload then begin
    (* GLSL doesn't have unions, so we use the largest payload type *)
    List.iter
      (fun (cname, args) ->
        match args with
        | [] -> ()
        | [ty] ->
            Buffer.add_string
              buf
              (Printf.sprintf "  %s %s_v;\n" (type_of_elttype ty) cname)
        | _ ->
            (* Multiple args - generate struct *)
            Buffer.add_string buf (Printf.sprintf "  struct { ") ;
            List.iteri
              (fun i ty ->
                if i > 0 then Buffer.add_string buf " " ;
                Buffer.add_string
                  buf
                  (Printf.sprintf "%s _%d;" (type_of_elttype ty) i))
              args ;
            Buffer.add_string buf (Printf.sprintf " } %s_v;\n" cname))
      constrs
  end ;
  Buffer.add_string buf "};\n\n" ;
  (* Constructor functions *)
  List.iteri
    (fun _i (cname, args) ->
      Buffer.add_string
        buf
        (Printf.sprintf "%s make_%s_%s(" mangled mangled cname) ;
      (match args with
      | [] -> ()
      | [ty] ->
          Buffer.add_string buf (type_of_elttype ty) ;
          Buffer.add_string buf " v"
      | _ ->
          List.iteri
            (fun j ty ->
              if j > 0 then Buffer.add_string buf ", " ;
              Buffer.add_string buf (type_of_elttype ty) ;
              Buffer.add_string buf (Printf.sprintf " v%d" j))
            args) ;
      Buffer.add_string buf ") {\n" ;
      Buffer.add_string buf (Printf.sprintf "  %s r;\n" mangled) ;
      Buffer.add_string buf (Printf.sprintf "  r.tag = %s;\n" cname) ;
      (match args with
      | [] -> ()
      | [_] -> Buffer.add_string buf (Printf.sprintf "  r.%s_v = v;\n" cname)
      | _ ->
          List.iteri
            (fun j _ ->
              Buffer.add_string
                buf
                (Printf.sprintf "  r.%s_v._%d = v%d;\n" cname j j))
            args) ;
      Buffer.add_string buf "  return r;\n}\n\n")
    constrs
