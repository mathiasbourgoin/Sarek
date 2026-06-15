(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Tail Recursion Analysis and Elimination
 *
 * This module detects tail-recursive functions and transforms them into
 * loops for GPU execution. Non-tail recursion can be inlined using
 * pragma ["sarek.inline N"].
 *
 * Pragmas:
 * - pragma ["unroll N"] body: For tail recursion transformed to loops.
 *   Passes #pragma unroll N to the GPU compiler.
 * - pragma ["sarek.inline N"] body: For non-tail recursion.
 *   PPX mechanically inlines N times, then verifies no calls remain.
 ******************************************************************************)

open Sarek_typed_ast

(** Re-export analysis functions for use by other modules *)
module Analysis = Sarek_tailrec_analysis

(** Re-export commonly used analysis functions *)
let count_recursive_calls = Sarek_tailrec_analysis.count_recursive_calls

let is_tail_recursive = Sarek_tailrec_analysis.is_tail_recursive

let analyze_recursion = Sarek_tailrec_analysis.analyze_recursion

(** {1 Kernel-Level Pass} *)

(** Maximum inline depth for bounded recursion. Beyond this, we refuse to
    compile as it would generate too much code. *)
let max_inline_limit = 16

(** Extract pragma options and inner body from a function body. Returns (Some
    opts, inner_body) if body is TEPragma, (None, body) otherwise. *)
let extract_pragma (body : texpr) : string list option * texpr =
  match body.te with
  | TEPragma (opts, inner) -> (Some opts, inner)
  | _ -> (None, body)

(** Transform all recursive module functions in a kernel:
    - Tail-recursive functions are transformed to loops
    - Non-tail recursion with pragma ["sarek.inline N"] is inlined N times
    - Validates pragma usage: unroll for tail, sarek.inline for non-tail

    This pass is run after type checking and before lowering to Kirc. *)
let transform_kernel (kernel : tkernel) : tkernel =
  Sarek_debug.log_enter "transform_kernel" ;
  let new_items =
    List.map
      (function
        | TMFun (name, is_rec, params, body) as orig -> (
            Sarek_debug.log "processing TMFun '%s'" name ;
            let pragma_opts, inner_body = extract_pragma body in
            let info =
              Sarek_tailrec_analysis.analyze_recursion name inner_body
            in
            if info.ri_call_count = 0 then
              (* Not recursive, leave as-is *)
              orig
            else if info.ri_is_tail then (
              (* Tail recursive - transform to loop *)
              (* Check for invalid pragma *)
              (match pragma_opts with
              | Some opts
                when Option.is_some
                       (Sarek_tailrec_pragma.parse_sarek_inline_pragma opts) ->
                  Format.eprintf
                    "Sarek error: function '%s' is tail-recursive.@."
                    name ;
                  Format.eprintf
                    "  Use 'pragma [\"unroll N\"]' instead of 'sarek.inline'.@." ;
                  failwith "Cannot use sarek.inline on tail-recursive function"
              | _ -> ()) ;
              if Sarek_debug.enabled then
                Format.eprintf
                  "Sarek: transforming tail-recursive function '%s' to loop@."
                  name ;
              let new_body =
                Sarek_tailrec_elim.eliminate_tail_recursion
                  name
                  params
                  inner_body
                  inner_body.te_loc
              in
              (* Preserve pragma wrapper if present (for unroll) *)
              let final_body =
                match pragma_opts with
                | Some opts -> {body with te = TEPragma (opts, new_body)}
                | None -> new_body
              in
              TMFun (name, is_rec, params, final_body))
            else
              (* Non-tail recursion *)
              (* Check for pragma *)
              match pragma_opts with
              | Some opts when Sarek_tailrec_pragma.is_unroll_pragma opts ->
                  Format.eprintf
                    "Sarek error: function '%s' is NOT tail-recursive.@."
                    name ;
                  Format.eprintf
                    "  Cannot use 'unroll' on non-tail recursion.@." ;
                  Format.eprintf
                    "  Use 'pragma [\"sarek.inline N\"]' to inline, or rewrite \
                     as tail-recursive.@." ;
                  failwith "Cannot use unroll on non-tail-recursive function"
              | Some opts -> (
                  match Sarek_tailrec_pragma.parse_sarek_inline_pragma opts with
                  | Some depth -> (
                      if Sarek_debug.enabled then
                        Format.eprintf
                          "Sarek: inlining recursive function '%s' %d times@."
                          name
                          depth ;
                      match
                        Sarek_tailrec_pragma.inline_with_pragma
                          name
                          params
                          inner_body
                          depth
                      with
                      | Ok new_body ->
                          if Sarek_debug.enabled then
                            Format.eprintf
                              "Sarek: successfully inlined '%s' (%d nodes)@."
                              name
                              (Sarek_tailrec_pragma.count_nodes new_body) ;
                          TMFun (name, is_rec, params, new_body)
                      | Error msg ->
                          Format.eprintf "Sarek error in '%s': %s@." name msg ;
                          failwith msg)
                  | None ->
                      (* Unknown pragma, warn and leave as-is *)
                      Format.eprintf
                        "Sarek warning: function '%s' is recursive but not \
                         tail-recursive@."
                        name ;
                      Format.eprintf
                        "  Use 'pragma [\"sarek.inline N\"]' to inline.@." ;
                      orig)
              | None ->
                  (* No pragma, unbounded non-tail recursion - error *)
                  Format.eprintf
                    "Sarek error: function '%s' is recursive but not \
                     tail-recursive@."
                    name ;
                  Format.eprintf
                    "  (recursive calls: %d, in loops: %b)@."
                    info.ri_call_count
                    info.ri_in_loop ;
                  Format.eprintf
                    "  Use 'pragma [\"sarek.inline N\"]' to inline, or rewrite \
                     as tail-recursive.@." ;
                  failwith
                    (Printf.sprintf
                       "Non-tail recursion in '%s' requires pragma \
                        [\"sarek.inline N\"]"
                       name))
        | item -> item)
      kernel.tkern_module_items
  in
  {kernel with tkern_module_items = new_items}
