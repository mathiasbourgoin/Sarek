(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Debug Logging
 *
 * Gated by SAREK_DEBUG environment variable. When not set, all logging
 * functions are no-ops with no performance impact.
 ******************************************************************************)

let enabled =
  match Sys.getenv_opt "SAREK_PPX_DEBUG" with
  | Some "1" | Some "true" | Some "yes" -> true
  | _ -> (
      (* Backward-compatible fallback to SAREK_DEBUG as a boolean *)
      match Sys.getenv_opt "SAREK_DEBUG" with
      | Some "1" | Some "true" | Some "yes" -> true
      | _ -> false)

let () = if enabled then Format.eprintf "SAREK DEBUG: Debug logging enabled@.%!"

let log fmt =
  if enabled then
    Format.kasprintf (fun s -> Format.eprintf "SAREK DEBUG: %s@.%!" s) fmt
  else Format.ikfprintf (fun _ -> ()) Format.err_formatter fmt

let log_enter name =
  if enabled then Format.eprintf "SAREK DEBUG: >>> %s@.%!" name

let log_exit name =
  if enabled then Format.eprintf "SAREK DEBUG: <<< %s@.%!" name

let log_int name v =
  if enabled then Format.eprintf "SAREK DEBUG: %s = %d@.%!" name v

let log_string name v =
  if enabled then Format.eprintf "SAREK DEBUG: %s = %s@.%!" name v

(** Log to file (bypasses dune's output capture) - only when SAREK_DEBUG is
    enabled *)
let log_to_file msg =
  if enabled then begin
    let log_file =
      open_out_gen
        [Open_creat; Open_append; Open_text]
        0o644
        "/tmp/sarek_ppx.log"
    in
    Printf.fprintf log_file "%s\n%!" msg ;
    close_out log_file
  end
