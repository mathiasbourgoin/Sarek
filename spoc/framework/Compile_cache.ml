(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Compile_cache - Shared compile-cache key builder
 *
 * See Compile_cache.mli for the rationale: every backend's cache key must
 * include the kernel/entry name, not just device + source digest, or a
 * second kernel compiled from the same source silently aliases the first.
 ******************************************************************************)

let canonicalize_options options =
  options
  |> List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2)
  |> List.map (fun (k, v) -> k ^ "=" ^ v)
  |> String.concat ","

let make_key ~device ~name ~source ?(options = []) () =
  let source_digest = Digest.string source |> Digest.to_hex in
  let opts_part = canonicalize_options options in
  if opts_part = "" then Printf.sprintf "%s:%s:%s" device name source_digest
  else Printf.sprintf "%s:%s:%s:%s" device name source_digest opts_part
