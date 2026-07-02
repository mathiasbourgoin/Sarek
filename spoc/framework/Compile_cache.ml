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

(* Each free-form component is digested on its own before being joined, so the
   ':'-separated fields below are always exactly 32 hex characters wide. That
   makes the join unambiguous regardless of what bytes appear inside [device],
   [name], or the canonicalized options string (e.g. a device name containing
   ':', or an option value containing ',' or '='). Concatenating the raw
   strings directly (the previous scheme) let two distinct (device, name,
   options) triples collapse to the same key, e.g. device="a:b" ^ name="c"
   digesting identically to device="a" ^ name="b:c". *)
let digest_component s = Digest.string s |> Digest.to_hex

let make_key ~device ~name ~source ?(options = []) () =
  let source_digest = digest_component source in
  let opts_digest = digest_component (canonicalize_options options) in
  Printf.sprintf
    "%s:%s:%s:%s"
    (digest_component device)
    (digest_component name)
    source_digest
    opts_digest
