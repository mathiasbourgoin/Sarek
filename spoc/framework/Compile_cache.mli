(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Shared compile-cache key builder for GPU backend plugins.

    Every backend (CUDA, OpenCL, Metal, Vulkan) memoizes compiled kernels keyed
    by some combination of device identity and kernel source. A source file
    frequently defines more than one kernel entry point; if the cache key omits
    the kernel/entry name, the second kernel compiled from a shared source
    silently resolves to whatever was compiled first under the same key. This
    module standardizes the key shape so every backend's key includes: device
    identifier, kernel/entry name, a digest of the source, and canonicalized
    (sorted) compile options.

    This module intentionally does not force backends to use a shared [Hashtbl]
    — each backend keeps its own cache table (they differ in what they store:
    resolved kernel handles, SPIR-V bytes, etc.). The contract this module
    enforces is only the key *shape*. *)

(** [make_key ~device ~name ~source ?options ()] builds a standardized cache key
    string of the form ["<device>:<name>:<source-digest>[:<opt>=<val>,...]"].

    - [device] should be a stable device identifier (e.g. device index, device
      name, or name+driver-version string — whatever uniquely identifies the
      compilation target for that backend).
    - [name] is the kernel/entry-point name. Required so that two kernels
      compiled from the same [source] never collide under the same key.
    - [source] is digested with {!Digest.string}; the raw source is never
      embedded in the key.
    - [options] is an optional association list of compile options (e.g.
      optimization flags). Entries are sorted by key before being folded into
      the digest so key order never affects cache hits. Defaults to [[]]. *)
val make_key :
  device:string ->
  name:string ->
  source:string ->
  ?options:(string * string) list ->
  unit ->
  string
