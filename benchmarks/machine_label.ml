(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The machine label's SHAPE, and the rules for accepting an operator override.
 *
 * Split out of system_info.ml so it depends on Stdlib alone: the red-path
 * harness scripts/check-no-machine-identifiers.test.sh runs this module
 * directly (`ocaml` + #use) against the very same case table it runs the gate's
 * bash regex against, and fails if the two disagree. That executed comparison
 * is what keeps the two implementations of one rule from drifting -- a comment
 * saying "these must agree" does not.
 *
 * WHY (backlog-168, then backlog-168b): the label replaced the hostname as the
 * dedup key and as the leading component of every result filename. Two boxes
 * with the same OS and GPU vendor derive an IDENTICAL label and their runs
 * merge in the dedup key, so SAREK_BENCH_MACHINE exists to tell them apart.
 * But the commit gate's filename allowlist had no suffix in it, so a
 * disambiguating label produced files that could not be committed: the override
 * worked everywhere except for the one purpose it exists for. Hence the
 * bounded suffix below, and hence this module validating the override against
 * the same shape the gate enforces -- an operator must not be able to set a
 * label the gate will refuse hours later, at commit time.
 ******************************************************************************)

(* MUST equal MACHINE_LABEL_SHAPE in scripts/machine-label-shape.sh, which is
   the prose authority for this rule and explains the bound. Restated here
   because OCaml cannot source a shell file; kept honest by the harness, not by
   this comment. *)
let shape_doc =
  "^(linux|darwin|windows)-(nvidia|amd|intel|apple|unknown)(-[a-z0-9]{1,8})?$"

(* Enumerated, not free-form. This is what keeps a bare hostname out: no
   widening of the suffix can give `drangleic` an <os>-<vendor> prefix. *)
let os_tokens = ["linux"; "darwin"; "windows"]

let vendor_tokens = ["nvidia"; "amd"; "intel"; "apple"; "unknown"]

let suffix_max_len = 8

let is_suffix_char c = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')

let is_legal_suffix s =
  let n = String.length s in
  n >= 1 && n <= suffix_max_len && String.for_all is_suffix_char s

(* Splitting on '-' is equivalent to the ERE above, not an approximation of it:
   every os and vendor token is dash-free, and the suffix character class
   excludes '-', so a legal label has exactly two or three dash-separated
   fields. Three fields with a repeated suffix (`linux-amd-a-b`) yields four and
   is refused, which is the intent -- one segment, so the bounded field cannot
   grow by repetition. *)
let is_wellformed label =
  match String.split_on_char '-' label with
  | [os; vendor] -> List.mem os os_tokens && List.mem vendor vendor_tokens
  | [os; vendor; suffix] ->
      List.mem os os_tokens
      && List.mem vendor vendor_tokens
      && is_legal_suffix suffix
  | _ -> false

let env_var = "SAREK_BENCH_MACHINE"

let norm s = String.lowercase_ascii (String.trim s)

(* [resolve ~derived ~override ~hostname] is the whole override policy, pure so
   it can be tested without an environment or a subprocess.

   [hostname] is a thunk: reading it is the one sanctioned use of the machine's
   hostname (backlog-168), and it must not be read at all when there is no
   override to compare it against.

   ORDER IS LOAD-BEARING. The hostname refusal comes FIRST and stays a hard
   [failwith]. A hostname that happens to have a legal shape is conceivable
   (`linux-amd` is a legal shape and could be somebody's hostname), so checking
   the shape first would let that one through with no complaint. An opt-in
   escape hatch that silently reintroduced the leak would be worse than no
   escape hatch, so neither check is a warning. *)
let resolve ~derived ~override ~hostname =
  match override with
  | None | Some "" -> derived
  | Some override ->
      if norm override = norm (hostname ()) then
        failwith
          (Printf.sprintf
             "%s is set to the machine's hostname. That is the identifier \
              benchmark output must not carry (backlog-168). Choose an opaque \
              label such as %S."
             env_var
             derived)
      else
        let label = String.trim override in
        if not (is_wellformed label) then
          failwith
            (Printf.sprintf
               "%s=%S does not have the machine-label shape %s. It would \
                produce result filenames that \
                scripts/check-no-machine-identifiers.sh refuses to commit. Use \
                the derived label %S, or that label plus a short suffix (1-%d \
                chars of [a-z0-9]) to tell two same-hardware machines apart, \
                e.g. %S."
               env_var
               label
               shape_doc
               derived
               suffix_max_len
               (derived ^ "-b"))
        else label
