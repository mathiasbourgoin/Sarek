(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The single spelling of "turn a float back into OCaml source".

    Every PPX site that reconstructs a float constant goes through here. There
    were three, byte-identical and independently maintained, and they all used
    [string_of_float] — which is ["%.12g"], so [3.14159265358979312] came back
    as [3.14159265359], losing about eleven bits. That is the whole point of the
    DSL's [G] float64 suffix defeated silently, and the existing regression
    guard could not see it because its literals (0.0G, 2.0G, 4.0G) are all exact
    in twelve digits.

    Two things this function owes its callers, and the reason it exists rather
    than being three copies:

    - ["%.17g"] round-trips binary64 exactly, where ["%.12g"] does not:
      [float_of_string (sprintf "%.17g" x) = x] holds for every finite [x];
    - it repairs the decimal point. ["%.17g" 3.0] is ["3"], where
      [string_of_float] gave ["3."]. Fed to [Ast_builder.efloat] that builds
      [Pconst_float ("3", None)], which types correctly in-process but renders
      as a bare [3] — an int literal in a float position — if the generated AST
      is ever printed and re-read ([-dsource], a source-emitting driver). Every
      backend float formatter in this tree already applies this repair
      (Sarek_ir_cuda.ml, Sarek_ir_glsl.ml, Sarek_ir_wgsl.ml, Sarek_ir_metal.ml);
      the PPX sites were the only ones without it.

    Non-finite values still format as ["inf"]/["nan"], which are not valid OCaml
    literals either — unchanged from [string_of_float], not a regression, and
    not reachable from the DSL's literal syntax. *)
let to_source (f : float) : string =
  let s = Printf.sprintf "%.17g" f in
  if String.contains s '.' || String.contains s 'e' || String.contains s 'E'
  then s
  else if String.contains s 'n' || String.contains s 'i' then s (* nan, inf *)
  else s ^ "."
