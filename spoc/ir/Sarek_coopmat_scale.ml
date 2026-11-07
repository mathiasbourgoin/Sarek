(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types

let scales op =
  match op with
  | CM_scale_add {row; col; offset; _} -> (
      (row :: Option.to_list col)
      @ match offset with None -> [] | Some o -> [o.off_row; o.off_col])
  | CM_decl _ | CM_load _ | CM_store _ | CM_muladd _ -> []

let buffers op = List.map (fun s -> s.scale_buf) (scales op)

let exprs op = List.map (fun s -> s.scale_index) (scales op)

(* Every [let] is explicit so the scales are visited in the documented order
   ([row], [col], [off_row], [off_col]): OCaml leaves the evaluation order of
   record fields unspecified, and a caller that hoists side computations out of
   the index expressions depends on the order. *)
let map ~buf ~expr op =
  let one s =
    let scale_buf = buf s.scale_buf in
    let scale_index = expr s.scale_index in
    {scale_buf; scale_index}
  in
  match op with
  | CM_scale_add r ->
      let row = one r.row in
      let col = Option.map one r.col in
      let offset =
        Option.map
          (fun o ->
            let off_row = one o.off_row in
            let off_col = one o.off_col in
            {off_row; off_col})
          r.offset
      in
      CM_scale_add {r with row; col; offset}
  | CM_decl _ | CM_load _ | CM_store _ | CM_muladd _ -> op

let map_exprs f op = map ~buf:Fun.id ~expr:f op

let fold_exprs f acc op = List.fold_left f acc (exprs op)
