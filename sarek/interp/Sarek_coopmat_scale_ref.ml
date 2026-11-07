(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

let round x = Int32.float_of_bits (Int32.bits_of_float x)

let of_int32 i = round (Int32.to_float i)

(* One [let] per step, in the order of the design document: the order is the
   semantics, and each step is rounded on its own (no step is fused with the
   next). *)
let element ~accumulate ~dst ~src ~row ~col ~offset =
  let t2 = round (src *. row) in
  let t3 = match col with None -> t2 | Some c -> round (t2 *. c) in
  let t4 = if accumulate then round (dst +. t3) else t3 in
  match offset with
  | None -> t4
  | Some (p, q) ->
      let t5 = round (p *. q) in
      round (t4 -. t5)

type scales = {
  row : float array;
  col : float array option;
  offset : (float array * float array) option;
}

let apply ~m ~n ~accumulate ~scales ~dst ~src =
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      element
        ~accumulate
        ~dst:dst.(idx)
        ~src:src.(idx)
        ~row:scales.row.(i)
        ~col:(Option.map (fun c -> c.(j)) scales.col)
        ~offset:(Option.map (fun (p, q) -> (p.(i), q.(j))) scales.offset))
