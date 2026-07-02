(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Kernel_args - Indexed kernel-argument container
 *
 * Shared implementation of "store a value at idx, last write wins" plus
 * strict pre-launch validation, so backend plugins stop hand-rolling
 * accumulate-by-call-order argument lists (see module doc in the .mli).
 ******************************************************************************)

type 'a t = {slots : (int, 'a) Hashtbl.t}

let create () = {slots = Hashtbl.create 8}

let set t idx v = Hashtbl.replace t.slots idx v

let count t = Hashtbl.length t.slots

let to_sorted_list t =
  Hashtbl.fold (fun idx v acc -> (idx, v) :: acc) t.slots []
  |> List.sort (fun (a, _) (b, _) -> compare a b)

let describe_indices idxs = String.concat ", " (List.map string_of_int idxs)

let plural n = if n = 1 then "" else "s"

let validate_and_extract t ~expected_count =
  let missing = ref [] in
  for i = expected_count - 1 downto 0 do
    if not (Hashtbl.mem t.slots i) then missing := i :: !missing
  done ;
  let extra =
    Hashtbl.fold
      (fun idx _ acc ->
        if idx < 0 || idx >= expected_count then idx :: acc else acc)
      t.slots
      []
    |> List.sort compare
  in
  match (!missing, extra) with
  | [], [] -> Ok (Array.init expected_count (fun i -> Hashtbl.find t.slots i))
  | missing, [] ->
      Error
        (Printf.sprintf
           "missing indices: [%s]; expected %d args, got %d"
           (describe_indices missing)
           expected_count
           (Hashtbl.length t.slots))
  | [], extra ->
      Error
        (Printf.sprintf
           "unexpected index%s: [%s]; expected contiguous 0..%d"
           (plural (List.length extra))
           (describe_indices extra)
           (expected_count - 1))
  | missing, extra ->
      Error
        (Printf.sprintf
           "missing indices: [%s]; unexpected index%s: [%s]; expected %d args, \
            got %d"
           (describe_indices missing)
           (plural (List.length extra))
           (describe_indices extra)
           expected_count
           (Hashtbl.length t.slots))
