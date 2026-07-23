(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test type declaration that should FAIL to compile:
   - [my_meters] is not a [@@sarek.type]-registered type and not a scalar, so
     using it as a GPU record field must be a hard PPX error. Before the fix
     (audit finding M6) the PPX silently assumed size/align 4/4, which
     desynchronizes the host byte layout from the device's aligned ABI. *)

type my_meters = float

type bad_record = {distance : my_meters; count : int32} [@@sarek.type]

let () =
  ignore (fun (r : bad_record) -> r.count) ;
  print_endline "This should not print - test should have failed to compile"
