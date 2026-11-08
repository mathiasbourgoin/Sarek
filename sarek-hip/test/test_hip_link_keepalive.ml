(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Link_keepalive on HIP hardware: first HIP device only (an iGPU must not be
   stressed); a command-line argument gives another index. Self-skips without a
   HIP device. *)

module Device = Spoc_core.Device

let () = Sarek_hip.Hip_plugin.register ()

let () =
  let devices = Device.init () in
  let hip =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  let index =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 0
  in
  match List.nth_opt hip index with
  | None ->
      print_endline "test_hip_link_keepalive: [SKIP] no such HIP device" ;
      exit 0
  | Some dev ->
      Link_keepalive_suite.run_on dev ;
      exit (Link_keepalive_suite.exit_code ())
