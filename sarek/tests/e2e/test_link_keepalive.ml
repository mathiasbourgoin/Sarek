(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Link_keepalive on the devices named on the command line ("Framework" or
   "Framework:index"); the runtest rule names the CPU backends, where it is a
   no-op, so a shared GPU host is never touched by accident. *)

module Device = Spoc_core.Device

let selected (devices : Device.t list) : Device.t list =
  let wanted = List.tl (Array.to_list Sys.argv) in
  let wanted = if wanted = [] then ["Native"; "Interpreter"] else wanted in
  let index = Hashtbl.create 4 in
  List.filter
    (fun (d : Device.t) ->
      let i =
        Option.value ~default:0 (Hashtbl.find_opt index d.Device.framework)
      in
      Hashtbl.replace index d.Device.framework (i + 1) ;
      List.exists
        (fun w ->
          w = d.Device.framework
          || w = Printf.sprintf "%s:%d" d.Device.framework i)
        wanted)
    devices

let () =
  Test_helpers.Benchmarks.init_backends () ;
  let devices = Array.to_list (Device.init ()) in
  List.iter Link_keepalive_suite.run_on (selected devices) ;
  exit (Link_keepalive_suite.exit_code ())
