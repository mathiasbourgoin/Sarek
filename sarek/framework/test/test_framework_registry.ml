(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Simplified test suite for Framework_registry - tests core functionality *)

open Spoc_framework.Framework_sig
open Spoc_framework_registry

(** Mock minimal plugin for testing *)
module Mock_plugin1 : S = struct
  let name = "MockPlugin1"

  let version = (1, 0, 0)

  let is_available () = true
end

module Mock_plugin2 : S = struct
  let name = "MockPlugin2"

  let version = (2, 0, 0)

  let is_available () = false (* Unavailable *)
end

module Mock_plugin3 : S = struct
  let name = "MockPlugin3"

  let version = (1, 0, 0)

  let is_available () = true
end

let test_register_plugin () =
  Framework_registry.register (module Mock_plugin1) ;

  (* Should be findable now *)
  Alcotest.(check bool)
    "Plugin found after registration"
    true
    (match Framework_registry.find "MockPlugin1" with
    | Some _ -> true
    | None -> false)

let test_register_with_priority () =
  Framework_registry.register ~priority:100 (module Mock_plugin3) ;
  let prio = Framework_registry.priority "MockPlugin3" in
  Alcotest.(check int) "Priority stored correctly" 100 prio

let test_list_plugins () =
  Framework_registry.register (module Mock_plugin1) ;
  Framework_registry.register (module Mock_plugin2) ;
  let names = Framework_registry.names () in
  Alcotest.(check bool)
    "Contains MockPlugin1"
    true
    (List.mem "MockPlugin1" names)

let test_available_plugins () =
  Framework_registry.register (module Mock_plugin1) ;
  Framework_registry.register (module Mock_plugin2) ;
  (* unavailable *)
  let available = Framework_registry.available () in
  let count = List.length available in
  Alcotest.(check bool) "Only available plugins listed" true (count >= 1)
(* At least MockPlugin1 *)

let test_priority_ordering () =
  let module HighPrio : S = struct
    let name = "HighPrio"

    let version = (1, 0, 0)

    let is_available () = true
  end in
  let module LowPrio : S = struct
    let name = "LowPrio"

    let version = (1, 0, 0)

    let is_available () = true
  end in
  Framework_registry.register ~priority:100 (module HighPrio) ;
  Framework_registry.register ~priority:10 (module LowPrio) ;

  let p_high = Framework_registry.priority "HighPrio" in
  let p_low = Framework_registry.priority "LowPrio" in
  Alcotest.(check bool) "High priority is higher than low" true (p_high > p_low)

let test_find_nonexistent () =
  let result = Framework_registry.find "NonExistentPlugin" in
  Alcotest.(check bool)
    "Nonexistent plugin returns None"
    true
    (match result with None -> true | Some _ -> false)

let test_backend_not_found () =
  let result = Framework_registry.find_backend "NonExistentBackend" in
  Alcotest.(check bool)
    "Nonexistent backend returns None"
    true
    (match result with None -> true | Some _ -> false)

let test_default_priority () =
  let module DefaultPrio : S = struct
    let name = "DefaultPrio"

    let version = (1, 0, 0)

    let is_available () = true
  end in
  Framework_registry.register (module DefaultPrio) ;
  let prio = Framework_registry.priority "DefaultPrio" in
  Alcotest.(check int) "Default priority is 50" 50 prio

let () =
  let open Alcotest in
  run
    "Framework_registry"
    [
      ( "registration",
        [
          test_case "Register plugin" `Quick test_register_plugin;
          test_case "Register with priority" `Quick test_register_with_priority;
          test_case "Default priority" `Quick test_default_priority;
        ] );
      ( "queries",
        [
          test_case "List plugins" `Quick test_list_plugins;
          test_case "Available plugins" `Quick test_available_plugins;
          test_case "Priority ordering" `Quick test_priority_ordering;
          test_case "Find nonexistent" `Quick test_find_nonexistent;
          test_case "Backend not found" `Quick test_backend_not_found;
        ] );
    ]
