(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cache_hooks tests - the cross-layer teardown back-channel.
 *
 * The load-bearing property is that a device-destroy notification carries the
 * BACKEND FAMILY as well as the backend-local index. Backend indices collide
 * across backends (OpenCL 0, Vulkan 0, HIP 0 all exist on the same machine), so
 * a listener matching on the index alone invalidates unrelated backends'
 * entries - which stopped being harmless once eviction gained the power to
 * abort in-flight builds.
 ******************************************************************************)

module CH = Spoc_framework.Cache_hooks

let test_device_destroy_carries_backend_and_index () =
  let seen = ref [] in
  CH.on_device_destroy (fun ~backend index -> seen := (backend, index) :: !seen) ;
  CH.notify_device_destroy ~backend:"HIP" 0 ;
  CH.notify_device_destroy ~backend:"CUDA" 3 ;
  Alcotest.(check (list (pair string int)))
    "listener receives (backend family, backend-local index)"
    [("HIP", 0); ("CUDA", 3)]
    (List.rev !seen)

let test_all_listeners_run_and_first_failure_is_reported () =
  let ran = ref [] in
  CH.on_clear_all (fun () -> ran := "c" :: !ran) ;
  CH.on_clear_all (fun () ->
      ran := "b" :: !ran ;
      failwith "listener b failed") ;
  CH.on_clear_all (fun () -> ran := "a" :: !ran) ;
  let raised =
    try
      CH.notify_clear_all () ;
      false
    with Failure _ -> true
  in
  Alcotest.(check bool) "the failing listener is reported" true raised ;
  (* Order across registrations is unspecified; what matters is that a failure
     in one did not skip the others - every layer must be dropped. *)
  Alcotest.(check int)
    "every listener ran despite the failure"
    3
    (List.length !ran) ;
  List.iter
    (fun tag ->
      Alcotest.(check bool)
        (Printf.sprintf "listener %s ran" tag)
        true
        (List.mem tag !ran))
    ["a"; "b"; "c"]

let () =
  Alcotest.run
    "Cache_hooks"
    [
      ( "notification",
        [
          Alcotest.test_case
            "device destroy carries backend family + index"
            `Quick
            test_device_destroy_carries_backend_and_index;
          Alcotest.test_case
            "a failing listener does not skip the others"
            `Quick
            test_all_listeners_run_and_first_failure_is_reported;
        ] );
    ]
