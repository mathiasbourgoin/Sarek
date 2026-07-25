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

(* [around_clear] is where the clear notification lives, so that a DIRECT
   backend clear cannot bypass it. Three properties, each of which a backend
   depends on. *)

let test_around_clear_notifies_on_both_sides () =
  let events = ref [] in
  CH.on_clear_all (fun () -> events := "notify" :: !events) ;
  CH.around_clear (fun () -> events := "clear" :: !events) ;
  (* The pre-notification drops what is memoized now; only the post one rejects
     a build that spans the handle release (Guarded_cache ~invalidated_by_clear).
     Neither alone is sufficient. *)
  Alcotest.(check (list string))
    "notify, clear, notify"
    ["notify"; "clear"; "notify"]
    (List.rev !events)

let test_around_clear_nesting_collapses () =
  let n = ref 0 in
  CH.on_clear_all (fun () -> incr n) ;
  (* Sarek.Kernel.clear_cache wraps the backend's own clear_cache, which is
     itself wrapped. The contract is two notifications per teardown, not two per
     nesting level. *)
  CH.around_clear (fun () -> CH.around_clear (fun () -> ())) ;
  Alcotest.(check int) "two notifications, not four" 2 !n

exception Boom

(* Runs LAST in the suite: [Cache_hooks] has no unregister, so the listener this
   case installs would otherwise poison every case declared after it. For the
   same reason it catches any exception rather than [Boom] specifically — by the
   time it runs, the notification cases above have also installed a raising
   listener, and [run_all] re-raises whichever failed first. *)
let test_around_clear_isolates_a_failing_listener () =
  let cleared = ref false in
  CH.on_clear_all (fun () -> raise Boom) ;
  let raised =
    try
      CH.around_clear (fun () -> cleared := true) ;
      false
    with _ -> true
  in
  (* A listener owns no handles: letting it escape early would skip the release
     the clear exists to perform and leak every backend handle. It must still be
     reported, just afterwards. *)
  Alcotest.(check bool) "the clear still ran" true !cleared ;
  Alcotest.(check bool) "the failure is re-raised afterwards" true raised

let () =
  Alcotest.run
    "Cache_hooks"
    [
      (* Declaration order is execution order, and the registry has no
         unregister: every case that installs a RAISING listener must come after
         every case that counts or orders notifications. *)
      ( "around_clear",
        [
          Alcotest.test_case
            "notifies on both sides of the clear"
            `Quick
            test_around_clear_notifies_on_both_sides;
          Alcotest.test_case
            "nested scopes notify once, not per level"
            `Quick
            test_around_clear_nesting_collapses;
        ] );
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
      ( "around_clear failure isolation",
        [
          Alcotest.test_case
            "a failing listener does not abort the clear"
            `Quick
            test_around_clear_isolates_a_failing_listener;
        ] );
    ]
