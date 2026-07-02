(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Hardware-verified check-funnel unification test (S3b item 4).
 *
 * Forces a real Vulkan FFI error - requesting a buffer allocation vastly
 * larger than any real device's memory, so vkCreateBuffer / vkAllocateMemory
 * fails at the driver level (a hard resource-limit check, independent of
 * whether validation layers are enabled) - and asserts the exception raised
 * by [Vulkan_api_base.check] is the canonical
 * [Sarek_backend_error.Backend_error], not the deprecated
 * [Vulkan_api_base.Vk_result_error] variant. Also proves a legacy handler
 * pattern-matching on the deprecated alias still compiles.
 *
 * Requires a real Vulkan device; skips (does not fail) if none is present.
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory

(* 1 << 40 float32 elements = ~4 TB - far beyond any real device's memory or
   maxBufferSize limit. *)
let absurd_element_count = 1 lsl 40

let test_absurd_allocation_raises_canonical_backend_error () =
  if not (Vulkan_api.is_available ()) then
    Printf.printf
      "[SKIP] No Vulkan device available - skipping hardware check-funnel test\n\
       %!"
  else begin
    Device.init () ;
    let device = Device.get 0 in
    let raised =
      try
        let (_ : float Memory.buffer) =
          Memory.alloc device absurd_element_count Bigarray.float32
        in
        None
      with e -> Some e
    in
    match raised with
    | None ->
        Alcotest.fail
          "expected an absurdly large allocation to fail on real hardware"
    | Some (Sarek_backend_error.Backend_error.Backend_error _) -> ()
    | Some e ->
        Alcotest.failf "expected Backend_error, got %s" (Printexc.to_string e)
  end

(** Compile-only: a legacy handler pattern-matching on the deprecated
    [Vulkan_api_base.Vk_result_error] alias must still type-check
    (opam-published library, out-of-tree code may still reference it). Never
    reached at runtime; [check] no longer raises it (see test above). *)
let _legacy_handler_still_compiles (f : unit -> unit) : unit =
  (try f () with Vulkan_api_base.Vk_result_error _ -> ())
  [@alert "-deprecated"]

let () =
  Alcotest.run
    "Vulkan_ffi_check_funnel"
    [
      ( "check_funnel_unification",
        [
          Alcotest.test_case
            "absurd allocation raises canonical Backend_error"
            `Quick
            test_absurd_allocation_raises_canonical_backend_error;
        ] );
    ]
