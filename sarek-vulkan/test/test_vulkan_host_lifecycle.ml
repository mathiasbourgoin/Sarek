(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Pinned and registered host memory across a device's destruction on Vulkan
 * (specs/vulkan-async-api.md): imports are destroyed with their device; a late
 * [unregister] of a dropped registration succeeds; a pinned allocation is
 * driver memory owned by its device, unmapped with it, and a late [free_pinned]
 * through the re-created device succeeds quietly.
 * Skips when no Vulkan device is present.
 ******************************************************************************)

open Sarek_vulkan
open Vulkan_async_support
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Transfer = Vulkan_api_transfer
module Host = Vulkan_api_host

let size = 100_000

let char_array n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let address_of ba =
  Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr ba)

(** Round-trip [size] bytes of [host] through a fresh buffer of [device]; the
    number of bytes that differ. *)
let roundtrip device host =
  let buf = Memory.alloc device size Bigarray.char in
  let out = char_array size in
  Transfer.host_to_device
    ~src_ptr:(address_of host)
    ~dst:buf
    ~dst_offset:0
    ~byte_size:size ;
  Transfer.device_to_host
    ~src:buf
    ~src_offset:0
    ~dst_ptr:(address_of out)
    ~byte_size:size ;
  let bad = ref 0 in
  for i = 0 to size - 1 do
    if Bigarray.Array1.get out i <> Bigarray.Array1.get host i then incr bad
  done ;
  Memory.free buf ;
  !bad

let test_import_alignment device =
  match device.Device.host_import_alignment with
  | None -> Alcotest.skip ()
  | Some alignment ->
      Alcotest.(check bool)
        "the import granule is a positive power of two"
        true
        (alignment > 0 && alignment land (alignment - 1) = 0)

let test_across_device_destroy device =
  if device.Device.host_import_alignment = None then Alcotest.skip () ;
  let reg = char_array size in
  Bigarray.Array1.fill reg 'r' ;
  let reg_addr = address_of reg in
  Host.register device reg_addr ~byte_size:size ;
  Device.destroy device ;
  (* The registration was dropped with the device: a late unregister is quiet. *)
  Host.unregister device reg_addr ;
  let fresh = Device.get 0 in
  Host.register fresh reg_addr ~byte_size:size ;
  Alcotest.(check int)
    "copies through a range registered on the re-created device"
    0
    (roundtrip fresh reg) ;
  Host.unregister fresh reg_addr ;
  ignore (Sys.opaque_identity reg)

(* Pinned memory needs no import: it is driver memory, mapped for the host. It
   goes with its device; the late free must neither raise nor touch it. *)
let test_pinned_across_device_destroy device =
  let pinned = Host.alloc_pinned device ~byte_size:size in
  let bytes =
    Ctypes.from_voidp Ctypes.char (Ctypes.ptr_of_raw_address pinned)
  in
  Ctypes.(bytes <-@ 'p') ;
  Alcotest.(check char) "writable before the destroy" 'p' Ctypes.(!@bytes) ;
  Device.destroy device ;
  let fresh = Device.get 0 in
  Host.free_pinned fresh pinned ;
  let again = Host.alloc_pinned fresh ~byte_size:size in
  Alcotest.(check bool) "the re-created device allocates" true (again <> 0n) ;
  Host.free_pinned fresh again ;
  match Host.free_pinned fresh pinned with
  | () -> Alcotest.fail "a second free of the dropped block was accepted"
  | exception Invalid_argument _ -> ()

(* A downstream LLM engine reads model files into a pinned block with pread on an O_DIRECT
   descriptor. The mapping of a driver allocation is not anonymous memory, so
   this is checked rather than assumed: get_user_pages must accept it. *)
let o_direct = 0x4000 (* Linux x86_64 / aarch64 differ; x86_64 here *)

let test_pinned_pread_o_direct device =
  let open Ctypes in
  let chunk = 1 lsl 20 in
  let path = Filename.temp_file ~temp_dir:(Sys.getcwd ()) "odirect" ".bin" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with _ -> ())
    (fun () ->
      let oc = open_out_bin path in
      output_string oc (String.init chunk (fun i -> Char.chr (i * 7 land 255))) ;
      close_out oc ;
      let c_open =
        Foreign.foreign
          ~check_errno:true
          "open"
          (string @-> int @-> returning int)
      in
      let c_pread =
        Foreign.foreign
          ~check_errno:true
          "pread"
          (int @-> ptr void @-> size_t @-> int64_t
          @-> returning PosixTypes.ssize_t)
      in
      let c_close = Foreign.foreign "close" (int @-> returning int) in
      match c_open path (0 lor o_direct) with
      | exception Unix.Unix_error _ ->
          print_endline "[SKIP] filesystem refuses O_DIRECT" ;
          Alcotest.skip ()
      | fd ->
          (* Slack so a 4096-aligned start exists even if the mapping is not. *)
          let block = Host.alloc_pinned device ~byte_size:(chunk + 4096) in
          let start = Nativeint.(logand (add block 4095n) (lognot 4095n)) in
          let n =
            c_pread
              fd
              (ptr_of_raw_address start)
              (Unsigned.Size_t.of_int chunk)
              0L
          in
          ignore (c_close fd) ;
          Alcotest.(check int)
            "pread O_DIRECT into the pinned block"
            chunk
            (PosixTypes.Ssize.to_int n) ;
          let bytes = from_voidp char (ptr_of_raw_address start) in
          let bad = ref 0 in
          for i = 0 to chunk - 1 do
            if !@(bytes +@ i) <> Char.chr (i * 7 land 255) then incr bad
          done ;
          Alcotest.(check int) "bytes read" 0 !bad ;
          Host.free_pinned device block)

(* The properties function is the owning driver's entry point: every device
   record resolves its own, none is shared through a process-wide cache. *)
let test_per_device_function device =
  match device.Device.host_pointer_properties_fn with
  | None ->
      Alcotest.(check bool)
        "no function without an import granule"
        true
        (device.Device.host_import_alignment = None)
  | Some old_fn -> (
      Alcotest.(check bool)
        "a function exactly when there is an import granule"
        true
        (device.Device.host_import_alignment <> None) ;
      Device.destroy device ;
      let fresh = Device.get 0 in
      Alcotest.(check bool) "a new device record" true (fresh != device) ;
      match fresh.Device.host_pointer_properties_fn with
      | Some fresh_fn ->
          Alcotest.(check bool)
            "the new device resolved its own function"
            true
            (fresh_fn != old_fn)
      | None -> Alcotest.fail "the re-created device lost the extension")

let () =
  Alcotest.run
    "vulkan_host_lifecycle"
    [
      ( "host memory",
        [
          Alcotest.test_case
            "import granule"
            `Quick
            (with_device test_import_alignment);
          Alcotest.test_case
            "pinned memory across a device destroy"
            `Quick
            (with_device test_pinned_across_device_destroy);
          Alcotest.test_case
            "pread O_DIRECT into a pinned block"
            `Quick
            (with_device test_pinned_pread_o_direct);
          Alcotest.test_case
            "per-device properties function"
            `Quick
            (with_device test_per_device_function);
          Alcotest.test_case
            "across device destruction"
            `Quick
            (with_device test_across_device_destroy);
        ] );
    ]
