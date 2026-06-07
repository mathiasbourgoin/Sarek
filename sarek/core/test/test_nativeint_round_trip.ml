(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Round-trip test for the nativeint boundary change (PR 2a).
 *
 * Verifies that host_ptr_to_device / device_to_host_ptr correctly round-trip
 * a known byte pattern through both the Native and Interpreter backends after
 * Framework_sig changed from unit Ctypes.ptr to nativeint.
 *
 * Load-bearing: if raw_address_of_ptr / ptr_of_raw_address were swapped or
 * dropped this test fails with mismatched bytes.
 *
 * Coverage: exercises the Custom_storage (alloc_custom) arm. The bigarray arm
 * of the same methods uses the identical conversion and is covered transitively
 * by the existing vector-transfer goldens (@sarek/tests/runtest).
 ******************************************************************************)

open Spoc_core

(* Force registration of both CPU backends before Device.init *)
let () =
  ignore (Lazy.force Sarek_native.Native_plugin.registered_backend) ;
  ignore (Lazy.force Sarek_interpreter.Interpreter_plugin.registered_backend)

let all_devices = Device.init ~frameworks:["Native"; "Interpreter"] ()

let elem_count = 16

(** Fill a ctypes uint8_t array with pattern i mod 256 and return void ptr. *)
let make_and_fill_buf n =
  let buf = Ctypes.(allocate_n uint8_t ~count:n) in
  for i = 0 to n - 1 do
    let v = Unsigned.UInt8.of_int (i mod 256) in
    Ctypes.(buf +@ i <-@ v)
  done ;
  (buf, Ctypes.to_voidp buf)

(** Read n bytes from void* into an int array. *)
let read_buf void_ptr n =
  let p = Ctypes.(from_voidp uint8_t void_ptr) in
  Array.init n (fun i -> Unsigned.UInt8.to_int Ctypes.(!@(p +@ i)))

(** Zero n bytes at void*. *)
let zero_buf void_ptr n =
  let p = Ctypes.(from_voidp uint8_t void_ptr) in
  for i = 0 to n - 1 do
    Ctypes.(p +@ i <-@ Unsigned.UInt8.zero)
  done

let test_backend_round_trip framework_name =
  let dev =
    match
      Array.to_list all_devices
      |> List.filter (fun d -> d.Device.framework = framework_name)
    with
    | d :: _ -> d
    | [] -> failwith (Printf.sprintf "No %s device found" framework_name)
  in
  let elem_size = 1 in
  let byte_count = elem_count * elem_size in
  let gpu_buf = Memory.alloc_custom dev ~size:elem_count ~elem_size in
  let _ocaml_buf, void_ptr = make_and_fill_buf byte_count in
  (* host -> device (exercises ptr -> nativeint -> ptr conversion) *)
  Memory.host_ptr_to_device ~src_ptr:void_ptr ~dst:gpu_buf ;
  (* Zero host buf to prove data lives in device buffer *)
  zero_buf void_ptr byte_count ;
  (* device -> host (exercises the reverse path) *)
  Memory.device_to_host_ptr ~src:gpu_buf ~dst_ptr:void_ptr ;
  let actual = read_buf void_ptr byte_count in
  Array.iteri
    (fun i v ->
      let expected = i mod 256 in
      if v <> expected then
        failwith
          (Printf.sprintf
             "%s round-trip FAIL at byte %d: expected %d got %d"
             framework_name
             i
             expected
             v))
    actual ;
  Memory.free gpu_buf ;
  Printf.printf
    "PASS: %s nativeint round-trip (%d bytes)\n"
    framework_name
    byte_count

let () =
  Printf.printf "nativeint round-trip tests:\n" ;
  test_backend_round_trip "Native" ;
  test_backend_round_trip "Interpreter" ;
  Printf.printf "All nativeint round-trip tests passed!\n"
