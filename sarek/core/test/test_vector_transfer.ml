(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Vector_transfer module
 *
 * Covers sync callback wiring and host pointer helpers.
 ******************************************************************************)

open Spoc_core

let make_caps () : Spoc_framework.Framework_sig.capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    supports_fp64 = true;
    supports_atomics = true;
    warp_size = 32;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu = false;
  }

let make_device id =
  {
    Device.id;
    backend_id = id;
    name = Printf.sprintf "Fake Device %d" id;
    framework = "Fake";
    capabilities = make_caps ();
  }

let make_buffer dev ba ~device_values ~host_to_device_called
    ~device_to_host_called ~freed : Vector.device_buffer =
  (module struct
    let device = dev

    let size = Bigarray.Array1.dim ba

    let elem_size = 4

    let device_ptr = Nativeint.of_int dev.Device.id

    let host_ptr_to_device _ptr ~byte_size =
      assert (byte_size = size * elem_size) ;
      host_to_device_called := true ;
      for i = 0 to size - 1 do
        device_values.(i) <- Bigarray.Array1.get ba i
      done

    let device_to_host_ptr _ptr ~byte_size =
      assert (byte_size = size * elem_size) ;
      device_to_host_called := true ;
      for i = 0 to size - 1 do
        Bigarray.Array1.set ba i device_values.(i)
      done

    let bind_to_kargs _kargs _idx = ()

    let free () = freed := true
  end : Vector.DEVICE_BUFFER)

let assert_float_array_equal expected actual =
  assert (Array.length expected = Array.length actual) ;
  Array.iteri
    (fun i expected_value ->
      assert (Float.abs (expected_value -. actual.(i)) < 0.00001))
    expected

let test_sync_callback () =
  let called = ref false in
  let cb =
    {
      Vector_transfer.sync =
        (fun _ ->
          called := true ;
          true);
    }
  in
  Vector.register_sync_callback cb ;
  let v = Vector.create_float32 1 in
  v.location <- Vector.Stale_CPU (make_device 0) ;
  Vector.ensure_cpu_sync v ;
  assert !called ;
  print_endline "  sync callback: OK"

let test_host_ptr_helpers () =
  let v = Vector.create_float32 1 in
  let ba = Vector.to_bigarray v in
  Bigarray.Array1.set ba 0 3.14 ;
  let ptr = Vector_transfer.host_ptr v in
  assert (ptr <> 0n) ;
  let void_ptr = Vector_transfer.to_ctypes_ptr v in
  assert (Ctypes.is_null void_ptr |> not) ;
  print_endline "  host_ptr helpers: OK"

let test_cross_device_transfer_preserves_authoritative_device_data () =
  let src = make_device 1 in
  let dst = make_device 2 in
  let v = Vector.create_float32 2 in
  let ba = Vector.to_bigarray v in
  Bigarray.Array1.set ba 0 1.0 ;
  Bigarray.Array1.set ba 1 2.0 ;
  let src_d2h = ref false in
  let dst_h2d = ref false in
  let src_free = ref false in
  let dst_free = ref false in
  let ignored = ref false in
  let src_values = [|10.0; 20.0|] in
  let dst_values = [|0.0; 0.0|] in
  let src_buf =
    make_buffer
      src
      ba
      ~device_values:src_values
      ~host_to_device_called:ignored
      ~device_to_host_called:src_d2h
      ~freed:src_free
  in
  let dst_buf =
    make_buffer
      dst
      ba
      ~device_values:dst_values
      ~host_to_device_called:dst_h2d
      ~device_to_host_called:ignored
      ~freed:dst_free
  in
  Hashtbl.replace v.device_buffers src.id src_buf ;
  Hashtbl.replace v.device_buffers dst.id dst_buf ;
  v.location <- Vector.Stale_CPU src ;
  Transfer.to_device v dst ;
  assert !src_d2h ;
  assert !dst_h2d ;
  assert_float_array_equal [|10.0; 20.0|] dst_values ;
  (match v.location with
  | Vector.Both d -> assert (d.id = dst.id)
  | _ -> assert false) ;
  print_endline "  cross-device authoritative transfer: OK"

let test_cross_device_transfer_preserves_gpu_authoritative_data () =
  let src = make_device 4 in
  let dst = make_device 5 in
  let v = Vector.create_float32 2 in
  let ba = Vector.to_bigarray v in
  Bigarray.Array1.set ba 0 (-1.0) ;
  Bigarray.Array1.set ba 1 (-2.0) ;
  let src_d2h = ref false in
  let dst_h2d = ref false in
  let src_free = ref false in
  let dst_free = ref false in
  let ignored = ref false in
  let src_values = [|50.0; 60.0|] in
  let dst_values = [|0.0; 0.0|] in
  let src_buf =
    make_buffer
      src
      ba
      ~device_values:src_values
      ~host_to_device_called:ignored
      ~device_to_host_called:src_d2h
      ~freed:src_free
  in
  let dst_buf =
    make_buffer
      dst
      ba
      ~device_values:dst_values
      ~host_to_device_called:dst_h2d
      ~device_to_host_called:ignored
      ~freed:dst_free
  in
  Hashtbl.replace v.device_buffers src.id src_buf ;
  Hashtbl.replace v.device_buffers dst.id dst_buf ;
  v.location <- Vector.GPU src ;
  Transfer.to_device v dst ;
  assert !src_d2h ;
  assert !dst_h2d ;
  assert_float_array_equal [|50.0; 60.0|] dst_values ;
  (match v.location with
  | Vector.Both d -> assert (d.id = dst.id)
  | _ -> assert false) ;
  print_endline "  cross-device GPU authoritative transfer: OK"

let test_free_buffer_preserves_stale_cpu_authoritative_device_data () =
  let dev = make_device 3 in
  let v = Vector.create_float32 2 in
  let ba = Vector.to_bigarray v in
  Bigarray.Array1.set ba 0 1.0 ;
  Bigarray.Array1.set ba 1 2.0 ;
  let h2d = ref false in
  let d2h = ref false in
  let freed = ref false in
  let device_values = [|30.0; 40.0|] in
  let buf =
    make_buffer
      dev
      ba
      ~device_values
      ~host_to_device_called:h2d
      ~device_to_host_called:d2h
      ~freed
  in
  Hashtbl.replace v.device_buffers dev.id buf ;
  v.location <- Vector.Stale_CPU dev ;
  Transfer.free_buffer v dev ;
  assert !d2h ;
  assert !freed ;
  assert (not (Hashtbl.mem v.device_buffers dev.id)) ;
  assert (not !h2d) ;
  assert_float_array_equal
    [|30.0; 40.0|]
    [|Bigarray.Array1.get ba 0; Bigarray.Array1.get ba 1|] ;
  (match v.location with Vector.CPU -> () | _ -> assert false) ;
  print_endline "  free_buffer Stale_CPU preservation: OK"

let test_free_all_buffers_preserves_authoritative_device_data () =
  let dev = make_device 6 in
  let other = make_device 7 in
  let v = Vector.create_float32 2 in
  let ba = Vector.to_bigarray v in
  Bigarray.Array1.set ba 0 1.0 ;
  Bigarray.Array1.set ba 1 2.0 ;
  let d2h = ref false in
  let dev_freed = ref false in
  let other_freed = ref false in
  let ignored = ref false in
  let device_values = [|70.0; 80.0|] in
  let other_values = [|0.0; 0.0|] in
  let dev_buf =
    make_buffer
      dev
      ba
      ~device_values
      ~host_to_device_called:ignored
      ~device_to_host_called:d2h
      ~freed:dev_freed
  in
  let other_buf =
    make_buffer
      other
      ba
      ~device_values:other_values
      ~host_to_device_called:ignored
      ~device_to_host_called:ignored
      ~freed:other_freed
  in
  Hashtbl.replace v.device_buffers dev.id dev_buf ;
  Hashtbl.replace v.device_buffers other.id other_buf ;
  v.location <- Vector.Stale_CPU dev ;
  Transfer.free_all_buffers v ;
  assert !d2h ;
  assert !dev_freed ;
  assert !other_freed ;
  assert (Hashtbl.length v.device_buffers = 0) ;
  assert_float_array_equal
    [|70.0; 80.0|]
    [|Bigarray.Array1.get ba 0; Bigarray.Array1.get ba 1|] ;
  (match v.location with Vector.CPU -> () | _ -> assert false) ;
  print_endline "  free_all_buffers authoritative preservation: OK"

let () =
  print_endline "Vector_transfer tests:" ;
  test_sync_callback () ;
  test_host_ptr_helpers () ;
  test_cross_device_transfer_preserves_authoritative_device_data () ;
  test_cross_device_transfer_preserves_gpu_authoritative_data () ;
  test_free_buffer_preserves_stale_cpu_authoritative_device_data () ;
  test_free_all_buffers_preserves_authoritative_device_data () ;
  print_endline "All Vector_transfer tests passed!"
