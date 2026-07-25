(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Runtime probe for dynamic shared memory (extern .shared).

    Generates PTX from hand-written IR with
    [DShared ("dynbuf", TFloat32, None)], loads it, and launches with an
    explicit [~shared_mem] byte size — proving both the emitter's
    [.extern .shared] declaration and the host plumbing that delivers the
    launch-time byte size (the same [Kernel.launch ~shared_mem] the
    Execute.run_source path uses).

    Kernel: one block; each thread stages out[tid] into the dynamic shared
    buffer, barriers, and reads back the block-reversed element: out[tid] <-
    dynbuf[ntid - 1 - tid]. A wrong or zero dynamic region size would trap or
    produce non-reversed output.

    Skips cleanly when no CUDA device is present. *)

open Sarek_cuda
open Sarek_ir_types

let block_size = 64

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(** out[tid] <- dynbuf[ntid-1-tid] after staging out into dynbuf. *)
let make_dynshared_kernel () : kernel =
  let out = make_var "out" (TVec TFloat32) in
  let tid = make_var "tid" TInt32 in
  let rev = make_var "rev" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "thread_idx_x", []),
        SSeq
          [
            SAssign
              (LArrayElem ("dynbuf", EVar tid), EArrayRead ("out", EVar tid));
            SBarrier;
            SLet
              ( rev,
                EBinop
                  ( Sub,
                    EBinop
                      ( Sub,
                        EIntrinsic ([], "block_dim_x", []),
                        EConst (CInt32 1l) ),
                    EVar tid ),
                SAssign
                  (LArrayElem ("out", EVar tid), EArrayRead ("dynbuf", EVar rev))
              );
          ] )
  in
  {
    kern_name = "dynshared_probe";
    kern_params =
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})];
    kern_locals = [DShared ("dynbuf", TFloat32, None)];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let emitted_ptx () =
  Sarek_codegen.Sarek_ir_ptx.generate (make_dynshared_kernel ())

(* Static half: always runs, needs no driver. *)
let test_dynshared_ptx () =
  let ptx = emitted_ptx () in
  (* The declaration must be the extern (launch-sized) form. *)
  Alcotest.(check bool)
    "PTX declares extern .shared dynbuf"
    true
    (let marker = ".extern .shared .align 4 .b32 dynbuf[];" in
     let mlen = String.length marker in
     let found = ref false in
     for i = 0 to String.length ptx - mlen do
       if String.sub ptx i mlen = marker then found := true
     done ;
     !found)

(* Device half: separate case so that skipping it does not report a green
   [OK] on a name claiming the kernel executed. The static half above keeps
   its own honest green. *)
let test_dynshared () =
  let ptx = emitted_ptx () in
  if not (Cuda_api.is_driver_available ()) then begin
    Printf.printf "  [SKIP] no CUDA device (PTX emission checked)\n%!" ;
    Alcotest.skip ()
  end
  else begin
    Cuda_api.Device.init () ;
    let dev = Cuda_api.Device.get 0 in
    let h_out =
      Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout block_size
    in
    for i = 0 to block_size - 1 do
      h_out.{i} <- float_of_int i
    done ;
    let d_out = Cuda_api.Memory.alloc dev block_size Bigarray.float32 in
    Cuda_api.Memory.host_to_device ~src:h_out ~dst:d_out ;
    let kernel =
      try Cuda_api.Kernel.load_from_ptx dev ~name:"dynshared_probe" ~ptx
      with e ->
        Printf.printf "LOAD ERROR: %s\nPTX:\n%s\n%!" (Printexc.to_string e) ptx ;
        raise e
    in
    Cuda_api.Kernel.launch
      kernel
      ~args:
        [
          Cuda_api.Kernel.ArgBuffer d_out;
          Cuda_api.Kernel.ArgInt32 (Int32.of_int block_size);
        ]
      ~grid:(1, 1, 1)
      ~block:(block_size, 1, 1)
      ~shared_mem:(block_size * 4) (* the dynamic region: ntid f32 lanes *)
      ~stream:None ;
    Cuda_api.Device.synchronize dev ;
    Cuda_api.Memory.device_to_host ~src:d_out ~dst:h_out ;
    let ok = ref true in
    for i = 0 to block_size - 1 do
      let expected = float_of_int (block_size - 1 - i) in
      if abs_float (h_out.{i} -. expected) > 1e-6 then begin
        Printf.printf
          "  FAIL at i=%d: expected %f got %f\n%!"
          i
          expected
          h_out.{i} ;
        ok := false
      end
    done ;
    Alcotest.(check bool) "block reversal through dynamic shared" true !ok ;
    Cuda_api.Memory.free d_out
  end

let () =
  Alcotest.run
    "ptx_dynshared_probe"
    [
      ( "dynshared",
        [
          Alcotest.test_case
            "PTX declares the extern .shared region"
            `Quick
            test_dynshared_ptx;
          Alcotest.test_case
            "extern .shared region sized at launch works"
            `Quick
            test_dynshared;
        ] );
    ]
