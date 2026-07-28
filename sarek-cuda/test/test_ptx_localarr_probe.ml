(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Runtime probe for per-thread local arrays (.local state space).

    Generates PTX from hand-written IR with
    [SLet (tmp, EArrayCreate (TFloat32, 8, Local), ...)], loads it and runs it.
    Pins that the function-scope [.local .align A .bXX name[n]] declaration form
    plus [mov.u64] symbol addressing and typed [ld.local]/[st.local] are
    accepted and behave per-thread on whatever driver is present (ZLUDA
    under-tests rare forms — the extern .shared probe caught exactly such a
    load-time rejection).

    Kernel: each thread fills its own 8-slot local array with tid*8+j, then sums
    it back into out[tid]. Expected out[tid] = 8*(tid*8) + 28. Any cross-thread
    interference (i.e. the array not being thread-private) or mis-addressing
    breaks the sum.

    Skips cleanly when no CUDA device is present. *)

open Sarek_cuda
open Sarek_ir_types

let block_size = 64

let slots = 8

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let make_localarr_kernel () : kernel =
  let out = make_var "out" (TVec TFloat32) in
  let tmp = make_var "tmp" (TArray (TFloat32, Local)) in
  let tid = make_var "tid" TInt32 in
  let acc = make_var "acc" TFloat32 in
  let j = make_var "j" TInt32 in
  let j2 = make_var "j2" TInt32 in
  let body =
    SLet
      ( tid,
        EIntrinsic ([], "thread_idx_x", []),
        SLet
          ( tmp,
            EArrayCreate (TFloat32, EConst (CInt32 (Int32.of_int slots)), Local),
            SSeq
              [
                SFor
                  ( j,
                    EConst (CInt32 0l),
                    EConst (CInt32 (Int32.of_int (slots - 1))),
                    Upto,
                    SAssign
                      ( LArrayElem ("tmp", EVar j),
                        ECast
                          ( TFloat32,
                            EBinop
                              ( Add,
                                EBinop
                                  ( Mul,
                                    EVar tid,
                                    EConst (CInt32 (Int32.of_int slots)) ),
                                EVar j ) ) ) );
                SLetMut
                  ( acc,
                    EConst (CFloat32 0.0),
                    SSeq
                      [
                        SFor
                          ( j2,
                            EConst (CInt32 0l),
                            EConst (CInt32 (Int32.of_int (slots - 1))),
                            Upto,
                            SAssign
                              ( LVar acc,
                                EBinop
                                  (Add, EVar acc, EArrayRead ("tmp", EVar j2))
                              ) );
                        SAssign (LArrayElem ("out", EVar tid), EVar acc);
                      ] );
              ] ) )
  in
  {
    default_kernel with
    kern_name = "localarr_probe";
    kern_params =
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})];
    kern_body = body;
  }

let emitted_ptx () =
  Sarek_codegen.Sarek_ir_ptx.generate (make_localarr_kernel ())

let contains_sub s sub =
  let n = String.length sub in
  let found = ref false in
  for i = 0 to String.length s - n do
    if String.sub s i n = sub then found := true
  done ;
  !found

(* Static half: always runs, needs no driver. *)
let test_localarr_ptx () =
  let ptx = emitted_ptx () in
  Alcotest.(check bool)
    "PTX declares .local tmp"
    true
    (contains_sub ptx ".local .align 4 .b32 tmp[8];") ;
  Alcotest.(check bool)
    "PTX stores via st.local"
    true
    (contains_sub ptx "st.local.f32") ;
  Alcotest.(check bool)
    "PTX loads via ld.local"
    true
    (contains_sub ptx "ld.local.f32")

(* Device half: separate case so that skipping it does not report a green
   [OK] on a name claiming the kernel executed. *)
let test_localarr () =
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
    Bigarray.Array1.fill h_out 0.0 ;
    let d_out = Cuda_api.Memory.alloc dev block_size Bigarray.float32 in
    Cuda_api.Memory.host_to_device ~src:h_out ~dst:d_out ;
    let kernel =
      try Cuda_api.Kernel.load_from_ptx dev ~name:"localarr_probe" ~ptx
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
      ~shared_mem:0
      ~stream:None ;
    Cuda_api.Device.synchronize dev ;
    Cuda_api.Memory.device_to_host ~src:d_out ~dst:h_out ;
    let ok = ref true in
    for i = 0 to block_size - 1 do
      (* sum_{j<8} (i*8 + j) = 8*(i*8) + 28 *)
      let expected = float_of_int ((8 * (i * slots)) + 28) in
      if abs_float (h_out.{i} -. expected) > 1e-3 then begin
        Printf.printf
          "  FAIL at i=%d: expected %f got %f\n%!"
          i
          expected
          h_out.{i} ;
        ok := false
      end
    done ;
    Alcotest.(check bool) "per-thread local array sums correct" true !ok ;
    Cuda_api.Memory.free d_out
  end

let () =
  Alcotest.run
    "ptx_localarr_probe"
    [
      ( "localarr",
        [
          Alcotest.test_case
            "PTX declares and uses the .local array"
            `Quick
            test_localarr_ptx;
          Alcotest.test_case
            ".local array declare/fill/sum executes correctly"
            `Quick
            test_localarr;
        ] );
    ]
