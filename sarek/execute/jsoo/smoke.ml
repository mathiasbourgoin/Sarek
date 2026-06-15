(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * smoke — jsoo Execute smoke test.
 *
 * Verifies that sarek_execute links under js_of_ocaml without ctypes.
 * Calls Execute.dims1d to prove a value is reachable, then attempts
 * Execute.run with a bogus device — expects Backend_error from the empty
 * registry (no backends registered in the jsoo build).
 ******************************************************************************)

let () =
  (* Prove Execute exports are accessible (no ctypes transitive dep) *)
  let dims = Execute.dims1d 256 in
  Printf.printf
    "dims1d 256 = { x=%d; y=%d; z=%d }\n"
    dims.Spoc_framework.Framework_sig.x
    dims.Spoc_framework.Framework_sig.y
    dims.Spoc_framework.Framework_sig.z ;

  (* Attempt a run with a bogus device — expect Backend_error *)
  let bogus_device : Spoc_framework.Device_type.t =
    {
      id = 0;
      backend_id = 0;
      name = "jsoo-test";
      framework = "TestBackend";
      capabilities =
        {
          is_cpu = true;
          supports_fp64 = false;
          supports_atomics = false;
          compute_capability = (0, 0);
          max_threads_per_block = 1;
          max_block_dims = (1, 1, 1);
          max_grid_dims = (1, 1, 1);
          shared_mem_per_block = 0;
          total_global_mem = 0L;
          multiprocessor_count = 1;
          clock_rate_khz = 0;
          max_registers_per_block = 0;
          warp_size = 1;
        };
    }
  in
  try
    Execute.run
      ~device:bogus_device
      ~name:"test"
      ~ir:None
      ~native_fn:None
      ~block:(Execute.dims1d 1)
      ~grid:(Execute.dims1d 1)
      [] ;
    Printf.printf "ERROR: expected Backend_error, got nothing\n"
  with
  | Execute_error.Execution_error
      (Execute_error.Backend_error {backend; message}) ->
      Printf.printf "OK: Backend_error (%s): %s\n" backend message
  | e ->
      Printf.printf "ERROR: unexpected exception: %s\n" (Printexc.to_string e)
