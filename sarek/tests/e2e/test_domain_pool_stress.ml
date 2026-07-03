(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression test - DomainPool oversubscription race
 *
 * Historical bug: [Sarek_ir_interp.DomainPool.create] (and its twin in
 * Sarek_cpu_runtime_pools.ml) built the pool record, spawned worker domains
 * closing over it, then rebuilt the record via [{pool with domains}] to fill
 * in the [domains] field. Because [active_tasks]/[shutdown] were plain
 * (unboxed) [mutable] fields rather than refs, that second record allocation
 * gave workers and [wait_all]/[submit] two different memory cells for the
 * same logical counter. [wait_all] then always observed [active_tasks = 0]
 * and could return as soon as the task queue drained - even if the last
 * popped block was still executing - dropping that block's writes.
 *
 * This surfaced on CI's 96-core runner (many domains -> many blocks in
 * flight -> wider race window) but was independent of core count: it
 * reproduced locally on a 32-core box too, just less often. [SAREK_DOMAIN_COUNT]
 * lets us force a high domain count to make the race close to certain within
 * a bounded number of trials, on any machine.
 *
 * This test runs a kernel over a grid with many blocks, forces a high domain
 * count, repeats many trials, and asserts EVERY output element was written -
 * i.e. no block was silently dropped.
 ******************************************************************************)

open Sarek_ir_types
open Sarek

(* Force a domain count well above typical local core counts *before* the
   interpreter's global pool is lazily created (first [run_kernel] call).
   Do not override an operator-supplied value. *)
let () =
  if Sys.getenv_opt "SAREK_DOMAIN_COUNT" = None then
    Unix.putenv "SAREK_DOMAIN_COUNT" "96"

(** Kernel IR: dst.(idx) <- idx + 1, built directly (no PPX) so this test has no
    dependency on kernel compilation - it only exercises
    [Sarek_ir_interp.run_kernel]'s block scheduling. *)
let make_ir () : kernel =
  let make_var id name ty =
    {var_name = name; var_id = id; var_type = ty; var_mutable = false}
  in
  let dst = make_var 0 "dst" (TVec TInt32) in
  let idx = make_var 1 "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar idx),
            EBinop (Add, EVar idx, EConst (CInt32 1l)) ) )
  in
  {
    kern_name = "domain_pool_stress";
    kern_params =
      [DParam (dst, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let block_size = 64

let num_blocks = 64

let n = block_size * num_blocks

let trials = 50

let run_one_trial ir =
  let dst = Array.make n (Sarek_ir_interp.VInt32 0l) in
  Sarek_ir_interp.run_kernel
    ir
    ~block:(block_size, 1, 1)
    ~grid:(num_blocks, 1, 1)
    [("dst", Sarek_ir_interp.ArgArray dst)] ;
  let dropped = ref [] in
  for i = 0 to n - 1 do
    let got = match dst.(i) with Sarek_ir_interp.VInt32 v -> v | _ -> -1l in
    if got <> Int32.of_int (i + 1) then dropped := i :: !dropped
  done ;
  !dropped

let () =
  let ir = make_ir () in
  Printf.printf
    "Stressing DomainPool: %d trials, %d blocks x %d threads, \
     SAREK_DOMAIN_COUNT=%s\n\
     %!"
    trials
    num_blocks
    block_size
    (Option.value ~default:"(unset)" (Sys.getenv_opt "SAREK_DOMAIN_COUNT")) ;
  let any_dropped = ref false in
  for trial = 1 to trials do
    match run_one_trial ir with
    | [] -> ()
    | dropped ->
        any_dropped := true ;
        Printf.printf
          "FAIL: trial %d dropped %d/%d output elements (first few indices: %s)\n\
           %!"
          trial
          (List.length dropped)
          n
          (String.concat
             ", "
             (List.filteri
                (fun i _ -> i < 5)
                (List.rev_map string_of_int dropped)))
  done ;
  if !any_dropped then begin
    print_endline "Some tests failed!" ;
    exit 1
  end
  else begin
    Printf.printf
      "PASS: all %d trials wrote every output element (no dropped blocks)\n"
      trials ;
    print_endline "All tests passed!" ;
    exit 0
  end
