(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * L14 S2 - E2E test for record-field static tag erasure (device path).
 *
 * A kernel-local immutable record whose variant-typed FIELD is written by a
 * literal constructor with a substitutable payload, and read only as a `match`
 * scrutinee, has that field's tag erased: the record is lowered to a
 * synthesized all-scalar record with positional fields ([_0], [_1], ..)
 * (Sarek_tag_erasure S2, device path only). This unblocks the backends that
 * reject a nested-variant record field today. Empirically (recorded when this
 * test was written, RX 7900 XTX host), a record with a variant field FAILS
 * before erasure on Interpreter and Vulkan (record layout rejected) and on
 * OpenCL (variant field read rejected), with malformed CUDA-C; only the Native
 * OCaml path tolerates it. After S2 every backend runs it.
 *
 * Self-verifying on two axes:
 *   1. OBSERVABLE erasure in the emitted device source (CUDA-C, generated from
 *      the lowered IR, no device needed):
 *        - erasable kernels carry NO variant tag/switch/enum and DO carry the
 *          synthesized [_erec_*] record (proof the field was erased, not just
 *          absent);
 *        - an all-scalar record (no variant field) is left untouched: no tag
 *          and NO [_erec_*] (the pass does not gratuitously rewrite records);
 *        - a record whose variant field's constructor is runtime-selected keeps
 *          its tag (proof the pass stays conservative).
 *   2. BEHAVIOURAL equivalence vs a pure-OCaml reference. Erasable kernels run
 *      on EVERY available device. The runtime-selected control retains a
 *      variant record field, which only the Native path can execute (that is
 *      the very limitation S2 removes for the erasable case), so its behaviour
 *      is checked on Native alone as the correctness oracle.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

let () = Test_helpers.Benchmarks.init ()

type float32 = float

(* A variant used as an init-time tag inside a record field. *)
type shape = Circle of float32 | Square of float32 [@@sarek.type]

(* A record with a variant-typed field [kind] and a scalar field [scale]. *)
type cell = {kind : shape; scale : float32} [@@sarek.type]

(* A nullary-constructor variant, and a record carrying it as a tag field. *)
type mode = Fast | Slow [@@sarek.type]

type job = {sel : mode; load : float32} [@@sarek.type]

(* A plain all-scalar record: no variant field, so S2 must leave it alone. *)
type vec2 = {vx : float32; vy : float32} [@@sarek.type]

(* ERASABLE, unary field: `kind = Circle src.(tid)`, matched, never taken as a
   whole value. Result: (v*v)*scale = (v*v)*2 with v = src.(tid). *)
let unary_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let c = {kind = Circle src.(tid); scale = 2.0} in
          let base = match c.kind with Circle r -> r *. r | Square x -> x in
          dst.(tid) <- base *. c.scale
        end]

(* ERASABLE, nullary field: `sel = Fast` is dropped entirely; the scalar field
   [load] survives. Result: v + 1.0 with v = src.(tid). *)
let nullary_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let j = {sel = Fast; load = src.(tid)} in
          match j.sel with
          | Fast -> dst.(tid) <- j.load +. 1.0
          | Slow -> dst.(tid) <- j.load
        end]

(* NEGATIVE CONTROL (no variant field): a plain scalar record must be left as
   its original named record (NOT rewritten to [_erec_*]) and runs everywhere.
   Result: vx*vy = (v)*(v+1). *)
let allscalar_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = {vx = src.(tid); vy = src.(tid) +. 1.0} in
          dst.(tid) <- p.vx *. p.vy
        end]

(* NEGATIVE CONTROL (runtime-selected field): the variant field's constructor
   is chosen at runtime, so the slot is NOT statically known and the field's
   tag must be RETAINED. This retains a variant record field, which only the
   Native path executes; behaviour is checked on Native as the oracle. *)
let runtime_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let c =
            {
              kind =
                (if tid mod 2 = 0 then Circle src.(tid) else Square src.(tid));
              scale = 2.0;
            }
          in
          match c.kind with
          | Circle r -> dst.(tid) <- r *. c.scale
          | Square x -> dst.(tid) <- x *. c.scale
        end]

let ir_of name kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith ("no IR for " ^ name)

(* ---- Axis 1: observable erasure in the emitted device source ------------- *)

let contains hay needle =
  let nh = String.length needle and h = String.length hay in
  let rec go i =
    if i + nh > h then false
    else if String.sub hay i nh = needle then true
    else go (i + 1)
  in
  go 0

let mentions_tag src =
  contains src "switch (" || contains src ".tag" || contains src "enum "

let observable_ok = ref true

let check_emitted name kirc ~expect_tag ~expect_erec =
  let src = Sarek_codegen.Sarek_ir_cuda.generate (ir_of name kirc) in
  let has_tag = mentions_tag src in
  let has_erec = contains src "_erec" in
  let ok = has_tag = expect_tag && has_erec = expect_erec in
  if not ok then observable_ok := false ;
  Printf.printf
    "  emitted[%s]: tag %s (exp %s), _erec %s (exp %s) -> %s\n%!"
    name
    (if has_tag then "present" else "absent")
    (if expect_tag then "present" else "absent")
    (if has_erec then "present" else "absent")
    (if expect_erec then "present" else "absent")
    (if ok then "OK" else "MISMATCH") ;
  if not ok then begin
    print_endline "  ---- emitted CUDA-C ----" ;
    print_endline src
  end

(* ---- Axis 2: behavioural equivalence ------------------------------------- *)

let all_must fw =
  match fw with
  | "CUDA" | "OpenCL" | "Vulkan" | "Metal" | "Native" | "Interpreter" -> true
  | _ -> false

let native_only fw = fw = "Native"

let any_failure = ref false

(* Initialise every backend ONCE and share the device set across kernels: each
   [Device.init] spins up a fresh Vulkan instance, and repeated init/teardown
   churn has been observed to crash RADV, so we avoid re-initialising per run. *)
let devs =
  Device.init
    ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"]
    ()

let run_behaviour ~must name kirc ~reference =
  Printf.printf "runtime[%s]:\n%!" name ;
  if Array.length devs = 0 then
    print_endline "  no devices - codegen-only (behaviour skipped)"
  else
    Array.iter
      (fun dev ->
        let fw = dev.Device.framework in
        Printf.printf "  [%s] %s: %!" fw dev.Device.name ;
        try
          let n = 64 in
          let src = Vector.create Vector.float32 n in
          let dst = Vector.create Vector.float32 n in
          for i = 0 to n - 1 do
            Vector.set src i (float_of_int (i + 1)) ;
            Vector.set dst i 0.0
          done ;
          let threads = min 64 n in
          let grid_x = (n + threads - 1) / threads in
          Sarek.Execute.run_vectors
            ~device:dev
            ~block:(Sarek.Execute.dims1d threads)
            ~grid:(Sarek.Execute.dims1d grid_x)
            ~ir:(ir_of name kirc)
            ~args:
              [
                Sarek.Execute.Vec src;
                Sarek.Execute.Vec dst;
                Sarek.Execute.Int32 (Int32.of_int n);
              ]
            () ;
          Transfer.flush dev ;
          let ok = ref true in
          for i = 0 to n - 1 do
            let got = Vector.get dst i and exp = reference i in
            if abs_float (got -. exp) > 1e-2 then begin
              ok := false ;
              if i < 4 then
                Printf.printf "\n    mismatch@%d got %.3f exp %.3f%!" i got exp
            end
          done ;
          if !ok then print_endline "PASSED"
          else begin
            if must fw then any_failure := true ;
            print_endline "FAILED"
          end
        with e ->
          if must fw then any_failure := true ;
          Printf.printf
            "%s %s\n%!"
            (if must fw then "ERROR"
             else "SKIP (backend lacks nested-variant record field)")
            (Printexc.to_string e))
      devs

let () =
  print_endline "=== L14 S2 record-field tag erasure E2E ===" ;
  print_endline "-- emitted-code snapshots --" ;
  check_emitted "unary(erasable)" unary_kirc ~expect_tag:false ~expect_erec:true ;
  check_emitted
    "nullary(erasable)"
    nullary_kirc
    ~expect_tag:false
    ~expect_erec:true ;
  check_emitted
    "all-scalar(untouched)"
    allscalar_kirc
    ~expect_tag:false
    ~expect_erec:false ;
  check_emitted
    "runtime-selected(control)"
    runtime_kirc
    ~expect_tag:true
    ~expect_erec:false ;
  print_endline "-- behaviour vs pure-OCaml reference --" ;
  run_behaviour ~must:all_must "unary(erasable)" unary_kirc ~reference:(fun i ->
      let v = float_of_int (i + 1) in
      v *. v *. 2.0) ;
  run_behaviour
    ~must:all_must
    "nullary(erasable)"
    nullary_kirc
    ~reference:(fun i -> float_of_int (i + 1) +. 1.0) ;
  run_behaviour
    ~must:all_must
    "all-scalar(untouched)"
    allscalar_kirc
    ~reference:(fun i ->
      let v = float_of_int (i + 1) in
      v *. (v +. 1.0)) ;
  run_behaviour
    ~must:native_only
    "runtime-selected(control)"
    runtime_kirc
    ~reference:(fun i -> float_of_int (i + 1) *. 2.0) ;
  Printf.printf
    "\n=== observable=%s behaviour=%s ===\n"
    (if !observable_ok then "OK" else "FAIL")
    (if !any_failure then "FAIL" else "OK") ;
  if !observable_ok && not !any_failure then
    print_endline "test_record_field_tag_erasure PASSED"
  else begin
    print_endline "test_record_field_tag_erasure FAILED" ;
    exit 1
  end
