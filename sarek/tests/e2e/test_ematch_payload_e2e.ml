(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E execution test for match-EXPRESSION payload bindings (#75).
 *
 * The IR-level test (codegen_golden/test_ematch_payload_binding.ml) pins WHAT
 * each backend emits. This one pins that the emitted kernel actually RUNS and
 * computes the right numbers on every device present.
 *
 * The kernel is deliberately built so neither the tag-erasure pass nor the
 * device compiler can paper over the defect:
 *
 *  - the constructor is chosen at RUNTIME (`if tid mod 2 = 0`), so the slot is
 *    not statically erasable and a real tagged value reaches the match;
 *  - the match is an EXPRESSION (its value is bound with `let`), which is the
 *    position that has nowhere to declare a payload binder;
 *  - both arms USE their payload, and the two arms compute different functions
 *    of it, so reading the wrong value cannot coincidentally agree;
 *  - `r` is also the name of a preceding local. Before #75 the discarded binder
 *    resolved to THAT local, so the kernel compiled cleanly on every C-family
 *    backend and returned a plausible wrong answer with no diagnostic at all.
 *    Without this shadow the failure mode is merely a device-compiler error
 *    ("use of undeclared identifier 'r'"), which is the lucky case.
 *
 * The pure-OCaml reference below is the oracle; the Interpreter and Native
 * paths (which bind payloads correctly and always did) are independent ones.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init ()

type float32 = float

type shape = Circle of float32 | Square of float32 [@@sarek.type]

let kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let v = src.(tid) in
          (* Same name as the Circle payload binder below: this is what turns
             the dropped binding from a compile error into a silent wrong
             answer. *)
          let r = 1000.0 in
          let s = if tid mod 2 = 0 then Circle v else Square v in
          let got =
            match s with Circle r -> r *. 2.0 | Square q -> q +. 7.0
          in
          dst.(tid) <- got +. (r *. 0.0)
        end]

(* THE PURELY SILENT VARIANT. Identical, except BOTH arms name their payload
   `r` — the name of the enclosing local. Every dropped reference then resolves
   to something that exists, so the emitted kernel is valid C:

     float got = ((s.tag == Circle) ? (r * 2.0f) : (r + 7.0f));

   It compiles on every vendor compiler and returns 2000.0 / 1007.0 for every
   element instead of the right answer, with no error, no warning and no crash
   anywhere in the stack. This is why "the device compiler catches it" is not a
   safety net: it only holds for the subset of programs where no in-scope name
   happens to collide, and the colliding name is chosen by the user, not by us. *)
let silent_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let v = src.(tid) in
          let r = 1000.0 in
          let s = if tid mod 2 = 0 then Circle v else Square v in
          let got =
            match s with Circle r -> r *. 2.0 | Square r -> r +. 7.0
          in
          dst.(tid) <- got +. (r *. 0.0)
        end]

type pick = Pick of int32 | NoPick [@@sarek.type]

(* CAPTURE. The value substituted for the outer arm's binder is built from the
   OUTER scrutinee and carries FREE VARIABLES — the index `tid` inside
   `src.(tid)`. The inner match rebinds that exact name.

   If the arms are rewritten one node at a time as the backend walks down, the
   inner rewrite reaches the already-injected outer term and substitutes `tid`
   inside it, so the kernel reads `src[p.data.Pick_v]` where the source says
   `src.(tid)`. It compiles on every vendor compiler and returns wrong numbers,
   which is the failure mode this whole change exists to remove — reintroduced
   by the fix itself. The CPU paths bind properly and disagree with the GPUs,
   which is what makes it observable here. *)
let capture_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = if tid < n then Pick 0 else NoPick in
          let got =
            match if tid < n then Circle src.(tid) else Square 99.0 with
            | Circle r -> ( match p with Pick tid -> r | NoPick -> 0.0)
            | Square q -> q
          in
          dst.(tid) <- got
        end]

(* Every element takes the Circle arm, so the answer is just src.(tid). Under
   capture the index collapses to Pick's payload (0) and every element reads
   src.(0) instead. *)
let capture_reference _i v = v

(* The oracle, shared by both kernels. *)
let reference i v = if i mod 2 = 0 then v *. 2.0 else v +. 7.0

let n = 1024

let ir_of name k =
  match k.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith ("no IR for " ^ name)

let any_failure = ref false

let run_on ~ir ~reference ~show dev =
  let src = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int (i + 1)) ;
    Vector.set dst i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec src;
        Sarek.Execute.Vec dst;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;
  (* Print the leading values when asked. For the capture case the numbers ARE
     the diagnosis: `1 2 3 4 5 6 7 8` is a correctly-indexed read, while
     `1 1 1 1 1 1 1 1` is every element collapsing onto src.(0) because the
     inner binder captured the index inside the injected term. *)
  if show > 0 then begin
    Printf.printf "      first %d:" show ;
    for i = 0 to min (show - 1) (n - 1) do
      Printf.printf " %g" (Vector.get dst i)
    done ;
    Printf.printf "\n%!"
  end ;
  (* Report the count AND the first offender: "got 1000.0 where 3.0 was
     expected" is the reading that identifies a stale same-named local, which a
     bare pass/fail would hide. *)
  let bad = ref 0 and first = ref None in
  for i = 0 to n - 1 do
    let got = Vector.get dst i and exp = reference i (Vector.get src i) in
    if abs_float (got -. exp) > 1e-3 then begin
      incr bad ;
      if !first = None then first := Some (i, got, exp)
    end
  done ;
  match !first with
  | None -> None
  | Some (i, got, exp) ->
      Some
        (Printf.sprintf
           "%d/%d elements wrong; first at %d: got %.3f, expected %.3f"
           !bad
           n
           i
           got
           exp)

let () =
  print_endline "=== #75 EMatch payload binding — E2E ===" ;
  let devs =
    Device.init
      ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"]
      ()
  in
  if Array.length devs = 0 then print_endline "  no devices — skipped"
  else
    List.iter
      (fun (name, k, reference, show) ->
        Printf.printf "-- %s --\n%!" name ;
        let ir = ir_of name k in
        Array.iter
          (fun dev ->
            let label =
              Printf.sprintf "[%s] %s" dev.Device.framework dev.Device.name
            in
            match run_on ~ir ~reference ~show dev with
            | None -> Printf.printf "  %s: PASSED\n%!" label
            | Some detail ->
                any_failure := true ;
                Printf.printf
                  "  %s: FAILED — the match arm did not read its payload (%s)\n\
                   %!"
                  label
                  detail
            | exception e ->
                any_failure := true ;
                Printf.printf
                  "  %s: FAILED — %s\n%!"
                  label
                  (Printexc.to_string e))
          devs)
      [
        ("distinct binders (undeclared-identifier shape)", kirc, reference, 0);
        ("colliding binders (silent-wrong shape)", silent_kirc, reference, 0);
        (* src.(i) = i+1, so a correctly-indexed read prints 1 2 3 4 5 6 7 8;
           under capture every element reads src.(0) and it prints 1 1 1 1 1 1
           1 1. *)
        ("nested rebinding (capture shape)", capture_kirc, capture_reference, 8);
      ] ;
  if !any_failure then exit 1 else print_endline "OK"
