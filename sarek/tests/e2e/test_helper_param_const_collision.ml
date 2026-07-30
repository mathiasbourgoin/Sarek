(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A helper PARAMETER whose name matches a module constant (backlog-180, H3).
 *
 * backlog-160 gives a helper its own copy of every module constant it
 * references, by prefixing that constant's [SLet] into the helper body. The set
 * of "referenced" names was collected with an EMPTY bound list for parameters,
 * so a parameter that merely shares a constant's name made the constant look
 * free — and its declaration got prefixed into a body that already binds that
 * identifier as a parameter.
 *
 * MEASURED pre-fix behaviour, per backend. It does NOT match the
 * "device backends hard-error, interpreter silently overwrites" split the
 * finding was reported with — the two halves are swapped, and the real one is
 * worse:
 *
 *   Interpreter x2  OK        — correct. It does NOT silently ignore the
 *                               argument, contrary to the report.
 *   Native          OK        — correct.
 *   CUDA/PTX        got 200, want 2   <- SILENTLY WRONG DATA
 *                               (measured THROUGH ZLUDA on an AMD RX 7900 XTX,
 *                               not on native NVIDIA — see the note below)
 *   OpenCL x2       compile failure (loud)
 *   Vulkan x2       compile failure (loud)
 *
 * PTX is the silent one because it does not emit a flat C scope of named
 * locals: it allocates registers, so a duplicated [SLet] never collides
 * textually. The prefixed constant simply overwrote the parameter's register
 * and the helper computed 100.0 *. 2.0 for every element.
 *
 * That inversion matters for how much this is worth. A silent bug on the
 * INTERPRETER would be caught by any cross-backend differential test, because
 * the interpreter is the oracle and it would disagree with the GPUs. Here the
 * oracle is RIGHT and only a real GPU backend is wrong, so agreement-with-
 * interpreter is exactly the check that would have caught it — and no test of
 * this shape existed.
 *
 * Metal and HIP are unmeasured (no such device on this host).
 *
 * HOW THE CUDA/PTX ROW WAS OBTAINED, because "on this host" was doing too much
 * work here. A bare run of this test enumerates four frameworks — Interpreter,
 * Native, OpenCL, Vulkan — and NO CUDA device; there is no NVIDIA hardware and
 * no nvidia-smi. The CUDA/PTX row exists only with
 * LD_LIBRARY_PATH=$HOME/opt/zluda, which puts ZLUDA's CUDA implementation on an
 * AMD GPU. So the row is a real execution of the real PTX the emitter produced,
 * but through a reimplementation of the CUDA driver, not through NVIDIA's.
 * Treat it as strong evidence about the EMITTED PTX and weaker evidence about
 * what an NVIDIA driver does with it. Verified: without ZLUDA the device list
 * is [Interpreter, Native, OpenCL, Vulkan]; with it, [CUDA/PTX, Interpreter,
 * Native, OpenCL, Vulkan].
 *
 * The [expr_names] header documents this exact redeclaration bug for LOCAL
 * binders ([SLet]/[SLetMut]/[SFor], "caught by review on #362"); the fix landed
 * for locals and stopped one binder class short of parameters. The refusal
 * message on that path even advises "pass the constant in as a parameter",
 * which was precisely the unchecked case.
 *
 * TWO CASES, both required:
 *
 *   collision — the helper's parameter is named [scale] and a module constant
 *     [scale] exists. The values are deliberately far apart (argument-derived
 *     result 2.0, constant-derived result 200.0) so the SILENT symptom is
 *     visible: a test whose two answers coincide cannot see it.
 *
 *   prefixing-still-works — a helper that references a constant it does NOT
 *     shadow must still get its copy. Without this case the fix could be
 *     "never prefix anything", which also makes the first case pass while
 *     breaking backlog-160 outright.
 *
 * WHICH MUTATION MAKES EACH CASE RED, and whether it reaches EXECUTION. The
 * table above records a pre-fix state that this test does not re-derive on each
 * run, so it is evidence only to the extent the mutations below were actually
 * observed. All three were run on 2026-07-30 with LD_LIBRARY_PATH pointed at
 * ZLUDA, so all 9 devices were live:
 *
 *   Revert BOTH halves of the fix in Sarek_lower_ir.ml (i.e. the file as it is
 *   on main). Builds cleanly, and the run REACHES EXECUTION and exits 1:
 *   CUDA/PTX x2 report "got 200 want 2", OpenCL x2 fail with "redefinition of
 *   'scale'", Vulkan x2 fail to compile, Interpreter x2 and Native pass. This
 *   is the mutation that reproduces the header table, and it is why the table
 *   is not merely asserted.
 *
 *   Revert ONLY the [referenced] seeding, keeping the parameter names in the
 *   binder table. This does NOT reach execution: it stops at a BUILD error, the
 *   refusal guard firing on the collision. Worth stating explicitly, because
 *   the seeding and the guard shipped in one commit, and the seeding alone is
 *   therefore not observable through this test — the guard masks it.
 *
 *   Replace the seeding call with [ignore] so nothing is ever considered
 *   referenced (the degenerate "never prefix anything" fix). Builds cleanly and
 *   REACHES EXECUTION, exit 1, on the prefixing-still-works case: "Unbound
 *   variable 'offset'" on the Interpreter, "PTX codegen: unbound variable:
 *   offset" on CUDA/PTX, a compile failure on OpenCL and Vulkan. NOTE that
 *   Native passes this mutation — it resolves [offset] from the enclosing OCaml
 *   scope — so Native alone is blind to a missing prefix, and case 2's value
 *   comes from the other backends.
 *
 * The refusal guard's own behaviour is NOT covered here, because it fires
 * during lowering and so is unobservable from a running test. It is pinned as a
 * negative-compile case instead:
 * sarek/tests/negative/test_helper_param_shadows_const.ml, asserted by
 * `make test_negative`.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

(* The helper parameter [scale] shadows the module constant [scale]. Correct
   behaviour: [boost x] is [x *. 2.0].

   Pre-fix, per the measured table in the header — NOT "interpreter silently
   wrong, everything else a compile error", which is what this comment used to
   say and which is the reported split rather than the observed one. Observed:
   the Interpreter and Native are CORRECT; CUDA/PTX silently computes
   [100.0 *. 2.0] because it allocates registers instead of a flat C scope, so
   the duplicated declaration overwrites the parameter rather than colliding;
   OpenCL and Vulkan fail to compile ("redefinition of 'scale'"). *)
let collision_kernel =
  snd
    [%kernel
      let open Std in
      let (scale : float32) = 100.0 in
      let boost (scale : float32) : float32 = scale *. 2.0 in
      fun (out : float32 vector) (src : float32 vector) (n : int32) ->
        let t = thread_idx_x + (block_idx_x * block_dim_x) in
        if t < n then out.(t) <- boost src.(t)]

(* No shadowing: the helper names a constant it does not bind, so the constant
   MUST still be prefixed into it. This is the backlog-160 feature the fix must
   not regress. *)
let prefixing_kernel =
  snd
    [%kernel
      let open Std in
      let (offset : float32) = 7.0 in
      let shifted (x : float32) : float32 = x +. offset in
      fun (out : float32 vector) (src : float32 vector) (n : int32) ->
        let t = thread_idx_x + (block_idx_x * block_dim_x) in
        if t < n then out.(t) <- shifted src.(t)]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let n = 32

let run_case (dev : Device.t) ~label ~kernel ~expected : bool =
  Printf.printf
    "helper-const %-22s [%s] %s: %!"
    label
    dev.Device.framework
    dev.Device.name ;
  let src = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int (i + 1)) ;
    Vector.set out i 0.0
  done ;
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of kernel)
      ~args:[Vec out; Vec src; Int n]
      ~block:(Sarek.Execute.dims1d n)
      ~grid:(Sarek.Execute.dims1d 1)
      ()
  with
  | exception e ->
      (* Not a skip. Both kernels are ordinary DSL and every backend listed
         executes ordinary scalar kernels; a refusal here IS the defect. *)
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok = ref true in
      let reported = ref 0 in
      for i = 0 to n - 1 do
        let want = expected (float_of_int (i + 1)) in
        let got = Vector.get out i in
        if Float.abs (got -. want) > 1e-4 then begin
          ok := false ;
          if !reported < 3 then begin
            incr reported ;
            Printf.printf "\n  @%d got %g want %g" i got want
          end
        end
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let () =
  let devs =
    Device.init
      ~frameworks:
        ["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"; "HIP"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found — nothing asserted, and that is a gap" ;
    exit 1
  end ;
  let any_failure = ref false in
  Array.iter
    (fun dev ->
      (* param shadows the constant -> the ARGUMENT must win (x *. 2.0). The
         wrong answer is 100.0 *. 2.0 = 200.0 for every element, two orders of
         magnitude away from the right one, which is what makes the SILENT
         backend's failure visible at all. Pre-fix that 200.0 was produced by
         CUDA/PTX (measured through ZLUDA on an AMD RX 7900 XTX); the
         Interpreter was correct here, so the expectation is not separating an
         interpreter answer. *)
      if
        not
          (run_case
             dev
             ~label:"param shadows const"
             ~kernel:collision_kernel
             ~expected:(fun x -> x *. 2.0))
      then any_failure := true ;
      (* no shadowing -> the constant must still reach the helper (x +. 7.0).
         If this regresses, the emitted device code names an identifier it never
         declared, which is a compile error rather than wrong data. *)
      if
        not
          (run_case
             dev
             ~label:"const still prefixed"
             ~kernel:prefixing_kernel
             ~expected:(fun x -> x +. 7.0))
      then any_failure := true)
    devs ;
  Printf.printf "%d device(s) exercised\n%!" (Array.length devs) ;
  if !any_failure then exit 1
