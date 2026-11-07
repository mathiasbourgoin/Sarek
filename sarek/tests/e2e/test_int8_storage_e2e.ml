(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The 8-bit storage types (uint8/int8) on every device, against the
 * interpreter, bit for bit.
 *
 * The kernels and the host model are in sarek/tests/unit/int8_kernels.ml:
 * zero- and sign-extending reads, low-byte stores, narrowing casts and a cast
 * of a cast on global buffers; u8 and s8 SHARED arrays written by one thread
 * and read by its mirror thread after a barrier; 8-bit local scalars (a let
 * and a mutable let reassigned out of range). Inputs cover every
 * byte value and int32 values that are negative and above 255.
 *
 * Three comparisons per device, all exact (integers, no tolerance):
 *   - the interpreter against the host model (the oracle is checked first);
 *   - the device against the interpreter;
 *   - a POSITIVE CONTROL: the device against a deliberately wrong model (zero
 *     extension in place of sign extension) must DIFFER, or the comparison
 *     would accept a sign-extension defect.
 *
 * A device whose capabilities lack 8-bit storage must be refused by the launch
 * gate rather than run; that is checked too. The Native backend is reported
 * and skipped: it runs only kernels that carry a PPX-generated native function,
 * and these kernels are IR.
 *
 * SAREK_TEST_EXCLUDE_DEVICES=sub1,sub2 skips devices whose name contains one of
 * the substrings (for instance an integrated GPU that must not be loaded).
 *
 * Run with: dune exec sarek/tests/e2e/test_int8_storage_e2e.exe
 ******************************************************************************)

open Sarek
module K = Int8_kernels
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let vec_i32 a =
  let v = Vector.create Vector.int32 (Array.length a) in
  Array.iteri (Vector.set v) a ;
  v

let vec_u8 a =
  let v = Vector.create Vector.char (Array.length a) in
  Array.iteri (fun i x -> Vector.set v i (Char.chr x)) a ;
  v

let vec_s8 a =
  let v = Vector.create Vector.int8 (Array.length a) in
  Array.iteri (Vector.set v) a ;
  v

(** One kernel's outputs, all as int32 so they compare as one array. *)
type outputs = int32 array

(** [rw] on a runner: [out] then [dst_u8] then [dst_s8], concatenated. *)
let run_rw (launch : Sarek_ir_types.kernel -> Execute.vector_arg list -> unit) :
    outputs =
  let out = vec_i32 (Array.make (K.rw_slots * K.n) 0l)
  and dst_u8 = vec_u8 (Array.make K.n 0)
  and dst_s8 = vec_s8 (Array.make K.n 0) in
  launch
    (K.rw_kernel ())
    [
      Execute.Vec (vec_u8 K.src_u8);
      Execute.Vec (vec_s8 K.src_s8);
      Execute.Vec (vec_i32 K.vals);
      Execute.Vec out;
      Execute.Vec dst_u8;
      Execute.Vec dst_s8;
    ] ;
  Array.concat
    [
      Vector.to_array out;
      Array.map (fun c -> Int32.of_int (Char.code c)) (Vector.to_array dst_u8);
      Array.map Int32.of_int (Vector.to_array dst_s8);
    ]

let run_shared launch : outputs =
  let out = vec_i32 (Array.make (2 * K.n) 0l) in
  launch (K.shared_kernel ()) [Execute.Vec (vec_i32 K.vals); Execute.Vec out] ;
  Vector.to_array out

(** The host model of each kernel, parameterised by the sign extension so the
    control below can swap in a wrong one. *)
let rw_model sext =
  let out, u, s =
    K.rw_model ~sext ~src_u8:K.src_u8 ~src_s8:K.src_s8 ~vals:K.vals ()
  in
  Array.concat [out; Array.map Int32.of_int u; Array.map Int32.of_int s]

let shared_model sext = K.shared_model ~sext ~vals:K.vals ()

let first_divergence x y =
  let rec go i =
    if i >= Array.length x then None
    else if x.(i) <> y.(i) then Some (i, x.(i), y.(i))
    else go (i + 1)
  in
  if Array.length x <> Array.length y then Some (-1, 0l, 0l) else go 0

let grid = Execute.dims1d (K.n / K.block)

let block = Execute.dims1d K.block

let interp_launch ir args =
  Execute.run_interpreter_vectors ~ir ~args ~block ~grid ~parallel:false

let device_launch dev ir args =
  Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () ;
  Transfer.flush dev

(** A wrong model: zero extension where the semantics sign-extend. *)
let zero_extend = K.u8

let run_scalar launch : outputs =
  let out = vec_i32 (Array.make (K.scalar_slots * K.n) 0l) in
  launch (K.scalar_kernel ()) [Execute.Vec (vec_i32 K.vals); Execute.Vec out] ;
  Vector.to_array out

let scalar_model sext = K.scalar_model ~sext ~vals:K.vals ()

let kernels =
  [
    ("global", run_rw, rw_model);
    ("shared", run_shared, shared_model);
    ("local scalars", run_scalar, scalar_model);
  ]

let excluded name =
  match Sys.getenv_opt "SAREK_TEST_EXCLUDE_DEVICES" with
  | None | Some "" -> false
  | Some s ->
      List.exists
        (fun sub ->
          sub <> ""
          && Test_helpers.string_contains
               ~needle:(String.lowercase_ascii sub)
               (String.lowercase_ascii name))
        (String.split_on_char ',' s)

let check_oracle () =
  List.iter
    (fun (label, run, model) ->
      match first_divergence (run interp_launch) (model K.s8) with
      | None -> Printf.printf "  OK  interpreter = host model (%s)\n%!" label
      | Some (i, g, e) ->
          fail "interpreter %s: element %d is %ld, model %ld" label i g e)
    kernels

(** s8 cooperative-matrix fragments loaded from int8 buffers, on a device that
    advertises s8 x s8 + s32 -> s32. One subgroup does the whole 16x16x16
    product, so the block is the device's subgroup. *)
let check_coopmat_int8 dev name =
  if
    Sarek_capability.permits
      (Sarek_coopmat.verdict
         ~support:dev.Device.capabilities.Spoc_framework.Framework_sig.coopmat
         K.s8_config)
  then begin
    let d = vec_i32 (Array.make (K.mnk * K.mnk) 0l) in
    let sg = dev.Device.capabilities.Spoc_framework.Framework_sig.warp_size in
    match
      Execute.run_vectors
        ~device:dev
        ~ir:(K.coopmat_s8_kernel ~buf_elt:Sarek_ir_types.TInt8)
        ~args:
          [
            Execute.Vec (vec_s8 K.coop_a);
            Execute.Vec (vec_s8 K.coop_b);
            Execute.Vec (vec_i32 K.coop_c);
            Execute.Vec d;
          ]
        ~block:(Execute.dims1d sg)
        ~grid:(Execute.dims1d 1)
        () ;
      Transfer.flush dev
    with
    | exception e ->
        fail
          "%s s8 coopmat from int8 buffers: raised %s"
          name
          (Printexc.to_string e)
    | () -> (
        match first_divergence (Vector.to_array d) (K.coopmat_s8_model ()) with
        | None ->
            Printf.printf
              "  OK  %s s8 coopmat from int8 buffers: equal to the model\n%!"
              name
        | Some (i, g, e) ->
            fail "%s s8 coopmat: element %d is %ld, model %ld" name i g e)
  end

let check_device dev =
  let name = Printf.sprintf "%s [%s]" dev.Device.name dev.Device.framework in
  let provides =
    List.mem
      Sarek_ir_analysis.Int8
      dev.Device.capabilities.Spoc_framework.Framework_sig.device_features
  in
  let gate =
    match Execute.check_device_capabilities ~device:dev (K.rw_kernel ()) with
    | () -> Ok ()
    | exception e -> Error (Printexc.to_string e)
  in
  match (provides, gate) with
  | false, Error why ->
      Printf.printf "  OK  %s: refused by the launch gate (%s)\n%!" name why
  | false, Ok () ->
      fail
        "%s does not report 8-bit storage, yet the launch gate permitted"
        name
  | true, Error why ->
      fail "%s reports 8-bit storage but was refused: %s" name why
  | true, Ok () ->
      check_coopmat_int8 dev name ;
      List.iter
        (fun (label, run, model) ->
          match run (device_launch dev) with
          | exception e ->
              fail "%s %s: raised %s" name label (Printexc.to_string e)
          | got -> (
              (match first_divergence got (run interp_launch) with
              | None ->
                  Printf.printf
                    "  OK  %s %s: bit-identical to the interpreter (%d values)\n\
                     %!"
                    name
                    label
                    (Array.length got)
              | Some (i, g, o) ->
                  fail
                    "%s %s: element %d is %ld, interpreter %ld"
                    name
                    label
                    i
                    g
                    o) ;
              match first_divergence got (model zero_extend) with
              | None ->
                  fail
                    "%s %s: the zero-extension control was ACCEPTED"
                    name
                    label
              | Some _ -> ()))
        kernels

(** CUDA C has no device here to execute on; when NVRTC and the CUDA headers are
    installed the emitted source is at least compiled for sm_80. *)
let check_nvrtc () =
  List.iter
    (fun (k : Sarek_ir_types.kernel) ->
      match
        Backend_cuda_c.ptx_of
          ~name:k.kern_name
          (Sarek_codegen.Sarek_ir_cuda.generate k)
      with
      | None ->
          Printf.printf
            "  skip CUDA C %s: NVRTC or the CUDA headers are not installed\n%!"
            k.kern_name
      | Some _ ->
          Printf.printf "  OK  CUDA C %s: NVRTC compiles it\n%!" k.kern_name
      | exception e ->
          fail "CUDA C %s: NVRTC refused: %s" k.kern_name (Printexc.to_string e))
    [K.rw_kernel (); K.shared_kernel (); K.scalar_kernel ()]

let () =
  print_endline "=== 8-bit storage types (uint8/int8) ===" ;
  check_oracle () ;
  check_nvrtc () ;
  let devices = Array.to_list (Device.init ()) in
  List.iter
    (fun dev ->
      if excluded dev.Device.name then
        Printf.printf
          "  skip %s [%s]: SAREK_TEST_EXCLUDE_DEVICES\n%!"
          dev.Device.name
          dev.Device.framework
      else if dev.Device.framework = "Native" then
        Printf.printf
          "  skip %s [Native]: runs PPX kernels only, these are IR kernels\n%!"
          dev.Device.name
      else check_device dev)
    devices ;
  if !failures > 0 then begin
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1
  end
  else print_endline "PASS"
