(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-62 slice 3 — the INTEGER cooperative-matrix path, executed.
 *
 * WHY INTEGER AND NOT f16, WHICH IS WHAT THE PLAN LEADS WITH.
 *
 * docs/design/f16-relaxed-accuracy.md §7 slice 4a schedules a hand-written
 * coopmat shader for f16 x f16 -> f32, measured against the gamma_16 bound of
 * §5.2 with the two positive controls of §5.4. That slice needs a numeric
 * contract that does not exist yet.
 *
 * The integer configurations need none of it. SPV_KHR_cooperative_matrix states
 * that integer accumulation is performed at the precision of the result type
 * and is EXACT, so u8 x u8 + s32 -> s32 computes the same function as a host
 * reference does — bit for bit, no bound, no allowlist, no opt-in, under
 * Sarek's EXISTING strict contract. Twelve of the fourteen configurations the
 * local RX 7900 XTX advertises are integer, and all twelve have 8-bit operands
 * with a 32-bit accumulator (measured; sarek-vulkan/probe/
 * probe_vulkan_coopmat_configs.ml prints the table).
 *
 * So this file is the integer twin of slice 4a: a hand-written GLSL coopmat
 * shader driven through Sarek's own Vulkan stack, with an exact host oracle.
 * It measures the DRIVER and the device plumbing, not Sarek's codegen — the
 * same distinction docs/fp-contraction-policy.md §7(c) draws about the f16
 * tripwire. The codegen-side gate is test_sarek_ir_glsl's coopmat cases.
 *
 * INPUT COVERAGE, AND WHY IT IS EXHAUSTIVE RATHER THAN SAMPLED.
 *
 * The full domain of a 16x16x16 u8 multiply-add is 256^512 and cannot be
 * enumerated. The domain that MATTERS for an exactness claim can be: there are
 * exactly 65536 ordered pairs (a, b) of u8 operand values, and one 16x16x16
 * multiply-add performs 4096 multiplications. Choosing
 *
 *     A[i][k] = 16*k + i        B[k][j] = 16*t + j
 *
 * makes the pair set of dispatch t equal to the union over k of
 * [16k, 16k+15] x [16t, 16t+15] — sixteen disjoint 16x16 blocks. Over
 * t = 0..15 those 256 blocks tile the whole 256x256 pair space exactly once.
 * SIXTEEN dispatches therefore exercise EVERY ordered pair of u8 operand
 * values exactly once, with no sampling anywhere.
 *
 * PROVING THE COMPARISON CAN GO RED.
 *
 * A bit-for-bit comparison that has never been observed failing is
 * indistinguishable from one that returns true. Two positive controls run on
 * the same measured data, mirroring §5.4's construction for the float path:
 * a reference that DROPS C, and a reference that TRANSPOSES B. Each must be
 * REJECTED by the same comparison that accepts the correct one. If either is
 * accepted, the test fails — the controls are assertions, not printouts.
 *
 * THE NEGATIVE DEVICE.
 *
 * The Raphael iGPU advertises no VK_KHR_cooperative_matrix. It is not merely
 * skipped: the test asserts that Sarek's capability verdict REFUSES the
 * configuration there, in the same run in which the RX 7900 XTX permits it.
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Kernel = Vulkan_api_kernel

let mnk = 16

let elems = mnk * mnk

(* u8 x u8 + s32 -> s32, 16x16x16, subgroup scope, non-saturating. Advertised as
   configuration [1] of 14 on the RX 7900 XTX under radv / Mesa 26.1.4-arch3.1.
   Non-saturating is the one whose host reference is plain two's-complement
   arithmetic; the saturating twin computes a different function and is not
   claimed here. *)
let cfg_u8_u8_s32 =
  {
    Sarek_coopmat.cfg_shape = {Sarek_coopmat.m = mnk; n = mnk; k = mnk};
    cfg_a = Sarek_coopmat.Uint8;
    cfg_b = Sarek_coopmat.Uint8;
    cfg_c = Sarek_coopmat.Sint32;
    cfg_result = Sarek_coopmat.Sint32;
    cfg_saturating = false;
    cfg_scope = Sarek_coopmat.Subgroup;
  }

(* Hand-written, deliberately: this file measures the driver. It is NOT produced
   by Sarek_ir_glsl and must not be, or a codegen defect would cancel against a
   driver defect and the run would still be green. *)
let coopmat_source ~local_size =
  Printf.sprintf
    {|#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(local_size_x = %d) in;
layout(std430, binding = 0) readonly buffer BufA { uint8_t a[]; };
layout(std430, binding = 1) readonly buffer BufB { uint8_t b[]; };
layout(std430, binding = 2) readonly buffer BufC { int32_t c[]; };
layout(std430, binding = 3) writeonly buffer BufD { int32_t d[]; };

void main() {
  coopmat<uint8_t, gl_ScopeSubgroup, %d, %d, gl_MatrixUseA> A;
  coopmat<uint8_t, gl_ScopeSubgroup, %d, %d, gl_MatrixUseB> B;
  coopmat<int32_t, gl_ScopeSubgroup, %d, %d, gl_MatrixUseAccumulator> C;
  coopMatLoad(A, a, 0, %d, gl_CooperativeMatrixLayoutRowMajor);
  coopMatLoad(B, b, 0, %d, gl_CooperativeMatrixLayoutRowMajor);
  coopMatLoad(C, c, 0, %d, gl_CooperativeMatrixLayoutRowMajor);
  C = coopMatMulAdd(A, B, C);
  coopMatStore(C, d, 0, %d, gl_CooperativeMatrixLayoutRowMajor);
}
|}
    local_size
    mnk
    mnk
    mnk
    mnk
    mnk
    mnk
    mnk
    mnk
    mnk
    mnk

(** [D = A x B + C] over the integers, exactly, in Int32 two's-complement.

    This is the oracle. It is written in terms of [Int32] operations rather than
    OCaml's native [int] so that a result which would overflow 32 bits wraps the
    way the specification says the device wraps, instead of quietly being right
    for the wrong reason on a 63-bit host int. *)
let reference ~a ~b ~c ~drop_c ~transpose_b =
  let d = Array.make elems 0l in
  for i = 0 to mnk - 1 do
    for j = 0 to mnk - 1 do
      let acc = ref (if drop_c then 0l else c.((i * mnk) + j)) in
      for k = 0 to mnk - 1 do
        let av = Int32.of_int a.((i * mnk) + k) in
        let bv =
          Int32.of_int
            (if transpose_b then b.((j * mnk) + k) else b.((k * mnk) + j))
        in
        acc := Int32.add !acc (Int32.mul av bv)
      done ;
      d.((i * mnk) + j) <- !acc
    done
  done ;
  d

let bit_identical (x : int32 array) (y : int32 array) =
  Array.length x = Array.length y
  &&
  let ok = ref true in
  Array.iteri (fun i v -> if v <> y.(i) then ok := false) x ;
  !ok

let first_divergence (x : int32 array) (y : int32 array) =
  let rec go i =
    if i >= Array.length x then None
    else if x.(i) <> y.(i) then Some (i, x.(i), y.(i))
    else go (i + 1)
  in
  go 0

(** One dispatch: upload A, B, C, run, read D back. *)
let dispatch device ~local_size ~a ~b ~c =
  let compiled =
    Kernel.compile_cached
      device
      ~name:(Printf.sprintf "coopmat_u8_s32_%d" local_size)
      ~source:(coopmat_source ~local_size)
  in
  let buf_a = Memory.alloc device elems Bigarray.int8_unsigned in
  let buf_b = Memory.alloc device elems Bigarray.int8_unsigned in
  let buf_c = Memory.alloc device elems Bigarray.int32 in
  let buf_d = Memory.alloc device elems Bigarray.int32 in
  let free_all () = List.iter Memory.free [buf_a; buf_b; buf_c; buf_d] in
  Fun.protect ~finally:free_all (fun () ->
      let ha =
        Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout elems
      and hb =
        Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout elems
      and hc = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout elems
      and hd = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout elems in
      for i = 0 to elems - 1 do
        ha.{i} <- a.(i) ;
        hb.{i} <- b.(i) ;
        hc.{i} <- c.(i) ;
        hd.{i} <- 0l
      done ;
      Memory.host_to_device ~src:ha ~dst:buf_a ;
      Memory.host_to_device ~src:hb ~dst:buf_b ;
      Memory.host_to_device ~src:hc ~dst:buf_c ;
      Memory.host_to_device ~src:hd ~dst:buf_d ;
      let args = Kernel.create_args () in
      Kernel.set_arg_buffer args 0 buf_a ;
      Kernel.set_arg_buffer args 1 buf_b ;
      Kernel.set_arg_buffer args 2 buf_c ;
      Kernel.set_arg_buffer args 3 buf_d ;
      let block = Spoc_framework.Framework_sig.dims_1d local_size in
      let grid = Spoc_framework.Framework_sig.dims_1d 1 in
      Kernel.launch compiled ~args ~grid ~block ~shared_mem:0 ~stream:None ;
      Device.synchronize device ;
      Memory.device_to_host ~src:buf_d ~dst:hd ;
      Array.init elems (fun i -> hd.{i}))

(* ------------------------------------------------------------------ *)
(* device selection                                                    *)
(* ------------------------------------------------------------------ *)

type classified = {
  dev : Device.t;
  permits : bool;  (** Sarek's own verdict for [cfg_u8_u8_s32]. *)
}

let classify () =
  let n = try Device.count () with _ -> 0 in
  List.init n (fun i ->
      let dev = Device.get i in
      let caps = Vulkan_plugin_base.Vulkan.Device.capabilities dev in
      let verdict =
        Sarek_coopmat.verdict
          ~support:caps.Spoc_framework.Framework_sig.coopmat
          cfg_u8_u8_s32
      in
      {dev; permits = Sarek_capability.permits verdict})

let with_vulkan f =
  if not (Vulkan_api.is_available ()) then Alcotest.skip ()
  else begin
    Device.init () ;
    f (classify ())
  end

(* ------------------------------------------------------------------ *)
(* tests                                                               *)
(* ------------------------------------------------------------------ *)

(** The device plumbing this path needs, reported as ENABLED rather than merely
    supported.

    Asserted only on a device whose verdict PERMITS the configuration: on such a
    device the three features are not optional extras but a precondition, since
    the shader below cannot be loaded legally without them. Asserting them
    unconditionally would fail on a device that legitimately has no coopmat at
    all, which is not the claim. *)
let test_features_enabled () =
  with_vulkan (fun devs ->
      let permitting = List.filter (fun d -> d.permits) devs in
      if permitting = [] then Alcotest.skip () ;
      List.iter
        (fun d ->
          let name = d.dev.Device.name in
          Alcotest.(check bool)
            (name ^ ": shaderInt8 enabled")
            true
            d.dev.Device.supports_int8 ;
          Alcotest.(check bool)
            (name ^ ": storageBuffer8BitAccess enabled")
            true
            d.dev.Device.storage_buffer_8bit ;
          Alcotest.(check bool)
            (name ^ ": vulkanMemoryModel enabled")
            true
            d.dev.Device.vulkan_memory_model)
        permitting)

(** The subgroup size is the calling convention, and it is read from the device.

    [Sarek_coopmat.config_fits_subgroup] must accept the configuration at the
    size this device actually reports, because a fragment that does not divide
    over the subgroup has no layout at all. Asserting PROBED and not merely
    positive is deliberate: [fallback_subgroup_size] guarantees positivity, so a
    positivity assertion cannot fail and is not evidence. *)
let test_subgroup_convention () =
  with_vulkan (fun devs ->
      let permitting = List.filter (fun d -> d.permits) devs in
      if permitting = [] then Alcotest.skip () ;
      List.iter
        (fun d ->
          let sg = d.dev.Device.subgroup_size in
          Alcotest.(check bool)
            (d.dev.Device.name ^ ": subgroup size is probed, not the fallback")
            true
            d.dev.Device.subgroup_size_probed ;
          Alcotest.(check bool)
            (Printf.sprintf
               "%s: 16x16x16 fits a subgroup of %d"
               d.dev.Device.name
               sg)
            true
            (Sarek_coopmat.config_fits_subgroup ~subgroup_size:sg cfg_u8_u8_s32) ;
          (* The whole point of slice 2's correction: 4 components per
             invocation at 64, not 8 at the 32 that used to be hard-coded. *)
          let frag_a, _, _, _ =
            Sarek_coopmat.fragments_of_config cfg_u8_u8_s32
          in
          match
            Sarek_coopmat.components_per_invocation ~subgroup_size:sg frag_a
          with
          | Ok cpi ->
              Alcotest.(check int)
                (Printf.sprintf
                   "%s: components per invocation"
                   d.dev.Device.name)
                (elems / sg)
                cpi
          | Error e -> Alcotest.fail e)
        permitting)

(** The gate refuses on a device that does not advertise the extension.

    NOT tautological: it does not ask the configuration list whether the
    configuration is in the configuration list. It asserts a relation between
    two INDEPENDENTLY OBSERVED facts — the extension bit that
    [vkEnumerateDeviceExtensionProperties] reported, and the verdict. A verdict
    that permitted on a device with no extension, or refused on one that has it
    and advertises the configuration, fails here. *)
let test_negative_device_refuses () =
  with_vulkan (fun devs ->
      if List.length devs < 2 then Alcotest.skip () ;
      let refusing =
        List.filter
          (fun d -> not d.dev.Device.coopmat_extension_advertised)
          devs
      in
      if refusing = [] then Alcotest.skip () ;
      List.iter
        (fun d ->
          Alcotest.(check bool)
            (d.dev.Device.name
           ^ ": no VK_KHR_cooperative_matrix, so the verdict must refuse")
            false
            d.permits ;
          (* And the refusal must be a DIAGNOSTIC, naming the capability. *)
          let cap = Sarek_coopmat.device_lacks_config cfg_u8_u8_s32 in
          let msg = Sarek_capability.explain ~target:d.dev.Device.name cap in
          Alcotest.(check bool)
            "the refusal names the device"
            true
            (String.length msg > 0
            && Astring.String.is_infix ~affix:d.dev.Device.name msg))
        refusing ;
      (* The separation itself: at least one device permits while another
         refuses, in the same run, under the same driver build. Without this the
         two arms above could both be vacuous. *)
      let permits = List.exists (fun d -> d.permits) devs in
      Alcotest.(check bool)
        "some device permits, so the refusal above is a separation"
        true
        permits)

(** Load, store and operand mapping, isolated from the sum.

    [B = I] and [C = 0] make [D = A]. A wrong row/column layout, a wrong stride,
    or A and B swapped all survive the exhaustive-product test below (which is
    symmetric in enough of the wrong ways to be fooled) and die here. *)
let test_identity_layout () =
  with_vulkan (fun devs ->
      let permitting = List.filter (fun d -> d.permits) devs in
      if permitting = [] then Alcotest.skip () ;
      List.iter
        (fun d ->
          let a = Array.init elems (fun i -> i * 7 land 0xff) in
          let b =
            Array.init elems (fun i -> if i / mnk = i mod mnk then 1 else 0)
          in
          let c = Array.make elems 0l in
          let got =
            dispatch d.dev ~local_size:d.dev.Device.subgroup_size ~a ~b ~c
          in
          let want = Array.init elems (fun i -> Int32.of_int a.(i)) in
          (match first_divergence got want with
          | None -> ()
          | Some (i, g, w) ->
              Alcotest.failf
                "%s: D = A x I + 0 diverges at %d: got %ld want %ld"
                d.dev.Device.name
                i
                g
                w) ;
          Alcotest.(check bool) "D = A x I" true (bit_identical got want))
        permitting)

(** Every ordered pair of u8 operand values, exactly once, with a nonzero C.

    Sixteen dispatches; see the header for why that is exhaustive rather than a
    sample. The two positive controls run on the SAME measured output, so a
    green result here is a claim about a comparison that has been observed to
    reject two specific wrong answers. *)
let test_exhaustive_operand_pairs () =
  with_vulkan (fun devs ->
      let permitting = List.filter (fun d -> d.permits) devs in
      if permitting = [] then Alcotest.skip () ;
      List.iter
        (fun d ->
          let controls_fired_drop_c = ref 0 and controls_fired_transp = ref 0 in
          for t = 0 to mnk - 1 do
            let a =
              Array.init elems (fun p ->
                  ((16 * (p mod mnk)) + (p / mnk)) land 0xff)
            and b =
              Array.init elems (fun p -> ((16 * t) + (p mod mnk)) land 0xff)
            and c =
              (* Nonzero, mixed sign, and not a constant — a C that is dropped
                 or added twice must change the answer at every position. *)
              Array.init elems (fun p -> Int32.of_int ((p * 37 mod 1009) - 500))
            in
            let got =
              dispatch d.dev ~local_size:d.dev.Device.subgroup_size ~a ~b ~c
            in
            let want = reference ~a ~b ~c ~drop_c:false ~transpose_b:false in
            (match first_divergence got want with
            | None -> ()
            | Some (i, g, w) ->
                Alcotest.failf
                  "%s: tile %d diverges at %d: got %ld want %ld"
                  d.dev.Device.name
                  t
                  i
                  g
                  w) ;
            Alcotest.(check bool)
              (Printf.sprintf "%s: tile %d exact" d.dev.Device.name t)
              true
              (bit_identical got want) ;
            (* Positive control 1 (§5.4's C-dropping reference). *)
            let no_c = reference ~a ~b ~c ~drop_c:true ~transpose_b:false in
            if not (bit_identical got no_c) then incr controls_fired_drop_c ;
            (* Positive control 2: B transposed. *)
            let tb = reference ~a ~b ~c ~drop_c:false ~transpose_b:true in
            if not (bit_identical got tb) then incr controls_fired_transp
          done ;
          Alcotest.(check int)
            (d.dev.Device.name
           ^ ": the C-dropping reference is rejected on every tile")
            mnk
            !controls_fired_drop_c ;
          Alcotest.(check int)
            (d.dev.Device.name
           ^ ": the transposed-B reference is rejected on every tile")
            mnk
            !controls_fired_transp)
        permitting)

(** Integer accumulation is EXACT, which is the whole reason this path lands
    under the strict contract — so it must also be exact where it wraps.

    A saturating configuration computes a different function; this one is the
    non-saturating variant and the specification says it wraps at the result
    precision. Driving C to just below [Int32.max_int] and requiring the
    two's-complement wrap makes "exact" a testable claim rather than a claim
    about small numbers only. *)
let test_wraparound_is_exact () =
  with_vulkan (fun devs ->
      let permitting = List.filter (fun d -> d.permits) devs in
      if permitting = [] then Alcotest.skip () ;
      List.iter
        (fun d ->
          let a = Array.init elems (fun _ -> 255)
          and b = Array.init elems (fun _ -> 255)
          and c = Array.make elems (Int32.sub Int32.max_int 1000l) in
          let got =
            dispatch d.dev ~local_size:d.dev.Device.subgroup_size ~a ~b ~c
          in
          let want = reference ~a ~b ~c ~drop_c:false ~transpose_b:false in
          (* The reference must actually be exercising the wrap, or this test
             proves nothing about wrapping. 16 * 255 * 255 = 1040400 >> 1000. *)
          Alcotest.(check bool)
            "the reference wrapped, so the case is not vacuous"
            true
            (Int32.compare want.(0) 0l < 0) ;
          (match first_divergence got want with
          | None -> ()
          | Some (i, g, w) ->
              Alcotest.failf
                "%s: wrap diverges at %d: got %ld want %ld"
                d.dev.Device.name
                i
                g
                w) ;
          Alcotest.(check bool)
            (d.dev.Device.name ^ ": wrapping accumulation is exact")
            true
            (bit_identical got want))
        permitting)

let () =
  Alcotest.run
    "Vulkan integer cooperative matrix"
    [
      ( "plumbing",
        [
          Alcotest.test_case
            "the device features the path needs are enabled"
            `Quick
            test_features_enabled;
          Alcotest.test_case
            "the subgroup calling convention is probed and divides"
            `Quick
            test_subgroup_convention;
        ] );
      ( "gate",
        [
          Alcotest.test_case
            "a device without the extension is refused"
            `Quick
            test_negative_device_refuses;
        ] );
      ( "numerics",
        [
          Alcotest.test_case
            "D = A x I + 0 recovers A"
            `Quick
            test_identity_layout;
          Alcotest.test_case
            "every u8 operand pair, exact, controls rejected"
            `Quick
            test_exhaustive_operand_pairs;
          Alcotest.test_case
            "wrapping accumulation is exact"
            `Quick
            test_wraparound_is_exact;
        ] );
    ]
