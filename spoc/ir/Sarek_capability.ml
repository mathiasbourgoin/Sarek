(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The kernel/backend capability vocabulary (#64, slice 1). See the .mli for
    why a capability is not one boolean per device. *)

type kind =
  | Backend_structural
  | Device_optional
  | Host_toolchain
  | Toolchain_semantic
  | Policy
  | Flag_legality

let kind_name = function
  | Backend_structural -> "backend-structural"
  | Device_optional -> "device-optional"
  | Host_toolchain -> "host-toolchain"
  | Toolchain_semantic -> "toolchain-semantic"
  | Policy -> "policy"
  | Flag_legality -> "flag-legality"

(* Backend_structural and Policy are decided by things we already know when we
   emit code: the target language, and our own recorded decisions. Everything
   else needs something outside the compiler to be interrogated first — a
   device, a host compiler, or a measurement. That split is exactly the
   static/dynamic split of #64, so it is one function rather than a comment. *)
let kind_needs_device = function
  | Backend_structural | Policy -> false
  | Device_optional | Host_toolchain | Toolchain_semantic | Flag_legality ->
      true

type evidence =
  | Measured of string
  | Quoted of string
  | By_construction of string

let evidence_text = function Measured s | Quoted s | By_construction s -> s

let evidence_provenance = function
  | Measured _ -> "measured"
  | Quoted _ -> "quoted"
  | By_construction _ -> "by construction"

type t = {
  cap_name : string;
  cap_kind : kind;
  cap_why : string;
  cap_evidence : evidence;
  cap_remedy : string option;
}

type verdict = Available | Unavailable of t | Unknown of string

(* The safety property. Written as an explicit match on all three constructors
   rather than [v = Available] or [function Unavailable _ -> false | _ -> true]
   so that adding a fourth verdict is a compile error here, at the one place
   that decides whether something is allowed to run. The failure mode this
   guards against is a future [Unknown]-like case defaulting to permitted. *)
let permits = function
  | Available -> true
  | Unavailable _ -> false
  | Unknown _ -> false

let first_refusal verdicts = List.find_opt (fun v -> not (permits v)) verdicts

let explain ~target cap =
  let remedy = match cap.cap_remedy with None -> "" | Some r -> " " ^ r in
  Printf.sprintf
    "%s: %s unavailable — %s [%s; %s: %s]%s"
    target
    cap.cap_name
    cap.cap_why
    (kind_name cap.cap_kind)
    (evidence_provenance cap.cap_evidence)
    (evidence_text cap.cap_evidence)
    remedy

let refuse_if_used ~raise_ ~target cap (feature : Sarek_ir_analysis.feature)
    (k : Sarek_ir_types.kernel) : unit =
  if Sarek_ir_analysis.kernel_uses feature k then raise_ (explain ~target cap)

let device_lacks_feature (f : Sarek_ir_analysis.feature) =
  let cap_why, cap_evidence, cap_remedy =
    match f with
    | Sarek_ir_analysis.Int64 ->
        ( "the device does not report VkPhysicalDeviceFeatures.shaderInt64 (or \
           the backend equivalent), and GLSL int64_t lowers to SPIR-V \
           declaring OpCapability Int64, which is legal only against a device \
           that enabled the feature",
          Quoted
            "Vulkan specification, VUID-VkShaderModuleCreateInfo-pCode-08740: \
             a declared SPIR-V capability must have its corresponding \
             requirement satisfied",
          Some "Use int32, or select a device that reports int64 support." )
    | Sarek_ir_analysis.Float64 ->
        ( "the device does not report double-precision support \
           (VkPhysicalDeviceFeatures.shaderFloat64 / cl_khr_fp64)",
          Quoted
            "Vulkan specification, VUID-VkShaderModuleCreateInfo-pCode-08740, \
             and OpenCL: cl_khr_fp64 is an optional extension",
          Some
            "Use float32, or Sarek_real64 — its Fallback_df64 substrate gives \
             software double precision on devices without native fp64." )
    | Sarek_ir_analysis.Float16 ->
        (* [shaderFloat16] is NOT in core VkPhysicalDeviceFeatures. It lives on
           VkPhysicalDeviceShaderFloat16Int8Features, queried and enabled by
           chaining that struct into VkPhysicalDeviceFeatures2 via pNext. The
           distinction is the difference between a feature you can request and
           one you cannot reach at all, in a file about where features must be
           requested — and this whole issue is about a feature that was never
           requested, so naming the wrong struct here is worse than average.

           The cl_khr_fp16 half was already correct: half precision genuinely
           is an optional OpenCL extension, and [half] is not a core OpenCL C
           arithmetic type (unlike [long], whose over-claim is the sibling
           defect fixed alongside this comment). *)
        ( "the device does not report half-precision support \
           (VkPhysicalDeviceShaderFloat16Int8Features.shaderFloat16 / \
           cl_khr_fp16)",
          Quoted
            "Vulkan specification: shaderFloat16 is an optional feature of \
             VkPhysicalDeviceShaderFloat16Int8Features, chained via \
             VkPhysicalDeviceFeatures2; OpenCL: cl_khr_fp16 is an optional \
             extension",
          Some "Use float32." )
    | Sarek_ir_analysis.Coopmat ->
        (* Reached only through [device_lacks_feature], i.e. by a caller that
           knows a kernel wants cooperative matrices but has no CONFIGURATION to
           name. {!Sarek_coopmat.device_lacks_config} is the richer refusal and
           is what the launch gate uses; this is the one a backend emits when it
           cannot spell the instruction at all, where naming a configuration
           would suggest that a different one might work. *)
        ( "the device advertises no cooperative-matrix support \
           (VK_KHR_cooperative_matrix and its cooperativeMatrix feature, or \
           the backend equivalent)",
          Quoted
            "Vulkan specification, VK_KHR_cooperative_matrix: cooperative \
             matrices are an optional device feature, and \
             SPV_KHR_cooperative_matrix states that INTEGER accumulation is \
             exact at the precision of the result type",
          Some
            "Use an ordinary multiply-accumulate kernel, or select a device \
             that advertises the cooperative-matrix configuration you need." )
  in
  {
    cap_name = Sarek_ir_analysis.feature_name f;
    (* Device_optional, emphatically not Backend_structural. The backend CAN
       spell every one of these types; a particular device may not provide it.
       Getting this wrong in the permissive direction is #142; getting it wrong
       in the other direction would refuse kernels that run fine on most
       hardware. [kind_needs_device] is [true] here, which is what says this
       must be a launch gate rather than a codegen refusal. *)
    cap_kind = Device_optional;
    cap_why;
    cap_evidence;
    cap_remedy;
  }

(** The verdict for [f] against a device advertising [provided].

    [provided] is the device's OWN report. A caller that cannot obtain one must
    pass [None] and gets [Unknown], which {!permits} refuses — the unprobed
    device must not land in the permitted bucket, which is the whole safety
    property of this module. *)
let device_verdict ~(provided : Sarek_ir_analysis.feature list option)
    (f : Sarek_ir_analysis.feature) : verdict =
  match provided with
  | None ->
      Unknown
        (Printf.sprintf
           "device capabilities were not probed, so %s cannot be confirmed"
           (Sarek_ir_analysis.feature_name f))
  | Some features ->
      if List.mem f features then Available
      else Unavailable (device_lacks_feature f)

let float64_absent_metal =
  {
    cap_name = "float64";
    cap_kind = Backend_structural;
    cap_why =
      "the Metal Shading Language has no double-precision scalar type, so \
       there is nothing for binary64 to lower to";
    cap_evidence =
      Quoted
        "Metal Shading Language Specification: `double` is not supported; \
         Metal's floating-point scalar types are `half` and `float`";
    cap_remedy =
      Some
        "Use float32, or Sarek_real64 — its Fallback_df64 substrate gives \
         software double precision on devices without native fp64.";
  }
