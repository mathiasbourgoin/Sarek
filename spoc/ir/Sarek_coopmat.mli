(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrix: the DEVICE-facing half.

    Every type and value of {!Sarek_coopmat_types} is re-exported here by
    [include module type of], with its type equalities intact, so
    [Sarek_coopmat.fragment], [Sarek_coopmat.Uint8] and the rest mean exactly
    what they meant before slice 3 split the module. Read
    {!Sarek_coopmat_types}'s interface for the vocabulary and the design
    rationale behind it.

    What is defined HERE is the part that consults {!Sarek_capability}, and it
    is separated for a mechanical reason as much as a conceptual one: the IR
    names a fragment, {!Sarek_capability} depends on the IR, and a vocabulary
    that also depended on {!Sarek_capability} would close a module cycle. *)

(* [struct include ... end] and not the bare module path: the bare form
   re-declares each type ABSTRACTLY, so [Sarek_coopmat.config] would be a
   distinct type from [Sarek_coopmat_types.config] and every value crossing
   between the IR and the capability layer would need a conversion that does
   nothing. This form preserves the type equalities, which is the whole point of
   the re-export. *)
include module type of struct
  include Sarek_coopmat_types
end

(** [verdict ~support cfg] judges a requested configuration against a device.

    [support = None] yields {!Sarek_capability.Unknown}, which
    {!Sarek_capability.permits} refuses. That is the reason this takes an option
    and the reason {!Framework_sig.capabilities} stores an option: an empty
    configuration list and an unprobed device are different facts, and only one
    of them is evidence.

    {b The gate must be keyed on the feature, not on the configuration list},
    and this is measured rather than argued. On this workstation
    [vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR] returns [VK_SUCCESS] and
    FOURTEEN configurations for the Raphael iGPU — a device that does not
    advertise [VK_KHR_cooperative_matrix] and whose [cooperativeMatrix] feature
    reads false. A probe that populated [ds_configs] from that call without
    first checking the extension would report the iGPU as fully capable, and
    this verdict would say [Available] for a device that cannot run the
    instruction. The Vulkan probe therefore leaves [ds_configs] empty unless the
    extension is advertised AND the feature is true; see
    [sarek-vulkan/Vulkan_api_device.ml]. *)
val verdict :
  support:device_support option -> config -> Sarek_capability.verdict

(** The {!Sarek_capability.Device_optional} record for a device that provides no
    cooperative-matrix support at all, or none matching the request. *)
val device_lacks_config : config -> Sarek_capability.t
