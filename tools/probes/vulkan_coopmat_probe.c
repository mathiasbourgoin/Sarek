/* vulkan_coopmat_probe.c — enumerate VK_KHR_cooperative_matrix support.
 *
 * Backs the coopmat availability table in docs/design/f16-relaxed-accuracy.md.
 * Reports, per physical device: whether the extension is advertised, whether
 * the cooperativeMatrix feature is enabled, whether shaderFloat16 and
 * storageBuffer16BitAccess are present (Vulkan requires shaderFloat16 before a
 * shader may use the SPIR-V Float16 capability at all, and Sarek does not
 * enable it today — see docs/fp-contraction-policy.md §7), and the full list of
 * supported M x N x K / component-type / scope configurations.
 *
 * Needs Vulkan headers new enough to declare VkCooperativeMatrixPropertiesKHR.
 * Arch's vulkan-icd-loader ships libvulkan but no headers; any upstream
 * Vulkan-Headers checkout will do.
 *
 * Build:
 *   gcc -O1 -I<vulkan-headers-include-dir> tools/probes/vulkan_coopmat_probe.c \
 *       -lvulkan -o vulkan_coopmat_probe
 *
 * Read-only: it creates an instance, queries, and destroys it. No device is
 * created and no shader is compiled.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const char *ctype(VkComponentTypeKHR t) {
  switch (t) {
  case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
  case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
  case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
  case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
  case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
  case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
  case VK_COMPONENT_TYPE_SINT64_KHR: return "s64";
  case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
  case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
  case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
  case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
  default: return "?";
  }
}

static const char *scope(VkScopeKHR s) {
  switch (s) {
  case VK_SCOPE_DEVICE_KHR: return "device";
  case VK_SCOPE_WORKGROUP_KHR: return "workgroup";
  case VK_SCOPE_SUBGROUP_KHR: return "subgroup";
  case VK_SCOPE_QUEUE_FAMILY_KHR: return "queuefamily";
  default: return "?";
  }
}

int main(void) {
  VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName = "sarek-coopmat-probe",
                           .apiVersion = VK_API_VERSION_1_3};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &app};
  VkInstance inst;
  if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
    fprintf(stderr, "vkCreateInstance failed\n");
    return 1;
  }

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, NULL);
  VkPhysicalDevice *pds = calloc(n, sizeof *pds);
  vkEnumeratePhysicalDevices(inst, &n, pds);

  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getProps =
      (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
          inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

  for (uint32_t i = 0; i < n; i++) {
    VkPhysicalDeviceDriverProperties drv = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
    vkGetPhysicalDeviceProperties2(pds[i], &p2);
    printf("=== device %u: %s\n", i, p2.properties.deviceName);
    printf("    driverName=%s driverInfo=%s apiVersion=%u.%u.%u\n", drv.driverName,
           drv.driverInfo, VK_VERSION_MAJOR(p2.properties.apiVersion),
           VK_VERSION_MINOR(p2.properties.apiVersion),
           VK_VERSION_PATCH(p2.properties.apiVersion));

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pds[i], NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pds[i], NULL, &en, exts);
    int has_cm = 0, has_f16int8 = 0, has_16bit = 0;
    for (uint32_t j = 0; j < en; j++) {
      if (!strcmp(exts[j].extensionName, "VK_KHR_cooperative_matrix")) has_cm = 1;
      if (!strcmp(exts[j].extensionName, "VK_KHR_shader_float16_int8")) has_f16int8 = 1;
      if (!strcmp(exts[j].extensionName, "VK_KHR_16bit_storage")) has_16bit = 1;
    }
    printf("    VK_KHR_cooperative_matrix=%s VK_KHR_shader_float16_int8=%s "
           "VK_KHR_16bit_storage=%s\n",
           has_cm ? "YES" : "no", has_f16int8 ? "YES" : "no",
           has_16bit ? "YES" : "no");

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat16Int8Features f16f = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .pNext = &cmf};
    VkPhysicalDevice16BitStorageFeatures s16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .pNext = &f16f};
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &s16};
    vkGetPhysicalDeviceFeatures2(pds[i], &f2);
    printf("    features: cooperativeMatrix=%d shaderFloat16=%d "
           "storageBuffer16BitAccess=%d\n",
           cmf.cooperativeMatrix, f16f.shaderFloat16, s16.storageBuffer16BitAccess);

    if (has_cm && getProps) {
      uint32_t cn = 0;
      getProps(pds[i], &cn, NULL);
      VkCooperativeMatrixPropertiesKHR *cps = calloc(cn, sizeof *cps);
      for (uint32_t j = 0; j < cn; j++)
        cps[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      getProps(pds[i], &cn, cps);
      printf("    %u cooperative-matrix configurations:\n", cn);
      for (uint32_t j = 0; j < cn; j++)
        printf("      M=%-3u N=%-3u K=%-3u  A=%-3s B=%-3s C=%-3s R=%-3s  "
               "scope=%s satAccum=%d\n",
               cps[j].MSize, cps[j].NSize, cps[j].KSize, ctype(cps[j].AType),
               ctype(cps[j].BType), ctype(cps[j].CType), ctype(cps[j].ResultType),
               scope(cps[j].scope), cps[j].saturatingAccumulation);
      free(cps);
    }
    free(exts);
  }
  vkDestroyInstance(inst, NULL);
  return 0;
}
