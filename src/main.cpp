// vkcontrast: how fast can a contrast slider update a RAW photo on this GPU?
//
// The photo is decoded once with LibRaw into linear 16-bit RGB and uploaded to the
// GPU once, with a full mip chain. On a zoom or resize, a compute pass resamples it
// to the exact on-screen size and keeps that image. While a slider moves, the only
// GPU work is one fragment pass over the on-screen pixels: one texel fetch,
// exposure, sRGB encoding and the contrast curve, with the slider values pushed as
// push constants. Shaders are compiled to SPIR-V at build time, pipelines are made
// once at startup from a pipeline cache kept on disk, and frames are drawn only
// when something changes.

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <libraw/libraw.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "recipe.h"
#include "tone_ref.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const uint32_t kVertSpv[] = {
#include "fullscreen.vert.inc"
};
const uint32_t kFragSpv[] = {
#include "develop.frag.inc"
};
#define SPIRV(name) const uint32_t k_##name[] = {
SPIRV(tone_display)
#include "tone_display.comp.inc"
};
SPIRV(measure_tone)
#include "measure_tone.comp.inc"
};
SPIRV(select_hist)
#include "select_hist.comp.inc"
};
SPIRV(select_scan)
#include "select_scan.comp.inc"
};
SPIRV(box_rows)
#include "box_rows.comp.inc"
};
SPIRV(box_cols)
#include "box_cols.comp.inc"
};
SPIRV(pivot_blocks)
#include "pivot_blocks.comp.inc"
};
SPIRV(pivot_sort)
#include "pivot_sort.comp.inc"
};
SPIRV(build_lut)
#include "build_lut.comp.inc"
};
const uint32_t kCompSpv[] = {
#include "resample.comp.inc"
};

// The photo's height on screen in RAWmakase's Fit view on this laptop, in physical
// pixels, measured from the screen recording (935 x 1402).
const int kRawmakaseHeight = 1402;
const int kFramesInFlight = 2;
const float kPanelWidth = 340.f; // logical pixels
const VkFormat kSourceFormat = VK_FORMAT_R16G16B16A16_UNORM;
const VkFormat kDisplayFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

[[noreturn]] void die(const char* what, VkResult r) {
    std::fprintf(stderr, "vkcontrast: %s failed (VkResult %d)\n", what, r);
    std::exit(1);
}
#define VK(x)                                                                        \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) die(#x, r_);                                           \
    } while (0)

// ---------------------------------------------------------------------------------
// Photo

struct Photo {
    uint32_t width = 0, height = 0;
    std::vector<uint16_t> rgba; // linear, sRGB primaries
    float baseline_exposure = 0.f;
    std::string camera;
    double decode_ms = 0;
};

Photo load_photo(const char* path, bool full) {
    auto t0 = Clock::now();
    auto raw = std::make_unique<LibRaw>();
    auto& p = raw->imgdata.params;
    p.output_bps = 16;
    p.gamm[0] = 1.0; // linear output
    p.gamm[1] = 1.0;
    p.no_auto_bright = 1;
    p.use_camera_wb = 1;
    p.output_color = 1; // sRGB primaries
    p.half_size = full ? 0 : 1;
    p.user_qual = 3; // AHD when demosaicing in full
    if (int e = raw->open_file(path); e != LIBRAW_SUCCESS)
        throw std::runtime_error(std::string("open: ") + libraw_strerror(e));
    if (int e = raw->unpack(); e != LIBRAW_SUCCESS)
        throw std::runtime_error(std::string("unpack: ") + libraw_strerror(e));
    if (int e = raw->dcraw_process(); e != LIBRAW_SUCCESS)
        throw std::runtime_error(std::string("process: ") + libraw_strerror(e));
    int err = 0;
    libraw_processed_image_t* img = raw->dcraw_make_mem_image(&err);
    if (!img || img->type != LIBRAW_IMAGE_BITMAP || img->colors != 3 || img->bits != 16)
        throw std::runtime_error("unexpected image from LibRaw");
    Photo photo;
    photo.width = img->width;
    photo.height = img->height;
    photo.rgba.resize(size_t(photo.width) * photo.height * 4);
    const auto* src = reinterpret_cast<const uint16_t*>(img->data);
    for (size_t i = 0, n = size_t(photo.width) * photo.height; i < n; ++i) {
        photo.rgba[i * 4 + 0] = src[i * 3 + 0];
        photo.rgba[i * 4 + 1] = src[i * 3 + 1];
        photo.rgba[i * 4 + 2] = src[i * 3 + 2];
        photo.rgba[i * 4 + 3] = 65535;
    }
    LibRaw::dcraw_clear_mem(img);
    photo.baseline_exposure = raw->imgdata.color.dng_levels.baseline_exposure;
    photo.camera = std::string(raw->imgdata.idata.make) + " " + raw->imgdata.idata.model;
    photo.decode_ms = ms_since(t0);
    return photo;
}

// ---------------------------------------------------------------------------------
// Vulkan helpers

struct Gpu {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool upload_pool = VK_NULL_HANDLE;
};

uint32_t memory_type(const Gpu& g, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g.physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    throw std::runtime_error("no suitable memory type");
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

Buffer make_buffer(const Gpu& g, VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags props) {
    Buffer b;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VK(vkCreateBuffer(g.device, &bi, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.device, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(g, req.memoryTypeBits, props);
    VK(vkAllocateMemory(g.device, &ai, nullptr, &b.memory));
    VK(vkBindBufferMemory(g.device, b.buffer, b.memory, 0));
    return b;
}

void destroy(const Gpu& g, Buffer& b) {
    vkDestroyBuffer(g.device, b.buffer, nullptr);
    vkFreeMemory(g.device, b.memory, nullptr);
    b = {};
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0, levels = 1;
};

Image make_image(const Gpu& g, uint32_t w, uint32_t h, uint32_t levels, VkFormat format,
                 VkImageUsageFlags usage) {
    Image im;
    im.width = w;
    im.height = h;
    im.levels = levels;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = levels;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK(vkCreateImage(g.device, &ci, nullptr, &im.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g.device, im.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(g, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(g.device, &ai, nullptr, &im.memory));
    VK(vkBindImageMemory(g.device, im.image, im.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1};
    VK(vkCreateImageView(g.device, &vi, nullptr, &im.view));
    return im;
}

void destroy(const Gpu& g, Image& im) {
    vkDestroyImageView(g.device, im.view, nullptr);
    vkDestroyImage(g.device, im.image, nullptr);
    vkFreeMemory(g.device, im.memory, nullptr);
    im = {};
}

void barrier(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags2 src_stage,
             VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
             VkAccessFlags2 dst_access, VkImageLayout from, VkImageLayout to,
             uint32_t base_level = 0, uint32_t levels = VK_REMAINING_MIP_LEVELS) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_level, levels, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

template <typename F>
void submit_once(const Gpu& g, F&& record) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g.upload_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(g.device, &ai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cmd, &bi));
    record(cmd);
    VK(vkEndCommandBuffer(cmd));
    VkCommandBufferSubmitInfo cs{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cs.commandBuffer = cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cs;
    VK(vkQueueSubmit2(g.queue, 1, &si, VK_NULL_HANDLE));
    VK(vkQueueWaitIdle(g.queue));
    vkFreeCommandBuffers(g.device, g.upload_pool, 1, &cmd);
}

// Uploads the photo and builds its mip chain with linear blits.
Image upload_photo(const Gpu& g, const Photo& photo) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(g.physical, kSourceFormat, &fp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                      VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                      VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if ((fp.optimalTilingFeatures & need) != need)
        throw std::runtime_error("RGBA16 UNORM cannot be filtered and blitted on this GPU");
    uint32_t levels = uint32_t(std::floor(std::log2(std::max(photo.width, photo.height)))) + 1;
    Image im = make_image(g, photo.width, photo.height, levels, kSourceFormat,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT);
    VkDeviceSize bytes = photo.rgba.size() * sizeof(uint16_t);
    Buffer staging = make_buffer(g, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = nullptr;
    VK(vkMapMemory(g.device, staging.memory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, photo.rgba.data(), bytes);
    vkUnmapMemory(g.device, staging.memory);

    submit_once(g, [&](VkCommandBuffer cmd) {
        barrier(cmd, im.image, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {photo.width, photo.height, 1};
        vkCmdCopyBufferToImage(cmd, staging.buffer, im.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        int32_t w = int32_t(photo.width), h = int32_t(photo.height);
        for (uint32_t level = 1; level < levels; ++level) {
            barrier(cmd, im.image, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, level - 1, 1);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
            blit.srcOffsets[1] = {w, h, 1};
            w = std::max(w / 2, 1);
            h = std::max(h / 2, 1);
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
            blit.dstOffsets[1] = {w, h, 1};
            vkCmdBlitImage(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        }
        // Every level but the last is TRANSFER_SRC now; the last is TRANSFER_DST.
        if (levels > 1)
            barrier(cmd, im.image, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, levels - 1);
        barrier(cmd, im.image, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                levels - 1, 1);
    });
    destroy(g, staging);
    return im;
}

VkShaderModule shader(const Gpu& g, const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule m;
    VK(vkCreateShaderModule(g.device, &ci, nullptr, &m));
    return m;
}

std::filesystem::path cache_path() {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    std::filesystem::path base = xdg && *xdg ? xdg : std::filesystem::path(std::getenv("HOME")) / ".cache";
    return base / "vkcontrast-poc" / "pipelines.bin";
}

// ---------------------------------------------------------------------------------
// Swapchain

struct Swapchain {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR format{};
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkSemaphore> rendered; // one per image, signalled for present
};

VkSurfaceFormatKHR pick_format(const Gpu& g, VkSurfaceKHR surface) {
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical, surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical, surface, &n, formats.data());
    // UNORM: the fragment shader encodes sRGB itself, as ImGui's colors expect.
    for (auto f : formats)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return formats.at(0);
}

VkPresentModeKHR pick_mode(const Gpu& g, VkSurfaceKHR surface) {
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g.physical, surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g.physical, surface, &n, modes.data());
    VkPresentModeKHR want = VK_PRESENT_MODE_MAILBOX_KHR; // newest frame, no tearing
    if (const char* env = std::getenv("VKCONTRAST_PRESENT")) {
        if (!std::strcmp(env, "fifo")) want = VK_PRESENT_MODE_FIFO_KHR;
        if (!std::strcmp(env, "immediate")) want = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    for (auto m : modes)
        if (m == want) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

void destroy(const Gpu& g, Swapchain& sc, bool keep_handle) {
    for (auto v : sc.views) vkDestroyImageView(g.device, v, nullptr);
    for (auto s : sc.rendered) vkDestroySemaphore(g.device, s, nullptr);
    sc.views.clear();
    sc.rendered.clear();
    sc.images.clear();
    if (!keep_handle) {
        vkDestroySwapchainKHR(g.device, sc.swapchain, nullptr);
        sc.swapchain = VK_NULL_HANDLE;
    }
}

void create_swapchain(const Gpu& g, VkSurfaceKHR surface, GLFWwindow* window, Swapchain& sc) {
    VkSurfaceCapabilitiesKHR caps;
    VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.physical, surface, &caps));
    if (caps.currentExtent.width != UINT32_MAX) {
        sc.extent = caps.currentExtent;
    } else {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        sc.extent = {std::clamp(uint32_t(w), caps.minImageExtent.width, caps.maxImageExtent.width),
                     std::clamp(uint32_t(h), caps.minImageExtent.height, caps.maxImageExtent.height)};
    }
    uint32_t count = std::max(caps.minImageCount + 1, 3u);
    if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
    VkSwapchainKHR old = sc.swapchain;
    destroy(g, sc, true);
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = count;
    ci.imageFormat = sc.format.format;
    ci.imageColorSpace = sc.format.colorSpace;
    ci.imageExtent = sc.extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = sc.mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = old;
    VK(vkCreateSwapchainKHR(g.device, &ci, nullptr, &sc.swapchain));
    if (old) vkDestroySwapchainKHR(g.device, old, nullptr);
    uint32_t n = 0;
    vkGetSwapchainImagesKHR(g.device, sc.swapchain, &n, nullptr);
    sc.images.resize(n);
    vkGetSwapchainImagesKHR(g.device, sc.swapchain, &n, sc.images.data());
    for (auto image : sc.images) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = sc.format.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView v;
        VK(vkCreateImageView(g.device, &vi, nullptr, &v));
        sc.views.push_back(v);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore s;
        VK(vkCreateSemaphore(g.device, &si, nullptr, &s));
        sc.rendered.push_back(s);
    }
}

const char* mode_name(VkPresentModeKHR m) {
    switch (m) {
    case VK_PRESENT_MODE_MAILBOX_KHR: return "mailbox";
    case VK_PRESENT_MODE_FIFO_KHR: return "fifo (vsync)";
    case VK_PRESENT_MODE_IMMEDIATE_KHR: return "immediate";
    default: return "other";
    }
}

struct Frame {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    bool timed = false; // its queries hold a finished frame's timestamps
    bool resampled = false;
    bool changed = false; // a slider changed for this frame
    int bench_tag = -1;
    std::string stages;
    std::vector<const char*> marks; // VKCONTRAST_PROFILE: a timestamp after each pass
};

// Push constants of every develop pass (common.glsl Params).
struct Params {
    float rect[4];
    float gain, black, gain0, black0;
    float contrast, highlights, shadows, whites, blacks;
    uint32_t n, w, h, flags;
    int32_t shift, radius;
    uint32_t tone_count, block_rows;
};
static_assert(sizeof(Params) == 84);
constexpr uint32_t F_MAP = 1, F_BASIC = 2, F_ADAPTIVE_WHITES = 4, F_SECOND = 8, F_TONE_UNIFORM = 16;
constexpr VkShaderStageFlags kAllStages =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

struct ResampleParams {
    int32_t size[2];
    float lod;
};

// The Basic panel, in RAWmakase's units: Exposure in EV, the others -1..1.
struct Sliders {
    float exposure = 0, contrast = 0, highlights = 0, shadows = 0, whites = 0, blacks = 0;
    bool operator==(const Sliders&) const = default;
};
const char* const kSliderNames[6] = {"Exposure", "Contrast", "Highlights", "Shadows", "Whites", "Blacks"};
float& slider_ref(Sliders& s, int i) {
    float* v[6] = {&s.exposure, &s.contrast, &s.highlights, &s.shadows, &s.whites, &s.blacks};
    return *v[i];
}

// What a change recomputes before the per-pixel pass.
struct Work {
    bool tone = false;  // the small image toned at the Exposure; keys and highlights
    bool map = false;   // the guided filter
    bool pivot = false; // the Contrast pivot
    bool lut = false;   // the Basic curve
    bool toned = false; // with the cache: the on-screen pixels toned at the Exposure
    std::string describe() const {
        std::string s;
        auto add = [&](bool on, const char* name) {
            if (on) s += s.empty() ? name : std::string(" + ") + name;
        };
        add(tone, "tone + percentiles");
        add(map, "guided filter");
        add(pivot, "pivot");
        add(lut, "curve");
        add(toned, "toned screen image");
        return s.empty() ? "photo pass only" : s + " + photo pass";
    }
};

// Smoothed readouts, so the numbers can be read while dragging.
struct Average {
    double value = 0;
    bool set = false;
    void add(double v) {
        value = set ? value * 0.85 + v * 0.15 : v;
        set = true;
    }
};

float half_to_float(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                --exp;
            }
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* photo_path = nullptr;
    std::string catalog; // RAWmakase's catalog, for the photo's saved edit and profile tone curve
    bool full = false, check = false, start_cached = true;
    int target_height = kRawmakaseHeight;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--full")) full = true;
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--no-cache")) start_cached = false;
        else if (!std::strcmp(argv[i], "--height") && i + 1 < argc) target_height = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--catalog") && i + 1 < argc) catalog = argv[++i];
        else photo_path = argv[i];
    }
    if (!photo_path) {
        std::fprintf(stderr,
                     "usage: vkcontrast PHOTO [--catalog CATALOG.rawmakase] [--full] [--height N] [--no-cache] "
                     "[--check]\n");
        return 2;
    }

    // The edit as RAWmakase saved it, read-only.
    recipe::Settings saved;
    try {
        if (!catalog.empty()) saved = recipe::load(catalog, photo_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vkcontrast: catalog %s: %s\n", catalog.c_str(), e.what());
    }
    ref::ProfileTone profile;
    if (saved.tone_x.size() >= 2) {
        profile.x = saved.tone_x;
        profile.y = saved.tone_y;
    } else {
        profile.x = {0.f, 1.f};
        profile.y = {0.f, 1.f};
    }
    if (saved.found)
        std::printf("Edit from %s: %s, tone curve %zu points, camera exposure %+.2f EV\n",
                    saved.source.c_str(), saved.profile.c_str(), saved.tone_x.size(), saved.camera_exposure);
    else
        std::printf("No saved edit%s%s: neutral sliders, no profile tone curve\n",
                    catalog.empty() ? "" : " in ", catalog.c_str());

    std::printf("Decoding %s (%s)...\n", photo_path, full ? "full, AHD" : "half size");
    Photo photo;
    try {
        photo = load_photo(photo_path, full);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vkcontrast: %s: %s\n", photo_path, e.what());
        return 1;
    }
    const float camera_exposure =
        float(saved.found ? saved.camera_exposure + saved.profile_exposure : photo.baseline_exposure);
    std::printf("%s, %ux%u, decoded in %.0f ms\n", photo.camera.c_str(), photo.width, photo.height,
                photo.decode_ms);

    // The 512-pixel copy the measures and the map are computed on, as RAWmakase reduces it.
    unsigned small_w = 0, small_h = 0;
    std::vector<ref::Rgb> small;
    {
        std::vector<ref::Rgb> linear(size_t(photo.width) * photo.height);
        for (size_t i = 0; i < linear.size(); ++i)
            for (int c = 0; c < 3; ++c) linear[i][c] = photo.rgba[i * 4 + size_t(c)] / 65535.f;
        small = ref::reduce(linear, photo.width, photo.height, ref::MAP_EDGE, small_w, small_h);
    }
    const uint32_t small_n = small_w * small_h;
    const uint32_t block_rows =
        uint32_t(std::max(1.f, std::round(float(48 * small_h) / float(std::max(small_w, 1u)))));
    const int32_t radius = std::max<int32_t>(1, int32_t(std::lround(ref::RADIUS * float(std::max(small_w, small_h)))));
    if (radius > 32) { // the box passes' shared tiles hold a halo of 32
        std::fprintf(stderr, "vkcontrast: guided filter radius %d is too large\n", radius);
        return 1;
    }

    // Window ------------------------------------------------------------------------
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    float scale = 1.f;
    glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &scale, nullptr);
    const float aspect = float(photo.width) / float(photo.height);
    const int margin = 48; // physical pixels around the photo
    int win_w = int((target_height * aspect + 2 * margin) / scale + kPanelWidth);
    int win_h = int((target_height + 2 * margin) / scale);
    GLFWwindow* window = glfwCreateWindow(win_w, win_h, "vkcontrast POC", nullptr, nullptr);
    if (!window) return 1;

    // Instance and device -------------------------------------------------------------
    Gpu g;
    {
        uint32_t n = 0;
        const char** ext = glfwGetRequiredInstanceExtensions(&n);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vkcontrast";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = n;
        ci.ppEnabledExtensionNames = ext;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        if (std::getenv("VKCONTRAST_VALIDATE")) {
            ci.enabledLayerCount = 1;
            ci.ppEnabledLayerNames = &layer;
        }
        VK(vkCreateInstance(&ci, nullptr, &g.instance));
    }
    VkSurfaceKHR surface;
    VK(glfwCreateWindowSurface(g.instance, window, nullptr, &surface));
    {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devices(n);
        vkEnumeratePhysicalDevices(g.instance, &n, devices.data());
        for (auto d : devices) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.apiVersion < VK_API_VERSION_1_3) continue;
            uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qf(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qf.data());
            for (uint32_t i = 0; i < qn; ++i) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
                if (present && (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                    (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) {
                    g.physical = d;
                    g.family = i;
                    g.props = p;
                    break;
                }
            }
            if (g.physical) break;
        }
        if (!g.physical) {
            std::fprintf(stderr, "vkcontrast: no Vulkan 1.3 GPU that can draw to the window\n");
            return 1;
        }
        float priority = 1.f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = g.family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f13;
        const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.pNext = &f2;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &qi;
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = &ext;
        VK(vkCreateDevice(g.physical, &ci, nullptr, &g.device));
        vkGetDeviceQueue(g.device, g.family, 0, &g.queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = g.family;
        pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VK(vkCreateCommandPool(g.device, &pi, nullptr, &g.upload_pool));
    }
    std::printf("GPU: %s\n", g.props.deviceName);

    // The photo on the GPU, once ------------------------------------------------------
    auto t_upload = Clock::now();
    Image source = upload_photo(g, photo);
    const double upload_ms = ms_since(t_upload);
    const uint32_t photo_w = photo.width, photo_h = photo.height;
    photo.rgba = {}; // the CPU copy is no longer needed
    std::printf("Uploaded with %u mip levels in %.0f ms\n", source.levels, upload_ms);

    // Buffers: host-visible, written once (tables, tone curve, small image) or by the
    // GPU only (work arrays, measures, curve). Small enough for any heap.
    auto host_buffer = [&](VkDeviceSize size, const void* data) {
        Buffer b = make_buffer(g, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        VK(vkMapMemory(g.device, b.memory, 0, size, 0, &mapped));
        if (data) std::memcpy(mapped, data, size);
        else std::memset(mapped, 0, size);
        vkUnmapMemory(g.device, b.memory);
        return b;
    };
    Buffer tables_buf = host_buffer(sizeof(tables::DATA), tables::DATA);
    std::vector<float> tone_data(profile.x);
    tone_data.insert(tone_data.end(), profile.y.begin(), profile.y.end());
    Buffer tone_buf = host_buffer(tone_data.size() * sizeof(float), tone_data.data());
    std::vector<float> small_data(size_t(small_n) * 4);
    for (size_t i = 0; i < small_n; ++i)
        for (int c = 0; c < 3; ++c) small_data[i * 4 + size_t(c)] = small[i][c];
    Buffer small_buf = host_buffer(small_data.size() * sizeof(float), small_data.data());
    const VkDeviceSize scratch_size = (VkDeviceSize(9) * small_n + 4096) * sizeof(float);
    Buffer scratch_buf = host_buffer(scratch_size, nullptr);
    const VkDeviceSize meas_size = 16 * 4 + 1024 * 4;
    Buffer meas_buf = host_buffer(meas_size, nullptr);
    Buffer lut_buf = host_buffer((1025 + 96) * sizeof(float), nullptr);
    // Knots at exactly i / (count - 1), as Adobe's curves have: indexed directly.
    bool tone_uniform = profile.x.size() >= 2;
    for (size_t i = 0; i < profile.x.size() && tone_uniform; ++i)
        tone_uniform = profile.x[i] == float(i) / float(profile.x.size() - 1);
    std::printf("Small copy %ux%u, guided filter radius %d, %u pivot blocks\n", small_w, small_h, radius,
                48 * block_rows);

    VkSampler source_sampler, display_sampler, nearest_sampler;
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = VK_LOD_CLAMP_NONE;
        VK(vkCreateSampler(g.device, &si, nullptr, &source_sampler));
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.maxLod = 0.f;
        VK(vkCreateSampler(g.device, &si, nullptr, &display_sampler));
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        VK(vkCreateSampler(g.device, &si, nullptr, &nearest_sampler));
    }

    // Descriptors ----------------------------------------------------------------------
    VkDescriptorSetLayout resample_layout, dev_layout;
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 2;
        li.pBindings = b;
        VK(vkCreateDescriptorSetLayout(g.device, &li, nullptr, &resample_layout));
    }
    {
        VkDescriptorSetLayoutBinding b[9]{};
        for (uint32_t i = 0; i < 9; ++i) {
            b[i].binding = i;
            b[i].descriptorCount = 1;
            b[i].descriptorType = i < 6 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                  : i < 8 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 9;
        li.pBindings = b;
        VK(vkCreateDescriptorSetLayout(g.device, &li, nullptr, &dev_layout));
    }
    VkDescriptorPool pool;
    {
        VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 2;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = sizes;
        VK(vkCreateDescriptorPool(g.device, &pi, nullptr, &pool));
    }
    VkDescriptorSet resample_set, dev_set;
    {
        VkDescriptorSetLayout layouts[2] = {resample_layout, dev_layout};
        VkDescriptorSet sets[2];
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 2;
        ai.pSetLayouts = layouts;
        VK(vkAllocateDescriptorSets(g.device, &ai, sets));
        resample_set = sets[0];
        dev_set = sets[1];
        VkDescriptorBufferInfo infos[6] = {{tables_buf.buffer, 0, VK_WHOLE_SIZE}, {tone_buf.buffer, 0, VK_WHOLE_SIZE},
                                           {small_buf.buffer, 0, VK_WHOLE_SIZE}, {scratch_buf.buffer, 0, VK_WHOLE_SIZE},
                                           {meas_buf.buffer, 0, VK_WHOLE_SIZE}, {lut_buf.buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[6]{};
        for (uint32_t i = 0; i < 6; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = dev_set;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(g.device, 6, w, 0, nullptr);
    }

    Swapchain sc;
    sc.format = pick_format(g, surface);
    sc.mode = pick_mode(g, surface);
    create_swapchain(g, surface, window, sc);

    // Pipelines, made once from the cache ---------------------------------------------------
    auto t_pipelines = Clock::now();
    VkPipelineCache cache;
    size_t cache_bytes_loaded = 0;
    {
        std::vector<char> data;
        if (std::ifstream in{cache_path(), std::ios::binary}) {
            data.assign(std::istreambuf_iterator<char>(in), {});
            cache_bytes_loaded = data.size();
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = data.size();
        ci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(g.device, &ci, nullptr, &cache) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            VK(vkCreatePipelineCache(g.device, &ci, nullptr, &cache));
        }
    }
    auto compute_pipeline = [&](VkPipelineLayout layout, const uint32_t* code, size_t bytes) {
        VkShaderModule m = shader(g, code, bytes);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        ci.layout = layout;
        VkPipeline p;
        VK(vkCreateComputePipelines(g.device, cache, 1, &ci, nullptr, &p));
        vkDestroyShaderModule(g.device, m, nullptr);
        return p;
    };
    VkPipelineLayout resample_pl, dev_pl;
    {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResampleParams)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &resample_layout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        VK(vkCreatePipelineLayout(g.device, &li, nullptr, &resample_pl));
        VkPushConstantRange dev_range{kAllStages, 0, sizeof(Params)};
        li.pSetLayouts = &dev_layout;
        li.pPushConstantRanges = &dev_range;
        VK(vkCreatePipelineLayout(g.device, &li, nullptr, &dev_pl));
    }
#define COMPUTE(name) compute_pipeline(dev_pl, k_##name, sizeof(k_##name))
    VkPipeline resample_pipeline = compute_pipeline(resample_pl, kCompSpv, sizeof(kCompSpv));
    VkPipeline p_measure = COMPUTE(measure_tone), p_hist = COMPUTE(select_hist), p_scan = COMPUTE(select_scan),
               p_rows = COMPUTE(box_rows), p_cols = COMPUTE(box_cols), p_blocks = COMPUTE(pivot_blocks),
               p_sort = COMPUTE(pivot_sort), p_lut = COMPUTE(build_lut), p_toned = COMPUTE(tone_display);
#undef COMPUTE
    VkPipeline draw_pipeline, draw_toned_pipeline;
    {
        VkShaderModule vert = shader(g, kVertSpv, sizeof(kVertSpv));
        VkShaderModule frag = shader(g, kFragSpv, sizeof(kFragSpv));
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                             VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &att;
        VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2;
        ds.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        ri.colorAttachmentCount = 1;
        ri.pColorAttachmentFormats = &sc.format.format;
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.pNext = &ri;
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vin;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vp;
        ci.pRasterizationState = &rs;
        ci.pMultisampleState = &ms;
        ci.pColorBlendState = &cb;
        ci.pDynamicState = &ds;
        ci.layout = dev_pl;
        VK(vkCreateGraphicsPipelines(g.device, cache, 1, &ci, nullptr, &draw_pipeline));
        // The same pass reading the cached toned image (TONED_INPUT).
        VkBool32 toned_input = VK_TRUE;
        VkSpecializationMapEntry entry{0, 0, sizeof(VkBool32)};
        VkSpecializationInfo spec{1, &entry, sizeof(toned_input), &toned_input};
        stages[1].pSpecializationInfo = &spec;
        VK(vkCreateGraphicsPipelines(g.device, cache, 1, &ci, nullptr, &draw_toned_pipeline));
        vkDestroyShaderModule(g.device, vert, nullptr);
        vkDestroyShaderModule(g.device, frag, nullptr);
    }
    const double pipelines_ms = ms_since(t_pipelines);
    std::printf("Pipelines ready in %.1f ms (cache: %zu bytes loaded)\n", pipelines_ms, cache_bytes_loaded);

    // Per-frame resources ------------------------------------------------------------------
    std::array<Frame, kFramesInFlight> frames;
    for (auto& f : frames) {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = g.family;
        pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VK(vkCreateCommandPool(g.device, &pi, nullptr, &f.pool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK(vkAllocateCommandBuffers(g.device, &ai, &f.cmd));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK(vkCreateFence(g.device, &fi, nullptr, &f.fence));
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK(vkCreateSemaphore(g.device, &si, nullptr, &f.acquired));
    }
    // Timestamps per frame: start, after resample, after the measures, after the curve,
    // after the photo pass, end.
    constexpr uint32_t kQueries = 6;
    constexpr uint32_t kMarks = 32;
    VkQueryPool queries, pass_queries;
    {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kQueries * kFramesInFlight;
        VK(vkCreateQueryPool(g.device, &qi, nullptr, &queries));
        qi.queryCount = kMarks * kFramesInFlight;
        VK(vkCreateQueryPool(g.device, &qi, nullptr, &pass_queries));
    }

    // ImGui -------------------------------------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FontScaleMain = 1.25f;
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = g.instance;
    ii.PhysicalDevice = g.physical;
    ii.Device = g.device;
    ii.QueueFamily = g.family;
    ii.Queue = g.queue;
    ii.DescriptorPoolSize = 16;
    ii.MinImageCount = 2;
    ii.ImageCount = uint32_t(sc.images.size());
    ii.PipelineCache = cache;
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &sc.format.format;
    ImGui_ImplVulkan_Init(&ii);

    // State -------------------------------------------------------------------------------
    Image display;           // the photo at its on-screen size, linear
    Image toned;             // with the cache: display through the tone stage
    bool resample = true;    // display must be (re)computed this frame
    bool fit_window = false; // otherwise RAWmakase's Fit size
    Sliders saved_sliders;
    if (saved.found)
        saved_sliders = {float(saved.exposure), float(saved.contrast), float(saved.highlights),
                         float(saved.shadows),  float(saved.whites),   float(saved.blacks)};
    Sliders sliders = saved_sliders;
    if (check) {
        // Every stage at work, with positive Whites for its adaptive table.
        auto set = [](float& v, float d) { if (v == 0.f) v = d; };
        set(sliders.contrast, 0.3f);
        set(sliders.highlights, -0.4f);
        set(sliders.shadows, 0.35f);
        sliders.whites = 0.25f;
        set(sliders.blacks, -0.2f);
    }
    bool use_cache = start_cached;
    // What the cache holds: measures at this Exposure, the pivot, the curve's sliders.
    bool measures_warm = false, pivot_warm = false, toned_warm = false;
    float measured_exposure = 0;
    Sliders curve_sliders;
    bool curve_warm = false;
    Sliders last_drawn;
    bool drawn_once = false;
    bool swapchain_stale = false;
    uint32_t frame_index = 0;
    int frames_to_draw = 3;
    Average gpu_measures, gpu_curve, gpu_photo, gpu_frame, gpu_change, input_to_present;
    std::string last_stages = "-";
    double last_resample_ms = 0;
    int presented = 0;
    double fps = 0;
    auto fps_since = Clock::now();
    const double ns_per_tick = g.props.limits.timestampPeriod;
    const char* dump_path = std::getenv("VKCONTRAST_DUMP");
    if (check && !dump_path) dump_path = "/tmp/vkcontrast-check.ppm";
    int frame_number = 0;
    Buffer dump{}, display_dump{};
    int dump_x0 = 0, dump_y0 = 0;
    Params dump_params{};

    // VKCONTRAST_PROFILE=1: every pass recomputed on every frame, timed one by one.
    const bool profiling = std::getenv("VKCONTRAST_PROFILE") != nullptr;
    std::vector<std::pair<std::string, std::vector<double>>> pass_times;
    if (profiling) {
        // VKCONTRAST_PROFILE=cache: a Shadows drag with the cache on.
        use_cache = !std::strncmp(std::getenv("VKCONTRAST_PROFILE"), "cache", 5);
        check = false;
        auto set = [](float& v, float d) { if (v == 0.f) v = d; };
        set(sliders.contrast, 0.3f);
        set(sliders.highlights, -0.4f);
        set(sliders.shadows, 0.35f);
        sliders.whites = 0.25f;
        set(sliders.blacks, -0.2f);
    }

    // VKCONTRAST_BENCH=1: each slider swept in both modes, one step per frame.
    const bool bench = std::getenv("VKCONTRAST_BENCH") != nullptr;
    const int kBenchSteps = 120;
    int bench_pass = 0, bench_step = -1; // pass = mode * 6 + slider
    std::vector<double> bench_gpu[12], bench_cpu[12];

    while (!glfwWindowShouldClose(window)) {
        if (frames_to_draw <= 0) {
            glfwWaitEvents();
            frames_to_draw = 3;
        } else {
            glfwPollEvents();
        }
        const auto woke = Clock::now();
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        if (fb_w == 0 || fb_h == 0) {
            frames_to_draw = 0;
            continue;
        }
        if (swapchain_stale || uint32_t(fb_w) != sc.extent.width || uint32_t(fb_h) != sc.extent.height) {
            vkDeviceWaitIdle(g.device);
            create_swapchain(g, surface, window, sc);
            ImGui_ImplVulkan_SetMinImageCount(2);
            swapchain_stale = false;
        }

        // UI ----------------------------------------------------------------------------
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const bool old_fit = fit_window, old_cache = use_cache;
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - kPanelWidth, 0));
        ImGui::SetNextWindowSize(ImVec2(kPanelWidth, io.DisplaySize.y));
        ImGui::Begin("Develop", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
        ImGui::SeparatorText("Basic");
        {
            float ev = sliders.exposure;
            if (ImGui::SliderFloat("Exposure", &ev, -5.f, 5.f, "%+.2f")) sliders.exposure = ev;
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) sliders.exposure = 0;
            for (int i = 1; i < 6; ++i) {
                float& v = slider_ref(sliders, i);
                int units = int(std::lround(v * 100.f));
                if (ImGui::SliderInt(kSliderNames[i], &units, -100, 100)) v = units / 100.f;
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) v = 0;
            }
            if (ImGui::Button("Reset to the saved edit")) sliders = saved_sliders;
        }
        ImGui::Checkbox("Cache what a slider doesn't change", &use_cache);
        ImGui::TextWrapped(use_cache ? "Measures are reused until what they depend on changes."
                                     : "Every change recomputes what RAWmakase does.");
        ImGui::SeparatorText("View");
        if (ImGui::RadioButton("RAWmakase Fit size", !fit_window)) fit_window = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("Fit window (F)", fit_window)) fit_window = true;
        if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F)) fit_window = !fit_window;
        ImGui::SeparatorText("Last change, GPU");
        ImGui::TextWrapped("%s", last_stages.c_str());
        ImGui::Text("Measures      %7.3f ms", gpu_measures.value);
        ImGui::Text("Curve         %7.3f ms", gpu_curve.value);
        ImGui::Text("Photo pass    %7.3f ms", gpu_photo.value);
        ImGui::Text("Change total  %7.3f ms", gpu_change.value);
        ImGui::SeparatorText("Frame");
        ImGui::Text("Whole frame (GPU) %6.3f ms", gpu_frame.value);
        ImGui::Text("Input to present  %6.2f ms", input_to_present.value);
        ImGui::Text("Frames per second %6.0f", fps);
        ImGui::Text("Last resample (GPU) %5.2f ms", last_resample_ms);
        ImGui::SeparatorText("Setup");
        ImGui::TextWrapped("%s, %s", photo.camera.c_str(), saved.found ? saved.profile.c_str() : "no saved edit");
        ImGui::Text("Decoded %ux%u in %.0f ms", photo_w, photo_h, photo.decode_ms);
        ImGui::Text("Upload + mips: %.0f ms", upload_ms);
        ImGui::Text("Pipelines: %.1f ms", pipelines_ms);
        ImGui::Text("On screen: %ux%u px", display.width, display.height);
        ImGui::TextWrapped("%s, %s", g.props.deviceName, mode_name(sc.mode));
        ImGui::End();
        ImGui::Render();

        // The benchmark moves the sliders itself.
        int bench_tag = -1;
        if (bench && frame_number >= 10 && bench_pass < 12) {
            if (bench_step < 0) {
                sliders = saved_sliders;
                use_cache = bench_pass >= 6;
                measures_warm = pivot_warm = curve_warm = toned_warm = false;
            } else {
                const int s = bench_pass % 6;
                const float t = float(bench_step) / float(kBenchSteps - 1);
                const float tri = t < 0.5f ? t * 2.f : 2.f - t * 2.f;
                float& v = slider_ref(sliders, s);
                v = s == 0 ? saved_sliders.exposure - 1.f + 2.f * tri : -1.f + 2.f * tri;
                bench_tag = bench_pass;
            }
            if (++bench_step >= kBenchSteps) {
                bench_step = -1;
                ++bench_pass;
            }
        }
        if (bench) frames_to_draw = 3;
        if (profiling) {
            // A different value each frame, so every pass runs (with the cache, Exposure
            // when VKCONTRAST_PROFILE=cache-exposure).
            if (!std::strcmp(std::getenv("VKCONTRAST_PROFILE"), "cache-exposure"))
                sliders.exposure = frame_number % 2 ? 0.f : 0.01f;
            else
                sliders.shadows = frame_number % 2 ? 0.35f : 0.36f;
            frames_to_draw = 3;
        }

        if (use_cache != old_cache) measures_warm = pivot_warm = curve_warm = toned_warm = false;
        const bool changed = !drawn_once || !(sliders == last_drawn);
        const bool exposure_moving = drawn_once && sliders.exposure != last_drawn.exposure;
        if (changed || fit_window != old_fit || io.WantCaptureMouse) frames_to_draw = std::max(frames_to_draw, 3);

        // What this frame recomputes.
        const bool needs_map = sliders.shadows != 0 || sliders.highlights != 0;
        const bool needs_whites = sliders.whites > 0;
        const bool needs_pivot = sliders.contrast != 0;
        const bool basic = sliders.contrast != 0 || sliders.whites != 0 || sliders.blacks != 0;
        Work work;
        if (!use_cache) {
            if (changed) {
                work.tone = needs_map || needs_whites;
                work.map = needs_map;
                work.pivot = needs_pivot;
                work.lut = basic;
            }
        } else {
            if (!measures_warm || sliders.exposure != measured_exposure) {
                work.tone = work.map = true;
                toned_warm = false;
                measures_warm = true;
                measured_exposure = sliders.exposure;
                curve_warm = false; // adaptive Whites follows the highlights
            }
            if (!pivot_warm) {
                work.pivot = true;
                pivot_warm = true;
                curve_warm = false;
            }
            // Toning the screen image pays off for the sliders after the tone stage; while
            // Exposure itself moves, the photo pass tones as it goes instead.
            if (!toned_warm && !exposure_moving) {
                work.toned = true;
                toned_warm = true;
            }
            const bool curve_moved = sliders.contrast != curve_sliders.contrast ||
                                     sliders.whites != curve_sliders.whites || sliders.blacks != curve_sliders.blacks;
            if (basic && (!curve_warm || curve_moved)) {
                work.lut = true;
                curve_warm = true;
                curve_sliders = sliders;
            }
        }
        last_drawn = sliders;
        drawn_once = true;

        // The photo's rectangle, in physical pixels, left of the panel.
        const float px_per_point = io.DisplayFramebufferScale.x;
        const int area_w = std::max(1, fb_w - int(kPanelWidth * px_per_point));
        const int area_h = fb_h;
        int disp_h = fit_window ? area_h - 2 * margin : target_height;
        int disp_w = int(std::lround(disp_h * aspect));
        if (disp_w > area_w - 2 * margin || disp_h > area_h - 2 * margin) {
            const float k = std::min(float(area_w - 2 * margin) / disp_w, float(area_h - 2 * margin) / disp_h);
            disp_w = std::max(1, int(disp_w * k));
            disp_h = std::max(1, int(disp_h * k));
        }
        if (uint32_t(disp_w) != display.width || uint32_t(disp_h) != display.height) {
            vkDeviceWaitIdle(g.device);
            if (display.image) destroy(g, display);
            if (toned.image) destroy(g, toned);
            display = make_image(g, disp_w, disp_h, 1, kDisplayFormat,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            toned = make_image(g, disp_w, disp_h, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            toned_warm = false;
            VkDescriptorImageInfo src{source_sampler, source.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo dst{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo shown{display_sampler, display.view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo toned_read{nearest_sampler, toned.view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo toned_write{VK_NULL_HANDLE, toned.view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[5]{};
            for (auto& x : w) {
                x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                x.descriptorCount = 1;
            }
            w[0].dstSet = resample_set;
            w[0].dstBinding = 0;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo = &src;
            w[1].dstSet = resample_set;
            w[1].dstBinding = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo = &dst;
            w[2].dstSet = dev_set;
            w[2].dstBinding = 6;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[2].pImageInfo = &shown;
            w[3].dstSet = dev_set;
            w[3].dstBinding = 7;
            w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[3].pImageInfo = &toned_read;
            w[4].dstSet = dev_set;
            w[4].dstBinding = 8;
            w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[4].pImageInfo = &toned_write;
            vkUpdateDescriptorSets(g.device, 5, w, 0, nullptr);
            resample = true;
        }
        const int x0 = (area_w - disp_w) / 2, y0 = (area_h - disp_h) / 2;

        // Parameters of every pass this frame.
        Params pc{};
        pc.rect[0] = 2.f * x0 / fb_w - 1.f;
        pc.rect[1] = 2.f * y0 / fb_h - 1.f;
        pc.rect[2] = 2.f * (x0 + disp_w) / fb_w - 1.f;
        pc.rect[3] = 2.f * (y0 + disp_h) / fb_h - 1.f;
        pc.gain = std::exp2(sliders.exposure + camera_exposure);
        pc.black = ref::DNG_SHADOWS_BLACK * pc.gain;
        pc.gain0 = std::exp2(camera_exposure);
        pc.black0 = ref::DNG_SHADOWS_BLACK * pc.gain0;
        pc.contrast = sliders.contrast;
        pc.highlights = sliders.highlights;
        pc.shadows = sliders.shadows;
        pc.whites = sliders.whites;
        pc.blacks = sliders.blacks;
        pc.n = small_n;
        pc.w = small_w;
        pc.h = small_h;
        pc.flags = (needs_map ? F_MAP : 0) | (basic ? F_BASIC : 0) | (needs_whites ? F_ADAPTIVE_WHITES : 0) |
                   (tone_uniform ? F_TONE_UNIFORM : 0);
        pc.radius = radius;
        pc.tone_count = uint32_t(profile.x.size());
        pc.block_rows = block_rows;

        // Frame -------------------------------------------------------------------------
        Frame& f = frames[frame_index];
        VK(vkWaitForFences(g.device, 1, &f.fence, VK_TRUE, UINT64_MAX));
        if (f.timed) {
            uint64_t t[kQueries];
            if (vkGetQueryPoolResults(g.device, queries, frame_index * kQueries, kQueries, sizeof(t), t,
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                auto ms = [&](uint64_t a, uint64_t b) { return double(b - a) * ns_per_tick / 1e6; };
                gpu_frame.add(ms(t[0], t[5]));
                if (f.resampled) last_resample_ms = ms(t[0], t[1]);
                if (f.changed) {
                    gpu_measures.add(ms(t[1], t[2]));
                    gpu_curve.add(ms(t[2], t[3]));
                    gpu_photo.add(ms(t[3], t[4]));
                    gpu_change.add(ms(t[1], t[4]));
                    last_stages = f.stages;
                }
                if (f.bench_tag >= 0) bench_gpu[f.bench_tag].push_back(ms(t[1], t[4]));
            }
            if (f.marks.size() > 1) {
                uint64_t m[kMarks];
                const uint32_t count = uint32_t(f.marks.size());
                if (vkGetQueryPoolResults(g.device, pass_queries, frame_index * kMarks, count, sizeof(m), m,
                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                    for (uint32_t i = 1; i < count; ++i) {
                        const double d = double(m[i] - m[i - 1]) * ns_per_tick / 1e6;
                        auto it = std::find_if(pass_times.begin(), pass_times.end(),
                                               [&](auto& p) { return p.first == f.marks[i]; });
                        if (it == pass_times.end()) pass_times.push_back({f.marks[i], {d}});
                        else it->second.push_back(d);
                    }
            }
            f.marks.clear();
        }
        uint32_t image_index;
        VkResult acquired = vkAcquireNextImageKHR(g.device, sc.swapchain, UINT64_MAX, f.acquired, VK_NULL_HANDLE,
                                                  &image_index);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain_stale = true;
            drawn_once = false;
            continue;
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) die("vkAcquireNextImageKHR", acquired);
        VK(vkResetFences(g.device, 1, &f.fence));
        VK(vkResetCommandPool(g.device, f.pool, 0));

        VkCommandBuffer cmd = f.cmd;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdResetQueryPool(cmd, queries, frame_index * kQueries, kQueries);
        vkCmdResetQueryPool(cmd, pass_queries, frame_index * kMarks, kMarks);
        auto mark = [&](const char* name) {
            if (!profiling || f.marks.size() >= kMarks) return;
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pass_queries,
                                 frame_index * kMarks + uint32_t(f.marks.size()));
            f.marks.push_back(name);
        };
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries, frame_index * kQueries + 0);
        f.resampled = resample;
        if (resample) {
            barrier(cmd, display.image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resample_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resample_pl, 0, 1, &resample_set, 0, nullptr);
            const float footprint = float(photo_h) / float(disp_h);
            ResampleParams rp{{disp_w, disp_h}, std::max(0.f, std::log2(footprint) - 1.f)};
            vkCmdPushConstants(cmd, resample_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rp), &rp);
            vkCmdDispatch(cmd, (disp_w + 15) / 16, (disp_h + 15) / 16, 1);
            barrier(cmd, display.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            barrier(cmd, toned.image, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            resample = false;
        }
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, frame_index * kQueries + 1);

        // The measures and the curve: compute passes on the work buffers, in order.
        auto sync = [&](VkPipelineStageFlags2 src, VkAccessFlags2 src_access) {
            VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            mb.srcStageMask = src;
            mb.srcAccessMask = src_access;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            d.memoryBarrierCount = 1;
            d.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &d);
        };
        auto after_compute = [&] {
            sync(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        };
        auto run = [&](const char* name, VkPipeline p, const Params& params, uint32_t x, uint32_t y = 1) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
            vkCmdPushConstants(cmd, dev_pl, kAllStages, 0, sizeof(params), &params);
            vkCmdDispatch(cmd, x, y, 1);
            after_compute();
            mark(name);
        };
        // Shadows' and Highlights' curves at their slider values, built per change as
        // RAWmakase builds them (96 values).
        const bool upload_curves = changed && needs_map;
        const bool any_work = work.tone || work.pivot || work.lut || work.toned || upload_curves;
        if (any_work) {
            // The previous frame's passes may still read what these write.
            sync(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dev_pl, 0, 1, &dev_set, 0, nullptr);
        }
        mark("start");
        if (work.tone) {
            const uint32_t groups = (small_n + 255) / 256;
            run("tone the small copy", p_measure, pc, groups);
            // Exact radix select of the three percentiles: the k-th of each.
            uint32_t state[8] = {0, 0, 0, 0,
                                 uint32_t(float(small_n - 1) * tables::DATA[tables::SHADOWS]),
                                 uint32_t(float(small_n - 1) * tables::DATA[tables::HIGHLIGHTS]),
                                 uint32_t(float(small_n - 1) * 0.98f), 0};
            vkCmdUpdateBuffer(cmd, meas_buf.buffer, 16, sizeof(state), state);
            vkCmdFillBuffer(cmd, meas_buf.buffer, 48, 1024 * 4, 0);
            sync(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            mark("reset select");
            for (int shift = 24; shift >= 0; shift -= 8) {
                Params p = pc;
                p.shift = shift;
                run("select: histograms", p_hist, p, 24);
                run("select: scans", p_scan, p, 1);
            }
            if (work.map) {
                Params p = pc;
                p.flags &= ~F_SECOND;
                const uint32_t gx = (small_w + 15) / 16, gy = (small_h + 15) / 16;
                run("guided filter: rows", p_rows, p, gx, gy);
                run("guided filter: columns", p_cols, p, gx, gy);
                p.flags |= F_SECOND;
                run("guided filter: rows", p_rows, p, gx, gy);
                run("guided filter: columns", p_cols, p, gx, gy);
            }
        }
        if (work.pivot) {
            run("pivot: blocks", p_blocks, pc, (48 * block_rows + 63) / 64);
            run("pivot: sort", p_sort, pc, 1);
        }
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, frame_index * kQueries + 2);
        if (work.lut) run("Basic curve", p_lut, pc, 1);
        if (upload_curves) {
            float curves[96];
            auto sh = ref::curve_table(tables::SHADOWS, sliders.shadows);
            auto hl = ref::curve_table(tables::HIGHLIGHTS, sliders.highlights);
            std::copy(sh.begin(), sh.end(), curves);
            std::copy(hl.begin(), hl.end(), curves + 48);
            vkCmdUpdateBuffer(cmd, lut_buf.buffer, 1025 * sizeof(float), sizeof(curves), curves);
            sync(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            mark("S/H curves upload");
        }
        if (work.toned) {
            barrier(cmd, toned.image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            run("toned screen image", p_toned, pc, (disp_w + 15) / 16, (disp_h + 15) / 16);
            barrier(cmd, toned.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, frame_index * kQueries + 3);
        f.changed = changed;
        f.bench_tag = bench_tag;
        f.stages = work.describe();

        VkImage target = sc.images[image_index];
        barrier(cmd, target, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = sc.views[image_index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        const float bg = 33.f / 255.f; // RAWmakase's viewport grey
        color.clearValue.color = {{bg, bg, bg, 1.f}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, sc.extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport viewport{0, 0, float(sc.extent.width), float(sc.extent.height), 0, 1};
        VkRect2D scissor{{0, 0}, sc.extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, use_cache && toned_warm ? draw_toned_pipeline : draw_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dev_pl, 0, 1, &dev_set, 0, nullptr);
        vkCmdPushConstants(cmd, dev_pl, kAllStages, 0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 4, 1, 0, 0);
        mark("photo pass");
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, frame_index * kQueries + 4);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, frame_index * kQueries + 5);

        ++frame_number;
        const bool dumping = dump_path && frame_number == 5;
        VkImageLayout drawn_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        if (dumping) {
            dump = make_buffer(g, VkDeviceSize(sc.extent.width) * sc.extent.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            barrier(cmd, target, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, drawn_layout,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {sc.extent.width, sc.extent.height, 1};
            vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dump.buffer, 1, &copy);
            drawn_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            if (check) {
                display_dump = make_buffer(g, VkDeviceSize(display.width) * display.height * 8,
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                barrier(cmd, display.image, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                        VK_IMAGE_LAYOUT_GENERAL);
                VkBufferImageCopy dc{};
                dc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                dc.imageExtent = {display.width, display.height, 1};
                vkCmdCopyImageToBuffer(cmd, display.image, VK_IMAGE_LAYOUT_GENERAL, display_dump.buffer, 1, &dc);
                dump_x0 = x0;
                dump_y0 = y0;
                dump_params = pc;
            }
        }
        barrier(cmd, target, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_NONE, 0,
                drawn_layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        VK(vkEndCommandBuffer(cmd));

        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = f.acquired;
        wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = sc.rendered[image_index];
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkCommandBufferSubmitInfo cs{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cs.commandBuffer = cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.waitSemaphoreInfoCount = 1;
        si.pWaitSemaphoreInfos = &wait;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cs;
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &signal;
        VK(vkQueueSubmit2(g.queue, 1, &si, f.fence));
        f.timed = true;

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &sc.rendered[image_index];
        pi.swapchainCount = 1;
        pi.pSwapchains = &sc.swapchain;
        pi.pImageIndices = &image_index;
        VkResult shown = vkQueuePresentKHR(g.queue, &pi);
        if (shown == VK_ERROR_OUT_OF_DATE_KHR || shown == VK_SUBOPTIMAL_KHR) swapchain_stale = true;
        else if (shown != VK_SUCCESS) die("vkQueuePresentKHR", shown);
        if (changed) input_to_present.add(ms_since(woke));
        if (bench_tag >= 0) bench_cpu[bench_tag].push_back(ms_since(woke));

        if (dumping) {
            VK(vkWaitForFences(g.device, 1, &f.fence, VK_TRUE, UINT64_MAX));
            void* px = nullptr;
            VK(vkMapMemory(g.device, dump.memory, 0, VK_WHOLE_SIZE, 0, &px));
            const bool bgr = sc.format.format == VK_FORMAT_B8G8R8A8_UNORM;
            const auto* p8 = static_cast<const uint8_t*>(px);
            {
                std::ofstream out(dump_path, std::ios::binary);
                out << "P6\n" << sc.extent.width << " " << sc.extent.height << "\n255\n";
                for (size_t i = 0, n = size_t(sc.extent.width) * sc.extent.height; i < n; ++i) {
                    const uint8_t rgb[3] = {p8[i * 4 + (bgr ? 2 : 0)], p8[i * 4 + 1], p8[i * 4 + (bgr ? 0 : 2)]};
                    out.write(reinterpret_cast<const char*>(rgb), 3);
                }
            }
            std::printf("Saved %s (%ux%u)\n", dump_path, sc.extent.width, sc.extent.height);
            if (check) {
                // The GPU's measures, map and curve against the C++ reference.
                auto read = [&](const Buffer& b, VkDeviceSize size) {
                    std::vector<char> data(size);
                    void* m = nullptr;
                    VK(vkMapMemory(g.device, b.memory, 0, size, 0, &m));
                    std::memcpy(data.data(), m, size);
                    vkUnmapMemory(g.device, b.memory);
                    return data;
                };
                auto meas = read(meas_buf, 16);
                auto scratch = read(scratch_buf, scratch_size);
                auto lutv = read(lut_buf, 1025 * 4);
                const float* m = reinterpret_cast<const float*>(meas.data());
                const float* S = reinterpret_cast<const float*>(scratch.data());
                const float* L = reinterpret_cast<const float*>(lutv.data());
                const Params& P = dump_params;
                std::vector<float> lum(small_n), enc(small_n);
                std::vector<ref::Rgb> encoded0(small_n);
                for (size_t i = 0; i < small_n; ++i) {
                    ref::Rgb t = ref::tone_stage(small[i], P.gain, P.black, profile);
                    lum[i] = std::max(ref::luminance(t), 6e-4f);
                    enc[i] = ref::srgb_encode(std::clamp(lum[i], 0.f, 1.f));
                    ref::Rgb t0 = ref::tone_stage(small[i], P.gain0, P.black0, profile);
                    for (int c = 0; c < 3; ++c) encoded0[i][c] = ref::srgb_encode(std::clamp(t0[c], 0.f, 1.f));
                }
                ref::Map map = ref::local_map(lum, small_w, small_h);
                float highlights = ref::select(enc, 0.98f);
                float pivot = ref::photo_pivot(encoded0, small_w, small_h);
                ref::Table whites = sliders.whites > 0 ? ref::whites_for_highlights(highlights)
                                                       : ref::table(tables::WHITES);
                auto lut = ref::basic_lut(sliders.contrast, sliders.whites, sliders.blacks, pivot, whites);
                double da = 0, db = 0, dl = 0;
                for (size_t i = 0; i < small_n; ++i) {
                    da = std::max(da, double(std::abs(S[7 * small_n + i] - map.a[i])));
                    db = std::max(db, double(std::abs(S[8 * small_n + i] - map.b[i])));
                }
                for (size_t i = 0; i < lut.size(); ++i) dl = std::max(dl, double(std::abs(L[i] - lut[i])));
                std::printf("Check against the C++ reference (sliders: exposure %+.2f, contrast %+.2f, "
                            "highlights %+.2f, shadows %+.2f, whites %+.2f, blacks %+.2f):\n",
                            sliders.exposure, sliders.contrast, sliders.highlights, sliders.shadows, sliders.whites,
                            sliders.blacks);
                std::printf("  pivot         GPU %.6f  ref %.6f\n", m[0], pivot);
                std::printf("  highlights    GPU %.6f  ref %.6f\n", m[1], highlights);
                std::printf("  shadows key   GPU %.6f  ref %.6f\n", m[2], map.keys[0]);
                std::printf("  highlight key GPU %.6f  ref %.6f\n", m[3], map.keys[1]);
                std::printf("  map a, b      max difference %.2e, %.2e\n", da, db);
                std::printf("  Basic curve   max difference %.2e\n", dl);
                // Final colours of on-screen pixels, from the same resampled input.
                void* dp = nullptr;
                VK(vkMapMemory(g.device, display_dump.memory, 0, VK_WHOLE_SIZE, 0, &dp));
                const auto* h16 = static_cast<const uint16_t*>(dp);
                auto bil = [&](const std::vector<float>& v, float u, float vv) {
                    float fx = std::clamp(u * float(small_w) - 0.5f, 0.f, float(small_w - 1));
                    float fy = std::clamp(vv * float(small_h) - 0.5f, 0.f, float(small_h - 1));
                    unsigned ix = unsigned(fx), iy = unsigned(fy);
                    unsigned jx = std::min(ix + 1, small_w - 1), jy = std::min(iy + 1, small_h - 1);
                    float tx = fx - float(ix), ty = fy - float(iy);
                    float top = v[iy * small_w + ix] * (1 - tx) + v[iy * small_w + jx] * tx;
                    float bottom = v[jy * small_w + ix] * (1 - tx) + v[jy * small_w + jx] * tx;
                    return top * (1 - ty) + bottom * ty;
                };
                auto lut_eval = [&](float x) {
                    float fl = std::clamp(x, 0.f, 1.f) * 1024.f;
                    size_t i = std::min<size_t>(size_t(fl), 1023);
                    return lut[i] + (lut[i + 1] - lut[i]) * (fl - float(i));
                };
                int worst = 0;
                size_t over1 = 0, samples = 0;
                for (uint32_t dy = 0; dy < display.height; dy += 7)
                    for (uint32_t dx = 0; dx < display.width; dx += 7) {
                        size_t di = (size_t(dy) * display.width + dx) * 4;
                        ref::Rgb base{half_to_float(h16[di]), half_to_float(h16[di + 1]), half_to_float(h16[di + 2])};
                        ref::Rgb rgb = ref::tone_stage(base, P.gain, P.black, profile);
                        float u = (dx + 0.5f) / float(display.width), v = (dy + 0.5f) / float(display.height);
                        if (P.flags & F_MAP) {
                            float lvl = bil(map.a, u, v) * std::log2(std::max(ref::luminance(rgb), 6e-4f)) +
                                        bil(map.b, u, v);
                            float ev = ref::family(tables::SHADOWS, sliders.shadows, map.keys[0], lvl) +
                                       ref::family(tables::HIGHLIGHTS, sliders.highlights, map.keys[1], lvl);
                            for (auto& c : rgb) c *= std::exp2(ev);
                        }
                        ref::Rgb p = ref::mul(ref::RGB_TO_PRO, rgb);
                        for (auto& c : p) c = ref::srgb_encode(std::clamp(c, 0.f, 1.f));
                        if (!lut.empty()) {
                            float lo = std::min({p[0], p[1], p[2]}), hi = std::max({0.f, p[0], p[1], p[2]});
                            float a = lut_eval(lo), b = lut_eval(hi);
                            for (auto& c : p) c = hi - lo > 1e-8f ? a + (b - a) * (c - lo) / (hi - lo) : a;
                        }
                        for (auto& c : p) c = ref::srgb_decode(std::clamp(c, 0.f, 1.f));
                        ref::Rgb lin = ref::mul(ref::PRO_TO_RGB, p);
                        size_t si = (size_t(dump_y0 + int(dy)) * sc.extent.width + size_t(dump_x0 + int(dx))) * 4;
                        const uint8_t got[3] = {p8[si + (bgr ? 2 : 0)], p8[si + 1], p8[si + (bgr ? 0 : 2)]};
                        for (int c = 0; c < 3; ++c) {
                            int want = int(std::lround(ref::srgb_encode(std::clamp(lin[c], 0.f, 1.f)) * 255.f));
                            int d = std::abs(want - int(got[c]));
                            worst = std::max(worst, d);
                            over1 += d > 1;
                        }
                        ++samples;
                    }
                vkUnmapMemory(g.device, display_dump.memory);
                destroy(g, display_dump);
                std::printf("  final colour  %zu pixels: largest difference %d/255, %zu channel values off by more "
                            "than 1\n", samples, worst, over1);
            }
            vkUnmapMemory(g.device, dump.memory);
            destroy(g, dump);
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        if (bench && bench_pass >= 12 && frame_number > 10) {
            static int tail = 0;
            if (++tail > kFramesInFlight + 1) {
                std::printf("\nPer slider change, %d changes each, medians (%s):\n", kBenchSteps, mode_name(sc.mode));
                std::printf("%-11s | %-28s | %-28s\n", "", "everything recomputed", "cache on");
                std::printf("%-11s | %12s %15s | %12s %15s\n", "slider", "GPU", "input-present", "GPU",
                            "input-present");
                for (int s = 0; s < 6; ++s)
                    std::printf("%-11s | %9.3f ms %12.2f ms | %9.3f ms %12.2f ms\n", kSliderNames[s],
                                median(bench_gpu[s]), median(bench_cpu[s]), median(bench_gpu[6 + s]),
                                median(bench_cpu[6 + s]));
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }

        if (profiling && frame_number >= 120) {
            std::printf("GPU time per pass, median of %d frames, %s:\n", frame_number - 3,
                        !use_cache ? "everything recomputed"
                        : !std::strcmp(std::getenv("VKCONTRAST_PROFILE"), "cache-exposure") ? "cache on, Exposure moving"
                                                                                           : "cache on, Shadows moving");
            double total = 0;
            for (auto& [name, times] : pass_times) {
                // Passes run more than once a frame are summed per frame.
                const size_t per_frame = std::max<size_t>(1, times.size() / size_t(std::max(1, frame_number - 3)));
                std::vector<double> sums;
                for (size_t i = 0; i + per_frame <= times.size(); i += per_frame) {
                    double s = 0;
                    for (size_t k = 0; k < per_frame; ++k) s += times[i + k];
                    sums.push_back(s);
                }
                const double m = median(sums);
                total += m;
                std::printf("  %-24s %zux %8.3f ms\n", name.c_str(), per_frame, m);
            }
            std::printf("  %-24s    %8.3f ms\n", "total", total);
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        ++presented;
        if (double s = ms_since(fps_since) / 1000.0; s >= 0.5) {
            fps = presented / s;
            presented = 0;
            fps_since = Clock::now();
        }
        frame_index = (frame_index + 1) % kFramesInFlight;
        --frames_to_draw;
    }

    // Shutdown ------------------------------------------------------------------------------
    vkDeviceWaitIdle(g.device);
    {
        size_t n = 0;
        vkGetPipelineCacheData(g.device, cache, &n, nullptr);
        std::vector<char> data(n);
        if (vkGetPipelineCacheData(g.device, cache, &n, data.data()) == VK_SUCCESS) {
            std::error_code ec;
            std::filesystem::create_directories(cache_path().parent_path(), ec);
            std::ofstream(cache_path(), std::ios::binary).write(data.data(), std::streamsize(n));
        }
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyQueryPool(g.device, queries, nullptr);
    vkDestroyQueryPool(g.device, pass_queries, nullptr);
    for (auto& f : frames) {
        vkDestroySemaphore(g.device, f.acquired, nullptr);
        vkDestroyFence(g.device, f.fence, nullptr);
        vkDestroyCommandPool(g.device, f.pool, nullptr);
    }
    if (display.image) destroy(g, display);
    if (toned.image) destroy(g, toned);
    destroy(g, source);
    for (VkPipeline p : {draw_pipeline, draw_toned_pipeline, p_toned, resample_pipeline, p_measure, p_hist, p_scan, p_rows, p_cols, p_blocks,
                         p_sort, p_lut})
        vkDestroyPipeline(g.device, p, nullptr);
    vkDestroyPipelineLayout(g.device, dev_pl, nullptr);
    vkDestroyPipelineLayout(g.device, resample_pl, nullptr);
    vkDestroyPipelineCache(g.device, cache, nullptr);
    vkDestroyDescriptorPool(g.device, pool, nullptr);
    vkDestroyDescriptorSetLayout(g.device, dev_layout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, resample_layout, nullptr);
    vkDestroySampler(g.device, display_sampler, nullptr);
    vkDestroySampler(g.device, nearest_sampler, nullptr);
    vkDestroySampler(g.device, source_sampler, nullptr);
    for (Buffer* b : {&tables_buf, &tone_buf, &small_buf, &scratch_buf, &meas_buf, &lut_buf}) destroy(g, *b);
    destroy(g, sc, false);
    vkDestroyCommandPool(g.device, g.upload_pool, nullptr);
    vkDestroyDevice(g.device, nullptr);
    vkDestroySurfaceKHR(g.instance, surface, nullptr);
    vkDestroyInstance(g.instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
