// Image I/O — ImageIO.framework (CGImageSource/CGImageDestination). Written from scratch.
//
// Load: JPG/PNG (and any other CGImageSource-supported format, auto-detected) → RGB f32 [H,W,3]
// (row 0 = top), 0..1 **as raw sRGB values** — no gamma (linearization) conversion (the gsplat
// training pipeline convention: both the image and the render output are compared with L1 in sRGB
// space). Alpha is ignored (RGB 3 channels are forced), and the color space is fixed to sRGB —
// images with a different ICC profile are color-converted to sRGB at load time.
// The default load does not resize — for factor downsampling, selecting the images_N folder is the
// first-class path, and load_image_rgb_f32_downscaled is the fallback only when that folder is
// absent (Integration Task 2 — convention below).
//
// Save: f32 [H,W,3] → 8-bit PNG (for render previews). Clamped to [0,1], then quantized with
// round(v*255) — the round-trip error is ≤ 0.5/255 (tested as a pair with the ImageIO load).
#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

struct ImageF32 {
    int width = 0, height = 0;
    std::vector<float> rgb; // [H,W,3] row-major, 0..1 sRGB
};

// The single source of truth for the factor-downsample dimension convention: round(orig/factor) —
// the gsplat/ImageMagick images_N folder convention (measured on garden: 5187/4 → 1297,
// 3361/4 → 840). The TrainC intrinsics scaling and the target dimensions of
// load_image_rgb_f32_downscaled share this one function (consistent with the GT dims guard).
inline int downscaled_dim(int v, int factor) {
    return static_cast<int>(std::lround(static_cast<double>(v) / factor));
}

ImageF32 load_image_rgb_f32(const std::string& path);                 // throws on failure
void save_png_rgb_f32(const ImageF32& img, const std::string& path); // throws on failure

// The fallback for a missing images_{factor} folder (Integration Task 2): the original is
// downscaled at decode time to exactly (downscaled_dim(w, factor), downscaled_dim(h, factor)) —
// w/h are the file's native pixel dimensions (EXIF orientation is not applied; this is the same
// orientation convention as load_image_rgb_f32, so a rotation-tag mismatch is caught by the
// caller's dims guard as before). The color pipeline is identical to the default load
// (sRGB 8-bit RGBX read → f32). factor 1 is equivalent to the default load.
// Throws on failure or on a dimension < 1.
ImageF32 load_image_rgb_f32_downscaled(const std::string& path, int factor);

// Single-channel PNG sample values read raw (no colour management, no gamma): a 16-bit greyscale
// PNG gives its 0..65535 codes, an 8-bit one its 0..255 codes (the depth sidecar convention:
// millimetres and ARConfidenceLevel). Images with more channels take channel 0. Throws on failure.
struct ImageU16 {
    int width = 0, height = 0;
    std::vector<uint16_t> data; // [H,W] row-major, row 0 = top
};
ImageU16 load_png_gray_raw(const std::string& path);
void save_png_gray16(const ImageU16& img, const std::string& path); // throws on failure
void save_png_gray8(const ImageU16& img, const std::string& path);  // values clamp to 255

} // namespace splatx
