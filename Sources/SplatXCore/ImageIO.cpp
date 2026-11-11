// Image I/O — ImageIO.framework (CGImageSource/CGImageDestination). Written from scratch.
//
// Load pipeline: CGImageSource (auto-detects the format: JPG, PNG, HEIC, etc.) → redraw into an sRGB
// 8bit RGBX CGBitmapContext (unifies the color space and the pixel format, discards alpha) → f32
// /255. No gamma conversion — the 8bit sRGB code values are carried over to 0..1 as they are (the
// gsplat common practice: the L1 comparison is in sRGB space as well).
// In CGBitmapContext memory, row 0 = the top of the image (the same orientation as imageio/PIL).
//
// Save: [0,1] clamp → lrintf(v·255) 8bit RGB (24bpp, no alpha) → PNG ("public.png").
#include "splatx/ImageIO.h"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splatx {

namespace {

// CFURL (file path) — the caller CFReleases it.
CFURLRef make_file_url(const std::string& path) {
    return CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault, reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.size()), /*isDirectory=*/false);
}

// CGImage → redraw into w×h sRGB 8bit RGBX → f32 (image is consumed — always CGImageRelease'd).
// Whatever the source pixel format or ICC profile is, everything converges on a single
// representation (an sRGB-tagged source passes through with its values unchanged). An alpha source
// is composited over the zero-initialized buffer (black) — training inputs (photographs) are opaque,
// so the real path is unaffected. If w/h differ from the image dimensions, it is resampled to those
// dimensions (kCGInterpolationHigh — the downscale fallback path; a 1:1 draw at identical dimensions
// does no resampling = byte-identical to the previous load).
ImageF32 draw_to_rgb_f32(CGImageRef image, size_t w, size_t h, const std::string& path) {
    if (w == 0 || h == 0) {
        CGImageRelease(image);
        throw std::runtime_error("load_image_rgb_f32: empty image " + path);
    }
    std::vector<uint8_t> rgbx(w * h * 4, 0);
    CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(
        rgbx.data(), w, h, /*bitsPerComponent=*/8, /*bytesPerRow=*/w * 4, srgb,
        static_cast<uint32_t>(kCGImageAlphaNoneSkipLast)); // R,G,B,X in memory
    CGColorSpaceRelease(srgb);
    if (!ctx) {
        CGImageRelease(image);
        throw std::runtime_error("load_image_rgb_f32: bitmap context failed for " + path);
    }
    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextDrawImage(ctx, CGRectMake(0, 0, static_cast<CGFloat>(w), static_cast<CGFloat>(h)),
                       image);
    CGContextRelease(ctx);
    CGImageRelease(image);

    ImageF32 out;
    out.width = static_cast<int>(w);
    out.height = static_cast<int>(h);
    out.rgb.resize(w * h * 3);
    constexpr float kInv255 = 1.0f / 255.0f;
    for (size_t p = 0; p < w * h; p++) {
        out.rgb[p * 3 + 0] = static_cast<float>(rgbx[p * 4 + 0]) * kInv255;
        out.rgb[p * 3 + 1] = static_cast<float>(rgbx[p * 4 + 1]) * kInv255;
        out.rgb[p * 3 + 2] = static_cast<float>(rgbx[p * 4 + 2]) * kInv255;
    }
    return out;
}

} // namespace

ImageF32 load_image_rgb_f32(const std::string& path) {
    CFURLRef url = make_file_url(path);
    CGImageSourceRef src = url ? CGImageSourceCreateWithURL(url, nullptr) : nullptr;
    if (url) CFRelease(url);
    if (!src) throw std::runtime_error("load_image_rgb_f32: cannot open " + path);
    CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!image) throw std::runtime_error("load_image_rgb_f32: cannot decode " + path);
    return draw_to_rgb_f32(image, CGImageGetWidth(image), CGImageGetHeight(image), path);
}

ImageF32 load_image_rgb_f32_downscaled(const std::string& path, int factor) {
    if (factor < 1)
        throw std::invalid_argument("load_image_rgb_f32_downscaled: factor must be >= 1");
    CFURLRef url = make_file_url(path);
    CGImageSourceRef src = url ? CGImageSourceCreateWithURL(url, nullptr) : nullptr;
    if (url) CFRelease(url);
    if (!src) throw std::runtime_error("load_image_rgb_f32_downscaled: cannot open " + path);

    // Target dimensions = downscaled_dim(native, factor) — the single source of truth shared with
    // the TrainC intrinsics scaling (ImageIO.h). So that the native dimensions come from a thumbnail
    // decode rather than being settled by a full decode, only a full-size CGImage handle is created
    // first (CGImageSourceCreateImageAtIndex decodes lazily — GetWidth/Height trigger header parsing
    // only, no pixel decode).
    CGImageRef full = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    if (!full) {
        CFRelease(src);
        throw std::runtime_error("load_image_rgb_f32_downscaled: cannot decode " + path);
    }
    const size_t w0 = CGImageGetWidth(full), h0 = CGImageGetHeight(full);
    const int tw = downscaled_dim(static_cast<int>(w0), factor);
    const int th = downscaled_dim(static_cast<int>(h0), factor);
    if (w0 == 0 || h0 == 0 || tw < 1 || th < 1) {
        CGImageRelease(full);
        CFRelease(src);
        throw std::runtime_error("load_image_rgb_f32_downscaled: image " + path + " (" +
                                 std::to_string(w0) + "x" + std::to_string(h0) +
                                 ") too small for factor " + std::to_string(factor));
    }

    // Thumbnail decode (for JPEG this is a DCT scaled decode — faster than a full decode and saves
    // memory): MaxPixelSize = max(tw, th) shrinks while preserving the aspect ratio. ImageIO's
    // rounding on the minor axis can land 1px away from the target, so the final dimensions are
    // settled by the draw_to_rgb_f32(tw, th) redraw below (nearly 1:1 — the quality impact is
    // negligible). kCGImageSourceCreateThumbnailWithTransform is left at its default of false —
    // EXIF orientation is not applied (the same orientation convention as the basic load).
    const long max_px = static_cast<long>(std::max(tw, th));
    CGImageRef image = nullptr;
    CFNumberRef max_px_num = CFNumberCreate(kCFAllocatorDefault, kCFNumberLongType, &max_px);
    if (max_px_num) {
        const void* keys[] = {kCGImageSourceCreateThumbnailFromImageAlways,
                              kCGImageSourceThumbnailMaxPixelSize};
        const void* vals[] = {kCFBooleanTrue, max_px_num};
        CFDictionaryRef opts = CFDictionaryCreate(kCFAllocatorDefault, keys, vals, 2,
                                                  &kCFTypeDictionaryKeyCallBacks,
                                                  &kCFTypeDictionaryValueCallBacks);
        if (opts) {
            image = CGImageSourceCreateThumbnailAtIndex(src, 0, opts);
            CFRelease(opts);
        }
        CFRelease(max_px_num);
    }
    CFRelease(src);
    if (image) {
        CGImageRelease(full); // thumbnail succeeded — the full-size handle is not needed
    } else {
        image = full; // thumbnail failed (unsupported per format, etc.) — shrink by redrawing straight from the full decode
    }
    return draw_to_rgb_f32(image, static_cast<size_t>(tw), static_cast<size_t>(th), path);
}

ImageU16 load_png_gray_raw(const std::string& path) {
    CFURLRef url = make_file_url(path);
    CGImageSourceRef src = url ? CGImageSourceCreateWithURL(url, nullptr) : nullptr;
    if (url) CFRelease(url);
    if (!src) throw std::runtime_error("load_png_gray_raw: cannot open " + path);
    CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!image) throw std::runtime_error("load_png_gray_raw: cannot decode " + path);

    const size_t w = CGImageGetWidth(image), h = CGImageGetHeight(image);
    const size_t bpc = CGImageGetBitsPerComponent(image);
    const size_t bpp = CGImageGetBitsPerPixel(image);
    const size_t stride = CGImageGetBytesPerRow(image);
    const CGBitmapInfo info = CGImageGetBitmapInfo(image);
    const CGImageAlphaInfo alpha = CGImageGetAlphaInfo(image);
    CFDataRef raw = CGDataProviderCopyData(CGImageGetDataProvider(image));
    CGImageRelease(image);
    if (!raw) throw std::runtime_error("load_png_gray_raw: no pixel data in " + path);
    if (w == 0 || h == 0 || (bpc != 8 && bpc != 16) || bpp % bpc != 0) {
        CFRelease(raw);
        throw std::runtime_error("load_png_gray_raw: unsupported pixel layout in " + path);
    }
    const size_t channels = bpp / bpc;
    const bool alpha_first = alpha == kCGImageAlphaFirst || alpha == kCGImageAlphaPremultipliedFirst ||
                             alpha == kCGImageAlphaNoneSkipFirst;
    const size_t ch = (channels > 1 && alpha_first) ? 1 : 0;
    const uint32_t order = info & kCGBitmapByteOrderMask;
    const bool little = order == kCGBitmapByteOrder16Little || order == kCGBitmapByteOrder32Little;
    const uint8_t* bytes = CFDataGetBytePtr(raw);
    if (static_cast<size_t>(CFDataGetLength(raw)) < stride * h) {
        CFRelease(raw);
        throw std::runtime_error("load_png_gray_raw: truncated pixel data in " + path);
    }

    ImageU16 out;
    out.width = static_cast<int>(w);
    out.height = static_cast<int>(h);
    out.data.resize(w * h);
    for (size_t y = 0; y < h; y++) {
        const uint8_t* row = bytes + y * stride;
        for (size_t x = 0; x < w; x++) {
            if (bpc == 8) {
                out.data[y * w + x] = row[x * channels + ch];
            } else {
                const uint8_t* p = row + (x * channels + ch) * 2;
                out.data[y * w + x] = little ? static_cast<uint16_t>(p[0] | (p[1] << 8))
                                             : static_cast<uint16_t>((p[0] << 8) | p[1]);
            }
        }
    }
    CFRelease(raw);
    return out;
}

namespace {

void save_png_gray(const ImageU16& img, const std::string& path, bool sixteen) {
    const size_t w = static_cast<size_t>(img.width), h = static_cast<size_t>(img.height);
    if (img.width <= 0 || img.height <= 0 || img.data.size() != w * h)
        throw std::invalid_argument("save_png_gray: data size != W*H (or empty)");
    std::vector<uint8_t> buf(w * h * (sixteen ? 2 : 1));
    for (size_t i = 0; i < w * h; i++) {
        if (sixteen) {
            buf[i * 2] = static_cast<uint8_t>(img.data[i] & 0xFF);
            buf[i * 2 + 1] = static_cast<uint8_t>(img.data[i] >> 8);
        } else {
            buf[i] = static_cast<uint8_t>(std::min<uint16_t>(img.data[i], 255));
        }
    }
    CGColorSpaceRef gray = CGColorSpaceCreateDeviceGray();
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, buf.data(), buf.size(), nullptr);
    const CGBitmapInfo info = static_cast<CGBitmapInfo>(kCGImageAlphaNone) |
                              (sixteen ? kCGBitmapByteOrder16Little : kCGBitmapByteOrderDefault);
    CGImageRef image = provider ? CGImageCreate(w, h, sixteen ? 16 : 8, sixteen ? 16 : 8,
                                                w * (sixteen ? 2 : 1), gray, info, provider,
                                                nullptr, false, kCGRenderingIntentDefault)
                                : nullptr;
    if (provider) CGDataProviderRelease(provider);
    CGColorSpaceRelease(gray);
    if (!image) throw std::runtime_error("save_png_gray: CGImageCreate failed");
    CFURLRef url = make_file_url(path);
    CGImageDestinationRef dst =
        url ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr) : nullptr;
    if (url) CFRelease(url);
    if (!dst) {
        CGImageRelease(image);
        throw std::runtime_error("save_png_gray: cannot create " + path);
    }
    CGImageDestinationAddImage(dst, image, nullptr);
    const bool ok = CGImageDestinationFinalize(dst);
    CFRelease(dst);
    CGImageRelease(image);
    if (!ok) throw std::runtime_error("save_png_gray: finalize failed for " + path);
}

} // namespace

void save_png_gray16(const ImageU16& img, const std::string& path) { save_png_gray(img, path, true); }
void save_png_gray8(const ImageU16& img, const std::string& path) { save_png_gray(img, path, false); }

void save_png_rgb_f32(const ImageF32& img, const std::string& path) {
    const size_t w = static_cast<size_t>(img.width);
    const size_t h = static_cast<size_t>(img.height);
    if (img.width <= 0 || img.height <= 0 || img.rgb.size() != w * h * 3)
        throw std::invalid_argument("save_png_rgb_f32: rgb size != W*H*3 (or empty)");

    // [0,1] clamp + round(v·255) quantization (NaN becomes 0 — a guard for the render preview)
    std::vector<uint8_t> rgb8(w * h * 3);
    for (size_t i = 0; i < rgb8.size(); i++) {
        float v = img.rgb[i];
        if (!(v > 0.0f)) v = 0.0f; // includes NaN
        if (v > 1.0f) v = 1.0f;
        rgb8[i] = static_cast<uint8_t>(lrintf(v * 255.0f));
    }

    CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGDataProviderRef provider =
        CGDataProviderCreateWithData(nullptr, rgb8.data(), rgb8.size(), nullptr);
    CGImageRef image =
        provider ? CGImageCreate(w, h, /*bitsPerComponent=*/8, /*bitsPerPixel=*/24,
                                 /*bytesPerRow=*/w * 3, srgb,
                                 static_cast<CGBitmapInfo>(kCGImageAlphaNone), provider,
                                 nullptr, false, kCGRenderingIntentDefault)
                 : nullptr;
    if (provider) CGDataProviderRelease(provider);
    CGColorSpaceRelease(srgb);
    if (!image) throw std::runtime_error("save_png_rgb_f32: CGImageCreate failed");

    CFURLRef url = make_file_url(path);
    // the "public.png" UTI literal — avoids depending on kUTTypePNG (CoreServices, deprecated)
    CGImageDestinationRef dst =
        url ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr) : nullptr;
    if (url) CFRelease(url);
    if (!dst) {
        CGImageRelease(image);
        throw std::runtime_error("save_png_rgb_f32: cannot create " + path);
    }
    CGImageDestinationAddImage(dst, image, nullptr);
    const bool ok = CGImageDestinationFinalize(dst);
    CFRelease(dst);
    CGImageRelease(image); // released after finalize — the provider references rgb8 (alive in scope)
    if (!ok) throw std::runtime_error("save_png_rgb_f32: finalize failed for " + path);
}

} // namespace splatx
