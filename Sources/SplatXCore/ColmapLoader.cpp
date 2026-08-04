// Parser for the COLMAP binary and text formats (format spec: https://colmap.github.io/format.html)
// Written from scratch — no derivation from external code.
// Auto-detection: if the sparse directory holds cameras.bin, use bin (the existing format takes
// precedence); if it does not and cameras.txt is there, use txt (the capture bundle format the
// iOS capture app's writer produces). If neither is present, it is an error.
#include "splatx/ColmapLoader.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace splatx {

namespace {

struct Reader {
    FILE* f;
    std::string path;
    explicit Reader(const std::string& p) : path(p) {
        f = std::fopen(p.c_str(), "rb");
        if (!f) throw std::runtime_error("colmap: cannot open " + p);
    }
    ~Reader() { if (f) std::fclose(f); }
    template <typename T> T get() {
        T v;
        if (std::fread(&v, sizeof(T), 1, f) != 1)
            throw std::runtime_error("colmap: truncated " + path);
        return v;
    }
    void skip(long bytes) {
        if (std::fseek(f, bytes, SEEK_CUR) != 0)
            throw std::runtime_error("colmap: seek failed " + path);
    }
    std::string cstring() {
        std::string s;
        for (;;) {
            int ch = std::fgetc(f);
            if (ch == EOF) throw std::runtime_error("colmap: truncated string " + path);
            if (ch == '\0') return s;
            s.push_back(static_cast<char>(ch));
        }
    }
};

// COLMAP camera model id → number of params (colmap/src/colmap/sensor/models.h)
int params_count(int model_id) {
    switch (model_id) {
        case 0: return 3;   // SIMPLE_PINHOLE
        case 1: return 4;   // PINHOLE
        case 2: return 4;   // SIMPLE_RADIAL
        case 3: return 5;   // RADIAL
        case 4: return 8;   // OPENCV
        case 5: return 8;   // OPENCV_FISHEYE
        case 6: return 12;  // FULL_OPENCV
        case 7: return 5;   // FOV
        case 8: return 4;   // SIMPLE_RADIAL_FISHEYE
        case 9: return 5;   // RADIAL_FISHEYE
        case 10: return 12; // THIN_PRISM_FISHEYE
        default: throw std::runtime_error("colmap: unknown camera model " + std::to_string(model_id));
    }
}

// COLMAP camera model id → name (for error messages — the inverse of params_count / model_id_from_name).
std::string model_name_from_id(int model_id) {
    switch (model_id) {
        case 0:  return "SIMPLE_PINHOLE";
        case 1:  return "PINHOLE";
        case 2:  return "SIMPLE_RADIAL";
        case 3:  return "RADIAL";
        case 4:  return "OPENCV";
        case 5:  return "OPENCV_FISHEYE";
        case 6:  return "FULL_OPENCV";
        case 7:  return "FOV";
        case 8:  return "SIMPLE_RADIAL_FISHEYE";
        case 9:  return "RADIAL_FISHEYE";
        case 10: return "THIN_PRISM_FISHEYE";
        default: return "model#" + std::to_string(model_id);
    }
}

// Derives fx/fy/cx/cy — shared by bin and txt. splatx-metal performs pure PINHOLE projection only,
// so only the distortion-free PINHOLE/SIMPLE_PINHOLE are let through. Distortion models are rejected
// explicitly instead (in the previous behavior SIMPLE_RADIAL silently discarded k1, and
// RADIAL/OPENCV/FISHEYE/FOV and the like corrupted the intrinsics to 0).
void derive_pinhole(ColmapCamera& c) {
    if (c.model_id == 0) {        // SIMPLE_PINHOLE: f, cx, cy
        c.fx = c.fy = static_cast<float>(c.params[0]);
        c.cx = static_cast<float>(c.params[1]);
        c.cy = static_cast<float>(c.params[2]);
    } else if (c.model_id == 1) { // PINHOLE: fx, fy, cx, cy
        c.fx = static_cast<float>(c.params[0]);
        c.fy = static_cast<float>(c.params[1]);
        c.cx = static_cast<float>(c.params[2]);
        c.cy = static_cast<float>(c.params[3]);
    } else {
        throw std::runtime_error(
            "colmap: unsupported camera model " + model_name_from_id(c.model_id) +
            " — splatx-metal supports pure PINHOLE only. Distortion cameras must be "
            "undistorted (e.g. with COLMAP image_undistorter) and converted to PINHOLE");
    }
}

// ===== Text format (cameras.txt / images.txt / points3D.txt) =====

struct TextReader {
    std::ifstream f;
    std::string path;
    long line_no = 0; // 1-based, the last physical line read
    explicit TextReader(const std::string& p) : f(p), path(p) {
        if (!f.is_open()) throw std::runtime_error("colmap: cannot open " + p);
    }
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("colmap: " + path + ":" + std::to_string(line_no) + ": " + msg);
    }
    // The next physical line (CRLF allowed — the trailing '\r' is stripped). false at EOF.
    bool raw_line(std::string& out) {
        if (!std::getline(f, out)) return false;
        line_no++;
        if (!out.empty() && out.back() == '\r') out.pop_back();
        return true;
    }
    // The next data line, skipping comment ('#') and blank lines. false at EOF.
    bool data_line(std::string& out) {
        while (raw_line(out)) {
            const size_t i = out.find_first_not_of(" \t");
            if (i == std::string::npos) continue; // blank line
            if (out[i] == '#') continue;          // comment
            return true;
        }
        return false;
    }
};

std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string tok;
    while (is >> tok) out.push_back(std::move(tok));
    return out;
}

long parse_long(const TextReader& r, const std::string& tok, const char* what) {
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(tok.c_str(), &end, 10);
    if (end == tok.c_str() || *end != '\0' || errno == ERANGE)
        r.fail(std::string("invalid ") + what + " '" + tok + "'");
    return v;
}

double parse_double(const TextReader& r, const std::string& tok, const char* what) {
    char* end = nullptr;
    const double v = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0')
        r.fail(std::string("invalid ") + what + " '" + tok + "'");
    return v;
}

// COLMAP camera model name → id (the same table as model_id in bin — the same supported range)
int model_id_from_name(const TextReader& r, const std::string& name) {
    if (name == "SIMPLE_PINHOLE") return 0;
    if (name == "PINHOLE") return 1;
    if (name == "SIMPLE_RADIAL") return 2;
    if (name == "RADIAL") return 3;
    if (name == "OPENCV") return 4;
    if (name == "OPENCV_FISHEYE") return 5;
    if (name == "FULL_OPENCV") return 6;
    if (name == "FOV") return 7;
    if (name == "SIMPLE_RADIAL_FISHEYE") return 8;
    if (name == "RADIAL_FISHEYE") return 9;
    if (name == "THIN_PRISM_FISHEYE") return 10;
    r.fail("unknown camera model " + name);
}

// cameras.txt: "CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]" (one line per camera)
void load_cameras_txt(const std::string& path, ColmapBundle& b) {
    TextReader r(path);
    std::string line;
    while (r.data_line(line)) {
        const auto t = split_ws(line);
        if (t.size() < 4)
            r.fail("expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS[], got " +
                   std::to_string(t.size()) + " token(s)");
        ColmapCamera c{};
        c.id = static_cast<int32_t>(parse_long(r, t[0], "CAMERA_ID"));
        c.model_id = model_id_from_name(r, t[1]);
        c.width = parse_long(r, t[2], "WIDTH");
        c.height = parse_long(r, t[3], "HEIGHT");
        const int np = params_count(c.model_id);
        if (static_cast<int>(t.size()) - 4 != np)
            r.fail(t[1] + " expects " + std::to_string(np) + " params, got " +
                   std::to_string(t.size() - 4));
        c.params.resize(np);
        for (int k = 0; k < np; k++) c.params[k] = parse_double(r, t[4 + k], "PARAM");
        derive_pinhole(c);
        b.cameras.push_back(std::move(c));
    }
}

// images.txt: two lines per image — "IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME" +
// a POINTS2D line (which may be empty — the SplatXKit writer always emits an empty one; its contents
// are skipped, the same as in bin).
void load_images_txt(const std::string& path, ColmapBundle& b) {
    TextReader r(path);
    std::string line;
    while (r.data_line(line)) {
        const auto t = split_ws(line);
        if (t.size() != 10)
            r.fail("expected 10 tokens (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME), got " +
                   std::to_string(t.size()));
        ColmapImage im{};
        im.id = static_cast<int32_t>(parse_long(r, t[0], "IMAGE_ID"));
        for (int k = 0; k < 4; k++) im.qvec[k] = parse_double(r, t[1 + k], "QVEC");
        for (int k = 0; k < 3; k++) im.tvec[k] = parse_double(r, t[5 + k], "TVEC");
        im.camera_id = static_cast<int32_t>(parse_long(r, t[8], "CAMERA_ID"));
        im.name = t[9];
        // Consumes exactly one physical line for the second line (POINTS2D) — unconditionally, the
        // same as the standard COLMAP reader.
        std::string p2d;
        if (!r.raw_line(p2d)) r.fail("truncated: missing POINTS2D line after pose line");
        b.images.push_back(std::move(im));
    }
}

// points3D.txt: "POINT3D_ID X Y Z R G B ERROR TRACK[]" (one line per point). An empty file is
// allowed (comments and blanks alone still yield 0 points — the seed can be injected separately).
void load_points3d_txt(const std::string& path, ColmapBundle& b) {
    TextReader r(path);
    std::string line;
    while (r.data_line(line)) {
        const auto t = split_ws(line);
        if (t.size() < 8)
            r.fail("expected POINT3D_ID X Y Z R G B ERROR TRACK[], got " +
                   std::to_string(t.size()) + " token(s)");
        if ((t.size() - 8) % 2 != 0)
            r.fail("TRACK[] must be (IMAGE_ID, POINT2D_IDX) pairs, got " +
                   std::to_string(t.size() - 8) + " trailing token(s)");
        ColmapPoint3D p{};
        parse_long(r, t[0], "POINT3D_ID"); // validation only (unused, the same as in bin)
        for (int k = 0; k < 3; k++) p.xyz[k] = parse_double(r, t[1 + k], "XYZ");
        for (int k = 0; k < 3; k++) {
            const long v = parse_long(r, t[4 + k], "RGB");
            if (v < 0 || v > 255) r.fail("RGB out of range 0..255: '" + t[4 + k] + "'");
            p.rgb[k] = static_cast<uint8_t>(v);
        }
        parse_double(r, t[7], "ERROR"); // validation only (unused)
        b.points.push_back(p);
    }
}

ColmapBundle load_colmap_bundle_txt(const std::string& dir) {
    ColmapBundle b;
    load_cameras_txt(dir + "/cameras.txt", b);
    load_images_txt(dir + "/images.txt", b);
    load_points3d_txt(dir + "/points3D.txt", b);
    return b;
}

ColmapBundle load_colmap_bundle_bin(const std::string& dir) {
    ColmapBundle b;
    {
        Reader r(dir + "/cameras.bin");
        uint64_t n = r.get<uint64_t>();
        b.cameras.reserve(n);
        for (uint64_t i = 0; i < n; i++) {
            ColmapCamera c{};
            c.id = r.get<int32_t>();
            c.model_id = r.get<int32_t>();
            c.width = static_cast<int64_t>(r.get<uint64_t>());
            c.height = static_cast<int64_t>(r.get<uint64_t>());
            int np = params_count(c.model_id);
            c.params.resize(np);
            for (int k = 0; k < np; k++) c.params[k] = r.get<double>();
            derive_pinhole(c);
            b.cameras.push_back(std::move(c));
        }
    }
    {
        Reader r(dir + "/images.bin");
        uint64_t n = r.get<uint64_t>();
        b.images.reserve(n);
        for (uint64_t i = 0; i < n; i++) {
            ColmapImage im{};
            im.id = r.get<int32_t>();
            for (int k = 0; k < 4; k++) im.qvec[k] = r.get<double>();
            for (int k = 0; k < 3; k++) im.tvec[k] = r.get<double>();
            im.camera_id = r.get<int32_t>();
            im.name = r.cstring();
            uint64_t np2d = r.get<uint64_t>();
            r.skip(static_cast<long>(np2d) * 24); // (x f64, y f64, point3D_id i64)
            b.images.push_back(std::move(im));
        }
    }
    {
        Reader r(dir + "/points3D.bin");
        uint64_t n = r.get<uint64_t>();
        b.points.reserve(n);
        for (uint64_t i = 0; i < n; i++) {
            ColmapPoint3D p{};
            r.get<uint64_t>(); // point3D_id (unused)
            for (int k = 0; k < 3; k++) p.xyz[k] = r.get<double>();
            for (int k = 0; k < 3; k++) p.rgb[k] = r.get<uint8_t>();
            r.get<double>();   // reprojection error (unused)
            uint64_t track = r.get<uint64_t>();
            r.skip(static_cast<long>(track) * 8); // (image_id i32, point2D_idx i32)
            b.points.push_back(p);
        }
    }
    return b;
}

bool file_exists(const std::string& p) {
    if (FILE* f = std::fopen(p.c_str(), "rb")) {
        std::fclose(f);
        return true;
    }
    return false;
}

} // namespace

ColmapBundle load_colmap_bundle(const std::string& dir) {
    if (file_exists(dir + "/cameras.bin")) return load_colmap_bundle_bin(dir); // bin takes precedence (existing)
    if (file_exists(dir + "/cameras.txt")) return load_colmap_bundle_txt(dir);
    throw std::runtime_error("colmap: cannot open " + dir + "/cameras.{bin,txt}");
}

} // namespace splatx
