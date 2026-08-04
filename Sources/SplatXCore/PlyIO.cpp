// Writer/reader for the INRIA 3DGS standard PLY layout, plus the export filter.
// Layout reference: field-compatible with save_ply in graphdeco-inria/gaussian-splatting
// (written from scratch from the field names and order alone — no code derivation. The Inria code is
// under a non-commercial license and MUST NOT be touched.)
// The export filter convention is checked against gsplat exporter.py export_splats:629–651 — see the
// PlyIO.h header comment.
#include "splatx/PlyIO.h"
#include "splatx/Trainer.h" // sigmoid_f32 (the single source of truth — same formula as KernelCommon.h sigmoid_standard)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace splatx {

void write_inria_ply(const SplatCloud& c, const std::string& path, bool pad_sh_full) {
    const int K = c.sh_rest_coeffs();
    // Viewer-compatibility zero-padding (design §8.1 — the basis for the minimum of 45 is in the
    // PlyIO.h comment: SplatIO 1.0.1 demands exactly 45 whenever f_rest_0 is present, while deg0 is
    // left unpadded because it takes the "no f_rest" branch).
    const int Kw = (pad_sh_full && K > 0 && K < 15) ? 15 : K;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("ply: cannot write " + path);

    std::string header = "ply\nformat binary_little_endian 1.0\n";
    header += "element vertex " + std::to_string(c.count) + "\n";
    auto prop = [&](const std::string& n) { header += "property float " + n + "\n"; };
    for (const char* n : {"x", "y", "z", "nx", "ny", "nz"}) prop(n);
    for (int i = 0; i < 3; i++) prop("f_dc_" + std::to_string(i));
    for (int i = 0; i < 3 * Kw; i++) prop("f_rest_" + std::to_string(i));
    prop("opacity");
    for (int i = 0; i < 3; i++) prop("scale_" + std::to_string(i));
    for (int i = 0; i < 4; i++) prop("rot_" + std::to_string(i));
    header += "end_header\n";
    std::fwrite(header.data(), 1, header.size(), f);

    std::vector<float> row(6 + 3 + 3 * Kw + 1 + 3 + 4);
    for (int64_t i = 0; i < c.count; i++) {
        size_t p = 0;
        for (int k = 0; k < 3; k++) row[p++] = c.means[i * 3 + k];
        for (int k = 0; k < 3; k++) row[p++] = 0.0f; // normals = 0
        for (int k = 0; k < 3; k++) row[p++] = c.sh0[i * 3 + k];
        // f_rest is channel-major: [R0..R(Kw-1), G0.., B0..]  (the internal shN is coefficient-major, [K,3]).
        // Within that channel-planar layout the padding writes 0 into the higher coefficients (k >= K)
        // of each channel — it is inserted at the end of every channel block, not as one zero tail
        // (consistent with the reader's transpose convention).
        for (int ch = 0; ch < 3; ch++)
            for (int k = 0; k < Kw; k++)
                row[p++] = k < K ? c.shN[(i * K + k) * 3 + ch] : 0.0f;
        row[p++] = c.opacities[i];
        for (int k = 0; k < 3; k++) row[p++] = c.scales[i * 3 + k];
        for (int k = 0; k < 4; k++) row[p++] = c.quats[i * 4 + k];
        std::fwrite(row.data(), 4, row.size(), f);
    }
    std::fclose(f);
}

SplatCloud read_inria_ply(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("ply: cannot open " + path);

    // --- Header parsing (text, line by line) ---
    std::vector<std::string> props;
    int64_t count = 0;
    char line[512];
    bool binary_le = false;
    for (;;) {
        if (!std::fgets(line, sizeof(line), f)) { std::fclose(f); throw std::runtime_error("ply: bad header"); }
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s == "end_header") break;
        if (s.rfind("format ", 0) == 0) binary_le = (s == "format binary_little_endian 1.0");
        else if (s.rfind("element vertex ", 0) == 0) count = std::stoll(s.substr(15));
        else if (s.rfind("property float ", 0) == 0) props.push_back(s.substr(15));
        else if (s.rfind("property ", 0) == 0) {
            // A non-float property breaks the row-size computation and silently corrupts data, so reject it immediately
            std::fclose(f);
            throw std::runtime_error("ply: only float properties supported (found: " + s + ")");
        }
    }
    if (!binary_le) { std::fclose(f); throw std::runtime_error("ply: only binary_little_endian supported"); }

    std::map<std::string, size_t> idx;
    for (size_t i = 0; i < props.size(); i++) idx[props[i]] = i;
    int rest_count = 0;
    while (idx.count("f_rest_" + std::to_string(rest_count))) rest_count++;
    if (rest_count % 3 != 0) { std::fclose(f); throw std::runtime_error("ply: f_rest not divisible by 3"); }
    const int K = rest_count / 3;

    int degree = -1;
    for (int d = 0; d <= 4; d++) if ((d + 1) * (d + 1) - 1 == K) degree = d;
    if (degree < 0) { std::fclose(f); throw std::runtime_error("ply: unsupported f_rest count " + std::to_string(rest_count)); }

    SplatCloud c;
    c.count = count;
    c.sh_degree = degree;
    c.means.resize(count * 3); c.scales.resize(count * 3); c.quats.resize(count * 4);
    c.opacities.resize(count); c.sh0.resize(count * 3); c.shN.resize(count * K * 3);

    std::vector<float> row(props.size());
    auto at = [&](const std::string& n) -> float {
        auto it = idx.find(n);
        if (it == idx.end()) throw std::runtime_error("ply: missing property " + n);
        return row[it->second];
    };
    for (int64_t i = 0; i < count; i++) {
        if (std::fread(row.data(), 4, row.size(), f) != row.size()) { std::fclose(f); throw std::runtime_error("ply: truncated body"); }
        c.means[i * 3 + 0] = at("x"); c.means[i * 3 + 1] = at("y"); c.means[i * 3 + 2] = at("z");
        for (int k = 0; k < 3; k++) c.sh0[i * 3 + k] = at("f_dc_" + std::to_string(k));
        for (int ch = 0; ch < 3; ch++)
            for (int k = 0; k < K; k++)
                c.shN[(i * K + k) * 3 + ch] = at("f_rest_" + std::to_string(ch * K + k));
        c.opacities[i] = at("opacity");
        for (int k = 0; k < 3; k++) c.scales[i * 3 + k] = at("scale_" + std::to_string(k));
        for (int k = 0; k < 4; k++) c.quats[i * 4 + k] = at("rot_" + std::to_string(k));
    }
    std::fclose(f);
    return c;
}

// ===== Export filter (design §5.4 — the basis for the convention is the PlyIO.h comment) =====
// Decision order: NaN/Inf is classified first (the upstream export_splats also drops these, because a
// NaN opacity makes the sigmoid comparison false, so the resulting set is identical — only the
// statistics classification is made explicit), then low opacity. Only the first count rows are
// consumed (the MCMC arena tail is ignored). If a vector is shorter than count rows it throws (a
// violation of the SoA size contract).
SplatCloud filter_cloud_for_export(const SplatCloud& c, float min_opacity,
                                   ExportFilterStats* stats) {
    const int64_t N = c.count;
    const int64_t K = c.sh_rest_coeffs();
    if (N < 0) throw std::invalid_argument("filter_cloud_for_export: negative count");
    const auto need = [&](size_t have, int64_t dim, const char* name) {
        if (have < static_cast<size_t>(N) * static_cast<size_t>(dim))
            throw std::invalid_argument(std::string("filter_cloud_for_export: ") + name +
                                        " smaller than count rows");
    };
    need(c.means.size(), 3, "means");
    need(c.scales.size(), 3, "scales");
    need(c.quats.size(), 4, "quats");
    need(c.opacities.size(), 1, "opacities");
    need(c.sh0.size(), 3, "sh0");
    need(c.shN.size(), K * 3, "shN");

    const auto row_finite = [&](int64_t i) {
        for (int k = 0; k < 3; k++) if (!std::isfinite(c.means[i * 3 + k])) return false;
        for (int k = 0; k < 3; k++) if (!std::isfinite(c.scales[i * 3 + k])) return false;
        for (int k = 0; k < 4; k++) if (!std::isfinite(c.quats[i * 4 + k])) return false;
        if (!std::isfinite(c.opacities[i])) return false;
        for (int k = 0; k < 3; k++) if (!std::isfinite(c.sh0[i * 3 + k])) return false;
        for (int64_t k = 0; k < K * 3; k++) if (!std::isfinite(c.shN[i * K * 3 + k])) return false;
        return true;
    };

    ExportFilterStats st;
    st.input = N;
    std::vector<int64_t> keep;
    keep.reserve(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; i++) {
        if (!row_finite(i)) { st.removed_nonfinite++; continue; }
        // strict `>` — the same predicate as gsplat exporter.py:236, the complement of dead (`<=` min_opacity).
        if (!(sigmoid_f32(c.opacities[i]) > min_opacity)) { st.removed_low_opacity++; continue; }
        keep.push_back(i);
    }
    st.kept = static_cast<int64_t>(keep.size());

    SplatCloud out;
    out.count = st.kept;
    out.sh_degree = c.sh_degree;
    out.means.resize(keep.size() * 3);
    out.scales.resize(keep.size() * 3);
    out.quats.resize(keep.size() * 4);
    out.opacities.resize(keep.size());
    out.sh0.resize(keep.size() * 3);
    out.shN.resize(keep.size() * static_cast<size_t>(K) * 3);
    for (size_t j = 0; j < keep.size(); j++) {
        const int64_t i = keep[j];
        for (int k = 0; k < 3; k++) out.means[j * 3 + k] = c.means[i * 3 + k];
        for (int k = 0; k < 3; k++) out.scales[j * 3 + k] = c.scales[i * 3 + k];
        for (int k = 0; k < 4; k++) out.quats[j * 4 + k] = c.quats[i * 4 + k];
        out.opacities[j] = c.opacities[i];
        for (int k = 0; k < 3; k++) out.sh0[j * 3 + k] = c.sh0[i * 3 + k];
        for (int64_t k = 0; k < K * 3; k++) out.shN[j * K * 3 + k] = c.shN[i * K * 3 + k];
    }
    if (stats) *stats = st;
    return out;
}

} // namespace splatx
