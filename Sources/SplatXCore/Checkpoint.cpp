// Checkpoint — serialization of the Trainer state snapshot (Phase 6 round 2 Task 2 — design
// §7.5/§8.1). The Checkpoint.h header comment is the authoritative record for the format and the
// modes. Written from scratch.
#include "splatx/Checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <fcntl.h>  // open — directory fsync (review fix 2)
#include <unistd.h> // fsync/close

namespace splatx {

namespace {

constexpr char kMagic[4] = {'S', 'X', 'C', 'K'};
constexpr uint32_t kVersion = 2;   // written; version 1 (no shape-reg/mask fields) still loads
constexpr uint32_t kVersionV1 = 1;
constexpr uint32_t kFlagFp16 = 1u << 0;

// Column count per group in D3 group order — shN is derived from sh_degree (the single source of
// truth for the serialized size)
inline int64_t group_dim(int g, int sh_degree) {
    constexpr int64_t fixed[5] = {3, 4, 3, 1, 3};
    if (g < 5) return fixed[g];
    return (static_cast<int64_t>(sh_degree) + 1) * (sh_degree + 1) * 3 - 3; // (K-1)·3
}

// ===== f32 ↔ IEEE 754 half (clang _Float16 — hardware conversion on arm64/x86-64, RNE) =====
// Beyond half's maximum finite value 65504 the value clamps to ±65504 (prevents inf contamination —
// a guard for saving a diverged state); NaN is preserved.
inline uint16_t f32_to_f16_bits(float x) {
    if (x > 65504.0f) x = 65504.0f;
    else if (x < -65504.0f) x = -65504.0f;
    const _Float16 h = static_cast<_Float16>(x);
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

inline float f16_bits_to_f32(uint16_t b) {
    _Float16 h;
    std::memcpy(&h, &b, sizeof(h));
    return static_cast<float>(h);
}

// ===== FILE RAII + failure = throw helpers (the partial file is the temp, discarded before rename) =====
struct File {
    std::FILE* f = nullptr;
    explicit File(const std::string& path, const char* mode) : f(std::fopen(path.c_str(), mode)) {
        if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
    }
    ~File() { if (f) std::fclose(f); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    // Verifies the close return value (detects a buffer flush failure — required before rename)
    void close_checked(const std::string& path) {
        const int rc = std::fclose(f);
        f = nullptr;
        if (rc != 0) throw std::runtime_error("checkpoint: close/flush failed for " + path);
    }
};

void write_bytes(std::FILE* f, const void* p, size_t n) {
    if (n && std::fwrite(p, 1, n, f) != n)
        throw std::runtime_error("checkpoint: write failed");
}

template <typename T>
void write_pod(std::FILE* f, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    write_bytes(f, &v, sizeof(T));
}

void read_bytes(std::FILE* f, void* p, size_t n) {
    if (n && std::fread(p, 1, n, f) != n)
        throw std::runtime_error("checkpoint: truncated file (short read)");
}

template <typename T>
T read_pod(std::FILE* f) {
    static_assert(std::is_trivially_copyable_v<T>);
    T v;
    read_bytes(f, &v, sizeof(T));
    return v;
}

// Writes/reads an f32 array — with fp16 it converts straight to half, and sqrt_compress is for
// exp_avg_sq only (v ↔ sqrt(v) — the basis is in Checkpoint.h). fp32 passes through byte for byte
// (lossless).
void write_f32_array(std::FILE* f, const float* p, size_t n, bool fp16, bool sqrt_compress) {
    if (!fp16) {
        write_bytes(f, p, n * sizeof(float));
        return;
    }
    std::vector<uint16_t> buf(n);
    for (size_t i = 0; i < n; i++)
        buf[i] = f32_to_f16_bits(sqrt_compress ? std::sqrt(p[i]) : p[i]);
    write_bytes(f, buf.data(), n * sizeof(uint16_t));
}

void read_f32_array(std::FILE* f, float* p, size_t n, bool fp16, bool sqrt_compress) {
    if (!fp16) {
        read_bytes(f, p, n * sizeof(float));
        return;
    }
    std::vector<uint16_t> buf(n);
    read_bytes(f, buf.data(), n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++) {
        const float v = f16_bits_to_f32(buf[i]);
        p[i] = sqrt_compress ? v * v : v;
    }
}

// Group vector views (serialization order = the D3 order of record — identical to ResidentParamSync)
inline std::vector<float>* cloud_group(SplatCloud& c, int g) {
    std::vector<float>* v[6] = {&c.means, &c.quats, &c.scales, &c.opacities, &c.sh0, &c.shN};
    return v[g];
}

inline const std::vector<float>* cloud_group(const SplatCloud& c, int g) {
    return cloud_group(const_cast<SplatCloud&>(c), g);
}

void mismatch(const char* field, const std::string& saved, const std::string& current,
              const char* extra = nullptr) {
    std::string msg = std::string("checkpoint resume: ") + field + " mismatch (saved " + saved +
                      ", current " + current + ")";
    if (extra) msg += std::string(" — ") + extra;
    throw std::runtime_error(msg);
}

std::string fstr(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return buf;
}

} // namespace

void verify_checkpoint_config(const CheckpointConfig& s, const CheckpointConfig& c) {
    // sh_degree first — §8.1 rejects unconditionally (there is no relaxation path; the message states the basis)
    if (s.sh_degree != c.sh_degree)
        mismatch("sh_degree", std::to_string(s.sh_degree), std::to_string(c.sh_degree),
                 "resume enforces fixed SH degree (design §8.1 — color corruption "
                 "incident lesson)");
    if (s.mcmc_enabled != c.mcmc_enabled)
        mismatch("mcmc", s.mcmc_enabled ? "on" : "off", c.mcmc_enabled ? "on" : "off");
    if (s.cap_max != c.cap_max)
        mismatch("cap_max", std::to_string(s.cap_max), std::to_string(c.cap_max));
    if (s.mcmc_seed != c.mcmc_seed)
        mismatch("mcmc seed", std::to_string(s.mcmc_seed), std::to_string(c.mcmc_seed),
                 "noise(Philox key)/multinomial stream would diverge");
    if (s.factor != c.factor)
        mismatch("factor", std::to_string(s.factor), std::to_string(c.factor));
    if (s.test_split != c.test_split) // the split convention — prevents contaminating the training view stream and eval (review fix 1)
        mismatch("test_split", std::to_string(s.test_split), std::to_string(c.test_split),
                 "train view stream (it % n_train) would diverge and trained views would "
                 "leak into the test split");
    if (s.total_iters != c.total_iters)
        mismatch("total_iters", std::to_string(s.total_iters), std::to_string(c.total_iters),
                 "ratio schedules (lr decay/SH promotion/refine) are normalized by "
                 "total_iters — resume iter would change meaning");
    if (s.scene_scale != c.scene_scale) // for the same dataset this is a deterministic computation, so it is bit-identical
        mismatch("scene_scale", fstr(s.scene_scale), fstr(c.scene_scale));
    if (s.ssim_weight != c.ssim_weight)
        mismatch("ssim_weight", fstr(s.ssim_weight), fstr(c.ssim_weight));
    if (s.antialiased != c.antialiased)
        mismatch("antialiased", s.antialiased ? "on" : "off", c.antialiased ? "on" : "off");
    if (s.flatten_weight != c.flatten_weight)
        mismatch("flatten_weight", fstr(s.flatten_weight), fstr(c.flatten_weight));
    if (s.needle_weight != c.needle_weight)
        mismatch("needle_weight", fstr(s.needle_weight), fstr(c.needle_weight));
    if (s.needle_ratio != c.needle_ratio)
        mismatch("needle_ratio", fstr(s.needle_ratio), fstr(c.needle_ratio));
    if (s.has_mask != c.has_mask)
        mismatch("region mask", s.has_mask ? "on" : "off", c.has_mask ? "on" : "off",
                 "the masked and unmasked losses train different objectives");
}

void save_checkpoint_file(const TrainerCheckpoint& ck, const std::string& path, bool fp16) {
    const int64_t n = ck.cloud.count;
    if (n < 1) throw std::invalid_argument("save_checkpoint_file: empty cloud");
    if (ck.cloud.sh_degree < 0 || ck.cloud.sh_degree > 3)
        throw std::invalid_argument("save_checkpoint_file: sh_degree must be in [0,3]");
    if (ck.cloud.sh_degree != ck.config.sh_degree)
        throw std::invalid_argument("save_checkpoint_file: cloud/config sh_degree mismatch");
    // Self-verification of sizes — a violation of the count/dim contract fails loudly (prevents a partial arena snapshot)
    for (int g = 0; g < 6; g++) {
        const size_t want = static_cast<size_t>(n) * group_dim(g, ck.cloud.sh_degree);
        if (cloud_group(ck.cloud, g)->size() != want)
            throw std::invalid_argument("save_checkpoint_file: param group size != count*dim");
        const CheckpointMoments& m = ck.moments[g];
        if (m.exp_avg.size() != want || m.exp_avg_sq.size() != want)
            throw std::invalid_argument("save_checkpoint_file: moment group size != count*dim");
    }

    // Atomic write: write to temp → rename (POSIX atomicity — the partial file is never exposed at
    // the final path). On a resign-active re-save the existing checkpoint also stays intact right up
    // to the moment of the rename (§7.5).
    const std::string tmp = path + ".tmp";
    try {
        File file(tmp, "wb");
        std::FILE* f = file.f;
        write_bytes(f, kMagic, 4);
        write_pod<uint32_t>(f, kVersion);
        write_pod<uint32_t>(f, fp16 ? kFlagFp16 : 0u);
        // config fingerprint (field by field — padding is not written)
        write_pod<int32_t>(f, ck.config.sh_degree);
        write_pod<int32_t>(f, ck.config.factor);
        write_pod<int32_t>(f, ck.config.test_split);
        write_pod<int32_t>(f, ck.config.total_iters);
        write_pod<uint8_t>(f, ck.config.mcmc_enabled);
        write_pod<uint8_t>(f, ck.config.antialiased);
        // reserved uint8 ×2 (fixed 0) — the first byte is the former bilateral_grid fingerprint slot
        // (the feature was removed 2026-07-17 — a non-zero file is rejected explicitly on load, see
        // read_header).
        write_pod<uint8_t>(f, 0);
        write_pod<uint8_t>(f, 0); // alignment padding (fixed 0)
        write_pod<int64_t>(f, ck.config.cap_max);
        write_pod<uint64_t>(f, ck.config.mcmc_seed);
        write_pod<float>(f, ck.config.scene_scale);
        write_pod<float>(f, ck.config.ssim_weight);
        write_pod<float>(f, ck.config.flatten_weight);
        write_pod<float>(f, ck.config.needle_weight);
        write_pod<float>(f, ck.config.needle_ratio);
        write_pod<uint8_t>(f, ck.config.has_mask ? 1 : 0);
        for (int k = 0; k < 3; k++) write_pod<uint8_t>(f, 0); // padding (fixed 0)
        // state
        write_pod<int32_t>(f, ck.iter);
        write_pod<int64_t>(f, n);
        write_pod<uint64_t>(f, static_cast<uint64_t>(ck.mcmc_rng_state.size()));
        write_bytes(f, ck.mcmc_rng_state.data(), ck.mcmc_rng_state.size());
        // params ×6 → moments ×6 (exp_avg and exp_avg_sq per group — only sq is sqrt-compressed)
        for (int g = 0; g < 6; g++) {
            const std::vector<float>* v = cloud_group(ck.cloud, g);
            write_f32_array(f, v->data(), v->size(), fp16, false);
        }
        for (int g = 0; g < 6; g++) {
            const CheckpointMoments& m = ck.moments[g];
            write_f32_array(f, m.exp_avg.data(), m.exp_avg.size(), fp16, false);
            write_f32_array(f, m.exp_avg_sq.data(), m.exp_avg_sq.size(), fp16, true);
        }
        // Data durability before rename (review fix 2): fflush (stdio buffer → kernel) followed by
        // fsync (kernel → storage queue). Without this order, on power loss only the rename is
        // journaled and a zero-length or partial file whose contents never landed is left at the
        // final path, losing the only checkpoint (iOS resign-active = saving right before the battery
        // dies is exactly that window).
        if (std::fflush(f) != 0)
            throw std::runtime_error("checkpoint: fflush failed for " + tmp);
        if (::fsync(::fileno(f)) != 0)
            throw std::runtime_error("checkpoint: fsync failed for " + tmp);
        file.close_checked(tmp);
    } catch (...) {
        std::remove(tmp.c_str()); // discards the partial temp (no-op if absent)
        throw;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("checkpoint: rename failed: " + tmp + " -> " + path);
    }
    // Directory entry durability (best-effort): fsync the rename itself — some filesystems do not
    // support it (EINVAL and the like), so failures are ignored (rename atomicity and the file data
    // fsync are already secured — the worst case is a rollback to "the intact checkpoint under the
    // old name", which is not corruption).
    {
        std::string dir = ".";
        if (const size_t slash = path.find_last_of('/'); slash != std::string::npos)
            dir = slash == 0 ? "/" : path.substr(0, slash);
        const int dfd = ::open(dir.c_str(), O_RDONLY);
        if (dfd >= 0) {
            (void)::fsync(dfd);
            (void)::close(dfd);
        }
    }
}

namespace {

// Reads the common prefix from magic through alive_count (the single source of truth shared by
// load_checkpoint_file and load_checkpoint_header — the format field order is in the Checkpoint.h
// header comment). It returns positioned immediately before the rng length.
CheckpointHeader read_header(std::FILE* f, const std::string& path) {
    char magic[4];
    read_bytes(f, magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("checkpoint: bad magic (not a splatx checkpoint): " + path);
    CheckpointHeader h;
    h.version = read_pod<uint32_t>(f);
    if (h.version != kVersion && h.version != kVersionV1)
        throw std::runtime_error("checkpoint: unsupported version " + std::to_string(h.version) +
                                 " (expected " + std::to_string(kVersionV1) + " or " +
                                 std::to_string(kVersion) + ")");
    const uint32_t flags = read_pod<uint32_t>(f);
    if (flags & ~kFlagFp16)
        throw std::runtime_error("checkpoint: unknown flags");
    h.fp16 = (flags & kFlagFp16) != 0;

    h.config.sh_degree = read_pod<int32_t>(f);
    h.config.factor = read_pod<int32_t>(f);
    h.config.test_split = read_pod<int32_t>(f);
    h.config.total_iters = read_pod<int32_t>(f);
    h.config.mcmc_enabled = read_pod<uint8_t>(f);
    h.config.antialiased = read_pod<uint8_t>(f);
    // reserved byte (the former bilateral_grid fingerprint slot): only 0 is valid — a non-zero value
    // means the file was trained with the removed feature, so it is rejected explicitly instead of
    // being silently restored wrong (the 2026-07-17 cleanup, spec §2).
    if (read_pod<uint8_t>(f) != 0)
        throw std::runtime_error(
            "checkpoint: bilateral grid checkpoints are no longer supported "
            "(feature removed): " + path);
    (void)read_pod<uint8_t>(f); // alignment padding
    h.config.cap_max = read_pod<int64_t>(f);
    h.config.mcmc_seed = read_pod<uint64_t>(f);
    h.config.scene_scale = read_pod<float>(f);
    h.config.ssim_weight = read_pod<float>(f);
    if (h.version >= 2) {
        h.config.flatten_weight = read_pod<float>(f);
        h.config.needle_weight = read_pod<float>(f);
        h.config.needle_ratio = read_pod<float>(f);
        const uint8_t has_mask = read_pod<uint8_t>(f);
        if (has_mask > 1) throw std::runtime_error("checkpoint: invalid region mask flag");
        h.config.has_mask = has_mask;
        for (int k = 0; k < 3; k++) (void)read_pod<uint8_t>(f); // padding
    }
    if (h.config.sh_degree < 0 || h.config.sh_degree > 3)
        throw std::runtime_error("checkpoint: sh_degree out of [0,3]");

    h.iter = read_pod<int32_t>(f);
    if (h.iter < 0) throw std::runtime_error("checkpoint: negative iter");
    h.alive_count = read_pod<int64_t>(f);
    if (h.alive_count < 1 || h.alive_count > (int64_t{1} << 33)) // guards against a corrupted magnitude
        throw std::runtime_error("checkpoint: invalid count");
    return h;
}

} // namespace

CheckpointHeader load_checkpoint_header(const std::string& path) {
    File file(path, "rb");
    return read_header(file.f, path);
}

TrainerCheckpoint load_checkpoint_file(const std::string& path) {
    File file(path, "rb");
    std::FILE* f = file.f;

    const CheckpointHeader h = read_header(f, path);
    const bool fp16 = h.fp16;
    TrainerCheckpoint ck;
    ck.config = h.config;
    ck.iter = h.iter;
    const int64_t n = h.alive_count;
    const uint64_t rng_len = read_pod<uint64_t>(f);
    if (rng_len > (1u << 20)) // the mt19937_64 text form is ~7KB — anything in the MB range is corruption
        throw std::runtime_error("checkpoint: rng state too large");
    ck.mcmc_rng_state.resize(static_cast<size_t>(rng_len));
    read_bytes(f, ck.mcmc_rng_state.data(), ck.mcmc_rng_state.size());

    ck.cloud.count = n;
    ck.cloud.sh_degree = ck.config.sh_degree;
    for (int g = 0; g < 6; g++) {
        const size_t want = static_cast<size_t>(n) * group_dim(g, ck.cloud.sh_degree);
        std::vector<float>* v = cloud_group(ck.cloud, g);
        v->resize(want);
        read_f32_array(f, v->data(), want, fp16, false);
    }
    for (int g = 0; g < 6; g++) {
        const size_t want = static_cast<size_t>(n) * group_dim(g, ck.cloud.sh_degree);
        CheckpointMoments& m = ck.moments[g];
        m.exp_avg.resize(want);
        m.exp_avg_sq.resize(want);
        read_f32_array(f, m.exp_avg.data(), want, fp16, false);
        read_f32_array(f, m.exp_avg_sq.data(), want, fp16, true);
        // paired zeroing (fp16 only — the basis is in Checkpoint.h): if sq underflowed to 0, exp_avg
        // is zeroed as well. (m ≠ 0, v = 0) is the combination that blows up as m/eps in fused_adam
        // (eps 1e-15, no bias correction).
        if (fp16)
            for (size_t i = 0; i < want; i++)
                if (m.exp_avg_sq[i] == 0.0f) m.exp_avg[i] = 0.0f;
    }

    // Trailing leftover bytes = corruption (every size in this format is derived — it must be consumed exactly)
    if (std::fgetc(f) != EOF)
        throw std::runtime_error("checkpoint: trailing bytes (corrupt or size mismatch)");
    return ck;
}

} // namespace splatx
