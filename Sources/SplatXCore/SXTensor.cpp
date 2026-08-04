#include "splatx/SXTensor.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace splatx {

static size_t dtype_size(int dtype) {
    if (dtype == 0 || dtype == 1) return 4;
    if (dtype == 3) return 1; // uint8 (2 is reserved for int64)
    throw std::runtime_error("SXTensor: unknown dtype " + std::to_string(dtype));
}

SXTensor SXTensor::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("SXTensor: cannot open " + path);
    SXTensor t;
    try {
        char magic[4];
        uint32_t dtype = 0, ndim = 0;
        if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "SXT1", 4) != 0)
            throw std::runtime_error("SXTensor: bad magic in " + path);
        if (std::fread(&dtype, 4, 1, f) != 1 || std::fread(&ndim, 4, 1, f) != 1)
            throw std::runtime_error("SXTensor: bad header in " + path);
        t.dtype = static_cast<int>(dtype);
        t.shape.resize(ndim);
        for (uint32_t i = 0; i < ndim; i++) {
            uint64_t d = 0;
            if (std::fread(&d, 8, 1, f) != 1) throw std::runtime_error("SXTensor: bad shape");
            t.shape[i] = static_cast<int64_t>(d);
        }
        size_t bytes = t.count() * dtype_size(t.dtype);
        t.data.resize(bytes);
        if (bytes > 0 && std::fread(t.data.data(), 1, bytes, f) != bytes)
            throw std::runtime_error("SXTensor: truncated payload in " + path);
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return t;
}

void SXTensor::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("SXTensor: cannot write " + path);
    uint32_t dtype32 = static_cast<uint32_t>(dtype);
    uint32_t ndim = static_cast<uint32_t>(shape.size());
    bool ok = std::fwrite("SXT1", 1, 4, f) == 4;
    ok = ok && std::fwrite(&dtype32, 4, 1, f) == 1;
    ok = ok && std::fwrite(&ndim, 4, 1, f) == 1;
    for (auto d : shape) {
        uint64_t d64 = static_cast<uint64_t>(d);
        ok = ok && std::fwrite(&d64, 8, 1, f) == 1;
    }
    ok = ok && std::fwrite(data.data(), 1, data.size(), f) == data.size();
    if (std::fclose(f) != 0) ok = false;
    if (!ok) throw std::runtime_error("SXTensor: write failed " + path);
}

} // namespace splatx
