#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

// Golden fixture tensor format (binary compatibility with tools/sxt.py MUST be maintained)
// layout: "SXT1"(4B) | uint32 dtype | uint32 ndim | uint64 shape[ndim] | payload (LE)
struct SXTensor {
    int dtype = 0; // 0=float32, 1=int32, 3=uint8 (2 is reserved for int64)
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;

    static SXTensor load(const std::string& path);
    void save(const std::string& path) const;

    size_t count() const {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }
    const float* f32() const { return reinterpret_cast<const float*>(data.data()); }
    const int32_t* i32() const { return reinterpret_cast<const int32_t*>(data.data()); }
    const uint8_t* u8() const { return data.data(); }
};

} // namespace splatx
