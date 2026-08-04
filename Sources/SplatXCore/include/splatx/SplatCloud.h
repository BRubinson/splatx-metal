// SplatCloud — SoA container for the training parameters (internal representation = gsplat convention)
#pragma once
#include <cstdint>
#include <vector>

namespace splatx {

struct SplatCloud {
    int64_t count = 0;
    int sh_degree = 3;                 // K = (degree+1)^2 - 1 band coefficients
    std::vector<float> means;          // [N,3]
    std::vector<float> scales;         // [N,3] log-space
    std::vector<float> quats;          // [N,4] w-first, unnormalized allowed
    std::vector<float> opacities;      // [N]   logit-space
    std::vector<float> sh0;            // [N,3] f_dc
    std::vector<float> shN;            // [N,K,3] f_rest (K = number of higher-band coefficients)
    int sh_rest_coeffs() const { return (sh_degree + 1) * (sh_degree + 1) - 1; }
};

} // namespace splatx
