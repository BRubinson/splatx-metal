#include "TestHarness.h"
#include "TestPaths.h" // fixture_dir() — single source of truth for paths (runtime injection)
#include "splatx/Math.h"
#include "splatx/SXTensor.h"
#include <string>

using namespace splatx;

static std::string fx(const char* rel) { return sxtest::fixture_dir() + "/" + rel; }

SX_TEST(quat_identity_gives_identity_rotmat) {
    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    Mat3 R = quat_to_rotmat(q);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            SX_CHECK_NEAR(R.m[i * 3 + j], i == j ? 1.0f : 0.0f, 0.0, 1e-7);
}

SX_TEST(quat_scale_to_covar_matches_gsplat_fixture) {
    SXTensor quats = SXTensor::load(fx("math_quat_scale_to_covar/quats.sxt"));
    SXTensor scales = SXTensor::load(fx("math_quat_scale_to_covar/scales.sxt"));
    SXTensor covars = SXTensor::load(fx("math_quat_scale_to_covar/covars.sxt"));
    const int64_t N = quats.shape[0];
    SX_CHECK(N == 64);
    for (int64_t i = 0; i < N; i++) {
        Mat3 c = quat_scale_to_covar(quats.f32() + i * 4, scales.f32() + i * 3);
        for (int k = 0; k < 9; k++)
            SX_CHECK_NEAR(c.m[k], covars.f32()[i * 9 + k], 1e-4, 1e-5);
    }
}
