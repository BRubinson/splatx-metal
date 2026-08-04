// SPZ Task 1 — a link smoke test for the vendored Niantic spz (v2.1.0, ThirdParty/spz).
// A single in-memory saveSpz→loadSpz round-trip checks every translation unit (load-spz,
// splat-types, and splat-c-types) together with the zlib (-lz) link (the deflate/inflate paths).
// Format consistency and the IO consumer code are Tasks 2 and 3.
#include "TestHarness.h"
#include "load-spz.h"

#include <cstdint>
#include <vector>

SX_TEST(spz_vendored_roundtrip_smoke) {
    spz::GaussianCloud cloud;
    cloud.numPoints = 2;
    cloud.shDegree = 0;
    cloud.positions = {0.5f, -1.25f, 2.0f, -3.5f, 0.75f, -0.125f};
    cloud.scales = {-2.0f, -2.0f, -2.0f, -1.5f, -1.5f, -1.5f}; // log-scale
    cloud.rotations = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // xyzw
    cloud.alphas = {0.8f, -0.4f}; // pre-sigmoid
    cloud.colors = {0.1f, 0.2f, 0.3f, -0.2f, 0.0f, 0.4f}; // SH DC

    std::vector<uint8_t> bytes;
    SX_CHECK(spz::saveSpz(cloud, {}, &bytes)); // the gzip (zlib deflate) path
    SX_CHECK(!bytes.empty());

    const spz::GaussianCloud loaded = spz::loadSpz(bytes, {}); // the gunzip (zlib inflate) path
    SX_CHECK(loaded.numPoints == 2);
    SX_CHECK(loaded.shDegree == 0);
    SX_CHECK(loaded.positions.size() == 6);
    // SPZ quantizes to 24-bit fixed point (fractionalBits=12) — compare approximately, with a
    // tolerance for the loss.
    for (size_t i = 0; i < 6; i++)
        SX_CHECK_NEAR(loaded.positions[i], cloud.positions[i], 0.0, 1e-3);
    SX_CHECK(loaded.alphas.size() == 2);
    SX_CHECK_NEAR(loaded.alphas[0], cloud.alphas[0], 0.0, 0.05); // alpha is quantized to 8 bits
}
