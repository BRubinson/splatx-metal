#include "TestHarness.h"
#include "TestPaths.h" // tmp_path() — a temp path compatible with the iOS sandbox
#include "splatx/SXTensor.h"
#include <cstdio>

using splatx::SXTensor;

SX_TEST(sxtensor_roundtrip_f32) {
    SXTensor t;
    t.dtype = 0;
    t.shape = {2, 3};
    float vals[6] = {1.5f, -2.0f, 0.0f, 3.25f, 1e-7f, -1e7f};
    t.data.assign(reinterpret_cast<uint8_t*>(vals), reinterpret_cast<uint8_t*>(vals) + sizeof(vals));

    const std::string path_s = sxtest::tmp_path("splatx_test_roundtrip.sxt"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    t.save(path);
    SXTensor u = SXTensor::load(path);
    std::remove(path);

    SX_CHECK(u.dtype == 0);
    SX_CHECK(u.shape.size() == 2 && u.shape[0] == 2 && u.shape[1] == 3);
    SX_CHECK(u.count() == 6);
    for (int i = 0; i < 6; i++) SX_CHECK(u.f32()[i] == vals[i]);

    // uint8 (dtype 3) round-trip — for the valid/masks kernel contract (uchar)
    SXTensor b;
    b.dtype = 3;
    b.shape = {5};
    b.data = {1, 0, 1, 1, 0};
    b.save(path);
    SXTensor c = SXTensor::load(path);
    std::remove(path);
    SX_CHECK(c.dtype == 3);
    SX_CHECK(c.count() == 5 && c.data.size() == 5);
    for (int i = 0; i < 5; i++) SX_CHECK(c.u8()[i] == b.data[i]);
}

SX_TEST(sxtensor_load_missing_throws) {
    bool threw = false;
    try { SXTensor::load(sxtest::tmp_path("splatx_no_such_file.sxt")); } catch (...) { threw = true; }
    SX_CHECK(threw);
}
