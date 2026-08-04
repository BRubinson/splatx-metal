#include "TestHarness.h"
#include "TestPaths.h" // fixture_dir() — single source of truth for paths (runtime injection)
#include "splatx/Json.h"
#include <cstdio>
#include <fstream>
#include <string>

using splatx::json_parse_flat;

SX_TEST(json_parse_flat_nested_numbers) {
    const std::string path_s = sxtest::tmp_path("splatx_test_meta.json"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    {
        std::ofstream f(path);
        f << R"({
  "width": 160,
  "eps2d": 0.3,
  "neg": -1.5e3,
  "golden_source": "gsplat@77ab983",
  "flags": {"calc_compensations": 1},
  "tolerances": {
    "means2d": {"rtol": 1e-5, "atol": 1e-5},
    "v_means": {"rtol": 2e-4, "atol": 1e-6, "rel_l2": 1e-4}
  },
  "tensors": {"inputs": ["means", "quats"], "counts": [1, 2, 3]},
  "has_bg": false,
  "extra": null
})";
    }
    auto m = json_parse_flat(path);
    std::remove(path);

    SX_CHECK(m.at("width") == 160.0);
    SX_CHECK_NEAR(m.at("eps2d"), 0.3, 1e-12, 0.0);
    SX_CHECK(m.at("neg") == -1500.0);
    SX_CHECK_NEAR(m.at("tolerances.means2d.rtol"), 1e-5, 1e-12, 0.0);
    SX_CHECK_NEAR(m.at("tolerances.means2d.atol"), 1e-5, 1e-12, 0.0);
    SX_CHECK_NEAR(m.at("tolerances.v_means.rel_l2"), 1e-4, 1e-12, 0.0);
    SX_CHECK(m.at("flags.calc_compensations") == 1.0);
    SX_CHECK(m.count("golden_source") == 0);   // strings ignored
    SX_CHECK(m.count("tensors.inputs") == 0);  // arrays ignored (including their elements)
    SX_CHECK(m.count("tensors.counts") == 0);
    SX_CHECK(m.count("has_bg") == 0);          // booleans ignored
    SX_CHECK(m.count("extra") == 0);           // null ignored

    // A real fixture meta.json must parse as well
    auto meta = json_parse_flat(sxtest::fixture_dir() + "/proj_fully_fused/meta.json");
    SX_CHECK(meta.at("width") == 160.0);
    SX_CHECK(meta.at("height") == 120.0);
}

SX_TEST(json_parse_flat_array_object_member_not_leaked) {
    // Regression: when an array element is an object, its numeric members must not leak out as
    // keys without a prefix.
    const std::string path_s = sxtest::tmp_path("splatx_test_arrobj.json"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    { std::ofstream f(path); f << R"({"a":[{"x":1}],"b":2})"; }
    auto m = json_parse_flat(path);
    std::remove(path);
    SX_CHECK(m.count("x") == 0);
    SX_CHECK(m.count("a.x") == 0);
    SX_CHECK(m.at("b") == 2.0);
}

SX_TEST(json_parse_flat_missing_or_bad_throws) {
    bool threw = false;
    try { json_parse_flat(sxtest::tmp_path("splatx_no_such_meta.json")); }
    catch (const std::exception&) { threw = true; }
    SX_CHECK(threw);

    const std::string path_s = sxtest::tmp_path("splatx_test_bad.json"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    { std::ofstream f(path); f << "{\"a\": }"; }
    threw = false;
    try { json_parse_flat(path); }
    catch (const std::exception&) { threw = true; }
    std::remove(path);
    SX_CHECK(threw);
}
