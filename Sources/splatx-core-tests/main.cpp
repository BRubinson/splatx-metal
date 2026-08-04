// splatx-core-tests CLI entry — the runner body is shared with the SplatXTests library
// (splatx_tests_c.h sx_tests_run_all) (Phase 6 Task 3 — a single canonical source with the
// SplatXBench iOS app, no duplication). This file is reduced to subcommand dispatch +
// function calls only.
#include "splatx_tests_c.h"

#include <cstdlib>
#include <cstring>
#include <string>
// Golden-producer subcommand (Task 5) — implemented in SplatXTests/IsectTests.cpp. AccuTile
// has no torch reference (07 Correction 2), so the C++ CPU reference is the golden producer:
// the CPU AccuTile output is dumped and committed as the fixture.
namespace sxtest { int dump_accutile_fixture(const std::string& out_dir); }
// Render-comparison subcommand (SPZ Task 3) — implemented in SplatXTests/RenderPsnrTool.cpp.
// Renders two PLYs with the same COLMAP camera and prints PSNR (a harness for measuring SPZ
// round-trip quantization loss).
namespace sxtest {
int render_psnr_compare(const std::string& ply_a, const std::string& ply_b,
                        const std::string& data_dir, int factor, int cam_index);
}
int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--dump-accutile") == 0)
        return sxtest::dump_accutile_fixture(argv[2]);
    if (argc == 7 && std::strcmp(argv[1], "--render-psnr") == 0)
        return sxtest::render_psnr_compare(argv[2], argv[3], argv[4], std::atoi(argv[5]),
                                           std::atoi(argv[6]));
    // All path arguments NULL = previous behavior (env vars SPLATX_FIXTURE_DIR/
    // SPLATX_KERNEL_SRC_DIR/SPLATX_TEST_METALLIB → build-time macro fallback —
    // TestPaths.h/TestUtils.h).
    return sx_tests_run_all(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}
