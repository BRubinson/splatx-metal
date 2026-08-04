#include "TestHarness.h"
#include "TestPaths.h" // fixture_dir() — the single source of truth for path resolution (runtime injection)
#include "splatx/CpuKernels.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"
#include <cmath>
#include <cstdlib>
#include <string>

using namespace splatx;

static std::string fx(const char* rel) { return sxtest::fixture_dir() + "/" + rel; }

SX_TEST(projection_fwd_matches_gsplat_fixture) {
    SXTensor means = SXTensor::load(fx("proj_fully_fused/means.sxt"));
    SXTensor covars = SXTensor::load(fx("proj_fully_fused/covars.sxt"));
    SXTensor viewmats = SXTensor::load(fx("proj_fully_fused/viewmats.sxt"));
    SXTensor Ks = SXTensor::load(fx("proj_fully_fused/Ks.sxt"));
    SXTensor g_radii = SXTensor::load(fx("proj_fully_fused/radii.sxt"));
    SXTensor g_means2d = SXTensor::load(fx("proj_fully_fused/means2d.sxt"));
    SXTensor g_depths = SXTensor::load(fx("proj_fully_fused/depths.sxt"));
    SXTensor g_conics = SXTensor::load(fx("proj_fully_fused/conics.sxt"));
    SXTensor g_comp = SXTensor::load(fx("proj_fully_fused/compensations.sxt"));

    ProjectionInputs in;
    in.means = means.f32(); in.covars = covars.f32();
    in.viewmats = viewmats.f32(); in.Ks = Ks.f32();
    in.N = means.shape[0]; in.C = viewmats.shape[0];
    in.width = 160; in.height = 120;                 // same as meta.json (the fixture convention)
    in.eps2d = 0.3f; in.near_plane = 0.01f; in.far_plane = 1e10f;
    in.calc_compensations = true;

    ProjectionOutputs out = projection_ewa_fwd_cpu(in);
    const size_t CN = static_cast<size_t>(in.C * in.N);

    // radii: integers. fp32 jitter at the ceil boundary is allowed — |diff|<=1 in at most 1% of cases
    size_t off_by_one = 0;
    for (size_t i = 0; i < CN * 2; i++) {
        int d = std::abs(out.radii[i] - g_radii.i32()[i]);
        SX_CHECK(d <= 1);
        if (d == 1) off_by_one++;
    }
    SX_CHECK(off_by_one <= CN * 2 / 100);

    for (size_t i = 0; i < CN * 2; i++) SX_CHECK_NEAR(out.means2d[i], g_means2d.f32()[i], 1e-4, 1e-5);
    for (size_t i = 0; i < CN; i++) SX_CHECK_NEAR(out.depths[i], g_depths.f32()[i], 1e-4, 1e-5);
    for (size_t i = 0; i < CN * 3; i++) SX_CHECK_NEAR(out.conics[i], g_conics.f32()[i], 1e-4, 1e-5);
    for (size_t i = 0; i < CN; i++) SX_CHECK_NEAR(out.compensations[i], g_comp.f32()[i], 1e-4, 1e-5);
}

SX_TEST(cpu_backend_matches_direct_call) {
    SXTensor means = SXTensor::load(fx("proj_fully_fused/means.sxt"));
    SXTensor covars = SXTensor::load(fx("proj_fully_fused/covars.sxt"));
    SXTensor viewmats = SXTensor::load(fx("proj_fully_fused/viewmats.sxt"));
    SXTensor Ks = SXTensor::load(fx("proj_fully_fused/Ks.sxt"));

    ProjectionInputs in;
    in.means = means.f32(); in.covars = covars.f32();
    in.viewmats = viewmats.f32(); in.Ks = Ks.f32();
    in.N = means.shape[0]; in.C = viewmats.shape[0];
    in.width = 160; in.height = 120; in.calc_compensations = true;

    auto backend = make_cpu_backend();
    SX_CHECK(std::string(backend->name()) == "cpu");
    ProjectionOutputs a = backend->projection_ewa_fwd(in);
    ProjectionOutputs b = projection_ewa_fwd_cpu(in);
    SX_CHECK(a.radii == b.radii);
    SX_CHECK(a.means2d == b.means2d);
    SX_CHECK(a.conics == b.conics);
    SX_CHECK(a.depths == b.depths);
    SX_CHECK(a.compensations == b.compensations);
}
