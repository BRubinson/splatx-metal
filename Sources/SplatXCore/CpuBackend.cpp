#include "splatx/KernelBackend.h"

namespace splatx {

namespace {
class CpuBackend final : public KernelBackend {
public:
    const char* name() const override { return "cpu"; }
    ProjectionOutputs projection_ewa_fwd(const ProjectionInputs& in) override {
        return projection_ewa_fwd_cpu(in);
    }
    void fused_adam(const AdamInputs& in) override { fused_adam_cpu(in); }
    ProjectionFusedOutputs projection_fused_fwd(const ProjectionFusedInputs& in) override {
        return projection_fused_fwd_cpu(in);
    }
    ShOutputs sh_fwd(const ShInputs& in) override { return sh_fwd_cpu(in); }
    IsectTilesOutputs isect_tiles(const IsectTilesInputs& in) override {
        return isect_tiles_cpu(in);
    }
    // The CPU backend keeps the host cumsum (Task 4 — the GPU 3-dispatch scan replaces the Metal side only).
    std::vector<int64_t> prefix_scan_i64(const int32_t* v, int64_t n) override {
        if (n < 0) throw std::invalid_argument("prefix_scan_i64: negative n");
        if (n > 0 && !v) throw std::invalid_argument("prefix_scan_i64: null input");
        return cumsum_i64(v, static_cast<std::size_t>(n));
    }
    // Spec 03 sort — index-based std::stable_sort (Task 6; ulong keys, ties preserve emit order).
    SortIsectOutputs sort_isect(const SortIsectInputs& in) override {
        return sort_isect_cpu(in);
    }
    std::vector<int32_t> isect_offset_encode(const IsectOffsetInputs& in) override {
        return isect_offset_encode_cpu(in);
    }
    RasterizeFwdOutputs rasterize_fwd(const RasterizeFwdInputs& in) override {
        return rasterize_fwd_cpu(in);
    }
    std::vector<float> rng_normal_f32(const RngNormalInputs& in) override {
        return rng_normal_f32_cpu(in);
    }
    std::vector<uint32_t> rng_philox_u32(const RngNormalInputs& in) override {
        return rng_philox_u32_cpu(in);
    }
    RelocationOutputs mcmc_relocation(const RelocationInputs& in) override {
        return mcmc_relocation_cpu(in);
    }
    void mcmc_perturb_positions(const PerturbInputs& in) override {
        mcmc_perturb_positions_cpu(in);
    }
    SsimFwdOutputs ssim_fwd(const SsimInputs& in) override { return ssim_fwd_cpu(in); }
    // sh_bwd/ssim_bwd have no CPU reference — they are compared directly against the autograd
    // golden fixtures (00-index §6 decision).
};
} // namespace

std::unique_ptr<KernelBackend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

} // namespace splatx
