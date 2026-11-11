// splatx-metal — the single source of truth for function constant index bands
// (shared by host C++ and device MSL)
//
// D4: index bands — projection 0–7, SH 8–15, isect 16–23, raster 24–31, adam/mcmc 32–39,
//     ssim 40–47, train step 48–55.
// D5-3: for a function constant, both the name and the index MUST be unique library-wide —
//        every .metal TU shares the single declaration below (docs/kernel-specs/00-index.md §2).
#ifndef SPLATX_FUNCTION_CONSTANTS_H
#define SPLATX_FUNCTION_CONSTANTS_H

enum SXFunctionConstantIndex {
    // --- projection (0–7) ---
    kFCProjUseCovars     = 0,
    kFCProjHasOpacities  = 1,
    kFCProjCalcComp      = 2,
    kFCProjSingleCamera  = 3,
    kFCProjViewmatsGrad  = 4,
    // --- spherical harmonics (8–15) ---
    kFCShHasMasks        = 8,
    kFCShComputeVDirs    = 9,
    // --- tile intersect (16–23) ---
    kFCIsectAccuTile     = 16,
    // --- rasterize (24–31) ---
    kFCRastHasBackground = 24,
    kFCRastHasMasks      = 25,
    kFCRastHasAbsGrad    = 26,
    kFCRastHasDepth      = 27,
    // --- adam / mcmc (32–39) ---
    kFCAdamHasValidMask  = 32,
    // --- ssim (40–47) ---
    kFCSsimNeedCoeffs    = 40,
    // --- train step (48–55) ---
    kFCTrainHasMask      = 48,
};

#ifdef __METAL_VERSION__
// Specialization declarations — the kernels reference only these names (values are injected
// when the pipeline is created).
constant bool kProjUseCovars     [[function_constant(kFCProjUseCovars)]];
constant bool kProjHasOpacities  [[function_constant(kFCProjHasOpacities)]];
constant bool kProjCalcComp      [[function_constant(kFCProjCalcComp)]];
constant bool kProjSingleCamera  [[function_constant(kFCProjSingleCamera)]];
constant bool kProjViewmatsGrad  [[function_constant(kFCProjViewmatsGrad)]];
constant bool kShHasMasks        [[function_constant(kFCShHasMasks)]];
constant bool kShComputeVDirs    [[function_constant(kFCShComputeVDirs)]];
constant bool kIsectAccuTile     [[function_constant(kFCIsectAccuTile)]];
constant bool kRastHasBackground [[function_constant(kFCRastHasBackground)]];
constant bool kRastHasMasks      [[function_constant(kFCRastHasMasks)]];
constant bool kRastHasAbsGrad    [[function_constant(kFCRastHasAbsGrad)]];
constant bool kRastHasDepth      [[function_constant(kFCRastHasDepth)]];
constant bool kAdamHasValidMask  [[function_constant(kFCAdamHasValidMask)]];
constant bool kSsimNeedCoeffs    [[function_constant(kFCSsimNeedCoeffs)]];
constant bool kTrainHasMask      [[function_constant(kFCTrainHasMask)]];
#endif // __METAL_VERSION__

#endif // SPLATX_FUNCTION_CONSTANTS_H
