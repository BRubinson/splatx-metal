// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "splatx-metal",
    // The command-line trainer and runtime-compiled Metal kernels use APIs
    // available on macOS 15.  The GUI remains free to require newer SDKs,
    // but keeping the package target at 15 lets the CLI run on current Macs.
    platforms: [.macOS("15.0"), .iOS("26.0")],
    products: [
        .library(name: "SplatXCore", targets: ["SplatXCore"]),
        .library(name: "SplatXMetal", targets: ["SplatXMetal"]),
        // Exposed so apps can consume the C surface (sx_convert_run PLY↔SPZ etc.) directly.
        // Formalizes the precedent of SplatXBench importing SplatXTrainC directly in SmokeTrainView.
        .library(name: "SplatXTrainC", targets: ["SplatXTrainC"]),
        .executable(name: "splatx-core-tests", targets: ["splatx-core-tests"]),
        .executable(name: "splatx", targets: ["splatx-cli"]),
    ],
    dependencies: [
        .package(path: "../swift-argument-parser"),
    ],
    targets: [
        // Niantic spz — SPZ format C++ library (vendored v2.1.0, MIT).
        // gzip (zlib) single-stream formats (SPZ v1–3) only — no zstd needed (rationale in ThirdParty/spz/VERSION.txt).
        .target(
            name: "SplatXSpz",
            path: "ThirdParty/spz",
            sources: ["src/cc"],
            publicHeadersPath: "src/cc",
            linkerSettings: [
                .linkedLibrary("z") // Apple SDK libz.tbd (common to macOS/iOS)
            ]
        ),
        .target(
            name: "SplatXCore",
            dependencies: ["SplatXSpz"],
            cxxSettings: [
                .headerSearchPath("../../ThirdParty/metal-cpp"),
                .headerSearchPath("../SplatXKernelSrc"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("CoreGraphics"), // ImageIO.cpp (CGBitmapContext/CGImage)
                .linkedFramework("ImageIO"),      // ImageIO.cpp (CGImageSource/Destination)
            ]
        ),
        // Swift consumption surface (probe + Phase 6 round 2 Task 5 SplatXTrainer — 1:1 with
        // TrainerBackend). SplatXTrainC dependency: training goes through the pure C surface
        // (sx_train_run_args etc.) (SplatXCore public headers are C++, so Swift cannot import
        // them directly — keeps the existing layering).
        .target(
            name: "SplatXMetal",
            dependencies: ["SplatXTrainC"],
            resources: [.copy("Kernels")]
        ),
        // Pure C surface for the Swift CLI (SplatXCore public headers are C++ — Swift cannot import them directly).
        .target(
            name: "SplatXTrainC",
            dependencies: ["SplatXCore"],
            cxxSettings: [
                .define("KERNEL_SRC_DIR", to: "\"\(Context.packageDirectory)/Sources/SplatXKernelSrc\""),
            ]
        ),
        .testTarget(name: "SplatXMetalTests", dependencies: ["SplatXMetal", "SplatXTests"]),
        .executableTarget(
            name: "splatx-cli",
            dependencies: [
                "SplatXMetal",
                "SplatXTrainC",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        // Test body library (Phase 6 Task 3 — every TU moved here except main.cpp). The single
        // canonical runner shared by the macOS CLI and the SplatXBench iOS app — the public
        // surface is include/splatx_tests_c.h (pure C — parsed by Swift ClangImporter, the
        // SplatXTrainC pattern).
        .target(
            name: "SplatXTests",
            // spz: link smoke test (SpzTests.cpp) / SplatXTrainC: C API integration assertions
            // (color-correction Task 2 — sx_train_run_args → SxTrainStats.final_cc_psnr E2E,
            // TrainerTests.cpp). The SplatXBenchKit dylib already carries all three targets, so
            // this internal dependency creates no double link (the contract in the comment near
            // the top of Package.swift is unchanged).
            dependencies: ["SplatXCore", "SplatXSpz", "SplatXTrainC"],
            cxxSettings: [
                .headerSearchPath("../SplatXKernelSrc"), // KernelConstants.h (single canonical source of the level-set t constants)
                .define("FIXTURE_DIR", to: "\"\(Context.packageDirectory)/fixtures\""),
                .define("KERNEL_SRC_DIR", to: "\"\(Context.packageDirectory)/Sources/SplatXKernelSrc\""),
            ]
        ),
        .executableTarget(
            name: "splatx-core-tests",
            dependencies: ["SplatXTests"] // runner/subcommand dispatch only — the TU bodies live in SplatXTests
        ),
    ],
    cxxLanguageStandard: .cxx17
)
