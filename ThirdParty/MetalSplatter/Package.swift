// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "MetalSplatter",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
        .visionOS(.v2),
    ],
    products: [
        .library(
            name: "PLYIO",
            targets: [ "PLYIO" ]
        ),
        .library(
            name: "SplatIO",
            targets: [ "SplatIO" ]
        ),
        .library(
            name: "MetalSplatter",
            targets: [ "MetalSplatter" ]
        ),
        // splatx-metal vendoring (Task 1): SampleBoxRenderer/SplatConverter products
        // and the swift-argument-parser dependency removed -- the SplatXMetalTrainer
        // app consumes only MetalSplatter/SplatIO. See VERSION.txt for the full
        // deviation list.
    ],
    dependencies: [
        .package(url: "https://github.com/scier/spz-swift.git", from: "2.1.0"),
    ],
    targets: [
        .target(
            name: "PLYIO",
            path: "PLYIO",
            sources: [ "Sources" ]
        ),
        .testTarget(
            name: "PLYIOTests",
            dependencies: [ "PLYIO" ],
            path: "PLYIO",
            sources: [ "Tests" ],
            resources: [ .copy("TestData") ]
        ),
        .target(
            name: "SplatIO",
            dependencies: [
                "PLYIO",
                .product(name: "spz", package: "spz-swift"),
            ],
            path: "SplatIO",
            sources: [ "Sources" ]
        ),
        .testTarget(
            name: "SplatIOTests",
            dependencies: [ "SplatIO" ],
            path: "SplatIO",
            sources: [ "Tests" ],
            resources: [ .copy("TestData") ]
        ),
        .target(
            name: "MetalSplatter",
            dependencies: [ "PLYIO", "SplatIO" ],
            path: "MetalSplatter",
            sources: [ "Sources" ],
            resources: [ .process("Resources") ]
        ),
        .testTarget(
            name: "MetalSplatterTests",
            dependencies: [ "MetalSplatter" ],
            path: "MetalSplatter",
            sources: [ "Tests" ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
