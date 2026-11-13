// swift-tools-version:5.10
import PackageDescription

/// Runtime hardening for on-device analysis: one-at-a-time job queue, thermal /
/// background / storage policy, the compact analysis archive, diagnostics.
let package = Package(
    name: "ShuttlRuntime",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlRuntime", targets: ["ShuttlRuntime"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        .package(path: "../ShuttlAnalysis"),
    ],
    targets: [
        .target(
            name: "ShuttlRuntime",
            dependencies: [
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlAnalysis", package: "ShuttlAnalysis"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlRuntimeTests",
            dependencies: [
                "ShuttlRuntime",
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlAnalysis", package: "ShuttlAnalysis"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
