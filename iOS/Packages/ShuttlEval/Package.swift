// swift-tools-version:5.10
import PackageDescription

// ShuttlEval — port of backend/app/services/tracking/metrics.py (+ the non-rendering
// half of sim.py) and a `shuttl-bench` CLI equivalent to backend/scripts/benchmark.py.
// The library builds for iOS and macOS; the CLI is macOS-only (its main is gated).
let package = Package(
    name: "ShuttlEval",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlEval", targets: ["ShuttlEval"]),
        .executable(name: "shuttl-bench", targets: ["shuttl-bench"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlEval",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .executableTarget(
            name: "shuttl-bench",
            dependencies: ["ShuttlEval", .product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlEvalTests",
            dependencies: ["ShuttlEval", .product(name: "ShuttlCore", package: "ShuttlCore")],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
