// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlAnalysis",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlAnalysis", targets: ["ShuttlAnalysis"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlAnalysis",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlAnalysisTests",
            dependencies: ["ShuttlAnalysis", .product(name: "ShuttlCore", package: "ShuttlCore")],
            exclude: ["Fixtures"],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
