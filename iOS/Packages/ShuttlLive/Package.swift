// swift-tools-version:5.10
import PackageDescription

/// The live analysis stages: glue between `ShuttlAnalysis.AnalysisStages` and the
/// per-stage packages. Kept separate so ShuttlAnalysis stays fast to test.
let package = Package(
    name: "ShuttlLive",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlLive", targets: ["ShuttlLive"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        .package(path: "../ShuttlAnalysis"),
        .package(path: "../ShuttlDetector"),
        .package(path: "../ShuttlLinker"),
        .package(path: "../ShuttlCourt"),
        .package(path: "../ShuttlHits"),
        .package(path: "../ShuttlPlayers"),
        .package(path: "../ShuttlReconstruct"),
        .package(path: "../ShuttlRally"),
        // The learned hit detector (HitNetDetector.mlpackage); see LiveStages.detectHits.
        .package(path: "../ShuttlHitModel"),
    ],
    targets: [
        .target(
            name: "ShuttlLive",
            dependencies: [
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlAnalysis", package: "ShuttlAnalysis"),
                .product(name: "ShuttlDetector", package: "ShuttlDetector"),
                .product(name: "ShuttlLinker", package: "ShuttlLinker"),
                .product(name: "ShuttlCourt", package: "ShuttlCourt"),
                .product(name: "ShuttlHits", package: "ShuttlHits"),
                .product(name: "ShuttlPlayers", package: "ShuttlPlayers"),
                .product(name: "ShuttlReconstruct", package: "ShuttlReconstruct"),
                .product(name: "ShuttlRally", package: "ShuttlRally"),
                .product(name: "ShuttlHitModel", package: "ShuttlHitModel"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlLiveTests",
            dependencies: ["ShuttlLive", .product(name: "ShuttlCourt", package: "ShuttlCourt"),
                           .product(name: "ShuttlDetector", package: "ShuttlDetector")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
