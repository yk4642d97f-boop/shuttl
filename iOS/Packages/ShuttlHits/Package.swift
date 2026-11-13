// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlHits",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlHits", targets: ["ShuttlHits"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlHits",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlHitsTests",
            dependencies: ["ShuttlHits"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
