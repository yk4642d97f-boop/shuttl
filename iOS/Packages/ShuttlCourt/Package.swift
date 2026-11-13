// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlCourt",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlCourt", targets: ["ShuttlCourt"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlCourt",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlCourtTests",
            dependencies: ["ShuttlCourt"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
