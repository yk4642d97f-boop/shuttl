// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlCore", targets: ["ShuttlCore"]),
    ],
    targets: [
        .target(
            name: "ShuttlCore",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlCoreTests",
            dependencies: ["ShuttlCore"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
