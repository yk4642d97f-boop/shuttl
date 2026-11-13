// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlDetector",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlDetector", targets: ["ShuttlDetector"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlDetector",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlDetectorTests",
            dependencies: ["ShuttlDetector", .product(name: "ShuttlCore", package: "ShuttlCore")],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
