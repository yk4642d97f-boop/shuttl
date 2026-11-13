// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlReconstruct",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlReconstruct", targets: ["ShuttlReconstruct"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlReconstruct",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlReconstructTests",
            dependencies: [
                "ShuttlReconstruct",
                .product(name: "ShuttlCore", package: "ShuttlCore"),
            ],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
