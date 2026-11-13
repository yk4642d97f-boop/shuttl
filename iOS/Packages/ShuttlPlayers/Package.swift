// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlPlayers",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlPlayers", targets: ["ShuttlPlayers"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlPlayers",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        // Vision-vs-YOLO benchmark on the TrackNetV2 Test clips (macOS tool, not shipped).
        .executableTarget(
            name: "ShuttlPlayersBench",
            dependencies: ["ShuttlPlayers", .product(name: "ShuttlCore", package: "ShuttlCore")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlPlayersTests",
            dependencies: ["ShuttlPlayers", .product(name: "ShuttlCore", package: "ShuttlCore")],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
