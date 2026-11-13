// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlRally",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlRally", targets: ["ShuttlRally"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        // The BST stroke model as a resource (71 MB); the app gets it through this package.
        .package(path: "../ShuttlStrokeModel"),
    ],
    targets: [
        .target(
            name: "ShuttlRally",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore"),
                           .product(name: "ShuttlStrokeModel", package: "ShuttlStrokeModel")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlRallyTests",
            dependencies: ["ShuttlRally", .product(name: "ShuttlCore", package: "ShuttlCore")],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
