// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlResultsUI",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlResultsUI", targets: ["ShuttlResultsUI"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        .package(path: "../ShuttlDesign"),
    ],
    targets: [
        .target(
            name: "ShuttlResultsUI",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore"),
                           .product(name: "ShuttlDesign", package: "ShuttlDesign")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlResultsUITests",
            dependencies: [
                "ShuttlResultsUI",
                .product(name: "ShuttlCore", package: "ShuttlCore"),
            ],
            resources: [.copy("Fixtures")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
