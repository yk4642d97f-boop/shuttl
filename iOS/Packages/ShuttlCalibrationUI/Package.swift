// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlCalibrationUI",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlCalibrationUI", targets: ["ShuttlCalibrationUI"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        .package(path: "../ShuttlDesign"),
    ],
    targets: [
        .target(
            name: "ShuttlCalibrationUI",
            dependencies: [
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlDesign", package: "ShuttlDesign"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlCalibrationUITests",
            dependencies: [
                "ShuttlCalibrationUI",
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlDesign", package: "ShuttlDesign"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
