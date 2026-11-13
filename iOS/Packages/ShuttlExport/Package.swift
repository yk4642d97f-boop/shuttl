// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "ShuttlExport",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlExport", targets: ["ShuttlExport"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        .package(path: "../ShuttlDesign"),
    ],
    targets: [
        .target(
            name: "ShuttlExport",
            dependencies: [
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlDesign", package: "ShuttlDesign"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlExportTests",
            dependencies: [
                "ShuttlExport",
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlDesign", package: "ShuttlDesign"),
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
