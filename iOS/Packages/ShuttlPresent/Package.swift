// swift-tools-version:5.10
import PackageDescription

/// Pure data / presentation logic that the record screen binds to.
/// No SwiftUI, no colours, no fonts: views live in the app / design packages.
///  • CourtLock: live "court locked" setup guide fed with camera frames.
let package = Package(
    name: "ShuttlPresent",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlPresent", targets: ["ShuttlPresent"]),
    ],
    dependencies: [
        .package(path: "../ShuttlCore"),
        .package(path: "../ShuttlCourt"),
    ],
    targets: [
        .target(
            name: "ShuttlPresent",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore"),
                           .product(name: "ShuttlCourt", package: "ShuttlCourt")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlPresentTests",
            dependencies: ["ShuttlPresent",
                           .product(name: "ShuttlCore", package: "ShuttlCore"),
                           .product(name: "ShuttlCourt", package: "ShuttlCourt")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
