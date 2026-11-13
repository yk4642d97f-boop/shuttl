// swift-tools-version:5.10
import PackageDescription

// The app's single design system ("Riso Court · Overprint"): colour, type, metrics,
// paper surfaces, shared components and the signature devices. The app and every UI
// package (results, calibration, export) depend on this instead of copying tokens.
let package = Package(
    name: "ShuttlDesign",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlDesign", targets: ["ShuttlDesign"]),
    ],
    targets: [
        .target(
            name: "ShuttlDesign",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlDesignTests",
            dependencies: ["ShuttlDesign"],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
