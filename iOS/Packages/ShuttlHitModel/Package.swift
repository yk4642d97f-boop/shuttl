// swift-tools-version:5.10
import PackageDescription

// The learned hit detector (HitNetDetector.mlpackage, 5.3 MB) as a CoreML resource, in a package
// of its own so the model is a build input rather than a source tree the app target copies —
// the same layout ShuttlStrokeModel uses for the stroke classifier. ShuttlLive depends on this
// package (Bundle.module), so the model ships inside the app through it.
//
// The package also owns the feature derivation and the decoder, because both are pinned by the
// same fixtures as the model: a port that derives the (T,51) matrix or decodes the peaks
// differently from research/hitnet/common.py is wrong in a way the model alone cannot show.
let package = Package(
    name: "ShuttlHitModel",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlHitModel", targets: ["ShuttlHitModel"]),
    ],
    dependencies: [
        // HitFeatures/HitDecoder are plain arithmetic, but they speak the packages' own types
        // (Track2D, PlayerObservation, HitEvent) so callers do not have to convert.
        .package(path: "../ShuttlCore"),
    ],
    targets: [
        .target(
            name: "ShuttlHitModel",
            dependencies: [.product(name: "ShuttlCore", package: "ShuttlCore")],
            resources: [.copy("Models/HitNetDetector.mlpackage")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "ShuttlHitModelTests",
            dependencies: ["ShuttlHitModel", .product(name: "ShuttlCore", package: "ShuttlCore")]
        ),
    ]
)
