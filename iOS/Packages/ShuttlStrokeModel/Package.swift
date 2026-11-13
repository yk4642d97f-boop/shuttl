// swift-tools-version:5.10
import PackageDescription

// The BST-CG-AP seq30 stroke classifier as a CoreML resource, in a package of its own so the
// 71 MB model is a build input rather than a source tree the app target copies. `ShuttlRally`
// depends on this package (Bundle.module), so the model ships inside the app through it.
let package = Package(
    name: "ShuttlStrokeModel",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ShuttlStrokeModel", targets: ["ShuttlStrokeModel"]),
    ],
    targets: [
        .target(
            name: "ShuttlStrokeModel",
            resources: [.copy("Models/BSTStroke.mlpackage")],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
    ]
)
