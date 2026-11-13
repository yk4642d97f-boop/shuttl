// swift-tools-version:5.10
import PackageDescription

let packages = "../../iOS/Packages"

let package = Package(
    name: "ShuttlHitsEval",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "hitseval", targets: ["hitseval"])],
    dependencies: [
        .package(path: "\(packages)/ShuttlCore"),
        .package(path: "\(packages)/ShuttlLive"),
        .package(path: "\(packages)/ShuttlAnalysis"),
        .package(path: "\(packages)/ShuttlHits"),
    ],
    targets: [
        .executableTarget(
            name: "hitseval",
            dependencies: [
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlLive", package: "ShuttlLive"),
                .product(name: "ShuttlAnalysis", package: "ShuttlAnalysis"),
                .product(name: "ShuttlHits", package: "ShuttlHits"),
            ]
        ),
    ]
)
