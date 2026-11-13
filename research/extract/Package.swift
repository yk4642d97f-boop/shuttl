// swift-tools-version:5.10
import PackageDescription

let packages = "../../iOS/Packages"

let package = Package(
    name: "ShuttlExtract",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "extract", targets: ["extract"])],
    dependencies: [
        .package(path: "\(packages)/ShuttlCore"),
        .package(path: "\(packages)/ShuttlDetector"),
        .package(path: "\(packages)/ShuttlLinker"),
        .package(path: "\(packages)/ShuttlCourt"),
        .package(path: "\(packages)/ShuttlPlayers"),
    ],
    targets: [
        .executableTarget(
            name: "extract",
            dependencies: [
                .product(name: "ShuttlCore", package: "ShuttlCore"),
                .product(name: "ShuttlDetector", package: "ShuttlDetector"),
                .product(name: "ShuttlLinker", package: "ShuttlLinker"),
                .product(name: "ShuttlCourt", package: "ShuttlCourt"),
                .product(name: "ShuttlPlayers", package: "ShuttlPlayers"),
            ]
        ),
    ]
)
