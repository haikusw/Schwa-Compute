// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Compute",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "Compute", targets: ["Compute"])
    ],
    targets: [
        .target(name: "Compute"),
        .executableTarget(name: "ComputeDemos", dependencies: ["Compute"]),
        .testTarget(name: "ComputeTests", dependencies: ["Compute"])
    ]
)