// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ai_edge_litert_lm",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "ai-edge-litert-lm",
            targets: ["ai_edge_litert_lm"]
        )
    ],
    dependencies: [
        // Official Google LiteRT-LM Swift SDK
        .package(url: "https://github.com/google-ai-edge/LiteRT-LM", from: "0.13.0")
    ],
    targets: [
        .target(
            name: "ai_edge_litert_lm",
            dependencies: [
                .product(name: "LiteRTLM", package: "LiteRT-LM")
            ],
            path: "Sources/ai_edge_litert_lm"
        )
    ]
)
