// Run on macOS 15+: swift tests/runtime_shapes.swift <exporter-binary> <fixtures-dir>
// Compilation uses the Rust executable, never coremlc or precompiled fixtures.
import CoreML
import Foundation

struct Failure: Error { let message: String }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message: message) }
}
func array(_ shape: [Int], _ values: [Int], type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
    let result = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: type)
    for (index, value) in values.enumerated() {
        result[coordinates(index, shape)] = NSNumber(value: value)
    }
    return result
}
func coordinates(_ index: Int, _ shape: [Int]) -> [NSNumber] {
    var remaining = index
    var result = [NSNumber](repeating: 0, count: shape.count)
    for axis in shape.indices.reversed() {
        result[axis] = NSNumber(value: remaining % shape[axis])
        remaining /= shape[axis]
    }
    return result
}
func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], shape: [Int], values: [Int]) throws {
    let features = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues(MLFeatureValue.init(multiArray:)))
    let prediction = try model.prediction(from: features)
    guard let output = prediction.featureValue(for: "output")?.multiArrayValue else {
        throw Failure(message: "missing output")
    }
    try require(output.shape.map(\.intValue) == shape, "incorrect output shape")
    try require(output.count == values.count, "incorrect element count")
    for (index, value) in values.enumerated() {
        try require(output[coordinates(index, shape)].doubleValue == Double(value), "incorrect element \(index)")
    }
}
func run() throws {
    try require(CommandLine.arguments.count == 3, "expected exporter executable and fixtures directory")
    let root = URL(fileURLWithPath: CommandLine.arguments[2])
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mlmodelc-runtime-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    var predictions = 0
    for fixture in ["flexible-range", "flexible-gather", "flexible-multifunction"] {
        let bundle = temporary.appendingPathComponent(fixture + ".mlmodelc")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
        process.arguments = [root.appendingPathComponent(fixture + "/input.mlmodel").path, bundle.path]
        try process.run()
        process.waitUntilExit()
        try require(process.terminationStatus == 0, "exporter failed")
        let functions: [String?] = fixture == "flexible-multifunction" ? [nil, "first", "second"] : [nil]
        for function in functions {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuOnly
            configuration.functionName = function
            let model = try MLModel(contentsOf: bundle, configuration: configuration)
            if fixture == "flexible-multifunction" {
                let expectedDefault = function == "first" ? 4 : 2
                try require(model.modelDescription.inputDescriptionsByName["input"]?.multiArrayConstraint?.shape == [NSNumber(value: expectedDefault)], "wrong default function/schema")
            }
            for count in [1, 2, 4, 1] {
                if fixture == "flexible-gather" {
                    let indices = Array([0, -1, 1, -2].prefix(count))
                    let data = [10, 11, 20, 21, 30, 31]
                    let expected = indices.flatMap { index -> [Int] in
                        let row = index < 0 ? index + 3 : index
                        return [data[row * 2], data[row * 2 + 1]]
                    }
                    try predict(model, ["data": array([3, 2], data), "indices": array([1, count], indices, type: .int32)], shape: [1, count, 2], values: expected)
                } else {
                    let input = (0..<count).map { $0 - 1 }
                    try predict(model, ["input": array([count], input)], shape: [count], values: input.map { max(0, $0) })
                }
                predictions += 1
            }
        }
    }
    print("PASS: \(predictions) exact predictions after local Rust compilation")
}
try run()
