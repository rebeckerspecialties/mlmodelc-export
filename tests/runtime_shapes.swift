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
    try predictExact(model, inputs, shape: shape, values: values.map(Double.init))
}
func predictExact(_ model: MLModel, _ inputs: [String: MLMultiArray], shape: [Int], values: [Double]) throws {
    let features = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues(MLFeatureValue.init(multiArray:)))
    let prediction = try model.prediction(from: features)
    guard let output = prediction.featureValue(for: "output")?.multiArrayValue else {
        throw Failure(message: "missing output")
    }
    try require(output.dataType == .float32, "incorrect output dtype")
    try require(output.shape.map(\.intValue) == shape, "incorrect output shape")
    try require(output.count == values.count, "incorrect element count")
    for (index, value) in values.enumerated() {
        let actual = output[coordinates(index, shape)].doubleValue
        try require(actual.isFinite && actual == value, "incorrect element \(index): \(actual), expected \(value)")
    }
}
func run() throws {
    try require(CommandLine.arguments.count == 3, "expected exporter executable and fixtures directory")
    let root = URL(fileURLWithPath: CommandLine.arguments[2])
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mlmodelc-runtime-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    var predictions = 0
    for fixture in ["flexible-range", "flexible-gather", "flexible-multifunction", "flexible-weighted-legacy", "fp16-subnormal-cast", "quantized-constexpr"] {
        let bundle = temporary.appendingPathComponent(fixture + ".mlmodelc")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
        process.arguments = [root.appendingPathComponent(fixture + "/input.mlmodel").path, bundle.path]
        let weights = root.appendingPathComponent(fixture + "/weights/weights.bin")
        if FileManager.default.fileExists(atPath: weights.path) {
            process.arguments! += ["--weights", weights.path]
        }
        try process.run()
        process.waitUntilExit()
        try require(process.terminationStatus == 0, "exporter failed")
        let functions: [String?] = fixture == "flexible-multifunction" ? [nil, "first", "second"] : [nil]
        for function in functions {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuOnly
            configuration.functionName = function
            let model = try MLModel(contentsOf: bundle, configuration: configuration)
            if fixture == "fp16-subnormal-cast" || fixture == "quantized-constexpr" {
                // Binary16 subnormals are integer multiples of 2^-24; the
                // smallest normal is 1024 such units. The second fixture uses
                // (int8 + 1) / 2. Neither oracle reads emitted model data.
                let values: [Double] = fixture == "fp16-subnormal-cast"
                    ? [1, 168, 1023, 1024, -168].map { Double($0) / 16_777_216 }
                    : [-63.5, 0, 0.5, 64]
                let offsets: [Double] = fixture == "fp16-subnormal-cast" ? [0, 1.0 / 16_384] : [0, 1]
                for offset in offsets {
                    let input = try array([values.count], [Int](repeating: 0, count: values.count))
                    for index in values.indices {
                        input[coordinates(index, [values.count])] = NSNumber(value: offset)
                    }
                    try predictExact(model, ["input": input], shape: [values.count], values: values.map { $0 + offset })
                    predictions += 1
                }
                continue
            }
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
                } else if fixture == "flexible-weighted-legacy" {
                    let input = (0..<(count * 4)).map { $0 - 8 }
                    let weights = [2, -3, 5, -7]
                    let expected = input.enumerated().map { $0.element + weights[$0.offset % 4] }
                    try predict(model, ["input": array([count, 4], input)], shape: [count, 4], values: expected)
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
