// Run on macOS 15+: swift tests/runtime_state.swift <exporter-binary> <fixtures-dir>
// Compilation uses the Rust executable, never coremlc or precompiled fixtures.
// Expected values are written out here; none is read from a fixture or generator.
import CoreML
import Foundation

struct Failure: Error { let message: String }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message: message) }
}
func array(_ shape: [Int], _ values: [Double], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
    let result = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: type)
    try require(result.count == values.count, "wrong element count for \(shape)")
    for (index, value) in values.enumerated() { result[index] = NSNumber(value: value) }
    return result
}
func values(_ array: MLMultiArray) -> [Double] {
    (0..<array.count).map { array[$0].doubleValue }
}
func compile(_ fixture: String, into directory: URL) throws -> URL {
    let root = URL(fileURLWithPath: CommandLine.arguments[2]).appendingPathComponent(fixture)
    let bundle = directory.appendingPathComponent(fixture + ".mlmodelc")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
    process.arguments = [root.appendingPathComponent("input.mlmodel").path, bundle.path]
    let weights = root.appendingPathComponent("weights/weights.bin")
    if FileManager.default.fileExists(atPath: weights.path) {
        process.arguments! += ["--weights", weights.path]
    }
    try process.run()
    process.waitUntilExit()
    try require(process.terminationStatus == 0, "exporter failed for \(fixture)")
    return bundle
}
func load(_ bundle: URL) throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuOnly
    return try MLModel(contentsOf: bundle, configuration: configuration)
}
/// One prediction; `state` is nil for stateless models. Checks dtype, shape and every value.
func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], _ state: MLState?, output name: String,
             type: MLMultiArrayDataType, shape: [Int], expected: [Double], _ label: String) throws {
    let features = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues(MLFeatureValue.init(multiArray:)))
    let prediction = try state.map { try model.prediction(from: features, using: $0) } ?? model.prediction(from: features)
    guard let output = prediction.featureValue(for: name)?.multiArrayValue else {
        throw Failure(message: "\(label): missing output \(name)")
    }
    let actual = values(output)
    print("\(label): \(name) = \(actual)")
    try require(output.dataType == type, "\(label): incorrect output dtype")
    try require(output.shape.map(\.intValue) == shape, "\(label): incorrect output shape")
    try require(actual.allSatisfy(\.isFinite) && actual == expected, "\(label): \(actual), expected \(expected)")
}
/// The stored value of one named state.
func stored(_ state: MLState, _ name: String, _ label: String) -> [Double] {
    let result = state.withMultiArray(for: name) { values($0) }
    print("\(label): \(name) holds \(result)")
    return result
}
func scaled(_ values: [Double], _ factor: Double) -> [Double] { values.map { $0 * factor } }

func run() throws {
    try require(CommandLine.arguments.count == 3, "expected exporter executable and fixtures directory")
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mlmodelc-state-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    var predictions = 0
    let x: [Double] = [1, 2, 3, 4]
    let zeros = [Double](repeating: 0, count: 4)

    // stateful-fp16: next_state_0 = state_0 + x, y = next_state_0 + x. After
    // step k the state holds k * x and y is (k + 1) * x.
    do {
        let model = try load(try compile("stateful-fp16", into: temporary))
        try require(Set(model.modelDescription.inputDescriptionsByName.keys) == ["x"], "state listed as an input")
        try require(Set(model.modelDescription.stateDescriptionsByName.keys) == ["state_0"], "state description missing")
        let input = ["x": try array([1, 4], x, .float16)]
        let state = model.makeState()
        try require(stored(state, "state_0", "stateful-fp16 new state") == zeros, "a new state must hold zeros")
        for step in 1...3 {
            try predict(model, input, state, output: "y", type: .float16, shape: [1, 4],
                        expected: scaled(x, Double(step + 1)), "stateful-fp16 step \(step)")
            predictions += 1
        }
        try require(stored(state, "state_0", "stateful-fp16 after 3 steps") == scaled(x, 3), "state did not accumulate")
        // A second state starts from zeros and leaves the first one alone.
        let fresh = model.makeState()
        try require(stored(fresh, "state_0", "stateful-fp16 second state") == zeros, "a second state must hold zeros")
        try predict(model, input, fresh, output: "y", type: .float16, shape: [1, 4],
                    expected: scaled(x, 2), "stateful-fp16 second state step 1")
        try predict(model, input, state, output: "y", type: .float16, shape: [1, 4],
                    expected: scaled(x, 5), "stateful-fp16 first state step 4")
        try require(stored(fresh, "state_0", "stateful-fp16 second state after 1 step") == x, "states are not independent")
        predictions += 2
    }

    // stateful-fp32-pair: next_state_0 = state_0 + x, next_state_1 = state_1 +
    // next_state_0, y = next_state_0 + next_state_1. Per unit of x the states
    // hold (1, 1), (2, 3), (3, 6) and y is 2, 5, 9.
    do {
        let model = try load(try compile("stateful-fp32-pair", into: temporary))
        try require(Set(model.modelDescription.stateDescriptionsByName.keys) == ["state_0", "state_1"], "state descriptions missing")
        let input = ["x": try array([1, 4], x, .float32)]
        let state = model.makeState()
        for (step, factor) in [2.0, 5, 9].enumerated() {
            try predict(model, input, state, output: "y", type: .float32, shape: [1, 4],
                        expected: scaled(x, factor), "stateful-fp32-pair step \(step + 1)")
            predictions += 1
        }
        try require(stored(state, "state_0", "stateful-fp32-pair after 3 steps") == scaled(x, 3), "state_0 did not accumulate")
        try require(stored(state, "state_1", "stateful-fp32-pair after 3 steps") == scaled(x, 6), "state_1 did not accumulate")
        try predict(model, input, model.makeState(), output: "y", type: .float32, shape: [1, 4],
                    expected: scaled(x, 2), "stateful-fp32-pair second state step 1")
        predictions += 1
    }

    // stateful-declaration-order: n0 = s0 + p, n1 = s1 + q, y = n0 + n1. The
    // exporter reorders the declarations; each name must keep its own value.
    do {
        let model = try load(try compile("stateful-declaration-order", into: temporary))
        let q: [Double] = [10, 20, 30, 40]
        let input = ["p": try array([1, 4], x, .float16), "q": try array([1, 4], q, .float16)]
        let state = model.makeState()
        for step in 1...2 {
            try predict(model, input, state, output: "y", type: .float16, shape: [1, 4],
                        expected: scaled([11, 22, 33, 44], Double(step)), "stateful-declaration-order step \(step)")
            predictions += 1
        }
        try require(stored(state, "s0", "stateful-declaration-order after 2 steps") == scaled(x, 2), "s0 holds the wrong value")
        try require(stored(state, "s1", "stateful-declaration-order after 2 steps") == scaled(q, 2), "s1 holds the wrong value")
    }

    // blockwise-uint8-offset: output[n] = sum over k of input[k] * W[n][k] with
    // W[n][k] = scale[n][k / 32] * (data[n][k] - 128),
    // data[0][k] - 128 = k % 7 - 3, data[1][k] - 128 = (3 * k) % 11 - 5,
    // scale = [[0.5, 0.25], [2, 0.125]].
    do {
        let model = try load(try compile("blockwise-uint8-offset", into: temporary))
        func input(_ value: (Int) -> Double) throws -> [String: MLMultiArray] {
            ["input": try array([1, 64], (0..<64).map(value), .float32)]
        }
        // Column 2, first block: 0.5 * (2 - 3) and 2 * (6 - 5).
        try predict(model, try input { $0 == 2 ? 1 : 0 }, nil, output: "output", type: .float32, shape: [1, 2],
                    expected: [-0.5, 2], "blockwise-uint8-offset column 2")
        // Column 33, second block: 0.25 * (5 - 3) and 0.125 * (0 - 5).
        try predict(model, try input { $0 == 33 ? 1 : 0 }, nil, output: "output", type: .float32, shape: [1, 2],
                    expected: [0.5, -0.625], "blockwise-uint8-offset column 33")
        // All ones. Row 0: the centered bytes sum to -6 over columns 0..<32 and
        // to 3 over 32..<64: 0.5 * -6 + 0.25 * 3. Row 1: -3 and 0: 2 * -3 + 0.125 * 0.
        try predict(model, try input { _ in 1 }, nil, output: "output", type: .float32, shape: [1, 2],
                    expected: [-2.25, -6], "blockwise-uint8-offset all ones")
        predictions += 3
    }

    // blockwise-int8-scales: output = input + data * 1 + data * rows with
    // data = [[-128, -1, 0, 127], [4, -8, 16, -32]] and rows = [[0.5], [0.25]].
    do {
        let model = try load(try compile("blockwise-int8-scales", into: temporary))
        let weights: [Double] = [-192, -1.5, 0, 190.5, 5, -10, 20, -40]
        for offset in [0.0, 1] {
            try predict(model, ["input": try array([2, 4], [Double](repeating: offset, count: 8), .float32)], nil,
                        output: "output", type: .float32, shape: [2, 4],
                        expected: weights.map { $0 + offset }, "blockwise-int8-scales input \(offset)")
            predictions += 1
        }
    }
    print("PASS: \(predictions) exact predictions with model states and blockwise constants after local Rust compilation")
}
do {
    try run()
} catch {
    FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
    exit(1)
}
