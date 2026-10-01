// macOS 15+: swift tests/runtime_pipelines.swift <exporter-binary> <fixtures-dir>
// Every bundle is emitted by Rust; the CoreML compiler is never invoked.
import CoreML
import Foundation

struct PipelineFailure: Error { let message: String }

func check(_ condition: Bool, _ message: String) throws {
    if !condition { throw PipelineFailure(message: message) }
}

func validatePipelineBundles(_ root: URL, units: MLComputeUnits = .cpuOnly) throws -> Int {
    let values: [Float] = [1.0003, -1.0003, 1.0007, -1.0007]
    let weights: [Float] = [0.25, -0.25, 0.5, -0.5]
    var predictions = 0
    for name in ["cast", "weighted", "dynamic", "masked", "masked-int32", "scalar", "indexed", "dynamic-boundary"] {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        configuration.allowLowPrecisionAccumulationOnGPU = false
        let model = try MLModel(contentsOf: root.appendingPathComponent(name + ".mlmodelc"), configuration: configuration)
        let counts = name == "dynamic" ? [1, 2, 4, 1]
            : name == "weighted" ? Array(repeating: 4, count: 32)
            : name == "dynamic-boundary" ? [1, 3, 7, 1] : name == "scalar" ? [1] : [4]
        for count in counts {
            let shape = name == "indexed" ? [2, 2] : [count]
            let input = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
            let source = input.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count { source[i] = name == "dynamic" ? Float(i - 2) : values[i % values.count] }
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: input)]))
            let resultName = name == "dynamic" ? "output" : "result"
            guard let result = output.featureValue(for: resultName)?.multiArrayValue else { throw PipelineFailure(message: "missing \(name) output") }
            try check(result.dataType == .float32 && result.shape.map(\.intValue) == shape, "wrong \(name) result dtype/shape")
            let data = result.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                let expected: Float = name == "dynamic" ? max(0, Float(i - 2))
                    : name == "weighted" ? Float(Float16(values[i] + weights[i])) + weights[i]
                    : name.hasPrefix("masked") && values[i] < 0 ? values[i]
                    : name == "indexed" ? Float(Float16(values[i / 2 * 2]))
                    : Float(Float16(values[i % values.count]))
                let offset = name == "indexed" ? (i / 2) * result.strides[0].intValue + (i % 2) * result.strides[1].intValue : i * result.strides[0].intValue
                try check(data[offset].bitPattern == expected.bitPattern, "\(name) index \(i): \(data[offset]), expected \(expected)")
            }
            if name == "weighted" {
                guard let rounded = output.featureValue(for: "rounded")?.multiArrayValue else { throw PipelineFailure(message: "missing rounded output") }
                try check(rounded.dataType == .float16 && rounded.count == 4, "wrong rounded dtype/shape")
                let bits = rounded.dataPointer.assumingMemoryBound(to: UInt16.self)
                for i in 0..<4 { try check(bits[i * rounded.strides[0].intValue] == Float16(values[i] + weights[i]).bitPattern, "wrong rounded half encoding") }
            }
            for i in 0..<count {
                try check(source[i].bitPattern == (name == "dynamic" ? Float(i - 2) : values[i % values.count]).bitPattern,
                          "\(name) changed input at \(i)")
            }
            predictions += 1
        }
    }
    return predictions
}

#if os(macOS)
func run() throws {
    try check(CommandLine.arguments.count == 3, "expected exporter binary and Pipeline fixture root")
    let fixtures = URL(fileURLWithPath: CommandLine.arguments[2])
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("pipeline-runtime-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for name in ["cast", "weighted", "dynamic", "masked", "masked-int32", "scalar", "indexed", "dynamic-boundary"] {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
        task.arguments = [fixtures.appendingPathComponent(name + "/input.mlmodel").path, root.appendingPathComponent(name + ".mlmodelc").path]
        if name == "weighted" { task.arguments! += ["--weights", fixtures.appendingPathComponent("weighted/weights.bin").path] }
        try task.run()
        task.waitUntilExit()
        try check(task.terminationStatus == 0, "Pipeline export failed")
    }
    // Match ordinary app staging: Foundation does not preserve hard links when
    // copying directory trees. Every stage must still find one root asset.
    let copied = root.appendingPathComponent("installed")
    try FileManager.default.createDirectory(at: copied, withIntermediateDirectories: true)
    for name in ["cast", "weighted", "dynamic", "masked", "masked-int32", "scalar", "indexed", "dynamic-boundary"] {
        try FileManager.default.copyItem(at: root.appendingPathComponent(name + ".mlmodelc"),
                                        to: copied.appendingPathComponent(name + ".mlmodelc"))
    }
    let weighted = copied.appendingPathComponent("weighted.mlmodelc")
    let assets = FileManager.default.enumerator(at: weighted, includingPropertiesForKeys: nil)!
        .compactMap { $0 as? URL }.filter { $0.lastPathComponent == "weights.bin" }
    try check(assets.count == 1, "copy transport duplicated weights per child: \(assets.map(\.path))")
    try check(assets[0].resolvingSymlinksInPath().path == weighted.appendingPathComponent("weights/weights.bin").resolvingSymlinksInPath().path,
              "copied asset is not at the shared root: \(assets[0].path)")
    let originalWeights = try Data(contentsOf: fixtures.appendingPathComponent("weighted/weights.bin"))
    try check(try Data(contentsOf: assets[0]) == originalWeights, "copy transport changed weights")
    var predictions = 0
    for units: MLComputeUnits in [.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all] {
        predictions += try validatePipelineBundles(copied, units: units)
    }
    print("PASS: \(predictions) exact copied-Pipeline predictions across four policies; one weight asset, repeated outputs, dynamic resizing, unchanged inputs")
}
try run()
#endif
