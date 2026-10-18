// swift tests/runtime_control_flow.swift <exporter> <tests/control-flow>
// Compile locally with Rust, then read only final CoreML outputs.
import CoreML
import Foundation

struct Failure: Error { let message: String }
func check(_ ok: Bool, _ message: String) throws {
    if !ok { throw Failure(message: message) }
}

func run() throws {
try check(CommandLine.arguments.count == 3, "expected exporter and fixture directory")
let root = FileManager.default.temporaryDirectory.appendingPathComponent("mil-control-" + UUID().uuidString)
try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: root) }
let fixture = URL(fileURLWithPath: CommandLine.arguments[2])
let compiled = root.appendingPathComponent("control.mlmodelc")
let task = Process()
task.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
task.arguments = [fixture.appendingPathComponent("input.mlmodel").path, compiled.path,
                  "--weights", fixture.appendingPathComponent("weights/weights.bin").path]
try task.run()
task.waitUntilExit()
try check(task.terminationStatus == 0, "export failed")
let installed = root.appendingPathComponent("installed.mlmodelc")
try FileManager.default.copyItem(at: compiled, to: installed)
var predictions = 0
for policy: MLComputeUnits in [.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all] {
    let config = MLModelConfiguration()
    config.computeUnits = policy
    config.allowLowPrecisionAccumulationOnGPU = false
    let model = try MLModel(contentsOf: installed, configuration: config)
    for step in 0..<8 {
        let input = try MLMultiArray(shape: [4], dataType: .float32)
        let values: [Float] = [Float(step) + 0.25, -3, 0.5, -8]
        for i in 0..<4 { input[i] = NSNumber(value: values[i]) }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": input]))
        guard let result = output.featureValue(for: "output")?.multiArrayValue else {
            throw Failure(message: "missing output")
        }
        try check(result.shape.map(\.intValue) == [4] && result.dataType == .float32, "wrong output type/shape")
        // Independent exact dyadic expectations for (input + weight) * weight.
        let expected: [Float] = [3.5 - 2 * Float(step), 4, 1.5, -12]
        for i in 0..<4 {
            try check(result[i].floatValue.bitPattern == expected[i].bitPattern, "policy \(policy.rawValue), step \(step), index \(i): \(result[i]) != \(expected[i])")
            try check(input[i].floatValue.bitPattern == values[i].bitPattern, "input modified")
        }
        predictions += 1
    }
}
print("PASS: \(predictions) exact nested-loop/branch predictions across four requested policies; no placement claim")
}

do { try run() }
catch {
    FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
    exit(1)
}
