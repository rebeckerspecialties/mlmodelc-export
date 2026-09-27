// macOS 15+: swift tests/runtime_identifiers.swift <exporter> <generator> [--check-invalid]
// Inputs are locally generated protobufs; no coremlc or precompiled model is used.
import CoreML
import Foundation

struct Failure: Error { let message: String }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message: message) }
}
func runProcess(_ executable: String, _ arguments: [String]) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    try process.run()
    process.waitUntilExit()
    try require(process.terminationStatus == 0, "command failed: \(executable)")
}
func array(_ values: [Double]) throws -> MLMultiArray {
    let result = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .float32)
    for (index, value) in values.enumerated() { result[index] = NSNumber(value: value) }
    return result
}
func run() throws {
    try require((3...4).contains(CommandLine.arguments.count), "expected exporter and generator")
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mlmodelc-identifiers-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    try runProcess("/usr/bin/env", ["python3", "-B", CommandLine.arguments[2], temporary.path])
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuOnly
    func compile(_ name: String) throws -> URL {
        let bundle = temporary.appendingPathComponent(name + ".mlmodelc")
        try runProcess(CommandLine.arguments[1], [temporary.appendingPathComponent(name + ".mlmodel").path, bundle.path])
        return bundle
    }
    if CommandLine.arguments.last == "--check-invalid" {
        let bundle = try compile("reserved")
        var rejected = false
        do {
            _ = try MLModel(contentsOf: bundle, configuration: configuration)
        } catch {
            rejected = true
            print("Expected reserved identifier rejection: \(error)")
        }
        try require(rejected, "this OS did not reproduce the reserved-state failure")
    }
    let bundle = try compile("escaped")
    let rawJSON = try Data(contentsOf: bundle.appendingPathComponent("metadata.json"))
    let metadata = try JSONSerialization.jsonObject(with: rawJSON) as! [[String: Any]]
    let userJSON = metadata[0]["userDefinedMetadata"] as? [String: String]
    try require(userJSON?["rustnn.coreml.name_encoding"] == "hex-v1", "metadata.json lost producer encoding marker")
    let model = try MLModel(contentsOf: bundle, configuration: configuration)
    let user = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
    try require(user?["rustnn.coreml.name_encoding"] == "hex-v1", "runtime model description lost producer encoding marker")
    let names = [
        "rustnn_escaped_7374617465",
        "rustnn_escaped_727573746e6e5f657363617065645f37333734363137343635",
        "rustnn_escaped_706173742e6b6579",
    ]
    let outputs = ["rustnn_escaped_72657475726e", "output"]
    try require(Set(model.modelDescription.inputDescriptionsByName.keys) == Set(names), "input bindings changed")
    try require(Set(model.modelDescription.outputDescriptionsByName.keys) == Set(outputs), "output bindings changed")
    var checked = 0
    for offset in [0.0, 1, 2] {
        let inputs = [
            [1.0, -2, 3.5, -4].map { $0 + offset },
            [8.0, 4, -1.5, 2],
            [4.0, -3, 0, 2],
        ]
        let expected = [[9.0, 2, 2, -2].map { $0 + offset }, [4.0, 0, 0, 2]]
        var features: [String: MLFeatureValue] = [:]
        for (name, values) in zip(names, inputs) {
            features[name] = MLFeatureValue(multiArray: try array(values))
        }
        let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
        for (name, values) in zip(outputs, expected) {
            guard let result = prediction.featureValue(for: name)?.multiArrayValue else {
                throw Failure(message: "missing output \(name)")
            }
            try require(result.dataType == .float32 && result.shape == [4], "wrong output type/shape")
            for (index, value) in values.enumerated() {
                let actual = result[index].doubleValue
                try require(actual.isFinite && actual == value, "\(name)[\(index)]: \(actual), expected \(value)")
                checked += 1
            }
        }
    }
    print("PASS: preserved producer metadata and feature bindings; \(checked) exact FP32 output values")
}
do {
    try run()
} catch {
    FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
    exit(1)
}
