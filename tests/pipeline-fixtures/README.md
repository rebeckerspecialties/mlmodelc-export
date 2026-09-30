# Pipeline fixtures

`cast/input.mlmodel` contains two MLPrograms with an FP16 boundary:
`input:fp32 -> rounded:fp16 -> result:fp32`. The input `[1.0003, -1.0003,
1.0007, -1.0007]` distinguishes a rounded cast from a compiler-eliminated cast.

`weighted/input.mlmodel` adds `[0.25, -0.25, 0.5, -0.5]` before and after that
boundary. Both stages reference the same external FP32 blob and the wrapper
exposes both the intermediate half tensor and final float tensor. These sources
were emitted by RustNN's Pipeline precision-boundary converter fixture on
2026-09-30; expected values are computed independently in `runtime_pipelines.swift`.

The `expected.mlmodelc` directories were captured with Xcode 27's `coremlc
3600.25.1`, targeting iOS 18. The tests compare wrapper loader data and child-name
data byte for byte, and compare JSON schemas and operation metadata structurally.
Child MIL programs remain separate. Native compilation duplicates the weight
sidecar; our output writes one root asset and links each child to it.

`dynamic/input.mlmodel` wraps the existing independently generated flexible ReLU
fixture in a Pipeline. `python3 -B tests/pipeline-fixtures/generate_dynamic.py
--check` verifies its source. Runtime coverage resizes `[1] -> [2] -> [4] -> [1]`
and checks every returned shape and value.

`dynamic-boundary/input.mlmodel` uses two precision stages with unknown MIL
extent and source bounds `[0, 8]`; runtime coverage checks `[1] -> [3] -> [7] ->
[1]` on one loaded model. `scalar` checks CoreML's one-element scalar interface.
`masked` carries a WebNN uint8 mask across the stage boundary through a float32
proxy and retains the original input for `where`; `indexed` carries argMax
indices as int32 into a gather after the FP16 boundary. Their expected values
are also calculated directly in Swift rather than from a second CoreML model.

Run `cargo test --test pipelines` and
`swift tests/runtime_pipelines.swift target/release/mlmodelc-export tests/pipeline-fixtures`.
The runtime test exports each source with Rust before loading it; it never invokes
Apple's compiler or loads the native golden bundles.
