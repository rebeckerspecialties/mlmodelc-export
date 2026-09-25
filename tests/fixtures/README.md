# Reference fixtures

Each subdirectory under `tests/fixtures/` is one test case. Layout:

```
tests/fixtures/<name>/
  input.mlmodel                       # raw CoreML protobuf (input)
  weights/weights.bin                 # optional source external weights
  expected-macos.mlmodelc/             # output produced by `xcrun coremlc compile`
    model.mil
    coremldata.bin
    metadata.json
    analytics/coremldata.bin
    weights/weights.bin                # only when input has external weights
  observed-watchos-broken.mlmodelc/    # what Apple's arm64_32 watchOS stub compiler emits
    coremldata.bin                     # truncated — missing the 16-byte loader trailer
    analytics/coremldata.bin           # this one comes through fine
    # NOTE: model.mil and metadata.json are deliberately absent — that is
    # exactly the bug. The watchOS stub refuses to emit them, which is what
    # makes `MLModel(contentsOfURL:)` fail to load anything compiled on
    # the watch directly.
```

The test harness in `tests/golden.rs` compares our compiler's output against
the macOS reference (positive path) and checks that the watchOS-broken pattern
is NOT reproduced (negative path).

## Adding a new fixture

```sh
# 1. Drop a CoreML protobuf as input.mlmodel.
mkdir tests/fixtures/<name>
cp /path/to/your/model.mlmodel tests/fixtures/<name>/input.mlmodel

# 2. Run Apple's coremlc on macOS to capture the reference.
cd tests/fixtures/<name>
xcrun coremlc compile input.mlmodel ./tmp
mv tmp/input.mlmodelc expected-macos.mlmodelc
rmdir tmp

# 3. (Optional) capture watchOS stub output. If you have arm64_32 hardware,
#    push the same input.mlmodel to a watchOS app and call MLModel.compileModel
#    on it, then `devicectl device copy from` the resulting .mlmodelc.
#    Otherwise, the heuristic synth used in tests/fixtures/neg-0d-scalar/
#    captures the failure mode (truncated coremldata.bin, absent model.mil
#    and metadata.json).
```

Then re-run `cargo test` — the harness auto-discovers the new directory.

## Why both fixtures matter

- **`expected-macos.mlmodelc/`** is the positive contract: our output should be
  byte-identical for MIL and the model-description container (modulo build info).
  JSON is compared structurally, excluding generated class names and optimization
  statistics; feature schemas and function selection must match.
- **`observed-watchos-broken.mlmodelc/`** documents the exact failure mode the
  crate exists to avoid. If our compiler ever degrades to producing that
  pattern, the negative-side test fails.

## Flexible-shape regressions

- `flexible-range`: rank-one ReLU with an unknown MIL extent, default 4 and
  range [0, 4].
- `flexible-gather`: dynamic rank-two indices with int32 feature type and
  negative-index normalization before gather. Both fixtures originate from the
  RustNN CoreML shape diagnostics; references use coremlc 3520.5.1 (MIL 3520.4.1).
- `flexible-multifunction`: two ReLU functions with different default input
  shapes, the second selected by default, and a description exceeding 255 bytes.
  Reference captured with Xcode 27.0 (27A266a), coremlc 3600.25.1.
- `flexible-weighted-legacy`: FP32 input `[?, 4]` plus an external four-element
  weight vector, retaining legacy input/output descriptions (no function list).
  Default shape `[1, 4]`, ranges `[[1, 4], [4, 4]]`. This small fixture covers
  the description/weight combination used by transformer models without model
  downloads. See [its provenance](flexible-weighted-legacy/PROVENANCE.md).

`generate_flexible.py` derives the multi-function fixture and an unsupported
enumerated-shape fixture from `flexible-range`. It uses `Model_pb2` generated
from [Apple's Core ML format schemas](https://github.com/apple/coremltools/tree/main/mlmodel/format).
`tests/unsupported/flexible-enumerated` retains native output for a future
implementation; the current test requires an explicit unsupported-format error.

`tests/runtime_shapes.swift` additionally checks 24 exact predictions, including
1 -> 2 -> 4 -> 1 input changes on each loaded model and default/named function
selection. The weighted fixture checks all 32 output values against ordinary
integer addition. Run it as documented in the repository README.

The golden harness supplies optional source weights, checks their output bytes
against both the source and native golden, and compares every emitted file with
the buffered API. `tests/weighted_legacy.rs` also checks CLI explicit weights,
package auto-detection, and rejection of missing explicit weight paths.

Regenerate the weighted fixture's inputs without external Python dependencies:

```sh
python3 tests/fixtures/generate_weighted.py
python3 tests/fixtures/generate_weighted.py --check
```

The generator does not create golden output. Capture that separately using
Apple's compiler; keep the source `weights/` beside `input.mlmodel` during
compilation. CI checks that committed inputs match the generator.
