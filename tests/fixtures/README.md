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

`tests/runtime_shapes.swift` additionally checks 28 exact predictions, including
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

## Precision regressions

`fp16-subnormal-cast` retains positive and negative FP16 subnormal constants,
casts to FP32 and adds a runtime input. `quantized-constexpr` dequantizes int8
blob weights with typed `axis`, `scale` and `zero_point` attributes before
adding a runtime input. Both source fixtures are smaller than 1 KB, use no
downloaded assets, and have independently specified exact runtime results.

```sh
python3 -B tests/fixtures/generate_precision.py
python3 -B tests/fixtures/generate_precision.py --check
```

See [precision fixture provenance and value formulas](PRECISION.md) for native
compiler capture details and hashes. The golden comparison detects the old
FP16 mantissa alignment and dropped constexpr attributes without executing
CoreML. Runtime checks use FP32 outputs with CPU-only execution and no tolerance
that could hide flushed subnormals or an incorrect dequantization parameter.

## Model state and blockwise constants

All five fixtures use the CoreML8 opset and specification version 9. Sources
are generated with Python's standard library; no model download, training data
or captured model output is used.

- `stateful-fp16`: FP16 input `x` and FP16 state `state_0`, both `[1, 4]`.
  `next_state_0 = read_state(state_0) + x`, `y = next_state_0 + x`, then
  `write_state`. The function declares `x` before the state; `coremlc` writes
  function inputs in name order.
- `stateful-fp32-pair`: FP32 input and two FP16 states. Each state is cast to
  FP32 after `read_state` and back to FP16 before `write_state`.
  `next_state_0 = state_0 + x`, `next_state_1 = state_1 + next_state_0`,
  `y = next_state_0 + next_state_1`.
- `stateful-declaration-order`: the function declares `s1, q, s0, p`; the
  description lists inputs `q, p` and states `s1, s0`. The MIL signature is
  `p, q, s0, s1`; `inputSchema` and `stateSchema` keep the description order.
- `blockwise-uint8-offset`: `constexpr_blockwise_shift_scale` with uint8 data
  `[2, 64]` and FP16 scales `[2, 2]` (one per row and 32-column block) from
  `weights.bin`, and an inline uint8 offset of 128. The FP16 result is the
  transposed right operand of a `matmul` with the FP16 cast of the input.
- `blockwise-int8-scales`: int8 data `[2, 4]` from `weights.bin`, decoded twice
  without an offset: with the FP32 `[1, 1]` scale 1 and with FP32 per-row
  scales `[[0.5], [0.25]]`. The output is the input plus both results.

In both blockwise sources `data`, `scale` and `offset` are `const` operations
referenced by name. `coremlc` keeps those operations and prints the same
values again as literal arguments of the `constexpr_*` operation.

```sh
python3 -B tests/fixtures/generate_state.py
python3 -B tests/fixtures/generate_state.py --check
python3 -B tests/fixtures/generate_blockwise.py
python3 -B tests/fixtures/generate_blockwise.py --check
```

The generators do not create golden output. The goldens were captured on
2026-11-16 with Xcode 27.2 (27B48), coremlc 3600.25.1, MIL component 3605.1.1,
by the procedure above, each blockwise model with its `weights/` beside it.
They are unmodified Apple compiler output. Capture invokes compilation only.

`tests/state_fixtures.rs` and `tests/blockwise_fixtures.rs` check the decoded
sources and weight payloads. `tests/metadata_statistics.rs` compares
`storagePrecision`, `computePrecision` and the operation histogram of every
fixture in this directory with its golden; `flexible-range` and
`quantized-constexpr` are listed there as histogram exceptions.

`tests/runtime_state.swift` loads the bundles written by the Rust exporter and
checks 16 exact CPU-only predictions: three accumulating steps on one
`MLState`, a second state starting from zeros while the first continues, the
stored value of each named state, and the blockwise weights against values
worked out in the script (single columns in each 32-column block, then all
ones). Run it as documented in the repository README.
