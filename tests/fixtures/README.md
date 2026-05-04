# Reference fixtures

Each subdirectory under `tests/fixtures/` is one test case. Layout:

```
tests/fixtures/<name>/
  input.mlmodel                       # raw CoreML protobuf (input)
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
  byte-identical to what Apple's official tool produces (modulo the
  build-info string identifying the producer).
- **`observed-watchos-broken.mlmodelc/`** documents the exact failure mode the
  crate exists to avoid. If our compiler ever degrades to producing that
  pattern, the negative-side test fails.
