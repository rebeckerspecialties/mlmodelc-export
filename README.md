# mlmodelc-export

Compile a CoreML MLProgram protobuf into a runnable `.mlmodelc` bundle in pure
Rust, without invoking Apple's `coremlc`.

[![Crates.io](https://img.shields.io/crates/v/mlmodelc-export.svg)](https://crates.io/crates/mlmodelc-export)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Why this exists

Apple's `coremlc` is the tool that turns a `.mlmodel` (or `.mlpackage`)
protobuf into the directory layout that `MLModel(contentsOfURL:)` actually
loads at runtime — it lives inside Xcode's toolchain and is callable only
from a developer host on macOS. There is no equivalent on:

- **watchOS** — sandboxed, no compiler, only the loader.
- **iOS App Extensions** — same restriction.
- **iOS App Clips** — same restriction.
- **tvOS sandboxed contexts** — same restriction.
- **Linux/Windows CI** — no Apple toolchain at all.

This crate fills the gap. Feed it the protobuf bytes you'd otherwise hand to
`coremlc`, and it emits the same four files Apple's compiler produces:

```
example.mlmodelc/
├─ model.mil                 UTF-8 MIL text
├─ coremldata.bin            FunctionDescription / defaultFunctionName trailer
├─ metadata.json             I/O schema, op histogram, availability matrix
├─ analytics/
│  └─ coremldata.bin         minimal stub satisfying CoreML's analytics check
└─ weights/
   └─ weights.bin            optional, when the input has external weights
```

The output is byte-identical to `coremlc`'s for the canonical files
(`model.mil`, `coremldata.bin`) and key-equivalent for `metadata.json` —
verified per-commit by [`tests/golden.rs`](tests/golden.rs) against fixtures
captured from `xcrun coremlc compile`.

## Quick start

### Library

```rust
use mlmodelc_export::compile_to_bundle;

let mlmodel_bytes = std::fs::read("model.mlmodel")?;
let bundle = compile_to_bundle(&mlmodel_bytes, None)?;
bundle.write_to_dir("model.mlmodelc")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Streaming variant (recommended for large models)

```rust
use mlmodelc_export::compile_to_dir;

let mlmodel_bytes = std::fs::read("model.mlmodel")?;
let stats = compile_to_dir(&mlmodel_bytes, None, "model.mlmodelc")?;
println!("emitted {} bytes of MIL text across {} ops", stats.output_bytes, stats.operation_count);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`compile_to_dir` writes the MIL text directly to disk through a 256 KB flush
buffer — keeping resident memory bounded by the chunk size rather than the
full MIL text. Recommended on memory-constrained devices like 32-bit Apple
Watches where the jetsam limit is ~150 MB.

### CLI

```sh
cargo install mlmodelc-export
mlmodelc-export model.mlmodel ./model.mlmodelc
```

Useful for CI / build pipelines where you want a runnable bundle without
depending on a macOS host with Xcode installed.

## Provenance

This crate originated as a Swift package (`MILTextCompiler`) inside the
`metal-info-app` watchOS validation harness for [rustnn]. After validating
the approach end-to-end on Apple Watch SE 2 hardware (arm64_32, watchOS 11)
running real WebNN workloads — LeNet MNIST, char-level transformer, MNIST
autoencoder, 44/44 WPT operator coverage — we ported it to Rust to align
with rustnn's Rust-first architecture.

The full motivation is documented in
[rustnn/rustnn#110](https://github.com/rustnn/rustnn/issues/110).

[rustnn]: https://github.com/rustnn/rustnn

## Testing methodology

The crate uses **golden-file directory comparison** against two committed
references per fixture:

| Reference | Source | Role |
|---|---|---|
| `expected-macos.mlmodelc/` | output of `xcrun coremlc compile` on macOS | **positive contract** — our output must match |
| `observed-watchos-broken.mlmodelc/` | output of Apple's arm64_32 watchOS stub | **negative contract** — our output must not match |

The watchOS stub is the same compiler binary but running with `arm64_32`
constraints — it returns silently after writing a truncated `coremldata.bin`
(missing the 16-byte loader trailer) and refusing to emit `model.mil` or
`metadata.json`. This is the failure mode the crate exists to avoid; pinning
it as a checked-in negative reference protects against future regressions.

See [`tests/fixtures/README.md`](tests/fixtures/README.md) for instructions
on adding new fixtures.

## Status

`v0.1` — works for the small-graph cases in our test fixtures. Round-trip
verified against `xcrun coremlc compile` (Xcode 26.4, `coremlc` 3520.4.1):

- `model.mil` byte-identical except for the `buildInfo` producer string
- `coremldata.bin` byte-identical (FunctionDescription + defaultFunctionName)
- `metadata.json` semantically equivalent (key-order tolerant)

For large dense Float32 constants (>10⁵ elements) the streaming path uses an
allocation-free hex-float byte formatter (see
[`src/hex_float.rs`](src/hex_float.rs)), giving ~3-5× faster emission than
the naive `String`-allocating implementation. This was the critical
optimisation for getting jetsam-safe compile times on arm64_32 watch SoCs.

## License

Dual licensed at your option:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
