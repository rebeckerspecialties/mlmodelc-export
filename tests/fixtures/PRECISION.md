# Precision fixture provenance

`generate_precision.py` constructs both source protobufs from documented Core
ML wire fields using Python's standard library. They use specification version
9, CoreML7 operations, legacy input/output descriptions, and FP32 boundaries.
No model download, training data or captured model output is used.

## Independent expected values

- **fp16-subnormal-cast**: 340-byte model, no external weights. Inline FP16 bit
  patterns `[0x0001, 0x00a8, 0x03ff, 0x0400, 0x80a8]` cover the smallest
  subnormal, positive/negative normalization epsilon, largest subnormal and
  smallest normal. It casts to FP32 before adding a runtime FP32 input. Expected
  zero-input values are `[1, 168, 1023, 1024, -168] / 2^24`. A second prediction
  adds `2^-14` to every element. All ten expected values are exactly
  representable in FP32; this does not test FP16 arithmetic or accumulation.
- **quantized-constexpr**: 404-byte model plus a 192-byte weight file. The int8
  payload is `[-128, -1, 0, 127]`; `constexpr_affine_dequantize` has typed
  attributes `axis=int32(0)`, `scale=fp32(0.5)` and `zero_point=int8(-1)`.
  Expected zero-input values are `(weight + 1) / 2 = [-63.5, 0, 0.5, 64]`.
  A second prediction adds 1 to each element; all eight values remain exact.

`runtime_shapes.swift` defines these arithmetic expectations separately from
the generator, loads freshly Rust-compiled bundles, selects CPU-only execution,
and checks FP32 dtype, exact shape, finiteness and every value. No tolerance or
subnormal-flush exception is applied. Source-payload tests independently check
the FP16 bits and signed/typed quantization parameters.

## Native golden capture

Captured on 2026-09-25 with Xcode 27.0 (27A266a), coremlc 3600.25.1,
MIL component 3600.16.1. Golden bundles are unmodified Apple compiler output.
Each fixture was compiled separately with its source weights beside the model:

```sh
python3 -B tests/fixtures/generate_precision.py --check
capture_dir=$(mktemp -d)
xcrun coremlc compile tests/fixtures/fp16-subnormal-cast/input.mlmodel "$capture_dir/fp16"
xcrun coremlc compile tests/fixtures/quantized-constexpr/input.mlmodel "$capture_dir/quantized"
```

The respective `input.mlmodelc` directories were copied to each fixture's
`expected-macos.mlmodelc`. This capture invokes compilation only, not prediction.
Numerical CI on macOS 15 is separate from physical iOS/watchOS validation.

SHA-256:

| Fixture/file | Digest |
| --- | --- |
| fp16 source | `c855b6bb5f85482f88eb8f0f2d966e989a5dbf6cb7ad293f23e35715c58dbb07` |
| fp16 golden model.mil | `5bd7193263c0afd46f0121522946193adc48a027ca784b842f83ad2a173f5c75` |
| fp16 golden coremldata.bin | `f1df53cf37b1d6e1e74b2ecf71bedcb25f581ed56b051de9a3a8c8dc374dceff` |
| fp16 golden metadata.json | `7c539b566dadd47bd7354623b78e4f5e9b4a6c813a4ada1a3fbeaf862c227558` |
| quantized source | `1a2c4ff3a6a049dc083e0cf2cba4b4e5e6beab81e14e7f0025a77a1c64feb45d` |
| quantized weights | `08201fd09931b7942ff512dce33403d3565d13524ca51c3e0bdbf73dbd55d87e` |
| quantized golden model.mil | `7c883a45ccf6928213051322ec4cd09b94f03054ef860b290830e06ba2a38cbe` |
| quantized golden coremldata.bin | `f3600551033ee25001fa58f8f84de6bb2ea28eccd0fe536043d46f1923a066d6` |
| quantized golden metadata.json | `05e38dbcff263c39fb1aaa312fa497bd000f3e1a39f4348f294f72ac286f7fbc` |
