"""Generate tiny `constexpr_blockwise_shift_scale` fixtures (CoreML8 opset).

Only the standard library is used. `data`, `scale` and `offset` are `const`
operations referenced by name, as iOS 18 producers serialize them. Expected
runtime values live separately in runtime_state.swift. This script does not
produce golden bundles; capture those with Apple's coremlc.
"""

import argparse
from pathlib import Path
import struct

from generate_precision import attribute, immediate, named_type, value_type
from generate_weighted import binding, message, scalar, text

BOOL, STRING, FP16, FP32, INT8, UINT8 = 1, 2, 10, 11, 21, 31
BLOB_FP16, BLOB_UINT8, BLOB_INT8 = 1, 3, 4
FEATURE_FP32 = 65568


def constant(name, dtype, shape, value):
    result = text(1, "const") + message(3, named_type(name, dtype, shape))
    return result + attribute("val", value)


def blob_value(dtype, shape, offset):
    blob = text(1, "@model_path/weights/weights.bin") + scalar(2, offset)
    return message(2, value_type(dtype, shape)) + message(5, blob)


def operation(kind, arguments, output, dtype, shape):
    result = text(1, kind) + b"".join(binding(key, name) for key, name in arguments)
    return result + message(3, named_type(output, dtype, shape))


def cast(source, output, dtype, shape):
    name = immediate(STRING, [], message(4, text(1, "fp16" if dtype == FP16 else "fp32")))
    dtype_binding = message(2, text(1, "dtype") + message(2, message(1, message(2, name))))
    result = text(1, "cast") + dtype_binding + binding("x", source)
    return result + message(3, named_type(output, dtype, shape))


def model(shape, operations, output_shape):
    block = text(2, "output") + b"".join(message(3, op) for op in operations)
    function = message(1, named_type("input", FP32, shape)) + text(2, "CoreML8")
    function += message(3, text(1, "CoreML8") + message(2, block))
    program = scalar(1, 1) + message(2, text(1, "main") + message(2, function))
    description = b"".join(
        message(field, text(1, name) + message(3, message(5, message(1, bytes(extents)) + scalar(2, FEATURE_FP32))))
        for field, name, extents in [(1, "input", shape), (10, "output", output_shape)]
    )
    return scalar(1, 9) + message(2, description) + message(502, program)


def weight_file(entries):
    """MIL blob storage v2: a 64-byte header, then 64-byte-aligned metadata and data."""
    result = bytearray(64)
    struct.pack_into("<II", result, 0, len(entries), 2)
    offsets = []
    for dtype, data in entries:
        offsets.append(len(result))
        metadata = bytearray(64)
        struct.pack_into("<IIQQ", metadata, 0, 0xDEADBEEF, dtype, len(data), len(result) + 64)
        result += metadata + data
        result += bytes(-len(result) % 64)
    return bytes(result), offsets


def uint8_rows():
    # Stored bytes straddle the uint8 offset 128 by a few steps.
    return [
        [128 + k % 7 - 3 for k in range(64)],
        [128 + (3 * k) % 11 - 5 for k in range(64)],
    ]


def uint8_weights():
    data = bytes(value for row in uint8_rows() for value in row)
    # One FP16 scale per row and per 32-column block: [[0.5, 0.25], [2, 0.125]].
    scale = struct.pack("<4e", 0.5, 0.25, 2.0, 0.125)
    return weight_file([(BLOB_UINT8, data), (BLOB_FP16, scale)])


def uint8_model():
    """output = fp32(matmul(fp16(input), W^T)), W = scale * (data - offset).

    uint8 data [2, 64] and FP16 scale [2, 2] come from weights.bin; the uint8
    offset [2, 2] is an inline uniform constant.
    """
    _, (data_offset, scale_offset) = uint8_weights()
    operations = [
        constant("data", UINT8, [2, 64], blob_value(UINT8, [2, 64], data_offset)),
        constant("scale", FP16, [2, 2], blob_value(FP16, [2, 2], scale_offset)),
        constant("offset", UINT8, [2, 2], immediate(UINT8, [2, 2], message(7, message(1, bytes([128] * 4))))),
        operation(
            "constexpr_blockwise_shift_scale",
            [("data", "data"), ("scale", "scale"), ("offset", "offset")],
            "weight", FP16, [2, 64],
        ),
        cast("input", "input_fp16", FP16, [1, 64]),
        constant("no", BOOL, [], immediate(BOOL, [], message(3, message(1, b"\x00")))),
        constant("yes", BOOL, [], immediate(BOOL, [], message(3, message(1, b"\x01")))),
        operation(
            "matmul",
            [("x", "input_fp16"), ("y", "weight"), ("transpose_x", "no"), ("transpose_y", "yes")],
            "product", FP16, [1, 2],
        ),
        cast("product", "output", FP32, [1, 2]),
    ]
    return model([1, 64], operations, [1, 2])


def int8_weights():
    return weight_file([(BLOB_INT8, struct.pack("<8b", -128, -1, 0, 127, 4, -8, 16, -32))])


def int8_model():
    """output = input + data * 1 + data * rows, without an offset.

    int8 data [2, 4] comes from weights.bin. `unit` is the FP32 [1, 1] scale 1
    of an exact int8 decode; `rows` is a per-row FP32 scale [[0.5], [0.25]].
    """
    _, (data_offset,) = int8_weights()
    unit = message(1, message(1, struct.pack("<f", 1.0)))
    rows = message(1, message(1, struct.pack("<2f", 0.5, 0.25)))
    operations = [
        constant("data", INT8, [2, 4], blob_value(INT8, [2, 4], data_offset)),
        constant("unit", FP32, [1, 1], immediate(FP32, [1, 1], unit)),
        operation("constexpr_blockwise_shift_scale", [("data", "data"), ("scale", "unit")], "exact", FP32, [2, 4]),
        constant("rows", FP32, [2, 1], immediate(FP32, [2, 1], rows)),
        operation("constexpr_blockwise_shift_scale", [("data", "data"), ("scale", "rows")], "scaled", FP32, [2, 4]),
        operation("add", [("x", "exact"), ("y", "scaled")], "sum", FP32, [2, 4]),
        operation("add", [("x", "input"), ("y", "sum")], "output", FP32, [2, 4]),
    ]
    return model([2, 4], operations, [2, 4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed sources without writing")
    args = parser.parse_args()
    root = Path(__file__).parent
    for relative, data in [
        ("blockwise-uint8-offset/input.mlmodel", uint8_model()),
        ("blockwise-uint8-offset/weights/weights.bin", uint8_weights()[0]),
        ("blockwise-int8-scales/input.mlmodel", int8_model()),
        ("blockwise-int8-scales/weights/weights.bin", int8_weights()[0]),
    ]:
        path = root / relative
        if args.check:
            if path.read_bytes() != data:
                raise SystemExit(f"fixture differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


if __name__ == "__main__":
    main()
