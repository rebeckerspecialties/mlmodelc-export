"""Generate tiny FP16 and quantized source fixtures using only the standard library.

Expected runtime values live separately in runtime_shapes.swift. This script
does not produce golden bundles; capture those with Apple's coremlc.
"""

import argparse
from pathlib import Path
import struct

from generate_weighted import binding, message, scalar, text


def value_type(dtype, shape):
    tensor = scalar(1, dtype) + scalar(2, len(shape))
    for extent in shape:
        tensor += message(3, message(1, scalar(1, extent)))
    return message(1, tensor)


def named_type(name, dtype, shape):
    return text(1, name) + message(2, value_type(dtype, shape))


def immediate(dtype, shape, tensor):
    return message(2, value_type(dtype, shape)) + message(3, message(1, tensor))


def attribute(name, value):
    return message(5, text(1, name) + message(2, value))


def blob_value(dtype, shape):
    blob = text(1, "@model_path/weights/weights.bin") + scalar(2, 64)
    return message(2, value_type(dtype, shape)) + message(5, blob)


def model(count, operations, constant_name):
    add = text(1, "add") + binding("x", "input") + binding("y", constant_name)
    add += message(3, named_type("output", 11, [count]))
    block = text(2, "output") + b"".join(message(3, op) for op in [*operations, add])
    function = message(1, named_type("input", 11, [count])) + text(2, "CoreML7")
    function += message(3, text(1, "CoreML7") + message(2, block))
    program = scalar(1, 1) + message(2, text(1, "main") + message(2, function))
    array = message(1, bytes([count])) + scalar(2, 65568)
    description = b"".join(
        message(field, text(1, name) + message(3, message(5, array)))
        for field, name in [(1, "input"), (10, "output")]
    )
    return scalar(1, 9) + message(2, description) + message(502, program)


def fp16_model():
    bits = [0x0001, 0x00A8, 0x03FF, 0x0400, 0x80A8]
    tensor = message(7, message(1, struct.pack("<5H", *bits)))
    constant = text(1, "const") + message(3, named_type("half", 10, [5]))
    constant += attribute("val", immediate(10, [5], tensor))
    dtype = immediate(2, [], message(4, text(1, "fp32")))
    dtype_binding = message(2, text(1, "dtype") + message(2, message(1, message(2, dtype))))
    cast = text(1, "cast") + dtype_binding + binding("x", "half")
    cast += message(3, named_type("constant", 11, [5]))
    return model(5, [constant, cast], "constant")


def quantized_model():
    operation = text(1, "constexpr_affine_dequantize")
    operation += message(3, named_type("constant", 11, [4]))
    # Keep non-sorted source attributes to exercise deterministic emission.
    operation += attribute("zero_point", immediate(21, [], message(7, message(1, b"\xff"))))
    operation += attribute("scale", immediate(11, [], message(1, message(1, struct.pack("<f", 0.5)))))
    operation += attribute("quantized_data", blob_value(21, [4]))
    operation += attribute("axis", immediate(23, [], message(2, message(1, b"\x00"))))
    return model(4, [operation], "constant")


def quantized_weights():
    result = bytearray(192)
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IIQQ", result, 64, 0xDEADBEEF, 4, 4, 128)
    struct.pack_into("<4b", result, 128, -128, -1, 0, 127)
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed sources without writing")
    args = parser.parse_args()
    root = Path(__file__).parent
    for relative, data in [
        ("fp16-subnormal-cast/input.mlmodel", fp16_model()),
        ("quantized-constexpr/input.mlmodel", quantized_model()),
        ("quantized-constexpr/weights/weights.bin", quantized_weights()),
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
