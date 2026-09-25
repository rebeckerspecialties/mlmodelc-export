"""Generate the small weighted-legacy fixture with only Python's standard library.

The protobuf follows Apple's Model.proto and MIL.proto schemas. The v2 weight
blob has one FP32 entry; its reference points to the metadata at offset 64.
This generates source inputs, never a golden .mlmodelc or expected predictions.
Capture golden output separately with Apple's coremlc, as documented in README.
"""

import argparse
from pathlib import Path
import struct


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def scalar(field, value):
    return varint(field << 3) + varint(value)


def message(field, value):
    return varint(field << 3 | 2) + varint(len(value)) + value


def text(field, value):
    return message(field, value.encode())


def tensor_type(shape):
    tensor = scalar(1, 11) + scalar(2, len(shape))  # FLOAT32
    for extent in shape:
        dimension = message(2, b"") if extent is None else message(1, scalar(1, extent))
        tensor += message(3, dimension)
    return message(1, tensor)


def named_type(name, shape):
    return text(1, name) + message(2, tensor_type(shape))


def feature(name):
    array = message(1, b"\x01\x04")  # default [1, 4]
    array += scalar(2, 65568)  # ArrayFeatureType.FLOAT32
    ranges = b"".join(message(1, scalar(1, lo) + scalar(2, hi)) for lo, hi in [(1, 4), (4, 4)])
    array += message(31, ranges)
    return text(1, name) + message(3, message(5, array))


def binding(argument, name):
    return message(2, text(1, argument) + message(2, message(1, text(1, name))))


def source_model():
    blob = text(1, "@model_path/weights/weights.bin") + scalar(2, 64)
    value = message(2, tensor_type([4])) + message(5, blob)
    constant = text(1, "const") + message(3, named_type("weight", [4]))
    constant += message(5, text(1, "val") + message(2, value))
    add = text(1, "add") + binding("x", "input") + binding("y", "weight")
    add += message(3, named_type("output", [None, 4]))
    block = text(2, "output") + message(3, constant) + message(3, add)
    function = message(1, named_type("input", [None, 4])) + text(2, "CoreML7")
    function += message(3, text(1, "CoreML7") + message(2, block))
    program = scalar(1, 1) + message(2, text(1, "main") + message(2, function))
    # Legacy input/output fields: deliberately no functions/defaultFunctionName.
    description = message(1, feature("input")) + message(10, feature("output"))
    return scalar(1, 9) + message(2, description) + message(502, program)


def weight_blob():
    result = bytearray(192)
    struct.pack_into("<II", result, 0, 1, 2)  # one entry, version 2
    struct.pack_into("<IIQQ", result, 64, 0xDEADBEEF, 2, 16, 128)
    struct.pack_into("<4f", result, 128, 2, -3, 5, -7)
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed sources without writing")
    args = parser.parse_args()
    root = Path(__file__).parent / "flexible-weighted-legacy"
    for relative, data in [("input.mlmodel", source_model()), ("weights/weights.bin", weight_blob())]:
        path = root / relative
        if args.check:
            if path.read_bytes() != data:
                raise SystemExit(f"fixture differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


if __name__ == "__main__":
    main()
