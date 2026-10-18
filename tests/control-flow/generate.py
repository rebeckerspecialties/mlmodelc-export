"""Reproducible nested MIL fixture, using only Python's standard library.

Two loop iterations: add packed weights, then multiply by those weights.
The Int8 BLOBFILE constant is inside the loop; a cond is inside its body.
This generates source protobuf only. Runtime expectations are independent.
"""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from generate_weighted import binding, message, scalar, text
from generate_precision import attribute, immediate, named_type, blob_value


def op(kind, inputs, outputs, blocks=(), attributes=()):
    data = text(1, kind)
    for key, values in inputs:
        data += message(2, text(1, key) + message(2, b"".join(message(1, text(1, v)) for v in values)))
    data += b"".join(message(3, named_type(*value)) for value in outputs)
    data += b"".join(message(4, block) for block in blocks)
    data += b"".join(attribute(key, value) for key, value in attributes)
    return data


def block(inputs, outputs, operations):
    return (b"".join(message(1, named_type(*v)) for v in inputs)
            + b"".join(text(2, v) for v in outputs)
            + b"".join(message(3, v) for v in operations))


def const(name, value):
    return op("const", [], [(name, 23, [])], attributes=[("val", immediate(23, [], message(2, message(1, bytes([value])))))])


def source(depth=0):
    yes = block([], ["added"], [op("add", [("x", ["carried"]), ("y", ["weight"])], [("added", 11, [4])])])
    no = block([], ["multiplied"], [op("mul", [("x", ["carried"]), ("y", ["weight"])], [("multiplied", 11, [4])])])
    choose = op("cond", [("pred", ["first"])], [("next", 11, [4])], [yes, no])
    # Deliberately excessive source for the decoder's nesting-limit regression.
    for _ in range(depth):
        choose = op("cond", [("pred", ["first"])], [("next", 11, [4])], [block([], ["next"], [choose]), no])
    cond = block([("ci", 23, []), ("unused", 11, [4])], ["continue"], [op("less", [("x", ["ci"]), ("y", ["two"])], [("continue", 1, [])])])
    body = block([("i", 23, []), ("carried", 11, [4])], ["incremented", "next"], [
        op("const", [], [("packed", 21, [4])], attributes=[("val", blob_value(21, [4]))]),
        op("cast", [("x", ["packed"]), ("dtype", ["dtype_fp32"])], [("weight", 11, [4])]),
        op("less", [("x", ["i"]), ("y", ["one"])], [("first", 1, [])]),
        choose,
        op("add", [("x", ["i"]), ("y", ["one"])], [("incremented", 23, [])]),
    ])
    operations = [const("zero", 0), const("one", 1), const("two", 2),
                  op("const", [], [("dtype_fp32", 2, [])], attributes=[("val", immediate(2, [], message(4, text(1, "fp32"))))]),
                  op("while_loop", [("loop_vars", ["zero", "input"])], [("end", 23, []), ("output", 11, [4])], [cond, body])]
    function = message(1, named_type("input", 11, [4])) + text(2, "CoreML8")
    function += message(3, text(1, "CoreML8") + message(2, block([], ["output"], operations)))
    program = scalar(1, 1) + message(2, text(1, "main") + message(2, function))
    array = message(1, b"\x04") + scalar(2, 65568)
    desc = b"".join(message(field, text(1, name) + message(3, message(5, array))) for field, name in [(1, "input"), (10, "output")])
    return scalar(1, 9) + message(2, desc) + message(502, program)


def weights():
    data = bytearray(192)
    struct.pack_into("<II", data, 0, 1, 2)
    struct.pack_into("<IIQQ", data, 64, 0xDEADBEEF, 4, 4, 128)
    struct.pack_into("<4b", data, 128, -2, -1, 1, 2)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    for name, data in [("input.mlmodel", source()), ("too-deep.mlmodel", source(64)), ("weights/weights.bin", weights())]:
        path = root / name
        if args.check:
            assert path.read_bytes() == data, path
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
