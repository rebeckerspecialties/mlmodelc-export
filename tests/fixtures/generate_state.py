"""Generate tiny model-state source fixtures using only the standard library.

All models use the CoreML8 opset with `read_state` / `write_state` and legacy
model-level input/output/state descriptions. Expected runtime values live
separately in runtime_state.swift. This script does not produce golden
bundles; capture those with Apple's coremlc.
"""

import argparse
from pathlib import Path

from generate_precision import immediate, named_type, value_type
from generate_weighted import binding, message, scalar, text

FP16, FP32 = 10, 11
FEATURE_FP16, FEATURE_FP32 = 65552, 65568
SHAPE = [1, 4]


def state_type(shape):
    # ValueType.stateType = 5 { StateType.wrappedType = 1 }; storage is always FP16.
    return message(5, message(1, value_type(FP16, shape)))


def state_input(name):
    return message(1, text(1, name) + message(2, state_type(SHAPE)))


def tensor_input(name, dtype):
    return message(1, named_type(name, dtype, SHAPE))


def operation(kind, arguments, output=None, dtype=FP16):
    result = text(1, kind) + b"".join(arguments)
    if output is not None:
        result += message(3, named_type(output, dtype, SHAPE))
    return result


def cast(source, output, dtype):
    name = immediate(2, [], message(4, text(1, "fp16" if dtype == FP16 else "fp32")))
    dtype_binding = message(2, text(1, "dtype") + message(2, message(1, message(2, name))))
    return operation("cast", [dtype_binding, binding("x", source)], output, dtype)


def array_feature(dtype):
    return message(1, bytes(SHAPE)) + scalar(2, dtype)


def feature(field, name, dtype):
    return message(field, text(1, name) + message(3, message(5, array_feature(dtype))))


def state_feature(name):
    # FeatureType.stateType = 8 { StateFeatureType.arrayType = 1 }.
    state = message(8, message(1, array_feature(FEATURE_FP16)))
    return message(13, text(1, name) + message(3, state))


def model(inputs, operations, description):
    block = text(2, "y") + b"".join(message(3, op) for op in operations)
    function = b"".join(inputs) + text(2, "CoreML8")
    function += message(3, text(1, "CoreML8") + message(2, block))
    program = scalar(1, 1) + message(2, text(1, "main") + message(2, function))
    return scalar(1, 9) + message(2, b"".join(description)) + message(502, program)


def fp16_model():
    """next_state_0 = state_0 + x, y = next_state_0 + x, all FP16."""
    operations = [
        operation("read_state", [binding("input", "state_0")], "state_0_read"),
        operation("add", [binding("x", "state_0_read"), binding("y", "x")], "next_state_0"),
        operation("add", [binding("x", "next_state_0"), binding("y", "x")], "y"),
        operation("write_state", [binding("input", "state_0"), binding("data", "next_state_0")]),
    ]
    return model(
        [tensor_input("x", FP16), state_input("state_0")],
        operations,
        [feature(1, "x", FEATURE_FP16), feature(10, "y", FEATURE_FP16), state_feature("state_0")],
    )


def fp32_pair_model():
    """Two FP32 values carried through FP16 states.

    next_state_0 = state_0 + x, next_state_1 = state_1 + next_state_0,
    y = next_state_0 + next_state_1. Each state is widened after the read and
    narrowed before the write.
    """
    operations = []
    for name in ["state_0", "state_1"]:
        operations.append(operation("read_state", [binding("input", name)], name + "_state_read"))
        operations.append(cast(name + "_state_read", name + "_read", FP32))
    operations += [
        operation("add", [binding("x", "state_0_read"), binding("y", "x")], "next_state_0", FP32),
        operation("add", [binding("x", "state_1_read"), binding("y", "next_state_0")], "next_state_1", FP32),
        operation("add", [binding("x", "next_state_0"), binding("y", "next_state_1")], "y", FP32),
    ]
    for name in ["state_0", "state_1"]:
        stored = "next_" + name + "_state_value"
        operations.append(cast("next_" + name, stored, FP16))
        operations.append(operation("write_state", [binding("input", name), binding("data", stored)]))
    return model(
        [tensor_input("x", FP32), state_input("state_0"), state_input("state_1")],
        operations,
        [
            feature(1, "x", FEATURE_FP32),
            feature(10, "y", FEATURE_FP32),
            state_feature("state_0"),
            state_feature("state_1"),
        ],
    )


def ordering_model():
    """Declaration orders that differ between the function and the description.

    The function declares `s1, q, s0, p`; the description lists inputs `q, p`
    and states `s1, s0`. n0 = s0 + p, n1 = s1 + q, y = n0 + n1.
    """
    operations = [
        operation("read_state", [binding("input", "s0")], "r0"),
        operation("read_state", [binding("input", "s1")], "r1"),
        operation("add", [binding("x", "r0"), binding("y", "p")], "n0"),
        operation("add", [binding("x", "r1"), binding("y", "q")], "n1"),
        operation("add", [binding("x", "n0"), binding("y", "n1")], "y"),
        operation("write_state", [binding("input", "s0"), binding("data", "n0")]),
        operation("write_state", [binding("input", "s1"), binding("data", "n1")]),
    ]
    return model(
        [state_input("s1"), tensor_input("q", FP16), state_input("s0"), tensor_input("p", FP16)],
        operations,
        [
            feature(1, "q", FEATURE_FP16),
            feature(1, "p", FEATURE_FP16),
            feature(10, "y", FEATURE_FP16),
            state_feature("s1"),
            state_feature("s0"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed sources without writing")
    args = parser.parse_args()
    root = Path(__file__).parent
    for relative, data in [
        ("stateful-fp16/input.mlmodel", fp16_model()),
        ("stateful-fp32-pair/input.mlmodel", fp32_pair_model()),
        ("stateful-declaration-order/input.mlmodel", ordering_model()),
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
