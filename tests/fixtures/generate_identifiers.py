"""Generate tiny identifier/metadata controls, with no model-tool dependency.

Names are already normalized by the producer. The exporter must preserve them;
it must not apply a second, independent name-encoding policy.
"""

import argparse

from generate_weighted import binding, message, named_type, scalar, text


def feature(name):
    array = message(1, b"\x04") + scalar(2, 65568)
    return text(1, name) + message(3, message(5, array))


def source_model(invalid=False):
    inputs = [
        "state" if invalid else "rustnn_escaped_7374617465",
        "rustnn_escaped_727573746e6e5f657363617065645f37333734363137343635",
        "rustnn_escaped_706173742e6b6579",
    ]
    outputs = ["rustnn_escaped_72657475726e", "output"]
    add = text(1, "add") + binding("x", inputs[0]) + binding("y", inputs[1])
    add += message(3, named_type(outputs[0], [4]))
    relu = text(1, "relu") + binding("x", inputs[2])
    relu += message(3, named_type(outputs[1], [4]))
    block = b"".join(text(2, name) for name in outputs)
    block += message(3, add) + message(3, relu)
    function = b"".join(message(1, named_type(name, [4])) for name in inputs)
    function += text(2, "CoreML8")
    function += message(3, text(1, "CoreML8") + message(2, block))
    program = scalar(1, 1) + message(2, text(1, "main") + message(2, function))
    description = b"".join(message(1, feature(name)) for name in inputs)
    description += b"".join(message(10, feature(name)) for name in outputs)
    metadata = message(100, text(1, "rustnn.coreml.name_encoding") + text(2, "hex-v1"))
    description += message(100, metadata)
    return scalar(1, 9) + message(2, description) + message(502, program)


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, invalid in [("escaped", False), ("reserved", True)]:
        (args.output / (name + ".mlmodel")).write_bytes(source_model(invalid))


if __name__ == "__main__":
    main()
