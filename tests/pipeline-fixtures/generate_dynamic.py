"""Wrap the existing independent flexible ReLU source in one Pipeline stage."""

import argparse
from pathlib import Path


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def message(field, data):
    return varint(field << 3 | 2) + varint(len(data)) + data


def read_varint(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7


def description(data):
    offset = 0
    while offset < len(data):
        tag, offset = read_varint(data, offset)
        if tag & 7 == 0:
            _, offset = read_varint(data, offset)
        elif tag & 7 == 2:
            length, offset = read_varint(data, offset)
            if tag >> 3 == 2:
                return data[offset:offset + length]
            offset += length
        else:
            raise ValueError("unexpected fixture wire type")
    raise ValueError("missing description")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    child = (root.parent / "fixtures/flexible-range/input.mlmodel").read_bytes()
    # Exercise an interface description beyond one-byte payload lengths too.
    metadata = message(1, ("Pipeline precision boundary. " * 20).encode())
    outer_description = description(child) + message(100, metadata)
    model = b"\x08\x09" + message(2, outer_description) + message(202, message(1, child))
    output = root / "dynamic/input.mlmodel"
    if args.check:
        if output.read_bytes() != model:
            raise SystemExit(f"fixture differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(model)


if __name__ == "__main__":
    main()
