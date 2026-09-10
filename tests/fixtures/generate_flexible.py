"""Derive small protobuf fixtures from the dynamic ReLU fixture.

Requires Apple's Model_pb2 generated from the Core ML format schemas on
PYTHONPATH. Capture expected bundles separately with xcrun coremlc compile.
"""
from pathlib import Path
from Model_pb2 import Model

root = Path(__file__).parent
source = Model.FromString((root / "flexible-range/input.mlmodel").read_bytes())

enumerated = Model()
enumerated.CopyFrom(source)
array = enumerated.description.input[0].type.multiArrayType
array.ClearField("shapeRange")
for size in [1, 2, 4]:
    array.enumeratedShapes.shapes.add().shape.append(size)

multi = Model()
multi.CopyFrom(source)
multi.specificationVersion = 9
multi.description.ClearField("input")
multi.description.ClearField("output")
multi.mlProgram.ClearField("functions")
for name in ["first", "second"]:
    function = multi.mlProgram.functions[name]
    function.CopyFrom(source.mlProgram.functions["main"])
    function.opset = "CoreML8"
    function.block_specializations["CoreML8"].CopyFrom(function.block_specializations["CoreML7"])
    del function.block_specializations["CoreML7"]
    description = multi.description.functions.add(name=name)
    description.input.extend(source.description.input)
    description.output.extend(source.description.output)
    if name == "second":
        description.input[0].type.multiArrayType.shape[0] = 2
        description.input[0].shortDescription = 'Default input: "second"\n' + "x" * 300
multi.description.defaultFunctionName = "second"

for name, model in [("flexible-enumerated", enumerated), ("flexible-multifunction", multi)]:
    directory = (root.parent / "unsupported" if name == "flexible-enumerated" else root) / name
    directory.mkdir(exist_ok=True)
    (directory / "input.mlmodel").write_bytes(model.SerializeToString(deterministic=True))
