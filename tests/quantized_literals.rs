use mlmodelc_export::{MILTensorData, compile_to_bundle, compile_to_dir, compile_to_text, decode};

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}

fn message(field: u64, contents: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(contents.len() as u64),
        contents.to_vec(),
    ]
    .concat()
}

fn scalar(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}

fn value_type(dtype: u64, shape: &[u64]) -> Vec<u8> {
    let mut tensor = [scalar(1, dtype), scalar(2, shape.len() as u64)].concat();
    for &size in shape {
        tensor.extend(message(3, &message(1, &scalar(1, size))));
    }
    message(1, &tensor)
}

fn byte_value(dtype: u64, shape: &[u64], bytes: &[u8]) -> Vec<u8> {
    let tensor = message(7, &message(1, bytes));
    [
        message(2, &value_type(dtype, shape)),
        message(3, &message(1, &tensor)),
    ]
    .concat()
}

fn attribute(name: &str, value: &[u8]) -> Vec<u8> {
    message(
        5,
        &[message(1, name.as_bytes()), message(2, value)].concat(),
    )
}

fn model(dtype: u64, shape: &[u64], operation_type: &str, attributes: &[u8]) -> Vec<u8> {
    let output = [message(1, b"result"), message(2, &value_type(dtype, shape))].concat();
    let op = [
        message(1, operation_type.as_bytes()),
        message(3, &output),
        attributes.to_vec(),
    ]
    .concat();
    let block = [message(2, b"result"), message(3, &op)].concat();
    let block_entry = [message(1, b"CoreML7"), message(2, &block)].concat();
    let function = [message(2, b"CoreML7"), message(3, &block_entry)].concat();
    let function_entry = [message(1, b"main"), message(2, &function)].concat();
    let program = [scalar(1, 1), message(2, &function_entry)].concat();
    [scalar(1, 8), message(502, &program)].concat()
}

fn constant_model(dtype: u64, shape: &[u64], bytes: &[u8]) -> Vec<u8> {
    model(
        dtype,
        shape,
        "const",
        &attribute("val", &byte_value(dtype, shape, bytes)),
    )
}

#[test]
fn scalar_int8_and_uint8_bytes_preserve_every_value() {
    for byte in 0..=u8::MAX {
        for (dtype, name, expected) in [(21, "int8", byte as i8 as i32), (31, "uint8", byte as i32)]
        {
            let protobuf = constant_model(dtype, &[], &[byte]);
            let text = compile_to_text(&protobuf).unwrap().mil_text;
            assert!(
                text.contains(&format!("[val = {name}({expected})];")),
                "{text}"
            );
        }
    }
}

#[test]
fn dense_and_uniform_byte_tensors_use_integer_elements_in_all_apis() {
    let directory = std::env::temp_dir().join(format!(
        "mlmodelc-quantized-literals-{}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    for (dtype, bytes, expected) in [
        (
            21,
            vec![128, 255, 0, 1, 7, 127],
            "tensor<int8, [2, 3]>([[-128, -1, 0], [1, 7, 127]])",
        ),
        (
            31,
            vec![0, 1, 127, 128, 254, 255],
            "tensor<uint8, [2, 3]>([[0, 1, 127], [128, 254, 255]])",
        ),
        (
            21,
            vec![255; 6],
            "tensor<int8, [2, 3]>([[-1, -1, -1], [-1, -1, -1]])",
        ),
        (
            31,
            vec![255; 6],
            "tensor<uint8, [2, 3]>([[255, 255, 255], [255, 255, 255]])",
        ),
    ] {
        let protobuf = constant_model(dtype, &[2, 3], &bytes);
        let text = compile_to_text(&protobuf).unwrap().mil_text;
        assert!(text.contains(&format!("[val = {expected}];")), "{text}");
        let bundle = compile_to_bundle(&protobuf, None).unwrap();
        assert_eq!(bundle.model_mil, text.as_bytes());
        compile_to_dir(&protobuf, None, &directory).unwrap();
        assert_eq!(
            std::fs::read(directory.join("model.mil")).unwrap(),
            bundle.model_mil
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn byte_payload_validation_rejects_truncation_and_unsupported_storage() {
    for (dtype, shape, bytes) in [
        (21, vec![], vec![]),
        (31, vec![2], vec![7]),
        (21, vec![1], vec![7, 8]),
        (10, vec![], vec![1]),
        (10, vec![2], vec![0, 0]),
        (23, vec![], vec![0, 0, 0, 0]),
    ] {
        assert!(decode(&constant_model(dtype, &shape, &bytes)).is_err());
    }
    for dtype in [21, 31] {
        let program = decode(&constant_model(dtype, &[0], &[])).unwrap();
        assert!(
            matches!(&program.functions[0].1.block.operations[0].attributes[0].1.tensor, MILTensorData::Ints(v) if v.is_empty())
        );
        assert!(
            compile_to_text(&constant_model(dtype, &[0], &[]))
                .unwrap()
                .mil_text
                .contains(">([])")
        );
    }
}

#[test]
fn constexpr_parameters_remain_sorted_typed_attributes() {
    let blob = [
        message(2, &value_type(21, &[4])),
        message(
            5,
            &[
                message(1, b"@model_path/weights/weights.bin"),
                scalar(2, 64),
            ]
            .concat(),
        ),
    ]
    .concat();
    let axis_tensor = message(2, &message(1, &[0]));
    let axis = [
        message(2, &value_type(23, &[])),
        message(3, &message(1, &axis_tensor)),
    ]
    .concat();
    let scale_tensor = message(1, &message(1, &0.5f32.to_le_bytes()));
    let scale = [
        message(2, &value_type(11, &[])),
        message(3, &message(1, &scale_tensor)),
    ]
    .concat();
    let attributes = [
        attribute("zero_point", &byte_value(21, &[], &[7])),
        attribute("scale", &scale),
        attribute("quantized_data", &blob),
        attribute("axis", &axis),
    ]
    .concat();
    let protobuf = model(11, &[4], "constexpr_affine_dequantize", &attributes);
    let expected = "constexpr_affine_dequantize()[axis = int32(0), quantized_data = tensor<int8, [4]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64(64))), scale = fp32(0x1p-1), zero_point = int8(7)];";
    let text = compile_to_text(&protobuf).unwrap().mil_text;
    assert!(text.contains(expected), "{text}");
    assert_eq!(
        compile_to_bundle(&protobuf, None).unwrap().model_mil,
        text.as_bytes()
    );
}
