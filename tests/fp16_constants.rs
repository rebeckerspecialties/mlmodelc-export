use mlmodelc_export::{compile_to_bundle, compile_to_dir, compile_to_text};

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

// Minimal Model/Program/Function/const protobuf using the real FP16
// TensorValue.bytes representation, including little-endian payload bytes.
fn constant_model(shape: &[u64], values: &[u16]) -> Vec<u8> {
    let mut tensor_type = [scalar(1, 10), scalar(2, shape.len() as u64)].concat();
    for &size in shape {
        tensor_type.extend(message(3, &message(1, &scalar(1, size))));
    }
    let value_type = message(1, &tensor_type);
    let payload: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor_value = message(7, &message(1, &payload));
    let value = [
        message(2, &value_type),
        message(3, &message(1, &tensor_value)),
    ]
    .concat();
    let named_type = [message(1, b"result"), message(2, &value_type)].concat();
    let attribute = [message(1, b"val"), message(2, &value)].concat();
    let operation = [
        message(1, b"const"),
        message(3, &named_type),
        message(5, &attribute),
    ]
    .concat();
    let block = [message(2, b"result"), message(3, &operation)].concat();
    let block_entry = [message(1, b"CoreML8"), message(2, &block)].concat();
    let function = [message(2, b"CoreML8"), message(3, &block_entry)].concat();
    let function_entry = [message(1, b"main"), message(2, &function)].concat();
    let program = [scalar(1, 1), message(2, &function_entry)].concat();
    [scalar(1, 9), message(502, &program)].concat()
}

#[test]
fn fp16_constants_preserve_scalar_uniform_and_dense_literals_in_all_apis() {
    let directory =
        std::env::temp_dir().join(format!("mlmodelc-fp16-constants-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    for (shape, values, expected) in [
        (vec![], vec![0x00a8], "fp16(0x1.5p-17)"),
        (
            vec![2, 2],
            vec![0x00a8; 4],
            "tensor<fp16, [2, 2]>([[0x1.5p-17, 0x1.5p-17], [0x1.5p-17, 0x1.5p-17]])",
        ),
        (
            vec![2, 3],
            vec![0x0003, 0x80a8, 0x03ff, 0x0400, 0x8000, 0x3c00],
            "tensor<fp16, [2, 3]>([[0x1.8p-23, -0x1.5p-17, 0x1.ff8p-15], [0x1p-14, -0x0p+0, 0x1p+0]])",
        ),
    ] {
        let model = constant_model(&shape, &values);
        let text = compile_to_text(&model).unwrap().mil_text;
        assert!(text.contains(&format!("[val = {expected}];")), "{text}");
        let bundle = compile_to_bundle(&model, None).unwrap();
        assert_eq!(bundle.model_mil, text.as_bytes());
        compile_to_dir(&model, None, &directory).unwrap();
        assert_eq!(
            std::fs::read(directory.join("model.mil")).unwrap(),
            bundle.model_mil
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}
