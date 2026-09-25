use mlmodelc_export::{MILDataType, MILTensorData, decode};

#[test]
fn fp16_runtime_fixture_keeps_subnormal_payloads_and_cast() {
    let model = include_bytes!("fixtures/fp16-subnormal-cast/input.mlmodel");
    let program = decode(model).unwrap();
    let function = &program.functions[0].1;
    assert_eq!(function.inputs[0].r#type.data_type, MILDataType::Float32);
    let operations = &function.block.operations;
    assert_eq!(operations.len(), 3);
    assert_eq!(operations[0].r#type, "const");
    let value = &operations[0].attributes[0].1;
    assert_eq!(value.r#type.data_type, MILDataType::Float16);
    let MILTensorData::Fp16Bytes(bytes) = &value.tensor else {
        panic!("the runtime fixture must retain inline FP16 bytes");
    };
    let bits: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect();
    assert_eq!(bits, [0x0001, 0x00a8, 0x03ff, 0x0400, 0x80a8]);
    assert_eq!(operations[1].r#type, "cast");
    assert_eq!(
        operations[1].outputs[0].r#type.data_type,
        MILDataType::Float32
    );
    assert_eq!(operations[2].r#type, "add");
}

#[test]
fn quantized_runtime_fixture_keeps_typed_attributes_and_signed_blob() {
    let model = include_bytes!("fixtures/quantized-constexpr/input.mlmodel");
    let weights = include_bytes!("fixtures/quantized-constexpr/weights/weights.bin");
    let program = decode(model).unwrap();
    let operation = &program.functions[0].1.block.operations[0];
    assert_eq!(operation.r#type, "constexpr_affine_dequantize");
    assert_eq!(operation.attributes.len(), 4);
    let value = |name| {
        &operation
            .attributes
            .iter()
            .find(|(key, _)| key == name)
            .unwrap()
            .1
    };
    assert_eq!(value("zero_point").r#type.data_type, MILDataType::Int8);
    assert!(matches!(&value("zero_point").tensor, MILTensorData::Ints(v) if v == &[-1]));
    assert_eq!(value("axis").r#type.data_type, MILDataType::Int32);
    assert!(matches!(&value("axis").tensor, MILTensorData::Ints(v) if v == &[0]));
    assert_eq!(value("scale").r#type.data_type, MILDataType::Float32);
    assert!(matches!(&value("scale").tensor, MILTensorData::Floats(v) if v == &[0.5]));
    let quantized = value("quantized_data");
    assert_eq!(quantized.r#type.data_type, MILDataType::Int8);
    assert_eq!(quantized.blob.as_ref().unwrap().offset, 64);
    assert_eq!(&weights[64..68], &0xdeadbeefu32.to_le_bytes());
    assert_eq!(&weights[68..72], &4u32.to_le_bytes()); // int8
    assert_eq!(&weights[72..80], &4u64.to_le_bytes());
    assert_eq!(&weights[80..88], &128u64.to_le_bytes());
    assert_eq!(
        weights[128..132]
            .iter()
            .map(|byte| *byte as i8)
            .collect::<Vec<_>>(),
        [-128, -1, 0, 127]
    );
}
