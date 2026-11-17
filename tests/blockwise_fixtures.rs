//! Source-level checks for the `constexpr_blockwise_shift_scale` fixtures.
//! Byte-for-byte comparison with the native goldens lives in golden.rs;
//! predictions in runtime_state.swift.

use mlmodelc_export::{
    MILBinding, MILDataType, MILOperation, MILTensorData, MILValue, compile_to_bundle,
    compile_to_text, decode, referenced_weight_paths,
};

const UINT8_MODEL: &[u8] = include_bytes!("fixtures/blockwise-uint8-offset/input.mlmodel");
const UINT8_WEIGHTS: &[u8] = include_bytes!("fixtures/blockwise-uint8-offset/weights/weights.bin");
const INT8_MODEL: &[u8] = include_bytes!("fixtures/blockwise-int8-scales/input.mlmodel");
const INT8_WEIGHTS: &[u8] = include_bytes!("fixtures/blockwise-int8-scales/weights/weights.bin");

fn constant<'a>(operations: &'a [MILOperation], name: &str) -> &'a MILValue {
    let op = operations
        .iter()
        .find(|op| op.r#type == "const" && op.outputs[0].name == name)
        .unwrap();
    &op.attributes
        .iter()
        .find(|(key, _)| key == "val")
        .unwrap()
        .1
}

fn references(op: &MILOperation) -> Vec<(&str, &str)> {
    op.inputs
        .iter()
        .map(|(key, bindings)| match bindings.as_slice() {
            [MILBinding::Reference(name)] => (key.as_str(), name.as_str()),
            other => panic!("{key}: {other:?}"),
        })
        .collect()
}

/// (data type code, payload) of the blob whose metadata starts at `offset`.
fn blob(weights: &[u8], offset: usize) -> (u32, &[u8]) {
    let word = |at: usize| u32::from_le_bytes(weights[at..at + 4].try_into().unwrap());
    let long = |at: usize| u64::from_le_bytes(weights[at..at + 8].try_into().unwrap()) as usize;
    assert_eq!(word(offset), 0xdead_beef);
    let (length, start) = (long(offset + 8), long(offset + 16));
    (word(offset + 4), &weights[start..start + length])
}

fn metadata(model: &[u8], weights: &[u8]) -> serde_json::Value {
    let bundle = compile_to_bundle(model, Some(weights)).unwrap();
    assert_eq!(bundle.weights_bin.as_deref(), Some(weights));
    serde_json::from_slice::<serde_json::Value>(&bundle.metadata_json).unwrap()[0].take()
}

#[test]
fn uint8_fixture_keeps_block_scales_and_offset_as_named_constants() {
    let program = decode(UINT8_MODEL).unwrap();
    let function = &program.functions[0].1;
    assert_eq!(function.opset, "CoreML8");
    let operations = &function.block.operations;
    let blockwise = operations
        .iter()
        .find(|op| op.r#type == "constexpr_blockwise_shift_scale")
        .unwrap();
    // The source refers to `const` operations by name; nothing is an attribute.
    assert!(blockwise.attributes.is_empty());
    assert_eq!(
        references(blockwise),
        [("data", "data"), ("scale", "scale"), ("offset", "offset")]
    );
    assert_eq!(blockwise.outputs[0].r#type.data_type, MILDataType::Float16);
    assert_eq!(
        blockwise.outputs[0].r#type.concrete_shape().unwrap(),
        [2, 64]
    );

    let data = constant(operations, "data");
    assert_eq!(data.r#type.data_type, MILDataType::Uint8);
    assert_eq!(data.blob.as_ref().unwrap().offset, 64);
    let scale = constant(operations, "scale");
    assert_eq!(scale.r#type.data_type, MILDataType::Float16);
    assert_eq!(scale.r#type.concrete_shape().unwrap(), [2, 2]);
    assert_eq!(scale.blob.as_ref().unwrap().offset, 256);
    let offset = constant(operations, "offset");
    assert_eq!(offset.r#type.data_type, MILDataType::Uint8);
    assert!(offset.blob.is_none());
    assert!(matches!(&offset.tensor, MILTensorData::Ints(values) if values == &[128; 4]));

    // Two entries, storage version 2: uint8 data, then FP16 scales.
    assert_eq!(&UINT8_WEIGHTS[..8], &[2, 0, 0, 0, 2, 0, 0, 0]);
    let (kind, bytes) = blob(UINT8_WEIGHTS, 64);
    assert_eq!(kind, 3);
    let expected: Vec<u8> = (0..64)
        .map(|k| (128 + k % 7 - 3) as u8)
        .chain((0..64).map(|k| (128 + (3 * k) % 11 - 5) as u8))
        .collect();
    assert_eq!(bytes, expected);
    let (kind, bytes) = blob(UINT8_WEIGHTS, 256);
    assert_eq!(kind, 1);
    // 0.5, 0.25, 2 and 0.125 as IEEE binary16.
    assert_eq!(bytes, [0x00, 0x38, 0x00, 0x34, 0x00, 0x40, 0x00, 0x30]);

    assert_eq!(
        referenced_weight_paths(UINT8_MODEL).unwrap(),
        ["weights/weights.bin"]
    );
    assert!(compile_to_bundle(UINT8_MODEL, None).is_err());
    let metadata = metadata(UINT8_MODEL, UINT8_WEIGHTS);
    assert_eq!(metadata["storagePrecision"], "Mixed (Float16, UInt8)");
    assert_eq!(
        metadata["mlProgramOperationTypeHistogram"],
        serde_json::json!({
            "Ios18.cast": 2,
            "Ios18.constexprBlockwiseShiftScale": 1,
            "Ios18.matmul": 1
        })
    );
}

#[test]
fn uint8_fixture_text_inlines_constants_only_into_the_constexpr() {
    let text = compile_to_text(UINT8_MODEL).unwrap().mil_text;
    let data = "tensor<uint8, [2, 64]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64(64)))";
    let scale = "tensor<fp16, [2, 2]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64(256)))";
    let offset = "tensor<uint8, [2, 2]>([[128, 128], [128, 128]])";
    for line in [
        format!("tensor<uint8, [2, 64]> data = const()[val = {data}];"),
        format!("tensor<fp16, [2, 2]> scale = const()[val = {scale}];"),
        format!("tensor<uint8, [2, 2]> offset = const()[val = {offset}];"),
        format!(
            "tensor<fp16, [2, 64]> weight = constexpr_blockwise_shift_scale(data = {data}, offset = {offset}, scale = {scale});"
        ),
        // matmul keeps the names of its boolean constants.
        "tensor<fp16, [1, 2]> product = matmul(transpose_x = no, transpose_y = yes, x = input_fp16, y = weight);".to_owned(),
    ] {
        assert!(text.contains(&format!("            {line}\n")), "{line}\n{text}");
    }
}

#[test]
fn int8_fixture_shares_one_blob_between_unit_and_per_row_scales() {
    let program = decode(INT8_MODEL).unwrap();
    let operations = &program.functions[0].1.block.operations;
    let blockwise: Vec<_> = operations
        .iter()
        .filter(|op| op.r#type == "constexpr_blockwise_shift_scale")
        .collect();
    assert_eq!(blockwise.len(), 2);
    // No offset argument in either form.
    assert_eq!(
        references(blockwise[0]),
        [("data", "data"), ("scale", "unit")]
    );
    assert_eq!(
        references(blockwise[1]),
        [("data", "data"), ("scale", "rows")]
    );
    let data = constant(operations, "data");
    assert_eq!(data.r#type.data_type, MILDataType::Int8);
    assert_eq!(data.blob.as_ref().unwrap().offset, 64);
    let unit = constant(operations, "unit");
    assert_eq!(unit.r#type.concrete_shape().unwrap(), [1, 1]);
    assert!(matches!(&unit.tensor, MILTensorData::Floats(values) if values == &[1.0]));
    let rows = constant(operations, "rows");
    assert_eq!(rows.r#type.concrete_shape().unwrap(), [2, 1]);
    assert!(matches!(&rows.tensor, MILTensorData::Floats(values) if values == &[0.5, 0.25]));

    assert_eq!(&INT8_WEIGHTS[..8], &[1, 0, 0, 0, 2, 0, 0, 0]);
    let (kind, bytes) = blob(INT8_WEIGHTS, 64);
    assert_eq!(kind, 4);
    let signed: Vec<_> = bytes.iter().map(|byte| *byte as i8).collect();
    assert_eq!(signed, [-128, -1, 0, 127, 4, -8, 16, -32]);

    let text = compile_to_text(INT8_MODEL).unwrap().mil_text;
    let data = "tensor<int8, [2, 4]>(BLOBFILE(path = string(\"@model_path/weights/weights.bin\"), offset = uint64(64)))";
    for line in [
        format!(
            "tensor<fp32, [2, 4]> exact = constexpr_blockwise_shift_scale(data = {data}, scale = tensor<fp32, [1, 1]>([[0x1p+0]]));"
        ),
        format!(
            "tensor<fp32, [2, 4]> scaled = constexpr_blockwise_shift_scale(data = {data}, scale = tensor<fp32, [2, 1]>([[0x1p-1], [0x1p-2]]));"
        ),
        "tensor<fp32, [2, 4]> sum = add(x = exact, y = scaled);".to_owned(),
    ] {
        assert!(
            text.contains(&format!("            {line}\n")),
            "{line}\n{text}"
        );
    }

    let metadata = metadata(INT8_MODEL, INT8_WEIGHTS);
    assert_eq!(metadata["storagePrecision"], "Int8");
    assert_eq!(
        metadata["mlProgramOperationTypeHistogram"],
        serde_json::json!({ "Ios18.add": 2, "Ios18.constexprBlockwiseShiftScale": 2 })
    );
}
