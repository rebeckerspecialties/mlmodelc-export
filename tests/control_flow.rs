use mlmodelc_export::{
    compile_to_bundle, compile_to_dir, compile_to_text, referenced_weight_paths,
};
use std::fs;

const MODEL: &[u8] = include_bytes!("control-flow/input.mlmodel");
const WEIGHTS: &[u8] = include_bytes!("control-flow/weights/weights.bin");

#[test]
fn nested_blocks_preserve_arguments_returns_constants_and_statistics() {
    let result = compile_to_text(MODEL).unwrap();
    assert_eq!(result.stats.operation_count, 13);
    assert_eq!(result.stats.const_count, 5);
    assert_eq!(result.stats.largest_const_elements, 4);
    assert!(
        result
            .mil_text
            .contains("(int32 i, tensor<fp32, [4]> carried)")
    );
    assert!(result.mil_text.contains("} -> (incremented, next)"));
    assert!(result.mil_text.contains("} -> (added)"));
    assert!(result.mil_text.contains("} -> (multiplied)"));
    assert_eq!(
        referenced_weight_paths(MODEL).unwrap(),
        ["weights/weights.bin"]
    );
    let bundle = compile_to_bundle(MODEL, Some(WEIGHTS)).unwrap();
    let dir = std::env::temp_dir().join(format!("nested-control-flow-{}", std::process::id()));
    compile_to_dir(MODEL, Some(WEIGHTS), &dir).unwrap();
    assert_eq!(
        fs::read(dir.join("model.mil")).unwrap(),
        result.mil_text.as_bytes()
    );
    assert_eq!(
        fs::read(dir.join("metadata.json")).unwrap(),
        bundle.metadata_json
    );
    assert_eq!(fs::read(dir.join("weights/weights.bin")).unwrap(), WEIGHTS);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_nested_weights_fail_before_creating_output() {
    let dir = std::env::temp_dir().join(format!("missing-nested-weights-{}", std::process::id()));
    assert!(
        compile_to_dir(MODEL, None, &dir)
            .unwrap_err()
            .to_string()
            .contains("missing external weight")
    );
    assert!(!dir.exists());
}

#[test]
fn excessive_control_flow_nesting_is_rejected() {
    assert!(
        compile_to_text(include_bytes!("control-flow/too-deep.mlmodel"))
            .unwrap_err()
            .to_string()
            .contains("64 nested blocks")
    );
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    while value >= 128 {
        result.push(value as u8 | 128);
        value >>= 7;
    }
    result.push(value as u8);
    result
}

fn message(field: u64, bytes: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(bytes.len() as u64),
        bytes.to_vec(),
    ]
    .concat()
}

#[test]
fn pipeline_relocates_weights_inside_nested_blocks() {
    let decoded = mlmodelc_export::decode(MODEL).unwrap();
    let pipeline = [
        vec![8, 9],
        message(2, &decoded.description_data),
        message(202, &message(1, MODEL)),
    ]
    .concat();
    let dir = std::env::temp_dir().join(format!("nested-pipeline-{}", std::process::id()));
    let stats = compile_to_dir(&pipeline, Some(WEIGHTS), &dir).unwrap();
    assert_eq!(stats.operation_count, 13);
    let mil = fs::read_to_string(dir.join("model0/model.mil")).unwrap();
    assert!(mil.contains("@model_path/../weights/weights.bin"));
    assert!(!dir.join("model0/weights").exists());
    assert_eq!(fs::read(dir.join("weights/weights.bin")).unwrap(), WEIGHTS);
    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("model0/metadata.json")).unwrap()).unwrap();
    assert!(metadata.to_string().contains("cond"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn invalid_nested_blob_paths_are_rejected() {
    let mut model = MODEL.to_vec();
    let original = b"@model_path/weights/weights.bin";
    let invalid = b"@model_path/../evil/weights.bin";
    let at = model
        .windows(original.len())
        .position(|value| value == original)
        .unwrap();
    model[at..at + original.len()].copy_from_slice(invalid);
    assert!(
        referenced_weight_paths(&model)
            .unwrap_err()
            .to_string()
            .contains("invalid external weight path")
    );
}

#[test]
fn streaming_accepts_borrowed_nonclone_weight_owners() {
    struct Borrowed<'a>(&'a [u8]);
    impl AsRef<[u8]> for Borrowed<'_> {
        fn as_ref(&self) -> &[u8] {
            self.0
        }
    }
    let files =
        std::collections::BTreeMap::from([("weights/weights.bin".to_owned(), Borrowed(WEIGHTS))]);
    let dir = std::env::temp_dir().join(format!("borrowed-nested-weights-{}", std::process::id()));
    mlmodelc_export::compile_to_dir_with_weight_files(MODEL, &files, &dir).unwrap();
    assert_eq!(fs::read(dir.join("weights/weights.bin")).unwrap(), WEIGHTS);
    fs::remove_dir_all(dir).unwrap();
}
