//! Wire-format/serialization controls, not CoreML execution or SSA validation.
//! Runtime numerical behavior remains covered by runtime_control_flow.swift.
use mlmodelc_export::{
    MILBinding, compile_to_bundle_with_weight_files, compile_to_dir_with_weight_files,
    compile_to_text, decode, referenced_weight_paths,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push(value as u8 | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}
fn message(field: u64, value: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}
fn text(field: u64, value: &str) -> Vec<u8> {
    message(field, value.as_bytes())
}
fn scalar(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}
fn ty(dtype: u64, shape: &[u64]) -> Vec<u8> {
    let mut tensor = [scalar(1, dtype), scalar(2, shape.len() as u64)].concat();
    for &extent in shape {
        tensor.extend(message(3, &message(1, &scalar(1, extent))));
    }
    message(1, &tensor)
}
fn named(name: &str, dtype: u64, shape: &[u64]) -> Vec<u8> {
    [text(1, name), message(2, &ty(dtype, shape))].concat()
}
fn reference(key: &str, names: &[&str]) -> Vec<u8> {
    let values = names
        .iter()
        .flat_map(|name| message(1, &text(1, name)))
        .collect::<Vec<_>>();
    message(2, &[text(1, key), message(2, &values)].concat())
}
fn blob(path: &str, offset: u64) -> Vec<u8> {
    [
        message(2, &ty(11, &[4])),
        message(5, &[text(1, path), scalar(2, offset)].concat()),
    ]
    .concat()
}
fn immediate(key: &str, value: &[u8]) -> Vec<u8> {
    message(
        2,
        &[text(1, key), message(2, &message(1, &message(2, value)))].concat(),
    )
}
fn attribute(key: &str, value: &[u8]) -> Vec<u8> {
    message(5, &[text(1, key), message(2, value)].concat())
}
fn operation(
    kind: &str,
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    blocks: &[Vec<u8>],
    attributes: &[Vec<u8>],
) -> Vec<u8> {
    [
        text(1, kind),
        inputs.concat(),
        outputs.iter().flat_map(|o| message(3, o)).collect(),
        blocks.iter().flat_map(|b| message(4, b)).collect(),
        attributes.concat(),
    ]
    .concat()
}
fn block(inputs: &[Vec<u8>], outputs: &[&str], operations: &[Vec<u8>]) -> Vec<u8> {
    [
        inputs
            .iter()
            .flat_map(|v| message(1, v))
            .collect::<Vec<_>>(),
        outputs.iter().flat_map(|v| text(2, v)).collect(),
        operations.iter().flat_map(|v| message(3, v)).collect(),
    ]
    .concat()
}
fn model(body: &[u8]) -> Vec<u8> {
    let function = [
        message(1, &named("input", 11, &[4])),
        message(1, &named("pred", 1, &[])),
        text(2, "CoreML8"),
        message(3, &[text(1, "CoreML8"), message(2, body)].concat()),
    ]
    .concat();
    let program = [
        scalar(1, 1),
        message(2, &[text(1, "main"), message(2, &function)].concat()),
    ]
    .concat();
    [scalar(1, 9), message(502, &program)].concat()
}
fn identity(input: &str, output: &str) -> Vec<u8> {
    operation(
        "identity",
        &[reference("x", &[input])],
        &[named(output, 11, &[4])],
        &[],
        &[],
    )
}
fn choose(output: &str, yes: Vec<u8>, no: Vec<u8>) -> Vec<u8> {
    operation(
        "cond",
        &[reference("pred", &["pred"])],
        &[named(output, 11, &[4])],
        &[yes, no],
        &[],
    )
}
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "nested-contract-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!dir.exists());
        Self(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn lexical_names_captured_returns_and_empty_blocks_are_preserved() {
    // Block names are deliberately reused in separate lexical scopes. This
    // checks preservation, not whether an Apple runtime admits all such names.
    let branch = choose(
        "selected",
        block(&[], &["input"], &[]),
        block(&[], &["input"], &[identity("outer", "input")]),
    );
    let loop_op = operation(
        "while_loop",
        &[reference("loop_vars", &["input"])],
        &[named("output", 11, &[4])],
        &[
            block(&[named("input", 11, &[4])], &["pred"], &[]),
            block(&[named("input", 11, &[4])], &["selected"], &[branch]),
        ],
        &[],
    );
    let bytes = model(&block(
        &[],
        &["output"],
        &[identity("input", "outer"), loop_op],
    ));
    let program = decode(&bytes).unwrap();
    let outer = &program.functions[0].1.block;
    let loop_body = &outer.operations[1].blocks[1];
    assert_eq!(loop_body.inputs[0].name, "input");
    assert_eq!(outer.operations[1].blocks[0].outputs, ["pred"]);
    assert!(outer.operations[1].blocks[0].operations.is_empty());
    let branches = &loop_body.operations[0].blocks;
    assert!(branches[0].operations.is_empty());
    assert_eq!(branches[0].outputs, ["input"]);
    assert_eq!(branches[1].outputs, ["input"]);
    assert!(
        matches!(&branches[1].operations[0].inputs[0].1[0], MILBinding::Reference(name) if name == "outer")
    );
    let result = compile_to_text(&bytes).unwrap();
    assert_eq!(result.stats.operation_count, 4);
    assert_eq!(
        result
            .mil_text
            .matches("(tensor<fp32, [4]> input) {")
            .count(),
        2
    );
    assert!(result.mil_text.contains("} -> (pred)"));
    assert_eq!(result.mil_text.matches("} -> (input)").count(), 2);
    assert!(result.mil_text.contains("identity(x = outer)"));
}

#[test]
fn preorder_keeps_siblings_and_returns_in_source_order() {
    let branch = choose(
        "middle",
        block(&[], &["local"], &[identity("input", "local")]),
        block(&[], &["local"], &[identity("input", "local")]),
    );
    let bytes = model(&block(
        &[],
        &["last", "middle", "first"],
        &[
            identity("input", "first"),
            branch,
            identity("middle", "last"),
        ],
    ));
    let decoded = decode(&bytes).unwrap();
    let top = &decoded.functions[0].1.block;
    let names: Vec<_> = top
        .all_operations()
        .map(|op| op.outputs[0].name.as_str())
        .collect();
    assert_eq!(names, ["first", "middle", "local", "local", "last"]);
    assert_eq!(top.outputs, ["last", "middle", "first"]);
    let text = compile_to_text(&bytes).unwrap().mil_text;
    assert!(text.contains("} -> (last, middle, first)"));
    assert_eq!(
        text.matches("tensor<fp32, [4]> local = identity(x = input)")
            .count(),
        2
    );
}

#[test]
fn accepts_exact_nesting_limit_and_rejects_next_level() {
    for depth in [0, 1, 63, 64, 65] {
        let mut nested = block(&[], &["input"], &[]);
        for _ in 0..depth {
            nested = block(
                &[],
                &["output"],
                &[choose("output", nested, block(&[], &["input"], &[]))],
            );
        }
        let result = compile_to_text(&model(&nested));
        if depth <= 64 {
            assert_eq!(result.unwrap().stats.operation_count, depth);
        } else {
            assert!(result.unwrap_err().to_string().contains("64 nested blocks"));
        }
    }
}

fn nested_assets(second_path: &str, second_offset: u64) -> Vec<u8> {
    let first = operation(
        "const",
        &[],
        &[named("weight", 11, &[4])],
        &[],
        &[attribute("val", &blob("@model_path/weights/a.bin", 4))],
    );
    let second = operation(
        "add",
        &[
            reference("x", &["input"]),
            immediate("y", &blob(second_path, second_offset)),
        ],
        &[named("added", 11, &[4])],
        &[],
        &[],
    );
    let inner = choose(
        "selected",
        block(&[], &["weight"], &[first]),
        block(&[], &["added"], &[second]),
    );
    model(&block(
        &[],
        &["output"],
        &[choose(
            "output",
            block(&[], &["selected"], &[inner]),
            block(&[], &["input"], &[]),
        )],
    ))
}

#[test]
fn recursive_attribute_and_immediate_assets_relocate_without_payload_changes() {
    // Deliberately small opaque files: these controls test paths/offsets, not
    // CoreML blob-header validity or native execution.
    let a = [0x31; 16];
    let b = [0xa7; 24];
    let files = BTreeMap::from([
        ("weights/a.bin".to_owned(), a.as_slice()),
        ("weights/b.bin".to_owned(), b.as_slice()),
    ]);
    let bytes = nested_assets("@model_path/weights/b.bin", 20);
    assert_eq!(
        referenced_weight_paths(&bytes).unwrap(),
        ["weights/a.bin", "weights/b.bin"]
    );
    let before = compile_to_text(&bytes).unwrap().mil_text;
    let owned = files
        .iter()
        .map(|(name, bytes)| (name.clone(), bytes.to_vec()))
        .collect();
    let bundle = compile_to_bundle_with_weight_files(&bytes, &owned).unwrap();
    let standalone = Scratch::new();
    bundle.write_to_dir(&standalone.0).unwrap();
    assert_eq!(fs::read(standalone.0.join("weights/a.bin")).unwrap(), a);
    assert_eq!(fs::read(standalone.0.join("weights/b.bin")).unwrap(), b);
    let pipeline = [scalar(1, 9), message(202, &message(1, &bytes))].concat();
    let output = Scratch::new();
    let stats = compile_to_dir_with_weight_files(&pipeline, &files, &output.0).unwrap();
    assert_eq!(stats.operation_count, 4);
    let mil = fs::read_to_string(output.0.join("model0/model.mil")).unwrap();
    assert!(mil.contains("@model_path/../weights/a.bin"));
    assert!(mil.contains("@model_path/../weights/b.bin"));
    assert!(!output.0.join("model0/weights").exists());
    assert_eq!(fs::read(output.0.join("weights/a.bin")).unwrap(), a);
    assert_eq!(fs::read(output.0.join("weights/b.bin")).unwrap(), b);
    assert_eq!(compile_to_text(&bytes).unwrap().mil_text, before);
}

#[test]
fn recursive_immediate_assets_are_validated_before_output_creation() {
    let a = [0_u8; 16];
    let b = [0_u8; 24];
    let both = BTreeMap::from([
        ("weights/a.bin".to_owned(), a.as_slice()),
        ("weights/b.bin".to_owned(), b.as_slice()),
    ]);
    for (path, offset, expected) in [
        ("@model_path/weights/b.bin", 24, "outside"),
        (
            "@model_path/../weights/b.bin",
            0,
            "invalid external weight path",
        ),
        ("weights/b.bin", 0, "must start with @model_path/"),
    ] {
        let output = Scratch::new();
        let error =
            compile_to_dir_with_weight_files(&nested_assets(path, offset), &both, &output.0)
                .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert!(!output.0.exists());
    }
    let output = Scratch::new();
    let missing = BTreeMap::from([("weights/a.bin".to_owned(), a.as_slice())]);
    assert!(
        compile_to_dir_with_weight_files(
            &nested_assets("@model_path/weights/b.bin", 0),
            &missing,
            &output.0
        )
        .unwrap_err()
        .to_string()
        .contains("missing external weight file")
    );
    assert!(!output.0.exists());
}

#[test]
fn invalid_nested_formal_and_output_types_propagate_decoder_errors() {
    for child in [
        block(&[named("bad", 999, &[4])], &["input"], &[]),
        block(
            &[],
            &["bad"],
            &[operation(
                "identity",
                &[reference("x", &["input"])],
                &[named("bad", 999, &[4])],
                &[],
                &[],
            )],
        ),
    ] {
        let bytes = model(&block(
            &[],
            &["output"],
            &[choose("output", child, block(&[], &["input"], &[]))],
        ));
        assert!(
            compile_to_text(&bytes)
                .unwrap_err()
                .to_string()
                .contains("unknown MIL DataType raw value: 999")
        );
    }
}
