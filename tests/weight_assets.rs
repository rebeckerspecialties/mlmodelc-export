use mlmodelc_export::{
    compile_to_bundle, compile_to_bundle_with_weight_files, compile_to_dir,
    compile_to_dir_with_weight_files, referenced_weight_paths,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const WEIGHTS: &[u8] = include_bytes!("fixtures/flexible-weighted-legacy/weights/weights.bin");

struct Scratch(PathBuf);
static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlmodelc-assets-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push(value as u8 | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}

fn message(field: u64, bytes: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(bytes.len() as u64),
        bytes.to_vec(),
    ]
    .concat()
}

fn scalar(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}

// Tiny const-only MLProgram. The separate runtime suite exercises real add
// predictions; these wire-format controls isolate asset discovery and offsets.
fn model(paths: &[&str]) -> Vec<u8> {
    let ty = message(
        1,
        &[
            scalar(1, 11),
            scalar(2, 1),
            message(3, &message(1, &scalar(1, 4))),
        ]
        .concat(),
    );
    let mut block = message(2, b"output0");
    for (index, path) in paths.iter().enumerate() {
        let value = [
            message(2, &ty),
            message(5, &[message(1, path.as_bytes()), scalar(2, 64)].concat()),
        ]
        .concat();
        let output = [
            message(1, format!("output{index}").as_bytes()),
            message(2, &ty),
        ]
        .concat();
        let op = [
            message(1, b"const"),
            message(3, &output),
            message(5, &[message(1, b"val"), message(2, &value)].concat()),
        ]
        .concat();
        block.extend(message(3, &op));
    }
    let function = [
        message(2, b"CoreML7"),
        message(3, &[message(1, b"CoreML7"), message(2, &block)].concat()),
    ]
    .concat();
    let program = [
        scalar(1, 1),
        message(2, &[message(1, b"main"), message(2, &function)].concat()),
    ]
    .concat();
    [scalar(1, 8), message(502, &program)].concat()
}

#[test]
fn single_weight_argument_preserves_the_referenced_filename() {
    let scratch = Scratch::new();
    let protobuf = model(&["@model_path/weights/weight.bin"]);
    let bundle = compile_to_bundle(&protobuf, Some(WEIGHTS)).unwrap();
    let buffered = scratch.0.join("buffered");
    bundle.write_to_dir(&buffered).unwrap();
    let streamed = scratch.0.join("streamed");
    compile_to_dir(&protobuf, Some(WEIGHTS), &streamed).unwrap();
    for directory in [buffered, streamed] {
        assert_eq!(
            fs::read(directory.join("weights/weight.bin")).unwrap(),
            WEIGHTS
        );
        assert!(!directory.join("weights/weights.bin").exists());
    }
}

#[test]
fn missing_referenced_weights_fail_before_output_is_written() {
    let scratch = Scratch::new();
    let protobuf = model(&["@model_path/weights/weight.bin"]);
    assert!(compile_to_bundle(&protobuf, None).is_err());
    let output = scratch.0.join("missing");
    assert!(compile_to_dir(&protobuf, None, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn multiple_nested_weights_preserve_bytes_offsets_and_streaming_parity() {
    let scratch = Scratch::new();
    let paths = ["weights/weights.bin", "weights/layer 1/weight.bin"];
    let protobuf = model(
        &paths
            .map(|path| format!("@model_path/{path}"))
            .each_ref()
            .map(String::as_str),
    );
    let files: BTreeMap<_, _> = paths
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            let mut bytes = WEIGHTS.to_vec();
            bytes[128..132].copy_from_slice(&(index as f32).to_le_bytes());
            (path.to_string(), bytes)
        })
        .collect();
    assert_eq!(
        referenced_weight_paths(&protobuf).unwrap(),
        files.keys().cloned().collect::<Vec<_>>()
    );
    assert!(
        compile_to_bundle(&protobuf, Some(WEIGHTS))
            .unwrap_err()
            .to_string()
            .contains("multiple")
    );
    let buffered = scratch.0.join("buffered");
    compile_to_bundle_with_weight_files(&protobuf, &files)
        .unwrap()
        .write_to_dir(&buffered)
        .unwrap();
    let streamed = scratch.0.join("streamed");
    compile_to_dir_with_weight_files(&protobuf, &files, &streamed).unwrap();
    for path in [
        "model.mil",
        "coremldata.bin",
        "metadata.json",
        "analytics/coremldata.bin",
    ]
    .into_iter()
    .chain(files.keys().map(String::as_str))
    {
        assert_eq!(
            fs::read(buffered.join(path)).unwrap(),
            fs::read(streamed.join(path)).unwrap(),
            "{path}"
        );
    }
    for (path, bytes) in &files {
        assert_eq!(fs::read(streamed.join(path)).unwrap(), *bytes);
        assert!(
            fs::read_to_string(streamed.join("model.mil"))
                .unwrap()
                .contains(&format!("@model_path/{path}\"), offset = uint64(64)"))
        );
    }
    let mut missing = files.clone();
    missing.remove("weights/layer 1/weight.bin");
    let output = scratch.0.join("incomplete");
    assert!(compile_to_dir_with_weight_files(&protobuf, &missing, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn invalid_paths_and_out_of_file_offsets_are_rejected_before_writing() {
    let scratch = Scratch::new();
    for path in [
        "/tmp/weights.bin",
        "@model_path/../weights.bin",
        "@model_path/weights/../x",
        "@model_path/weights//x",
        "@model_path/weights/./x",
        "@model_path/C:/x",
        "@model_path/weights\\x",
        "@model_path/weights/x\"y",
        "@model_path/model.mil",
        "@model_path/analytics/data",
        "@model_path/weights/x\ny",
    ] {
        let protobuf = model(&[path]);
        assert!(referenced_weight_paths(&protobuf).is_err(), "{path}");
        assert!(
            compile_to_bundle(&protobuf, Some(WEIGHTS)).is_err(),
            "{path}"
        );
        let output = scratch.0.join("invalid");
        assert!(
            compile_to_dir(&protobuf, Some(WEIGHTS), &output).is_err(),
            "{path}"
        );
        assert!(!output.exists());
    }
    let protobuf = model(&["@model_path/weights/weight.bin"]);
    assert!(
        compile_to_bundle(&protobuf, Some(&WEIGHTS[..64]))
            .unwrap_err()
            .to_string()
            .contains("offset 64")
    );
}

#[test]
fn repeated_references_need_only_one_file() {
    let protobuf = model(&[
        "@model_path/weights/weight.bin",
        "@model_path/weights/weight.bin",
    ]);
    assert_eq!(
        referenced_weight_paths(&protobuf).unwrap(),
        ["weights/weight.bin"]
    );
    compile_to_bundle(&protobuf, Some(WEIGHTS)).unwrap();
}

#[cfg(feature = "cli")]
#[test]
fn cli_discovers_all_referenced_package_and_raw_model_assets() {
    let scratch = Scratch::new();
    let paths = ["weights/weight.bin", "weights/layer/second.bin"];
    let protobuf = model(
        &paths
            .map(|p| format!("@model_path/{p}"))
            .each_ref()
            .map(String::as_str),
    );
    let package = scratch.0.join("input.mlpackage");
    let root = package.join("Data/com.apple.CoreML");
    fs::create_dir_all(root.join("weights/layer")).unwrap();
    fs::write(root.join("model.mlmodel"), &protobuf).unwrap();
    for path in paths {
        fs::write(root.join(path), WEIGHTS).unwrap();
    }
    for (name, input) in [("package", package), ("raw", root.join("model.mlmodel"))] {
        let output = scratch.0.join(name);
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_mlmodelc-export"))
            .arg(input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        for path in paths {
            assert_eq!(fs::read(output.join(path)).unwrap(), WEIGHTS);
        }
    }
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_mlmodelc-export"))
        .current_dir(&root)
        .arg("model.mlmodel")
        .arg(scratch.0.join("relative"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::remove_file(root.join(paths[1])).unwrap();
    let output = scratch.0.join("missing");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_mlmodelc-export"))
        .arg(root.join("model.mlmodel"))
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains(paths[1]));
    assert!(!output.exists());
}

#[test]
fn conflicting_file_and_directory_names_fail_before_writing() {
    let scratch = Scratch::new();
    let protobuf = model(&["@model_path/weights/data", "@model_path/weights/data/part"]);
    let files = BTreeMap::from([
        ("weights/data".into(), WEIGHTS.to_vec()),
        ("weights/data/part".into(), WEIGHTS.to_vec()),
    ]);
    let output = scratch.0.join("invalid");
    assert!(compile_to_bundle_with_weight_files(&protobuf, &files).is_err());
    assert!(compile_to_dir_with_weight_files(&protobuf, &files, &output).is_err());
    assert!(!output.exists());
}

#[cfg(unix)]
#[test]
fn output_symlinks_are_rejected_without_overwriting_other_files() {
    let scratch = Scratch::new();
    let outside = scratch.0.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("weight.bin"), b"unchanged").unwrap();
    let protobuf = model(&["@model_path/weights/weight.bin"]);
    for name in ["streamed", "buffered"] {
        let output = scratch.0.join(name);
        fs::create_dir(&output).unwrap();
        std::os::unix::fs::symlink(&outside, output.join("weights")).unwrap();
        if name == "streamed" {
            assert!(compile_to_dir(&protobuf, Some(WEIGHTS), &output).is_err());
        } else {
            assert!(
                compile_to_bundle(&protobuf, Some(WEIGHTS))
                    .unwrap()
                    .write_to_dir(&output)
                    .is_err()
            );
        }
        assert!(!output.join("model.mil").exists());
    }
    assert_eq!(fs::read(outside.join("weight.bin")).unwrap(), b"unchanged");
}

#[cfg(all(unix, feature = "cli"))]
#[test]
fn cli_does_not_follow_weight_symlinks_outside_the_model_directory() {
    let scratch = Scratch::new();
    let root = scratch.0.join("source");
    fs::create_dir_all(root.join("weights")).unwrap();
    fs::write(
        root.join("model.mlmodel"),
        model(&["@model_path/weights/weight.bin"]),
    )
    .unwrap();
    fs::write(scratch.0.join("outside.bin"), WEIGHTS).unwrap();
    std::os::unix::fs::symlink(
        scratch.0.join("outside.bin"),
        root.join("weights/weight.bin"),
    )
    .unwrap();
    let output = scratch.0.join("output");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_mlmodelc-export"))
        .arg(root.join("model.mlmodel"))
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("escapes model directory"));
    assert!(!output.exists());
}
