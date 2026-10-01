use std::fs;
use std::path::PathBuf;

use mlmodelc_export::{
    compile_to_bundle, compile_to_bundle_with_weight_files, compile_to_dir,
    compile_to_dir_with_weight_files, referenced_weight_paths,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/pipeline-fixtures")
        .join(name)
}

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("mlmodelc-pipeline-{}-{name}", std::process::id()))
}

#[test]
fn cast_pipeline_matches_native_wrapper_and_preserves_half_boundary() {
    let fx = fixture("cast");
    let data = fs::read(fx.join("input.mlmodel")).unwrap();
    let dir = temporary("cast");
    let stats = compile_to_dir(&data, None, &dir).unwrap();
    assert_eq!(stats.operation_count, 2);
    assert_eq!(
        fs::read(dir.join("coremldata.bin")).unwrap(),
        fs::read(fx.join("expected.mlmodelc/coremldata.bin")).unwrap()
    );
    assert_eq!(
        fs::read(dir.join("modelNames/coremldata.bin")).unwrap(),
        fs::read(fx.join("expected.mlmodelc/modelNames/coremldata.bin")).unwrap()
    );
    assert!(!dir.join("model.mil").exists());
    assert!(
        fs::read_to_string(dir.join("model0/model.mil"))
            .unwrap()
            .contains("tensor<fp16, [4]> rounded = cast")
    );
    assert!(
        fs::read_to_string(dir.join("model1/model.mil"))
            .unwrap()
            .contains("tensor<fp16, [4]> rounded")
    );
    let mut actual: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    let mut expected: serde_json::Value =
        serde_json::from_slice(&fs::read(fx.join("expected.mlmodelc/metadata.json")).unwrap())
            .unwrap();
    actual[0]
        .as_object_mut()
        .unwrap()
        .remove("generatedClassName");
    expected[0]
        .as_object_mut()
        .unwrap()
        .remove("generatedClassName");
    assert_eq!(actual, expected);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn weighted_pipeline_retains_one_shared_asset() {
    let fx = fixture("weighted");
    let data = fs::read(fx.join("input.mlmodel")).unwrap();
    let weights = fs::read(fx.join("weights.bin")).unwrap();
    assert_eq!(
        referenced_weight_paths(&data).unwrap(),
        ["weights/weights.bin"]
    );
    assert!(
        compile_to_bundle(&data, None)
            .unwrap_err()
            .to_string()
            .contains("missing external weight")
    );
    let bundle = compile_to_bundle(&data, Some(&weights)).unwrap();
    let dir = temporary("weighted");
    bundle.write_to_dir(&dir).unwrap();
    assert_eq!(fs::read(dir.join("weights/weights.bin")).unwrap(), weights);
    for child in ["model0", "model1"] {
        assert!(!dir.join(child).join("weights/weights.bin").exists());
        let mil = fs::read_to_string(dir.join(child).join("model.mil")).unwrap();
        assert!(mil.contains("@model_path/../weights/weights.bin"));
        assert!(!mil.contains("@model_path/weights/weights.bin"));
    }
    assert_eq!(
        fs::read(dir.join("coremldata.bin")).unwrap(),
        fs::read(fx.join("expected.mlmodelc/coremldata.bin")).unwrap()
    );
    let mut actual: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    let mut expected: serde_json::Value =
        serde_json::from_slice(&fs::read(fx.join("expected.mlmodelc/metadata.json")).unwrap())
            .unwrap();
    actual[0]
        .as_object_mut()
        .unwrap()
        .remove("generatedClassName");
    expected[0]
        .as_object_mut()
        .unwrap()
        .remove("generatedClassName");
    assert_eq!(actual, expected);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pipeline_named_assets_resolve_per_child_without_whole_file_clones() {
    let fx = fixture("weighted");
    let mut data = fs::read(fx.join("input.mlmodel")).unwrap();
    let old = b"@model_path/weights/weights.bin";
    let new = b"@model_path/weights/another.bin";
    let at = data
        .windows(old.len())
        .position(|bytes| bytes == old)
        .unwrap();
    data[at..at + old.len()].copy_from_slice(new);
    let weights = fs::read(fx.join("weights.bin")).unwrap();
    assert!(
        compile_to_bundle(&data, Some(&weights))
            .unwrap_err()
            .to_string()
            .contains("multiple")
    );
    let mut files = std::collections::BTreeMap::from([
        ("weights/weights.bin".to_owned(), weights.clone()),
        ("weights/another.bin".to_owned(), weights.clone()),
    ]);
    let buffered = compile_to_bundle_with_weight_files(&data, &files).unwrap();
    let dir = temporary("multiple");
    compile_to_dir_with_weight_files(&data, &files, &dir).unwrap();
    assert_eq!(buffered.pipeline.as_ref().unwrap().models.len(), 2);
    assert_eq!(fs::read(dir.join("weights/another.bin")).unwrap(), weights);
    assert_eq!(fs::read(dir.join("weights/weights.bin")).unwrap(), weights);
    assert!(!dir.join("model0/weights/weights.bin").exists());
    assert!(!dir.join("model1/weights/another.bin").exists());
    assert!(
        fs::read_to_string(dir.join("model0/model.mil"))
            .unwrap()
            .contains("@model_path/../weights/another.bin")
    );
    assert!(
        fs::read_to_string(dir.join("model1/model.mil"))
            .unwrap()
            .contains("@model_path/../weights/weights.bin")
    );
    files.remove("weights/another.bin");
    assert!(
        compile_to_bundle_with_weight_files(&data, &files)
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    files.insert("model0/conflict.bin".into(), vec![0]);
    assert!(
        compile_to_bundle_with_weight_files(&data, &files)
            .unwrap_err()
            .to_string()
            .contains("collides")
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pipeline_rejects_untrusted_parent_paths_before_internal_relocation() {
    let fx = fixture("weighted");
    let mut data = fs::read(fx.join("input.mlmodel")).unwrap();
    let old = b"@model_path/weights/weights.bin";
    let new = b"@model_path/../evil/weights.bin";
    assert_eq!(old.len(), new.len());
    let offset = data
        .windows(old.len())
        .position(|bytes| bytes == old)
        .unwrap();
    data[offset..offset + old.len()].copy_from_slice(new);
    let weights = fs::read(fx.join("weights.bin")).unwrap();
    let dir = temporary("untrusted-parent");
    assert!(
        referenced_weight_paths(&data)
            .unwrap_err()
            .to_string()
            .contains("invalid external weight path")
    );
    assert!(
        compile_to_bundle(&data, Some(&weights))
            .unwrap_err()
            .to_string()
            .contains("invalid external weight path")
    );
    assert!(
        compile_to_dir(&data, Some(&weights), &dir)
            .unwrap_err()
            .to_string()
            .contains("invalid external weight path")
    );
    assert!(!dir.exists());
}

#[test]
fn pipeline_reexport_removes_only_obsolete_child_assets() {
    let fx = fixture("weighted");
    let data = fs::read(fx.join("input.mlmodel")).unwrap();
    let weights = fs::read(fx.join("weights.bin")).unwrap();
    for buffered in [false, true] {
        let dir = temporary(if buffered {
            "old-buffered"
        } else {
            "old-streaming"
        });
        for child in ["model0", "model1"] {
            fs::create_dir_all(dir.join(child).join("weights")).unwrap();
            fs::write(dir.join(child).join("weights/weights.bin"), &weights).unwrap();
            fs::write(dir.join(child).join("weights/unrelated.bin"), b"preserve").unwrap();
        }
        if buffered {
            compile_to_bundle(&data, Some(&weights))
                .unwrap()
                .write_to_dir(&dir)
                .unwrap();
        } else {
            compile_to_dir(&data, Some(&weights), &dir).unwrap();
        }
        assert_eq!(fs::read(dir.join("weights/weights.bin")).unwrap(), weights);
        for child in ["model0", "model1"] {
            assert!(!dir.join(child).join("weights/weights.bin").exists());
            assert_eq!(
                fs::read(dir.join(child).join("weights/unrelated.bin")).unwrap(),
                b"preserve"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
#[cfg(unix)]
fn pipeline_rejects_child_output_symlinks_before_writing_payloads() {
    use std::os::unix::fs::symlink;
    let data = fs::read(fixture("cast").join("input.mlmodel")).unwrap();
    let dir = temporary("symlinks");
    fs::create_dir_all(&dir).unwrap();
    let outside = temporary("outside");
    fs::create_dir_all(&outside).unwrap();
    symlink(&outside, dir.join("model0")).unwrap();
    assert!(
        compile_to_dir(&data, None, &dir)
            .unwrap_err()
            .to_string()
            .contains("symlink")
    );
    assert!(!dir.join("coremldata.bin").exists());
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(outside).unwrap();
}

fn varint(mut n: u64) -> Vec<u8> {
    let mut result = Vec::new();
    while n >= 128 {
        result.push((n as u8) | 128);
        n >>= 7;
    }
    result.push(n as u8);
    result
}

fn field(tag: u32, data: &[u8]) -> Vec<u8> {
    let mut result = varint((u64::from(tag) << 3) | 2);
    result.extend(varint(data.len() as u64));
    result.extend(data);
    result
}

fn pipeline(children: &[Vec<u8>]) -> Vec<u8> {
    let mut body = Vec::new();
    for child in children {
        body.extend(field(1, child));
    }
    let mut model = vec![8, 9];
    if let Some(child) = children.first() {
        let mut offset = 0;
        while offset < child.len() {
            let tag = read_varint(child, &mut offset);
            match tag & 7 {
                0 => {
                    read_varint(child, &mut offset);
                }
                2 => {
                    let length = read_varint(child, &mut offset) as usize;
                    let bytes = &child[offset..offset + length];
                    if tag >> 3 == 2 {
                        model.extend(field(2, bytes));
                        break;
                    }
                    offset += length;
                }
                _ => panic!("unexpected fixture wire type"),
            }
        }
    }
    model.extend(field(202, &body));
    model
}

fn read_varint(data: &[u8], offset: &mut usize) -> u64 {
    let mut n = 0;
    let mut shift = 0;
    loop {
        let byte = data[*offset];
        *offset += 1;
        n |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return n;
        }
        shift += 7;
    }
}

fn message_fields(data: &[u8], selected: u64) -> Vec<&[u8]> {
    let mut offset = 0;
    let mut result = Vec::new();
    while offset < data.len() {
        let tag = read_varint(data, &mut offset);
        match tag & 7 {
            0 => {
                read_varint(data, &mut offset);
            }
            2 => {
                let length = read_varint(data, &mut offset) as usize;
                if tag >> 3 == selected {
                    result.push(&data[offset..offset + length]);
                }
                offset += length;
            }
            _ => panic!("unexpected fixture wire type"),
        }
    }
    result
}

fn producer_metadata(entries: &[(&str, &str)]) -> Vec<u8> {
    let entries = entries
        .iter()
        .flat_map(|(key, value)| {
            field(
                100,
                &[field(1, key.as_bytes()), field(2, value.as_bytes())].concat(),
            )
        })
        .collect::<Vec<_>>();
    field(100, &entries)
}

#[test]
fn pipeline_preserves_wrapper_and_child_producer_metadata_independently() {
    let source = fs::read(fixture("cast").join("input.mlmodel")).unwrap();
    let original_description = message_fields(&source, 2)[0];
    let children = message_fields(message_fields(&source, 202)[0], 1);
    let mut child = children[0].to_vec();
    let mut child_description = message_fields(&child, 2)[0].to_vec();
    child_description.extend(producer_metadata(&[(
        "child.only",
        "do not lift to wrapper",
    )]));
    child.extend(field(2, &child_description));
    let body = [field(1, &child), field(1, children[1])].concat();
    let mut description = original_description.to_vec();
    description.extend(producer_metadata(&[
        ("rustnn.coreml.name_encoding", "hex-v1"),
        ("rustnn.coreml.input_views", "{\"cache\":[1,4]}"),
        ("quoted\"key", "line1\nline2\\\t\u{1} café"),
        ("duplicate", "old"),
        ("duplicate", "new"),
    ]));
    let data = [vec![8, 9], field(2, &description), field(202, &body)].concat();
    let buffered = compile_to_bundle(&data, None).unwrap();
    let wrapper: serde_json::Value = serde_json::from_slice(&buffered.metadata_json).unwrap();
    let user = &wrapper[0]["userDefinedMetadata"];
    assert_eq!(user["rustnn.coreml.name_encoding"], "hex-v1");
    assert_eq!(user["rustnn.coreml.input_views"], "{\"cache\":[1,4]}");
    assert_eq!(user["quoted\"key"], "line1\nline2\\\t\u{1} café");
    assert_eq!(user["duplicate"], "new");
    assert_eq!(user.as_object().unwrap().len(), 4);

    let models = &buffered.pipeline.as_ref().unwrap().models;
    let first: serde_json::Value = serde_json::from_slice(&models[0].0.metadata_json).unwrap();
    assert_eq!(
        first[0]["userDefinedMetadata"]["child.only"],
        "do not lift to wrapper"
    );
    assert_eq!(
        first[0]["userDefinedMetadata"].as_object().unwrap().len(),
        1
    );
    let second: serde_json::Value = serde_json::from_slice(&models[1].0.metadata_json).unwrap();
    assert_eq!(second[0]["userDefinedMetadata"], serde_json::json!({}));

    let binary = &buffered.coremldata_bin;
    let target_length = u64::from_le_bytes(binary[28..36].try_into().unwrap()) as usize;
    let length_offset = 68 + target_length;
    let length =
        u64::from_le_bytes(binary[length_offset..length_offset + 8].try_into().unwrap()) as usize;
    assert_eq!(
        &binary[length_offset + 8..length_offset + 8 + length],
        description
    );

    let dir = temporary("producer-metadata");
    compile_to_dir(&data, None, &dir).unwrap();
    assert_eq!(
        fs::read(dir.join("metadata.json")).unwrap(),
        buffered.metadata_json
    );
    assert_eq!(
        fs::read(dir.join("coremldata.bin")).unwrap(),
        buffered.coremldata_bin
    );
    for (index, (model, _)) in models.iter().enumerate() {
        assert_eq!(
            fs::read(dir.join(format!("model{index}/metadata.json"))).unwrap(),
            model.metadata_json
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nested_and_nonprogram_children_fail_before_writing() {
    let ordinary = fs::read(fixture("cast").join("input.mlmodel")).unwrap();
    for (data, expected) in [
        (pipeline(&[ordinary]), "nested"),
        (pipeline(&[vec![8, 9]]), "MLProgram"),
        (pipeline(&[]), "empty"),
    ] {
        let dir = temporary(expected);
        assert!(
            compile_to_dir(&data, None, &dir)
                .unwrap_err()
                .to_string()
                .contains(expected)
        );
        assert!(!dir.exists());
    }
}

#[test]
fn flexible_pipeline_keeps_source_bounds_and_unknown_mil_dimensions() {
    let fx = fixture("dynamic");
    let data = fs::read(fx.join("input.mlmodel")).unwrap();
    let bundle = compile_to_bundle(&data, None).unwrap();
    let dir = temporary("flexible");
    bundle.write_to_dir(&dir).unwrap();
    let mil = fs::read_to_string(dir.join("model0/model.mil")).unwrap();
    assert!(mil.contains("tensor<fp32, [?]>"));
    assert!(mil.contains("RangeDims"));
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("model0/metadata.json")).unwrap()).unwrap();
    assert_eq!(meta[0]["inputSchema"][0]["hasShapeFlexibility"], "1");
    let outer: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(outer[0]["inputSchema"][0]["hasShapeFlexibility"], "1");
    let bytes = fs::read(dir.join("coremldata.bin")).unwrap();
    assert_eq!(
        bytes,
        fs::read(fx.join("expected.mlmodelc/coremldata.bin")).unwrap()
    );
    let target = u64::from_le_bytes(bytes[28..36].try_into().unwrap()) as usize;
    let offset = 36 + target + 32;
    assert!(u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) > 255);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pipeline_proxy_scalar_and_dynamic_interfaces_match_native_metadata() {
    for name in [
        "masked",
        "masked-int32",
        "indexed",
        "scalar",
        "dynamic-boundary",
    ] {
        let fx = fixture(name);
        let data = fs::read(fx.join("input.mlmodel")).unwrap();
        let bundle = compile_to_bundle(&data, None).unwrap();
        assert_eq!(
            bundle.coremldata_bin,
            fs::read(fx.join("expected.mlmodelc/coremldata.bin")).unwrap(),
            "{name} wrapper"
        );
        assert_eq!(
            bundle.pipeline.as_ref().unwrap().model_names_bin,
            fs::read(fx.join("expected.mlmodelc/modelNames/coremldata.bin")).unwrap(),
            "{name} names"
        );
        let mut actual: serde_json::Value = serde_json::from_slice(&bundle.metadata_json).unwrap();
        let mut expected: serde_json::Value =
            serde_json::from_slice(&fs::read(fx.join("expected.mlmodelc/metadata.json")).unwrap())
                .unwrap();
        actual[0]
            .as_object_mut()
            .unwrap()
            .remove("generatedClassName");
        expected[0]
            .as_object_mut()
            .unwrap()
            .remove("generatedClassName");
        assert_eq!(actual, expected, "{name} metadata");
    }
}

#[test]
fn pipeline_streaming_and_buffered_bundles_match_every_file() {
    for name in [
        "cast",
        "weighted",
        "dynamic",
        "masked",
        "masked-int32",
        "scalar",
        "indexed",
        "dynamic-boundary",
    ] {
        let fx = fixture(name);
        let data = fs::read(fx.join("input.mlmodel")).unwrap();
        let weights = (name == "weighted").then(|| fs::read(fx.join("weights.bin")).unwrap());
        let buffered = temporary(&format!("parity-buffered-{name}"));
        let streaming = temporary(&format!("parity-streaming-{name}"));
        compile_to_bundle(&data, weights.as_deref())
            .unwrap()
            .write_to_dir(&buffered)
            .unwrap();
        compile_to_dir(&data, weights.as_deref(), &streaming).unwrap();
        let files = |root: &PathBuf| {
            walkdir::WalkDir::new(root)
                .into_iter()
                .map(Result::unwrap)
                .filter(|entry| entry.file_type().is_file())
                .map(|entry| {
                    (
                        entry.path().strip_prefix(root).unwrap().to_path_buf(),
                        fs::read(entry.path()).unwrap(),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(files(&buffered), files(&streaming), "{name}");
        fs::remove_dir_all(buffered).unwrap();
        fs::remove_dir_all(streaming).unwrap();
    }
}

#[test]
fn child_metadata_is_deterministic_when_precision_counts_tie() {
    let data = fs::read(fixture("indexed").join("input.mlmodel")).unwrap();
    let mut metadata = std::collections::BTreeSet::new();
    for _ in 0..64 {
        let bundle = compile_to_bundle(&data, None).unwrap();
        metadata.insert(
            bundle
                .pipeline
                .unwrap()
                .models
                .into_iter()
                .map(|(model, _)| model.metadata_json)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(
        metadata.len(),
        1,
        "repeated exports selected different tied precisions"
    );
}
