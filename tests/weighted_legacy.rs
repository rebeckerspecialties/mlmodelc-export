use mlmodelc_export::{MILDimension, compile_to_bundle, decode};

const MODEL: &[u8] = include_bytes!("fixtures/flexible-weighted-legacy/input.mlmodel");
const WEIGHTS: &[u8] = include_bytes!("fixtures/flexible-weighted-legacy/weights/weights.bin");

#[test]
fn weighted_legacy_model_preserves_description_and_flexible_shape() {
    let program = decode(MODEL).unwrap();
    assert_eq!(program.functions.len(), 1);
    let (name, function) = &program.functions[0];
    assert_eq!(name, "main");
    assert_eq!(
        function.inputs[0].r#type.shape,
        [MILDimension::Unknown, MILDimension::Constant(4)]
    );
    let blob = function.block.operations[0].attributes[0]
        .1
        .blob
        .as_ref()
        .unwrap();
    assert_eq!(blob.filename, "@model_path/weights/weights.bin");
    assert_eq!(blob.offset, 64);

    let bundle = compile_to_bundle(MODEL, Some(WEIGHTS)).unwrap();
    assert_eq!(bundle.weights_bin.as_deref(), Some(WEIGHTS));
    let metadata: serde_json::Value = serde_json::from_slice(&bundle.metadata_json).unwrap();
    assert!(metadata[0].get("functions").is_none());
    assert!(metadata[0].get("defaultFunctionName").is_none());
    assert_eq!(metadata[0]["inputSchema"][0]["shape"], "[1, 4]");
    assert_eq!(
        metadata[0]["inputSchema"][0]["shapeRange"],
        "[[1, 4], [4, 4]]"
    );
    let description_length =
        u64::from_le_bytes(bundle.coremldata_bin[75..83].try_into().unwrap()) as usize;
    assert_eq!(description_length, program.description_data.len() + 3);
    assert_eq!(
        &bundle.coremldata_bin[83..83 + program.description_data.len()],
        &program.description_data
    );
    assert_eq!(
        &bundle.coremldata_bin[83 + program.description_data.len()..83 + description_length],
        &[0xa2, 6, 0]
    );
}

#[cfg(feature = "cli")]
mod cli {
    use super::*;
    use std::{fs, path::PathBuf, process::Command};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let directory = std::env::temp_dir().join(format!(
                "mlmodelc-weighted-{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&directory).unwrap();
            Self(directory)
        }

        fn package(&self) -> PathBuf {
            let package = self.0.join("source.mlpackage");
            let data = package.join("Data/com.apple.CoreML");
            fs::create_dir_all(data.join("weights")).unwrap();
            fs::write(data.join("model.mlmodel"), MODEL).unwrap();
            fs::write(data.join("weights/weights.bin"), WEIGHTS).unwrap();
            package
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn explicit_and_package_weights_produce_the_same_bundle() {
        let scratch = Scratch::new("cli");
        let package = scratch.package();
        let raw = package.join("Data/com.apple.CoreML/model.mlmodel");
        let weights = package.join("Data/com.apple.CoreML/weights/weights.bin");
        let expected = compile_to_bundle(MODEL, Some(WEIGHTS)).unwrap();
        for (name, input) in [
            ("explicit", &raw),
            ("detected", &package),
            ("override", &package),
        ] {
            let output = scratch.0.join(name);
            let mut command = Command::new(env!("CARGO_BIN_EXE_mlmodelc-export"));
            command.arg(input).arg(&output);
            if name != "detected" {
                command.arg("--weights").arg(&weights);
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            for (relative, bytes) in [
                ("model.mil", expected.model_mil.as_slice()),
                ("coremldata.bin", expected.coremldata_bin.as_slice()),
                ("metadata.json", expected.metadata_json.as_slice()),
                (
                    "analytics/coremldata.bin",
                    expected.analytics_coremldata_bin.as_slice(),
                ),
                ("weights/weights.bin", WEIGHTS),
            ] {
                assert_eq!(
                    fs::read(output.join(relative)).unwrap(),
                    bytes,
                    "{name}: {relative}"
                );
            }
        }
    }

    #[test]
    fn missing_explicit_weights_fail_before_writing_a_bundle() {
        let scratch = Scratch::new("missing");
        let package = scratch.package();
        let raw = package.join("Data/com.apple.CoreML/model.mlmodel");
        for (name, input) in [("raw", &raw), ("package", &package)] {
            let output = scratch.0.join(name);
            let result = Command::new(env!("CARGO_BIN_EXE_mlmodelc-export"))
                .arg(input)
                .arg(&output)
                .arg("--weights")
                .arg(scratch.0.join("missing.bin"))
                .output()
                .unwrap();
            assert!(
                !result.status.success(),
                "{name} accepted missing explicit weights"
            );
            assert!(!output.exists(), "{name} wrote an incomplete bundle");
        }
    }
}
