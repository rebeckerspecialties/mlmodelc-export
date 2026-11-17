//! `metadata.json` statistics against the native goldens.
//!
//! golden.rs leaves `storagePrecision`, `computePrecision` and the operation
//! histogram out of its comparison because `coremlc` derives them from its
//! optimized program. For the committed fixtures the source graph gives the
//! same numbers, with the two histogram exceptions listed below.

use std::fs;
use std::path::Path;

use mlmodelc_export::compile_to_bundle;

/// Fixtures whose native histogram is not a function of the source graph.
const HISTOGRAM_EXCEPTIONS: [&str; 2] = [
    // Captured with coremlc 3520.5.1, which reports the CoreML7 relu as `Ios16.relu`.
    "flexible-range",
    // `constexpr_affine_dequantize` is reported under the opset that introduced it (`Ios16`).
    "quantized-constexpr",
];

#[test]
fn source_statistics_match_native_metadata() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut checked = 0;
    for entry in fs::read_dir(&root).unwrap() {
        let directory = entry.unwrap().path();
        let Ok(model) = fs::read(directory.join("input.mlmodel")) else {
            continue;
        };
        let name = directory.file_name().unwrap().to_str().unwrap().to_owned();
        let weights = fs::read(directory.join("weights/weights.bin")).ok();
        let bundle = compile_to_bundle(&model, weights.as_deref()).unwrap();
        let ours: serde_json::Value = serde_json::from_slice(&bundle.metadata_json).unwrap();
        let native = fs::read(directory.join("expected-macos.mlmodelc/metadata.json")).unwrap();
        let native: serde_json::Value = serde_json::from_slice(&native).unwrap();
        for key in ["storagePrecision", "computePrecision"] {
            assert_eq!(ours[0].get(key), native[0].get(key), "{name}: {key}");
        }
        let key = "mlProgramOperationTypeHistogram";
        if HISTOGRAM_EXCEPTIONS.contains(&name.as_str()) {
            assert_ne!(ours[0][key], native[0][key], "{name}: stale exception");
        } else {
            assert_eq!(ours[0][key], native[0][key], "{name}: {key}");
        }
        checked += 1;
    }
    assert!(checked >= 12, "only {checked} fixtures were found");
}
