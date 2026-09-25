//! Directory-comparison harness for committed fixtures.
//!
//! Walks every subdirectory under `tests/fixtures/`, compiles the
//! `input.mlmodel`, and checks two things:
//!
//! 1. **Positive**: our output's `model.mil` and `coremldata.bin` are
//!    byte-identical to `expected-macos.mlmodelc/` (modulo the buildInfo
//!    component string in `model.mil`, which legitimately identifies a
//!    different producer). `metadata.json` is compared structurally, excluding
//!    generated class names and optimization statistics: those describe Apple's
//!    compiler passes rather than the source MIL graph. I/O schemas, defaults,
//!    constraints and function selection must match.
//! 2. **Negative**: our output does NOT match the broken pattern in
//!    `observed-watchos-broken.mlmodelc/`. Specifically:
//!    - we always emit `model.mil` (the watchOS stub doesn't),
//!    - our `coremldata.bin` is the full 165+ bytes (the watchOS stub
//!      truncates it), and
//!    - we always emit `metadata.json` (the watchOS stub doesn't).

use std::fs;
use std::path::{Path, PathBuf};

use mlmodelc_export::{compile_to_bundle, compile_to_dir};

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn discover_fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(fixtures_root()).expect("fixtures directory") {
        let path = entry.expect("readable entry").path();
        if path.is_dir() && path.join("input.mlmodel").exists() {
            out.push(path);
        }
    }
    out.sort();
    out
}

#[test]
fn fixtures_round_trip() {
    let fixtures = discover_fixtures();
    assert!(
        !fixtures.is_empty(),
        "no fixtures discovered under tests/fixtures/"
    );
    let mut failures: Vec<String> = Vec::new();
    for fx in fixtures {
        let name = fx.file_name().unwrap().to_string_lossy().to_string();
        if let Err(e) = check_fixture(&fx) {
            failures.push(format!("{name}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "fixture comparison failures:\n  {}",
        failures.join("\n  ")
    );
}

fn check_fixture(dir: &Path) -> Result<(), String> {
    let input = dir.join("input.mlmodel");
    let model_bytes = fs::read(&input).map_err(|e| format!("read input: {e}"))?;
    let weights = match fs::read(dir.join("weights/weights.bin")) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("read weights: {error}")),
    };

    let tmp = std::env::temp_dir().join(format!(
        "mlmodelc-export-test-{}-{}",
        std::process::id(),
        dir.file_name().unwrap().to_string_lossy()
    ));
    fs::create_dir(&tmp).map_err(|e| format!("create temporary directory: {e}"))?;
    compile_to_dir(&model_bytes, weights.as_deref(), &tmp)
        .map_err(|e| format!("compile: {e:?}"))?;
    let buffered = compile_to_bundle(&model_bytes, weights.as_deref())
        .map_err(|e| format!("buffered compile: {e:?}"))?;
    for (name, bytes) in [
        ("model.mil", &buffered.model_mil),
        ("coremldata.bin", &buffered.coremldata_bin),
        ("metadata.json", &buffered.metadata_json),
        (
            "analytics/coremldata.bin",
            &buffered.analytics_coremldata_bin,
        ),
    ] {
        let streamed = fs::read(tmp.join(name)).map_err(|e| format!("read {name}: {e}"))?;
        if &streamed != bytes {
            return Err(format!("streaming/buffered output differs: {name}"));
        }
    }
    if buffered.weights_bin != weights {
        return Err("buffered weights differ from input".into());
    }

    // === Positive: matches expected-macos ===
    let expected = dir.join("expected-macos.mlmodelc");
    compare_model_mil(&expected.join("model.mil"), &tmp.join("model.mil"))?;
    compare_bytes_exact(
        &expected.join("coremldata.bin"),
        &tmp.join("coremldata.bin"),
    )?;
    compare_metadata_json(&expected.join("metadata.json"), &tmp.join("metadata.json"))?;
    if weights.is_some() {
        compare_bytes_exact(
            &dir.join("weights/weights.bin"),
            &tmp.join("weights/weights.bin"),
        )?;
        compare_bytes_exact(
            &expected.join("weights/weights.bin"),
            &tmp.join("weights/weights.bin"),
        )?;
    } else if tmp.join("weights").exists() {
        return Err("unexpected weights for an inline-only fixture".into());
    }
    // analytics/coremldata.bin is structurally tolerated:
    // - coremlc emits a full record (NeuralNetworkModelDetails + Specification-
    //   Details with modelHash + modelName tied to the input filename)
    // - we emit a minimal stub (NeuralNetworkModelDetails only)
    // Both are accepted by `MLModel(contentsOfURL:)`. We only require that
    // the file exists and starts with the expected `NeuralNetworkModelDetails`
    // header.
    require_analytics_present(&tmp)?;

    // === Negative: must not match observed-watchos-broken ===
    let broken = dir.join("observed-watchos-broken.mlmodelc");
    if broken.exists() {
        check_not_watchos_broken(&broken, &tmp)?;
    }

    let _ = fs::remove_dir_all(&tmp);
    Ok(())
}

/// Compare `model.mil` byte-for-byte except for the `buildInfo` line, which
/// legitimately identifies a different producer.
fn compare_model_mil(expected: &Path, actual: &Path) -> Result<(), String> {
    let exp = fs::read_to_string(expected).map_err(|e| format!("read {expected:?}: {e}"))?;
    let act = fs::read_to_string(actual).map_err(|e| format!("read {actual:?}: {e}"))?;
    let exp_lines: Vec<&str> = exp.lines().collect();
    let act_lines: Vec<&str> = act.lines().collect();
    if exp_lines.len() != act_lines.len() {
        return Err(format!(
            "model.mil line-count mismatch: expected {}, got {}\n--- expected ---\n{exp}\n--- actual ---\n{act}",
            exp_lines.len(),
            act_lines.len(),
        ));
    }
    for (i, (e, a)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
        if e == a {
            continue;
        }
        if e.starts_with("[buildInfo") && a.starts_with("[buildInfo") {
            continue;
        }
        return Err(format!(
            "model.mil line {} differs:\n  expected: {e}\n  actual:   {a}",
            i + 1
        ));
    }
    Ok(())
}

fn compare_bytes_exact(expected: &Path, actual: &Path) -> Result<(), String> {
    let e = fs::read(expected).map_err(|err| format!("read {expected:?}: {err}"))?;
    let a = fs::read(actual).map_err(|err| format!("read {actual:?}: {err}"))?;
    if e == a {
        return Ok(());
    }
    let first_diff = e
        .iter()
        .zip(a.iter())
        .position(|(x, y)| x != y)
        .unwrap_or(e.len().min(a.len()));
    Err(format!(
        "{} bytes differ: expected {} bytes, got {} bytes; first divergence at offset {}",
        expected.file_name().unwrap().to_string_lossy(),
        e.len(),
        a.len(),
        first_diff
    ))
}

/// Compare JSON structure, including array order and field ownership.
fn compare_metadata_json(expected: &Path, actual: &Path) -> Result<(), String> {
    let e = fs::read_to_string(expected).map_err(|err| format!("read {expected:?}: {err}"))?;
    let a = fs::read_to_string(actual).map_err(|err| format!("read {actual:?}: {err}"))?;
    let mut e: serde_json::Value = serde_json::from_str(&e).map_err(|e| e.to_string())?;
    let mut a: serde_json::Value = serde_json::from_str(&a).map_err(|e| e.to_string())?;
    normalize_metadata(&mut e);
    normalize_metadata(&mut a);
    if e == a {
        return Ok(());
    }
    Err(format!(
        "metadata.json differs:\n--- expected ---\n{e:#}\n--- actual ---\n{a:#}",
    ))
}

fn normalize_metadata(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            for key in [
                "generatedClassName",
                "computePrecision",
                "storagePrecision",
                "mlProgramOperationTypeHistogram",
            ] {
                object.remove(key);
            }
            for child in object.values_mut() {
                normalize_metadata(child);
            }
        }
        serde_json::Value::Array(array) => array.iter_mut().for_each(normalize_metadata),
        _ => {}
    }
}

fn require_analytics_present(ours: &Path) -> Result<(), String> {
    let path = ours.join("analytics").join("coremldata.bin");
    let bytes = fs::read(&path).map_err(|e| format!("read {path:?}: {e}"))?;
    // First 8 bytes are a little-endian uint64 prefix (length of the header
    // string), then "NeuralNetworkModelDetails".
    let header = b"NeuralNetworkModelDetails";
    if !bytes.windows(header.len()).any(|w| w == header) {
        return Err(
            "analytics/coremldata.bin is missing the NeuralNetworkModelDetails header".to_string(),
        );
    }
    Ok(())
}

/// Verify our compile output does NOT exhibit the watchOS-stub failure
/// pattern: missing model.mil, missing metadata.json, truncated coremldata.bin.
fn check_not_watchos_broken(broken: &Path, ours: &Path) -> Result<(), String> {
    if !ours.join("model.mil").exists() {
        return Err(
            "our output is missing model.mil — same defect as the watchOS stub".to_string(),
        );
    }
    if !ours.join("metadata.json").exists() {
        return Err(
            "our output is missing metadata.json — same defect as the watchOS stub".to_string(),
        );
    }
    let our_coreml = fs::read(ours.join("coremldata.bin")).map_err(|e| e.to_string())?;
    let broken_coreml = fs::read(broken.join("coremldata.bin")).map_err(|e| e.to_string())?;
    if our_coreml == broken_coreml {
        return Err(format!(
            "our coremldata.bin matches the watchOS-broken reference ({} bytes) — the loader trailer is missing",
            our_coreml.len()
        ));
    }
    if our_coreml.len() <= broken_coreml.len() {
        return Err(format!(
            "our coremldata.bin ({} bytes) is no larger than the truncated watchOS-broken one ({} bytes); trailer likely missing",
            our_coreml.len(),
            broken_coreml.len()
        ));
    }
    Ok(())
}
