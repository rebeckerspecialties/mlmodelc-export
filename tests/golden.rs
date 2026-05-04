//! Directory-comparison harness for committed fixtures.
//!
//! Walks every subdirectory under `tests/fixtures/`, compiles the
//! `input.mlmodel`, and checks two things:
//!
//! 1. **Positive**: our output's `model.mil` and `coremldata.bin` are
//!    byte-identical to `expected-macos.mlmodelc/` (modulo the buildInfo
//!    component string in `model.mil`, which legitimately identifies a
//!    different producer). `metadata.json` is compared with whitespace and
//!    JSON-key-order tolerance (`semantic_json_eq`) — Apple's coremlc and
//!    our emitter both produce the same fields and values, but in slightly
//!    different declaration order.
//! 2. **Negative**: our output does NOT match the broken pattern in
//!    `observed-watchos-broken.mlmodelc/`. Specifically:
//!    - we always emit `model.mil` (the watchOS stub doesn't),
//!    - our `coremldata.bin` is the full 165+ bytes (the watchOS stub
//!      truncates it), and
//!    - we always emit `metadata.json` (the watchOS stub doesn't).

use std::fs;
use std::path::{Path, PathBuf};

use mlmodelc_export::compile_to_dir;

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

    let tmp = std::env::temp_dir().join(format!(
        "mlmodelc-export-test-{}",
        dir.file_name().unwrap().to_string_lossy()
    ));
    let _ = fs::remove_dir_all(&tmp);
    compile_to_dir(&model_bytes, None, &tmp).map_err(|e| format!("compile: {e:?}"))?;

    // === Positive: matches expected-macos ===
    let expected = dir.join("expected-macos.mlmodelc");
    compare_model_mil(&expected.join("model.mil"), &tmp.join("model.mil"))?;
    compare_bytes_exact(
        &expected.join("coremldata.bin"),
        &tmp.join("coremldata.bin"),
    )?;
    compare_metadata_json(&expected.join("metadata.json"), &tmp.join("metadata.json"))?;
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

/// Compare `metadata.json` with key-order tolerance: tokenise both files,
/// strip whitespace and quoted commas, and check the resulting key/value
/// multisets match. This accepts coremlc's declaration order or ours, since
/// both are accepted by `MLModel(contentsOfURL:)` in practice.
fn compare_metadata_json(expected: &Path, actual: &Path) -> Result<(), String> {
    let e = fs::read_to_string(expected).map_err(|err| format!("read {expected:?}: {err}"))?;
    let a = fs::read_to_string(actual).map_err(|err| format!("read {actual:?}: {err}"))?;
    let e_tokens = tokenize_metadata_json(&e);
    let a_tokens = tokenize_metadata_json(&a);
    if e_tokens == a_tokens {
        return Ok(());
    }
    Err(format!(
        "metadata.json semantic content differs (key-order tolerant):\n--- expected ({} tokens) ---\n{}\n--- actual ({} tokens) ---\n{}",
        e_tokens.len(),
        e_tokens
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join(" | "),
        a_tokens.len(),
        a_tokens
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join(" | "),
    ))
}

/// Cheap tokeniser: split on whitespace and structural punctuation, then
/// sort the resulting bag. Good enough for "do these two JSONs say the same
/// thing without caring about declaration order"; not a proper JSON parser.
///
/// `generatedClassName` is masked because Apple's `coremlc` defaults it to
/// the input filename stem (e.g. `"input"` for `input.mlmodel`), whereas
/// we always emit a stable `"model"`. Both are accepted by `MLModel`.
fn tokenize_metadata_json(s: &str) -> Vec<String> {
    let masked = mask_generated_class_name(s);
    let mut out: Vec<String> = masked
        .split(|c: char| c.is_whitespace() || c == ',')
        .map(|t| {
            t.trim_matches(|c: char| c == ',' || c.is_whitespace())
                .to_string()
        })
        .filter(|t| !t.is_empty())
        .collect();
    out.sort();
    out
}

fn mask_generated_class_name(s: &str) -> String {
    let key = "\"generatedClassName\" :";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(idx) = rest.find(key) {
        out.push_str(&rest[..idx]);
        out.push_str(key);
        rest = &rest[idx + key.len()..];
        if let Some(end) = rest.find(',').or_else(|| rest.find('\n')) {
            out.push_str(" \"<masked>\"");
            rest = &rest[end..];
        } else {
            out.push_str(rest);
            return out;
        }
    }
    out.push_str(rest);
    out
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
