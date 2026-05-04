//! Compile a CoreML MLProgram protobuf into a runnable `.mlmodelc` bundle in
//! pure Rust, without invoking Apple's `coremlc`.
//!
//! `coremlc` is the Apple tool that turns a `.mlmodel` (or `.mlpackage`)
//! protobuf into the directory layout that `MLModel(contentsOfURL:)` actually
//! loads at runtime. It exists only on macOS, only inside Xcode's toolchain,
//! and is not callable from sandboxed environments — watchOS apps, iOS App
//! Extensions, App Clips, tvOS sandbox, or any non-Apple host.
//!
//! This crate replaces `coremlc` for the MLProgram path with a pure-Rust
//! emitter. It accepts the protobuf bytes that you'd otherwise hand to
//! `coremlc` and produces the same four files Apple's compiler emits:
//!
//! ```text
//! example.mlmodelc/
//!   model.mil
//!   coremldata.bin
//!   metadata.json
//!   analytics/coremldata.bin
//!   weights/weights.bin   (optional, when the input has external weights)
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use mlmodelc_export::compile_to_bundle;
//!
//! let mlmodel_bytes = std::fs::read("model.mlmodel")?;
//! let bundle = compile_to_bundle(&mlmodel_bytes, None)?;
//! bundle.write_to_dir("model.mlmodelc")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Provenance
//!
//! Originated as a Swift package (`MILTextCompiler`) inside the
//! `metal-info-app` watchOS validation harness for [rustnn]. Ported to Rust
//! to align with rustnn's Rust-first architecture and so the output of
//! `CoremlMlProgramConverter` can be consumed end-to-end without leaving
//! the Cargo ecosystem.
//!
//! [rustnn]: https://github.com/rustnn/rustnn

mod bundle;
mod decoder;
mod emitter;
mod hex_float;
mod pb_reader;
mod sink;
mod types;

use std::fs;
use std::path::Path;

pub use bundle::{
    MlmodelcBundle, build_bundle, generate_analytics_bin, generate_coremldata_bin,
    generate_metadata_json,
};
pub use decoder::{MILDecoderError, decode};
pub use emitter::{emit, emit_to_string};
pub use hex_float::{hex_float16, hex_float32, hex_float32_bytes};
pub use sink::MILOutputSink;
pub use types::{
    MILBinding, MILBlobRef, MILBlock, MILDataType, MILFunction, MILNamedType, MILOperation,
    MILProgram, MILTensorData, MILType, MILValue,
};

/// Statistics computed from a single compile pass.
#[derive(Debug, Clone)]
pub struct CompileStats {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub operation_count: usize,
    pub const_count: usize,
    pub largest_const_elements: usize,
}

/// Result of an in-memory compile.
#[derive(Debug, Clone)]
pub struct CompileResult {
    pub mil_text: String,
    pub stats: CompileStats,
}

/// Errors that can be returned by the public API.
#[derive(Debug)]
pub enum Error {
    Decode(MILDecoderError),
    Io(std::io::Error),
}

impl From<MILDecoderError> for Error {
    fn from(e: MILDecoderError) -> Self {
        Error::Decode(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "decode error: {e}"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Compile a CoreML protobuf into MIL text only (no bundle assembly).
///
/// Useful for debugging or for callers that want to handle the bundle layout
/// themselves. Most users want [`compile_to_bundle`] instead.
pub fn compile_to_text(protobuf: &[u8]) -> Result<CompileResult, Error> {
    let program = decode(protobuf)?;
    let mil_text = emit_to_string(&program);
    let stats = compute_stats(protobuf, &mil_text, &program);
    Ok(CompileResult { mil_text, stats })
}

/// Compile a CoreML protobuf into a complete `.mlmodelc` bundle in memory.
///
/// `weights` is the contents of `weights/weights.bin` if the input model
/// references external blob weights (`BlobFileValue` entries in the MLProgram
/// protobuf). Pass `None` for inline-constant models.
pub fn compile_to_bundle(protobuf: &[u8], weights: Option<&[u8]>) -> Result<MlmodelcBundle, Error> {
    let program = decode(protobuf)?;
    let mil_text = emit_to_string(&program);
    let bundle = build_bundle(&program, mil_text.into_bytes(), weights.map(|w| w.to_vec()));
    Ok(bundle)
}

/// Streaming variant of [`compile_to_bundle`]: write the bundle directly to
/// `directory` without materialising the full MIL text in memory. Recommended
/// for large models on memory-constrained devices.
pub fn compile_to_dir(
    protobuf: &[u8],
    weights: Option<&[u8]>,
    directory: impl AsRef<Path>,
) -> Result<CompileStats, Error> {
    let program = decode(protobuf)?;
    let dir = directory.as_ref();
    fs::create_dir_all(dir)?;

    let mil_path = dir.join("model.mil");
    if mil_path.exists() {
        fs::remove_file(&mil_path)?;
    }
    let file = fs::File::create(&mil_path)?;
    let mut sink = MILOutputSink::streaming(file);
    emit(&program, &mut sink);
    let bytes_written = sink.bytes_written();
    let _ = sink.finalize()?;

    fs::write(
        dir.join("coremldata.bin"),
        generate_coremldata_bin(&program),
    )?;
    fs::write(dir.join("metadata.json"), generate_metadata_json(&program))?;

    let analytics_dir = dir.join("analytics");
    fs::create_dir_all(&analytics_dir)?;
    fs::write(
        analytics_dir.join("coremldata.bin"),
        generate_analytics_bin(),
    )?;

    if let Some(w) = weights {
        let weights_dir = dir.join("weights");
        fs::create_dir_all(&weights_dir)?;
        fs::write(weights_dir.join("weights.bin"), w)?;
    }

    Ok(CompileStats {
        input_bytes: protobuf.len(),
        output_bytes: bytes_written,
        operation_count: count_operations(&program),
        const_count: count_consts(&program),
        largest_const_elements: largest_const(&program),
    })
}

fn compute_stats(protobuf: &[u8], mil_text: &str, program: &MILProgram) -> CompileStats {
    CompileStats {
        input_bytes: protobuf.len(),
        output_bytes: mil_text.len(),
        operation_count: count_operations(program),
        const_count: count_consts(program),
        largest_const_elements: largest_const(program),
    }
}

fn count_operations(program: &MILProgram) -> usize {
    program
        .functions
        .iter()
        .map(|(_, f)| f.block.operations.len())
        .sum()
}

fn count_consts(program: &MILProgram) -> usize {
    program
        .functions
        .iter()
        .flat_map(|(_, f)| f.block.operations.iter())
        .filter(|op| op.r#type == "const")
        .count()
}

fn largest_const(program: &MILProgram) -> usize {
    let mut largest = 0;
    for (_, f) in &program.functions {
        for op in &f.block.operations {
            if op.r#type == "const" {
                for (k, v) in &op.attributes {
                    if k == "val" {
                        largest = largest.max(v.element_count());
                    }
                }
            }
        }
    }
    largest
}
