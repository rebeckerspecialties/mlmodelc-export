//! `mlmodelc-export` CLI — compile a `.mlmodel` (or `.mlpackage`) into a
//! `.mlmodelc` bundle without Apple's `coremlc`.

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use mlmodelc_export::compile_to_dir;

#[derive(Parser, Debug)]
#[command(name = "mlmodelc-export")]
#[command(about = "Compile a CoreML MLProgram into a .mlmodelc bundle in pure Rust", long_about = None)]
#[command(version)]
struct Args {
    /// Input `.mlmodel` (raw protobuf) or `.mlpackage` directory containing
    /// `Data/com.apple.CoreML/model.mlmodel`.
    input: PathBuf,

    /// Output `.mlmodelc` directory. Created if missing; existing files
    /// inside are overwritten.
    output: PathBuf,

    /// Optional path to an external `weights.bin` to copy into
    /// `<output>/weights/weights.bin`. Defaults to auto-detection inside
    /// `.mlpackage` inputs.
    #[arg(long)]
    weights: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (model_bytes, weights_bytes) = read_input(&args.input, args.weights.as_deref())?;
    let stats = compile_to_dir(&model_bytes, weights_bytes.as_deref(), &args.output)?;
    eprintln!(
        "wrote {} ({} ops, {} consts, MIL {} bytes)",
        args.output.display(),
        stats.operation_count,
        stats.const_count,
        stats.output_bytes
    );
    Ok(())
}

fn read_input(
    input: &std::path::Path,
    explicit_weights: Option<&std::path::Path>,
) -> std::io::Result<(Vec<u8>, Option<Vec<u8>>)> {
    if input.is_dir() {
        let model = input
            .join("Data")
            .join("com.apple.CoreML")
            .join("model.mlmodel");
        let model_bytes = fs::read(&model)?;
        let weights_path = explicit_weights
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| {
                input
                    .join("Data")
                    .join("com.apple.CoreML")
                    .join("weights")
                    .join("weights.bin")
            });
        let weights_bytes = if weights_path.exists() {
            Some(fs::read(&weights_path)?)
        } else {
            None
        };
        return Ok((model_bytes, weights_bytes));
    }
    let model_bytes = fs::read(input)?;
    let weights_bytes = explicit_weights.map(fs::read).transpose()?;
    Ok((model_bytes, weights_bytes))
}
