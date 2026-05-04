//! Generate the contents of a `.mlmodelc` bundle from a [`MILProgram`].
//!
//! A bundle is a directory containing four files:
//!
//! ```text
//! example.mlmodelc/
//!   model.mil          UTF-8 MIL text
//!   coremldata.bin     binary container with FunctionDescription / defaultFunctionName
//!   metadata.json      I/O schema, op histogram, availability matrix
//!   analytics/
//!     coremldata.bin   minimal stub satisfying CoreML's analytics check
//! ```
//!
//! The wire layout of `coremldata.bin` matches Apple `coremlc` 3520.x
//! byte-for-byte for the small models we've validated.

use std::fs;
use std::io;
use std::path::Path;

use crate::types::*;

/// In-memory representation of a `.mlmodelc` bundle, ready to write to disk.
#[derive(Debug, Clone)]
pub struct MlmodelcBundle {
    pub model_mil: Vec<u8>,
    pub coremldata_bin: Vec<u8>,
    pub metadata_json: Vec<u8>,
    pub analytics_coremldata_bin: Vec<u8>,
    /// Mirror of the input `weights_data`, written to `weights/weights.bin`
    /// when present. None for graphs without external weights.
    pub weights_bin: Option<Vec<u8>>,
}

impl MlmodelcBundle {
    /// Write the bundle to `directory`. Creates the directory and any
    /// missing parents. Always overwrites if files already exist.
    pub fn write_to_dir(&self, directory: impl AsRef<Path>) -> io::Result<()> {
        let dir = directory.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(dir.join("model.mil"), &self.model_mil)?;
        fs::write(dir.join("coremldata.bin"), &self.coremldata_bin)?;
        fs::write(dir.join("metadata.json"), &self.metadata_json)?;

        let analytics_dir = dir.join("analytics");
        fs::create_dir_all(&analytics_dir)?;
        fs::write(
            analytics_dir.join("coremldata.bin"),
            &self.analytics_coremldata_bin,
        )?;

        if let Some(weights) = &self.weights_bin {
            let weights_dir = dir.join("weights");
            fs::create_dir_all(&weights_dir)?;
            fs::write(weights_dir.join("weights.bin"), weights)?;
        }
        Ok(())
    }
}

/// Build the four-file bundle from a decoded `MILProgram` and the MIL text
/// produced by [`crate::emitter::emit`].
pub fn build_bundle(
    program: &MILProgram,
    model_mil: Vec<u8>,
    weights_bin: Option<Vec<u8>>,
) -> MlmodelcBundle {
    MlmodelcBundle {
        model_mil,
        coremldata_bin: generate_coremldata_bin(program),
        metadata_json: generate_metadata_json(program),
        analytics_coremldata_bin: generate_analytics_bin(),
        weights_bin,
    }
}

// === coremldata.bin ===

/// Generate `coremldata.bin` from a decoded `MILProgram`.
///
/// iOS 18 / watchOS 11 runtime change: MLPrograms that use external weights
/// (BlobFileValue refs) require a `FunctionDescription` (proto field 20) +
/// `defaultFunctionName` (field 21) in the trailer protobuf. A non-empty
/// `ModelDescription` alone is rejected with
/// "This MLModel doesn't support the multi-function description sytnax"
/// (Apple's typo). Older MLPrograms with inline constants still load fine
/// with just the `ModelDescription`, so we emit both to preserve
/// compatibility across the two on-device runtimes.
///
/// Wire layout (matches `coremlc` 3520.x byte-equivalent):
///
/// ```text
/// 0x00  4    uint32 LE 502 (model type = mlProgram)
/// 0x04  4    uint32 LE spec_version
/// 0x08  20   zeros (reserved)
/// 0x1C  4    uint32 LE 7 (target string length)
/// 0x20  4    zeros
/// 0x24  7    "generic"
/// 0x2B  1    spec_version as uint8
/// 0x2C  31   zeros
/// 0x4B  1    payload_size = trailer_proto_len + 3
/// 0x4C  7    zeros
/// 0x53  var  trailer protobuf:
///              field 20: FunctionDescription for each function
///              field 21: defaultFunctionName (string)
/// var   3    a2 06 00 (ModelDescription closing tag, len=0)
/// var   4    uint32 LE 502 (model type, repeated)
/// var   12   zeros
/// ```
pub fn generate_coremldata_bin(program: &MILProgram) -> Vec<u8> {
    let mut trailer = Vec::new();
    let mut default_name: Option<&str> = None;
    for (name, function) in &program.functions {
        let desc = encode_function_description(name, function);
        append_proto_length_delimited(&mut trailer, 20, &desc);
        if default_name.is_none() {
            default_name = Some(name);
        }
    }
    if let Some(name) = default_name {
        append_proto_string(&mut trailer, 21, name);
    }

    let desc_len = trailer.len();
    let mut bin = Vec::with_capacity(desc_len + 102);

    append_u32_le(&mut bin, 502);
    append_u32_le(&mut bin, program.spec_version as u32);
    bin.extend(std::iter::repeat_n(0u8, 20));
    append_u32_le(&mut bin, 7);
    append_u32_le(&mut bin, 0);
    bin.extend_from_slice(b"generic");
    bin.push((program.spec_version & 0xFF) as u8);
    bin.extend(std::iter::repeat_n(0u8, 31));
    bin.push(((desc_len + 3) & 0xFF) as u8);
    bin.extend(std::iter::repeat_n(0u8, 7));

    bin.extend_from_slice(&trailer);

    bin.extend_from_slice(&[0xa2, 0x06, 0x00]);
    append_u32_le(&mut bin, 502);
    bin.extend(std::iter::repeat_n(0u8, 12));
    bin
}

/// Generate a minimal `analytics/coremldata.bin` — CoreML expects the file
/// to exist; the contents are tolerated as long as the header matches.
pub fn generate_analytics_bin() -> Vec<u8> {
    let mut bin = Vec::new();
    let header = "NeuralNetworkModelDetails";
    append_u64_le(&mut bin, header.len() as u64);
    bin.extend_from_slice(header.as_bytes());
    append_u64_le(&mut bin, 2);
    append_analytics_entry(&mut bin, "containsCustomLayer", "0");
    append_analytics_entry(&mut bin, "modelDimension", "0");
    bin
}

fn append_analytics_entry(bin: &mut Vec<u8>, key: &str, value: &str) {
    append_u64_le(bin, key.len() as u64);
    bin.extend_from_slice(key.as_bytes());
    append_u64_le(bin, value.len() as u64);
    bin.extend_from_slice(value.as_bytes());
}

// === metadata.json ===

/// Generate `metadata.json`.
///
/// The output shape matches Apple `coremlc` 3520.x for CoreML8 / iOS18
/// MLPrograms. iOS 18+ loaders require the `functions` array and
/// `defaultFunctionName` at the top level when the model uses external
/// weights — without them MLModel rejects with the same multi-function
/// syntax error noted above.
pub fn generate_metadata_json(program: &MILProgram) -> Vec<u8> {
    let main_func = program.functions.first();
    let main_name: String = main_func
        .map(|(n, _)| n.clone())
        .unwrap_or_else(|| "main".to_string());

    let mut hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    if let Some((_, f)) = main_func {
        let prefix = opset_prefix(&f.opset);
        for op in &f.block.operations {
            *hist.entry(format!("{prefix}.{}", op.r#type)).or_insert(0) += 1;
        }
    }

    let inputs: Vec<(String, MILType)> = main_func
        .map(|(_, f)| {
            f.inputs
                .iter()
                .map(|i| (i.name.clone(), i.r#type.clone()))
                .collect()
        })
        .unwrap_or_default();

    let output_types: Vec<(String, MILType)> = main_func
        .map(|(_, f)| {
            f.block
                .outputs
                .iter()
                .map(|name| (name.clone(), find_output_type(name, &f.block)))
                .collect()
        })
        .unwrap_or_default();

    // Dominant storage/compute precision (matches `coremlc` shape).
    let mut counts: std::collections::HashMap<MILDataType, usize> =
        std::collections::HashMap::new();
    if let Some((_, f)) = main_func {
        for op in &f.block.operations {
            for o in &op.outputs {
                *counts.entry(o.r#type.data_type).or_insert(0) += 1;
            }
        }
    }
    let dominant = counts
        .iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| *k)
        .unwrap_or(MILDataType::Float32);
    let precision = format_data_type_for_meta(dominant);
    let avail = availability_for_spec(program.spec_version);

    let mut json = String::new();
    json.push_str("[\n  {\n");
    json.push_str("    \"metadataOutputVersion\" : \"3.0\",\n");
    json.push_str(&format!(
        "    \"outputSchema\" : {},\n",
        schema_list(&output_types, "    ")
    ));
    json.push_str("    \"modelParameters\" : [\n\n    ],\n");
    json.push_str(&format!(
        "    \"specificationVersion\" : {},\n",
        program.spec_version
    ));

    // Functions array (required for iOS 18+). Field order matches Apple's
    // coremlc 3520.x output: computePrecision, outputSchema, stateSchema,
    // name, mlProgramOperationTypeHistogram, inputSchema. coremlc does NOT
    // emit `storagePrecision` here — keeping our output aligned.
    json.push_str("    \"functions\" : [\n");
    json.push_str("      {\n");
    json.push_str(&format!(
        "        \"computePrecision\" : \"{precision}\",\n"
    ));
    json.push_str(&format!(
        "        \"outputSchema\" : {},\n",
        schema_list(&output_types, "        ")
    ));
    json.push_str("        \"stateSchema\" : [\n\n        ],\n");
    json.push_str(&format!("        \"name\" : \"{main_name}\",\n"));
    json.push_str("        \"mlProgramOperationTypeHistogram\" : {\n");
    let hist_entries: Vec<(&String, &usize)> = hist.iter().collect();
    for (i, (k, v)) in hist_entries.iter().enumerate() {
        json.push_str(&format!("          \"{k}\" : {v}"));
        if i < hist_entries.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("        },\n");
    json.push_str(&format!(
        "        \"inputSchema\" : {}\n",
        schema_list(&inputs, "        ")
    ));
    json.push_str("      }\n");
    json.push_str("    ],\n");

    json.push_str("    \"mlProgramOperationTypeHistogram\" : {\n");
    for (i, (k, v)) in hist_entries.iter().enumerate() {
        json.push_str(&format!("      \"{k}\" : {v}"));
        if i < hist_entries.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("    },\n");

    json.push_str("    \"isUpdatable\" : \"0\",\n");
    json.push_str("    \"stateSchema\" : [\n\n    ],\n");

    json.push_str("    \"availability\" : {\n");
    for (i, (k, v)) in avail.iter().enumerate() {
        json.push_str(&format!("      \"{k}\" : \"{v}\""));
        if i < avail.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("    },\n");

    json.push_str(&format!("    \"computePrecision\" : \"{precision}\",\n"));
    json.push_str("    \"modelType\" : {\n");
    json.push_str("      \"name\" : \"MLModelType_mlProgram\"\n");
    json.push_str("    },\n");
    json.push_str(&format!(
        "    \"inputSchema\" : {},\n",
        schema_list(&inputs, "    ")
    ));
    json.push_str(&format!("    \"defaultFunctionName\" : \"{main_name}\",\n"));
    json.push_str("    \"generatedClassName\" : \"model\",\n");
    json.push_str("    \"userDefinedMetadata\" : {\n\n    },\n");
    json.push_str("    \"method\" : \"predict\"\n");
    json.push_str("  }\n");
    json.push(']');

    json.into_bytes()
}

fn schema_list(entries: &[(String, MILType)], indent: &str) -> String {
    if entries.is_empty() {
        return format!("[\n\n{indent}]");
    }
    let mut s = String::from("[\n");
    for (i, (name, ty)) in entries.iter().enumerate() {
        s.push_str(&schema_entry(name, ty, &format!("{indent}  ")));
        if i < entries.len() - 1 {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str(indent);
    s.push(']');
    s
}

fn schema_entry(name: &str, r#type: &MILType, indent: &str) -> String {
    let dt_str = format_data_type_for_meta(r#type.data_type);
    let shape_str: String = r#type
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let formatted_shape: String = r#type
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" \u{00d7} ");
    let mut s = format!("{indent}{{\n");
    s.push_str(&format!("{indent}  \"hasShapeFlexibility\" : \"0\",\n"));
    s.push_str(&format!("{indent}  \"isOptional\" : \"0\",\n"));
    s.push_str(&format!("{indent}  \"dataType\" : \"{dt_str}\",\n"));
    s.push_str(&format!(
        "{indent}  \"formattedType\" : \"MultiArray ({dt_str} {formatted_shape})\",\n"
    ));
    s.push_str(&format!("{indent}  \"shortDescription\" : \"\",\n"));
    s.push_str(&format!("{indent}  \"shape\" : \"[{shape_str}]\",\n"));
    s.push_str(&format!("{indent}  \"name\" : \"{name}\",\n"));
    s.push_str(&format!("{indent}  \"type\" : \"MultiArray\"\n"));
    s.push_str(&format!("{indent}}}"));
    s
}

fn format_data_type_for_meta(dt: MILDataType) -> &'static str {
    match dt {
        MILDataType::Float32 => "Float32",
        MILDataType::Float16 => "Float16",
        MILDataType::Int32 => "Int32",
        MILDataType::Int64 => "Int64",
        MILDataType::Bool => "Bool",
        MILDataType::String => "String",
        _ => dt.text_name(),
    }
}

fn opset_prefix(opset: &str) -> &'static str {
    match opset {
        "CoreML8" => "Ios18",
        "CoreML7" => "Ios17",
        "CoreML6" => "Ios16",
        "CoreML5" => "Ios15",
        _ => "Generic",
    }
}

fn availability_for_spec(spec: i64) -> Vec<(&'static str, &'static str)> {
    // Key order matches Apple's coremlc 3520.x output exactly: macOS, tvOS,
    // visionOS, watchOS, iOS, macCatalyst. JSON wouldn't care about order
    // semantically, but byte-exact tooling comparisons (and our own golden
    // tests) do.
    match spec {
        9 => vec![
            ("macOS", "15.0"),
            ("tvOS", "18.0"),
            ("visionOS", "2.0"),
            ("watchOS", "11.0"),
            ("iOS", "18.0"),
            ("macCatalyst", "18.0"),
        ],
        8 => vec![
            ("macOS", "14.0"),
            ("tvOS", "17.0"),
            ("visionOS", "1.0"),
            ("watchOS", "10.0"),
            ("iOS", "17.0"),
            ("macCatalyst", "17.0"),
        ],
        _ => vec![
            ("macOS", "14.0"),
            ("tvOS", "17.0"),
            ("watchOS", "10.0"),
            ("iOS", "17.0"),
        ],
    }
}

fn find_output_type(name: &str, block: &MILBlock) -> MILType {
    for op in block.operations.iter().rev() {
        for output in &op.outputs {
            if output.name == name {
                return output.r#type.clone();
            }
        }
    }
    MILType::new(MILDataType::Float32, Vec::new())
}

fn encode_function_description(name: &str, function: &MILFunction) -> Vec<u8> {
    let mut bin = Vec::new();
    append_proto_string(&mut bin, 1, name);
    for input in &function.inputs {
        let desc = encode_feature_description(&input.name, &input.r#type);
        append_proto_length_delimited(&mut bin, 2, &desc);
    }
    for output_name in &function.block.outputs {
        let ty = find_output_type(output_name, &function.block);
        let desc = encode_feature_description(output_name, &ty);
        append_proto_length_delimited(&mut bin, 3, &desc);
    }
    bin
}

fn encode_feature_description(name: &str, ty: &MILType) -> Vec<u8> {
    let mut bin = Vec::new();
    append_proto_string(&mut bin, 1, name);
    let feature_type = encode_feature_type(&ty.shape, ty.data_type);
    append_proto_length_delimited(&mut bin, 3, &feature_type);
    bin
}

fn encode_feature_type(shape: &[usize], dt: MILDataType) -> Vec<u8> {
    let mut bin = Vec::new();
    let arr = encode_array_feature_type(shape, dt);
    append_proto_length_delimited(&mut bin, 5, &arr);
    bin
}

fn encode_array_feature_type(shape: &[usize], dt: MILDataType) -> Vec<u8> {
    let mut bin = Vec::new();
    let mut shape_bytes = Vec::new();
    for d in shape {
        append_proto_varint(&mut shape_bytes, *d as u64);
    }
    append_proto_length_delimited(&mut bin, 1, &shape_bytes);
    let dt_value: u64 = match dt {
        MILDataType::Float32 => 65568,
        MILDataType::Float16 => 65552,
        MILDataType::Float64 => 65600,
        MILDataType::Int32 => 131072,
        MILDataType::Int64 => 131104,
        MILDataType::Int16 => 131040,
        MILDataType::Int8 => 131024,
        MILDataType::Uint8 => 131136,
        MILDataType::Uint16 => 131152,
        MILDataType::Bool => 131168,
        MILDataType::String => 65696,
    };
    append_proto_varint_field(&mut bin, 2, dt_value);
    bin
}

// === minimal protobuf writer ===

fn append_proto_varint(bin: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        bin.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    bin.push((v & 0x7F) as u8);
}

fn append_proto_tag(bin: &mut Vec<u8>, field: u32, wire_type: u64) {
    append_proto_varint(bin, ((field as u64) << 3) | wire_type);
}

fn append_proto_length_delimited(bin: &mut Vec<u8>, field: u32, value: &[u8]) {
    append_proto_tag(bin, field, 2);
    append_proto_varint(bin, value.len() as u64);
    bin.extend_from_slice(value);
}

fn append_proto_string(bin: &mut Vec<u8>, field: u32, value: &str) {
    append_proto_length_delimited(bin, field, value.as_bytes());
}

fn append_proto_varint_field(bin: &mut Vec<u8>, field: u32, value: u64) {
    append_proto_tag(bin, field, 0);
    append_proto_varint(bin, value);
}

fn append_u32_le(bin: &mut Vec<u8>, value: u32) {
    bin.extend_from_slice(&value.to_le_bytes());
}

fn append_u64_le(bin: &mut Vec<u8>, value: u64) {
    bin.extend_from_slice(&value.to_le_bytes());
}
