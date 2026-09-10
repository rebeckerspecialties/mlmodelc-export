use mlmodelc_export::{
    compile_to_bundle, compile_to_dir, decode, generate_coremldata_bin, generate_metadata_json,
};

const RANGE: &[u8] = include_bytes!("fixtures/flexible-range/input.mlmodel");
const MULTI: &[u8] = include_bytes!("fixtures/flexible-multifunction/input.mlmodel");

#[test]
fn preserves_large_description_and_per_function_trailers() {
    let program = decode(MULTI).unwrap();
    assert!(program.description_data.len() > 255);
    let binary = generate_coremldata_bin(&program);
    let length = u64::from_le_bytes(binary[75..83].try_into().unwrap()) as usize;
    assert_eq!(length, program.description_data.len() + 3); // empty metadata field
    assert_eq!(
        &binary[83..83 + program.description_data.len()],
        &program.description_data
    );
    assert_eq!(
        &binary[83 + program.description_data.len()..83 + length],
        &[0xa2, 6, 0]
    );
    assert_eq!(binary.len(), 83 + length + 16 * 2);
    for trailer in binary[83 + length..].chunks_exact(16) {
        assert_eq!(&trailer[..4], &502u32.to_le_bytes());
        assert!(trailer[4..].iter().all(|&byte| byte == 0));
    }
}

#[test]
fn preserves_existing_metadata_instead_of_appending_another_field() {
    let mut program = decode(RANGE).unwrap();
    // ModelDescription.metadata.shortDescription = "note".
    program
        .description_data
        .extend_from_slice(&[0xa2, 6, 6, 0x0a, 4, b'n', b'o', b't', b'e']);
    let binary = generate_coremldata_bin(&program);
    let length = u64::from_le_bytes(binary[75..83].try_into().unwrap()) as usize;
    assert_eq!(length, program.description_data.len());
    assert_eq!(&binary[83..83 + length], &program.description_data);
}

#[test]
fn metadata_uses_source_default_function_and_shapes() {
    let program = decode(MULTI).unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&generate_metadata_json(&program)).unwrap();
    let model = &metadata[0];
    assert_eq!(model["defaultFunctionName"], "second");
    assert_eq!(model["inputSchema"][0]["shape"], "[2]");
    assert_eq!(model["inputSchema"][0]["shapeRange"], "[[0, 4]]");
    let functions = model["functions"].as_array().unwrap();
    assert_eq!(functions.len(), 2);
    for function in functions {
        assert_eq!(
            function["inputSchema"][0]["shape"],
            if function["name"] == "second" {
                "[2]"
            } else {
                "[4]"
            }
        );
    }
    let single: serde_json::Value =
        serde_json::from_slice(&generate_metadata_json(&decode(RANGE).unwrap())).unwrap();
    assert!(single[0].get("functions").is_none());
    assert!(single[0].get("defaultFunctionName").is_none());
}

#[test]
fn streaming_matches_buffered_for_flexible_models() {
    let directory =
        std::env::temp_dir().join(format!("mlmodelc-flexible-parity-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    for model in [RANGE, MULTI] {
        let bundle = compile_to_bundle(model, None).unwrap();
        compile_to_dir(model, None, &directory).unwrap();
        for (name, bytes) in [
            ("model.mil", bundle.model_mil),
            ("coremldata.bin", bundle.coremldata_bin),
            ("metadata.json", bundle.metadata_json),
        ] {
            assert_eq!(std::fs::read(directory.join(name)).unwrap(), bytes);
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn enumerated_inputs_fail_explicitly_until_supported() {
    let model = include_bytes!("unsupported/flexible-enumerated/input.mlmodel");
    let error = compile_to_bundle(model, None).unwrap_err();
    assert!(error.to_string().contains("enumerated input shapes"));
}
