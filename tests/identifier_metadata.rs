use mlmodelc_export::{compile_to_bundle, compile_to_dir, decode, generate_coremldata_bin};

const INPUTS: [&str; 3] = [
    "rustnn_escaped_7374617465", // state
    "rustnn_escaped_727573746e6e5f657363617065645f37333734363137343635", // literal encoded prefix
    "rustnn_escaped_706173742e6b6579", // past.key
];
const OUTPUTS: [&str; 2] = ["rustnn_escaped_72657475726e", "output"];

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push((value as u8 & 127) | 128);
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

fn tensor_type() -> Vec<u8> {
    message(
        1,
        &[
            scalar(1, 11),
            scalar(2, 1),
            message(3, &message(1, &scalar(1, 4))),
        ]
        .concat(),
    )
}

fn named_type(name: &str) -> Vec<u8> {
    [message(1, name.as_bytes()), message(2, &tensor_type())].concat()
}

fn feature(name: &str) -> Vec<u8> {
    let array = [message(1, &[4]), scalar(2, 65568)].concat();
    [message(1, name.as_bytes()), message(3, &message(5, &array))].concat()
}

fn binding(argument: &str, name: &str) -> Vec<u8> {
    message(
        2,
        &[
            message(1, argument.as_bytes()),
            message(2, &message(1, &message(1, name.as_bytes()))),
        ]
        .concat(),
    )
}

fn model() -> Vec<u8> {
    let metadata = [
        ("rustnn.coreml.name_encoding", "hex-v1"),
        ("test.quoted\"key", "line1\nline2\\\t\u{1} café"),
        ("test.duplicate", "old"),
        ("test.duplicate", "new"),
    ]
    .into_iter()
    .flat_map(|(key, value)| {
        message(
            100,
            &[message(1, key.as_bytes()), message(2, value.as_bytes())].concat(),
        )
    })
    .collect::<Vec<_>>();
    let mut description = INPUTS
        .iter()
        .flat_map(|name| message(1, &feature(name)))
        .collect::<Vec<_>>();
    description.extend(OUTPUTS.iter().flat_map(|name| message(10, &feature(name))));
    description.extend(message(100, &metadata));
    let add = [
        message(1, b"add"),
        binding("x", INPUTS[0]),
        binding("y", INPUTS[1]),
        message(3, &named_type(OUTPUTS[0])),
    ]
    .concat();
    let relu = [
        message(1, b"relu"),
        binding("x", INPUTS[2]),
        message(3, &named_type(OUTPUTS[1])),
    ]
    .concat();
    let mut block = OUTPUTS
        .iter()
        .flat_map(|name| message(2, name.as_bytes()))
        .collect::<Vec<_>>();
    block.extend(message(3, &add));
    block.extend(message(3, &relu));
    let mut function = INPUTS
        .iter()
        .flat_map(|name| message(1, &named_type(name)))
        .collect::<Vec<_>>();
    function.extend(message(2, b"CoreML8"));
    function.extend(message(
        3,
        &[message(1, b"CoreML8"), message(2, &block)].concat(),
    ));
    let program = [
        scalar(1, 1),
        message(2, &[message(1, b"main"), message(2, &function)].concat()),
    ]
    .concat();
    [
        scalar(1, 9),
        message(2, &description),
        message(502, &program),
    ]
    .concat()
}

#[test]
fn preserves_user_metadata_and_already_normalized_feature_bindings() {
    let source = model();
    let program = decode(&source).unwrap();
    let bundle = compile_to_bundle(&source, None).unwrap();
    let binary = generate_coremldata_bin(&program);
    let length = u64::from_le_bytes(binary[75..83].try_into().unwrap()) as usize;
    assert_eq!(&binary[83..83 + length], &program.description_data);
    let metadata: serde_json::Value = serde_json::from_slice(&bundle.metadata_json).unwrap();
    let user = &metadata[0]["userDefinedMetadata"];
    assert_eq!(user["rustnn.coreml.name_encoding"], "hex-v1");
    assert_eq!(user["test.quoted\"key"], "line1\nline2\\\t\u{1} café");
    assert_eq!(user["test.duplicate"], "new");
    assert_eq!(user.as_object().unwrap().len(), 3);
    for (field, names) in [
        ("inputSchema", INPUTS.as_slice()),
        ("outputSchema", OUTPUTS.as_slice()),
    ] {
        let actual = metadata[0][field]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, names);
    }
    let mil = String::from_utf8(bundle.model_mil.clone()).unwrap();
    assert!(
        mil.contains(&format!("add(x = {}, y = {})", INPUTS[0], INPUTS[1])),
        "{mil}"
    );
    assert!(mil.contains(&format!("relu(x = {})", INPUTS[2])), "{mil}");
    let directory =
        std::env::temp_dir().join(format!("mlmodelc-identifiers-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    compile_to_dir(&source, None, &directory).unwrap();
    for (name, bytes) in [
        ("model.mil", bundle.model_mil),
        ("coremldata.bin", bundle.coremldata_bin),
        ("metadata.json", bundle.metadata_json),
    ] {
        assert_eq!(std::fs::read(directory.join(name)).unwrap(), bytes);
    }
    std::fs::remove_dir_all(directory).unwrap();
}
