//! Source-level checks for the model-state fixtures. Byte-for-byte comparison
//! with the native goldens lives in golden.rs; predictions in runtime_state.swift.

use mlmodelc_export::{
    Error, MILBinding, MILDataType, MILDecoderError, MILProgram, MILType, compile_to_bundle,
    compile_to_text, decode, generate_coremldata_bin, generate_metadata_json,
};

const FP16_STATE: &[u8] = include_bytes!("fixtures/stateful-fp16/input.mlmodel");
const FP32_PAIR: &[u8] = include_bytes!("fixtures/stateful-fp32-pair/input.mlmodel");
const DECLARATION_ORDER: &[u8] =
    include_bytes!("fixtures/stateful-declaration-order/input.mlmodel");

fn metadata(program: &MILProgram) -> serde_json::Value {
    serde_json::from_slice::<serde_json::Value>(&generate_metadata_json(program)).unwrap()[0].take()
}

fn names(schema: &serde_json::Value) -> Vec<&str> {
    schema
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect()
}

fn reference<'a>(op: &'a mlmodelc_export::MILOperation, key: &str) -> &'a str {
    let (_, bindings) = op.inputs.iter().find(|(name, _)| name == key).unwrap();
    match bindings.as_slice() {
        [MILBinding::Reference(name)] => name,
        other => panic!("{key}: {other:?}"),
    }
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}

fn message(field: u64, contents: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(contents.len() as u64),
        contents.to_vec(),
    ]
    .concat()
}

/// Fields of one protobuf message: (field number, length-delimited payload).
fn fields(mut data: &[u8]) -> Vec<(u64, &[u8])> {
    fn read_varint(data: &mut &[u8]) -> u64 {
        let mut value = 0;
        for shift in (0..).step_by(7) {
            let byte = data[0];
            *data = &data[1..];
            value |= u64::from(byte & 127) << shift;
            if byte < 128 {
                break;
            }
        }
        value
    }
    let mut result = Vec::new();
    while !data.is_empty() {
        let tag = read_varint(&mut data);
        assert_eq!(tag & 7, 2, "only length-delimited fields are expected");
        let length = read_varint(&mut data) as usize;
        let (payload, rest) = data.split_at(length);
        result.push((tag >> 3, payload));
        data = rest;
    }
    result
}

#[test]
fn fp16_fixture_reads_adds_and_writes_one_state() {
    let program = decode(FP16_STATE).unwrap();
    assert_eq!(program.spec_version, 9);
    let (name, function) = &program.functions[0];
    assert_eq!(
        (name.as_str(), function.opset.as_str()),
        ("main", "CoreML8")
    );
    // Source order: the tensor input, then the state.
    let fp16 = MILType::new(MILDataType::Float16, vec![1, 4]);
    assert_eq!(function.inputs[0].name, "x");
    assert_eq!(function.inputs[0].r#type, fp16);
    assert_eq!(function.inputs[1].name, "state_0");
    assert_eq!(function.inputs[1].r#type, fp16.clone().into_state());
    let operations = &function.block.operations;
    let kinds: Vec<_> = operations.iter().map(|op| op.r#type.as_str()).collect();
    assert_eq!(kinds, ["read_state", "add", "add", "write_state"]);
    assert_eq!(reference(&operations[0], "input"), "state_0");
    assert_eq!(operations[0].outputs[0].r#type, fp16);
    assert!(operations[3].outputs.is_empty());
    assert_eq!(reference(&operations[3], "input"), "state_0");
    assert_eq!(reference(&operations[3], "data"), "next_state_0");
    assert_eq!(function.block.outputs, ["y"]);

    let text = compile_to_text(FP16_STATE).unwrap();
    assert_eq!(text.stats.operation_count, 4);
    let lines: Vec<_> = text.mil_text.lines().map(str::trim).collect();
    assert_eq!(
        lines[3..],
        [
            "func main<ios18>(state<tensor<fp16, [1, 4]>> state_0, tensor<fp16, [1, 4]> x) {",
            "tensor<fp16, [1, 4]> state_0_read = read_state(input = state_0);",
            "tensor<fp16, [1, 4]> next_state_0 = add(x = state_0_read, y = x);",
            "tensor<fp16, [1, 4]> y = add(x = next_state_0, y = x);",
            "write_state(data = next_state_0, input = state_0);",
            "} -> (y);",
            "}",
        ]
    );

    let metadata = metadata(&program);
    assert_eq!(names(&metadata["inputSchema"]), ["x"]);
    assert_eq!(names(&metadata["outputSchema"]), ["y"]);
    assert_eq!(names(&metadata["stateSchema"]), ["state_0"]);
    assert_eq!(metadata["stateSchema"][0]["type"], "State");
    assert_eq!(
        metadata["stateSchema"][0]["formattedType"],
        "State (Float16 1 \u{d7} 4)"
    );
    assert_eq!(metadata["inputSchema"][0]["type"], "MultiArray");
    assert!(metadata.get("functions").is_none());
    assert!(metadata.get("storagePrecision").is_none());

    // The source description, state included, reaches the loader unchanged.
    let binary = generate_coremldata_bin(&program);
    let length = u64::from_le_bytes(binary[75..83].try_into().unwrap()) as usize;
    let payload = &binary[83..83 + length];
    assert!(payload.starts_with(&program.description_data));
    let state = fields(payload)
        .into_iter()
        .find(|(field, _)| *field == 13)
        .unwrap()
        .1;
    assert_eq!(fields(state)[0], (1, &b"state_0"[..]));
}

#[test]
fn fp32_pair_fixture_casts_around_two_fp16_states() {
    let program = decode(FP32_PAIR).unwrap();
    let function = &program.functions[0].1;
    let fp16 = MILType::new(MILDataType::Float16, vec![1, 4]);
    let fp32 = MILType::new(MILDataType::Float32, vec![1, 4]);
    assert_eq!(function.inputs[0].r#type, fp32);
    for (input, name) in function.inputs[1..].iter().zip(["state_0", "state_1"]) {
        assert_eq!(input.name, name);
        assert_eq!(input.r#type, fp16.clone().into_state());
    }
    let operations = &function.block.operations;
    for (index, name) in ["state_0", "state_1"].into_iter().enumerate() {
        // read_state yields FP16; the graph value is its FP32 cast.
        let read = &operations[index * 2];
        assert_eq!(read.r#type, "read_state");
        assert_eq!(reference(read, "input"), name);
        assert_eq!(read.outputs[0].r#type, fp16);
        let widen = &operations[index * 2 + 1];
        assert_eq!(widen.r#type, "cast");
        assert_eq!(reference(widen, "x"), read.outputs[0].name);
        assert_eq!(widen.outputs[0].r#type, fp32);
        // The FP32 result is narrowed to FP16 before write_state.
        let narrow = &operations[7 + index * 2];
        assert_eq!(narrow.r#type, "cast");
        assert_eq!(reference(narrow, "x"), format!("next_{name}"));
        assert_eq!(narrow.outputs[0].r#type, fp16);
        let write = &operations[8 + index * 2];
        assert_eq!(write.r#type, "write_state");
        assert!(write.outputs.is_empty());
        assert_eq!(reference(write, "input"), name);
        assert_eq!(reference(write, "data"), narrow.outputs[0].name);
    }
    let text = compile_to_text(FP32_PAIR).unwrap().mil_text;
    assert!(
        text.contains(
            "func main<ios18>(state<tensor<fp16, [1, 4]>> state_0, state<tensor<fp16, [1, 4]>> state_1, tensor<fp32, [1, 4]> x) {"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "            write_state(data = next_state_1_state_value, input = state_1);\n"
        ),
        "{text}"
    );
    let metadata = metadata(&program);
    assert_eq!(names(&metadata["stateSchema"]), ["state_0", "state_1"]);
    assert_eq!(names(&metadata["inputSchema"]), ["x"]);
    assert_eq!(metadata["inputSchema"][0]["dataType"], "Float32");
    assert_eq!(metadata["stateSchema"][1]["dataType"], "Float16");
}

#[test]
fn declaration_order_follows_names_in_mil_and_the_description_in_schemas() {
    let program = decode(DECLARATION_ORDER).unwrap();
    let source: Vec<_> = program.functions[0]
        .1
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect();
    assert_eq!(source, ["s1", "q", "s0", "p"]);
    let text = compile_to_text(DECLARATION_ORDER).unwrap().mil_text;
    assert!(
        text.contains(
            "func main<ios18>(tensor<fp16, [1, 4]> p, tensor<fp16, [1, 4]> q, state<tensor<fp16, [1, 4]>> s0, state<tensor<fp16, [1, 4]>> s1) {"
        ),
        "{text}"
    );
    let metadata = metadata(&program);
    assert_eq!(names(&metadata["inputSchema"]), ["q", "p"]);
    assert_eq!(names(&metadata["stateSchema"]), ["s1", "s0"]);
}

#[test]
fn synthesized_description_declares_states_in_their_own_field() {
    // Callers constructing a MILProgram directly supply no description.
    let mut program = decode(FP16_STATE).unwrap();
    program.description_data.clear();
    let binary = generate_coremldata_bin(&program);
    let length = u64::from_le_bytes(binary[75..83].try_into().unwrap()) as usize;
    // ModelDescription: functions = 20, defaultFunctionName = 21, metadata = 100.
    let description = fields(&binary[83..83 + length]);
    let top: Vec<_> = description.iter().map(|(field, _)| *field).collect();
    assert_eq!(top, [20, 21, 100]);
    assert_eq!(description[1].1, b"main");
    let function = description[0].1;
    let array = [10, 2, 1, 4, 16, 0x90, 0x80, 4];
    let entries = fields(function);
    assert_eq!(entries[0], (1, &b"main"[..]));
    // FunctionDescription: input = 2, output = 3, state = 6.
    let kinds: Vec<_> = entries[1..].iter().map(|(field, _)| *field).collect();
    assert_eq!(kinds, [2, 6, 3]);
    let x = [message(1, b"x"), message(3, &message(5, &array))].concat();
    let state = [
        message(1, b"state_0"),
        message(3, &message(8, &message(1, &array))),
    ]
    .concat();
    let y = [message(1, b"y"), message(3, &message(5, &array))].concat();
    assert_eq!(entries[1].1, x);
    assert_eq!(entries[2].1, state);
    assert_eq!(entries[3].1, y);

    let metadata = metadata(&program);
    assert_eq!(metadata["defaultFunctionName"], "main");
    for scope in [&metadata, &metadata["functions"][0]] {
        assert_eq!(names(&scope["inputSchema"]), ["x"]);
        assert_eq!(names(&scope["stateSchema"]), ["state_0"]);
        assert_eq!(
            scope["stateSchema"][0]["formattedType"],
            "State (Float16 1 \u{d7} 4)"
        );
        assert_eq!(scope["stateSchema"][0]["type"], "State");
    }

    // A stateless function synthesizes no state entry and an empty stateSchema.
    let function = &mut program.functions[0].1;
    function.inputs.truncate(1);
    function.block.operations.retain(|op| op.r#type == "add");
    let binary = generate_coremldata_bin(&program);
    assert!(!binary.windows(state.len()).any(|window| window == state));
    let metadata = self::metadata(&program);
    assert_eq!(metadata["stateSchema"], serde_json::json!([]));
    assert_eq!(
        metadata["functions"][0]["stateSchema"],
        serde_json::json!([])
    );
}

/// A model whose only function input has the given `ValueType`.
fn model_with_input_type(value_type: &[u8]) -> Vec<u8> {
    let input = [message(1, b"state_0"), message(2, value_type)].concat();
    let block = message(3, &[message(1, b"CoreML8"), message(2, &[])].concat());
    let function = [message(1, &input), message(2, b"CoreML8"), block].concat();
    let program = message(2, &[message(1, b"main"), message(2, &function)].concat());
    [vec![8, 9], message(502, &program)].concat()
}

#[test]
fn malformed_state_inputs_fail_to_compile_with_a_decode_error() {
    let tensor = message(1, &[8, 10, 16, 1, 26, 4, 10, 2, 8, 4]);
    let valid = model_with_input_type(&message(5, &message(1, &tensor)));
    let text = compile_to_text(&valid).unwrap().mil_text;
    assert!(
        text.contains("func main<ios18>(state<tensor<fp16, [4]>> state_0) {"),
        "{text}"
    );

    // StateType.wrappedType declares 20 bytes; 3 follow.
    let truncated = message(5, &[0x0a, 20, 0x0a, 1, 8]);
    for (value_type, expected) in [
        (message(5, &[]), "state type without a wrapped tensor type"),
        (
            message(5, &message(1, &[])),
            "state type without a wrapped tensor type",
        ),
        (
            message(5, &message(1, &message(5, &message(1, &tensor)))),
            "state types wrapping another state type",
        ),
        (truncated, "truncated type message"),
    ] {
        let model = model_with_input_type(&value_type);
        for result in [
            compile_to_text(&model).map(|_| ()),
            compile_to_bundle(&model, None).map(|_| ()),
        ] {
            match result {
                Err(Error::Decode(MILDecoderError::UnsupportedFormat(message))) => {
                    assert_eq!(message, expected)
                }
                other => panic!("{expected}: {other:?}"),
            }
        }
    }
}
