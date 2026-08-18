//! Integration tests for the prover CLI (issue #44).
//!
//! Acceptance criteria: `prove` output round-trips through `bundle_from_json`
//! with a `model_hash` matching `model_commitment`, `validate` fails loudly on
//! an out-of-tolerance dataset, and a malformed `--input` exits non-zero rather
//! than silently dropping the bad field.
//!
//! The process-level checks use `env!("CARGO_BIN_EXE_zkml-prover")` rather than
//! a new `assert_cmd` dev-dependency.

use std::path::{Path, PathBuf};
use std::process::Command as ProcCommand;

use zkml_prover::cli::{cmd_prove, cmd_validate, CliError};
use zkml_prover::model_io::import_json;
use zkml_prover::prover::{bundle_from_json, model_commitment};
use zkml_prover::quantization::QuantizationConfig;

const CREDIT: &str = "../../examples/models/credit_lr.json";

/// Per-test scratch directory under the target dir, so parallel tests and
/// repeated runs never collide on a fixed filename.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("zkml_cli_integration").join(name);
    std::fs::create_dir_all(&dir).expect("scratch dir is creatable");
    dir
}

fn run_bin(args: &[&str]) -> std::process::Output {
    ProcCommand::new(env!("CARGO_BIN_EXE_zkml-prover"))
        .args(args)
        .output()
        .expect("prover binary runs")
}

#[test]
fn prove_output_round_trips_and_binds_the_model() {
    let mut buf = Vec::new();
    cmd_prove(Path::new(CREDIT), "0.5,0.2,0.9,0.1", None, &mut buf).expect("prove succeeds");
    let json = String::from_utf8(buf).expect("bundle json is utf-8");

    let bundle = bundle_from_json(json.trim()).expect("bundle_from_json accepts CLI output");

    let model = import_json(&std::fs::read(CREDIT).unwrap()).unwrap();
    assert_eq!(
        bundle.public_inputs.model_hash,
        model_commitment(&model),
        "bundle model_hash must equal the model commitment registered on-chain"
    );
    // Phase 1: proof bytes are a placeholder until issue #11.
    assert!(bundle.proof.data.is_empty());
}

#[test]
fn prove_to_file_round_trips() {
    let path = scratch("prove_to_file").join("bundle.json");
    let mut buf = Vec::new();
    cmd_prove(Path::new(CREDIT), "0.5,0.2,0.9,0.1", Some(&path), &mut buf).expect("prove succeeds");

    let written = std::fs::read_to_string(&path).expect("bundle file exists");
    assert!(bundle_from_json(&written).is_ok());
}

#[test]
fn validate_succeeds_on_the_credit_example() {
    let cfg = QuantizationConfig::default();
    let mut buf = Vec::new();
    cmd_validate(Path::new(CREDIT), None, &cfg, &mut buf).expect("credit_lr validates");
    let out = String::from_utf8(buf).unwrap();
    assert!(out.contains("range check: ok"));
    assert!(out.contains("overflow bounds: ok"));
}

#[test]
fn validate_fails_with_agreement_percentage_on_a_bad_dataset() {
    // Expected outputs deliberately far from what the model produces, so the
    // accuracy pass drops to 0% agreement.
    let dataset = r#"[
      { "inputs": [0.5, 0.2, 0.9, 0.1], "expected": 99.0 },
      { "inputs": [0.0, 0.0, 0.0, 0.0], "expected": -99.0 }
    ]"#;
    let path = scratch("bad_dataset").join("dataset.json");
    std::fs::write(&path, dataset).unwrap();

    let cfg = QuantizationConfig::default();
    let err = cmd_validate(Path::new(CREDIT), Some(&path), &cfg, &mut Vec::new())
        .expect_err("mismatched dataset must fail validation");

    assert!(matches!(err, CliError::Validation(_)));
    let message = err.to_string();
    assert!(
        message.contains("agreement rate") && message.contains('%'),
        "message should carry the agreement percentage, got: {message}"
    );
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn validate_passes_with_a_matching_dataset() {
    // Expected values computed by the quantized model itself, so agreement is 100%.
    let model = import_json(&std::fs::read(CREDIT).unwrap()).unwrap();
    let rows: Vec<String> = [
        vec![0.5, 0.2, 0.9, 0.1],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![-0.5, 0.25, 0.125, 0.75],
    ]
    .iter()
    .map(|inputs| {
        let fp: Vec<_> = inputs
            .iter()
            .map(|&x| zkml_common::fixed_point::FixedPoint::quantize(x))
            .collect();
        let expected = zkml_common::inference::run_inference(&model, &fp).dequantize();
        format!(r#"{{ "inputs": {inputs:?}, "expected": {expected} }}"#)
    })
    .collect();

    let path = scratch("good_dataset").join("dataset.json");
    std::fs::write(&path, format!("[{}]", rows.join(","))).unwrap();

    let cfg = QuantizationConfig::default();
    let mut buf = Vec::new();
    cmd_validate(Path::new(CREDIT), Some(&path), &cfg, &mut buf).expect("dataset validates");
    let out = String::from_utf8(buf).unwrap();
    assert!(out.contains("accuracy: ok over 3 sample(s)"));
    assert!(out.contains("agreement:      100.00%"));
}

#[test]
fn malformed_input_exits_two_and_writes_to_stderr() {
    // The silent-drop regression: `0.5,o.2,0.9,0.1` used to yield a 3-element
    // vector and run inference against the wrong feature count.
    let out = run_bin(&["infer", CREDIT, "-i", "0.5,o.2,0.9,0.1"]);

    assert_eq!(out.status.code(), Some(2), "malformed --input must exit 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("o.2"),
        "stderr should name the bad token: {stderr}"
    );
    assert!(out.stdout.is_empty(), "no inference output on a bad vector");
}

#[test]
fn wrong_feature_count_exits_one_without_panicking() {
    let out = run_bin(&["infer", CREDIT, "-i", "0.5,0.2"]);

    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("expects 4 feature(s), got 2"), "{stderr}");
    assert!(
        !stderr.contains("panicked"),
        "must be an error, not a panic: {stderr}"
    );
}

#[test]
fn commit_succeeds_and_prints_hex() {
    let out = run_bin(&["commit", CREDIT]);
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.trim().len(), 64);
}

#[test]
fn unknown_subcommand_exits_two() {
    let out = run_bin(&["frobnicate"]);
    assert_eq!(out.status.code(), Some(2), "clap's own bad-invocation code");
}

#[test]
fn missing_model_exits_one() {
    let out = run_bin(&["commit", "/nonexistent/model.json"]);
    assert_eq!(out.status.code(), Some(1));
}
