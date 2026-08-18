//! Command-line surface for the prover binary.
//!
//! The whole CLI lives here rather than in `main.rs` so every command body is
//! reachable from tests without spawning a process: each `cmd_*` writes to a
//! `&mut impl Write`, and `main.rs` is reduced to argument dispatch plus
//! exit-code mapping.
//!
//! Models are loaded from the JSON exchange format only
//! ([`crate::model_io::import_json`]). ONNX is deliberately not dispatched here
//! — extraction currently handles single-node `TreeEnsembleClassifier` /
//! `LinearClassifier` graphs, so most real `.onnx` files would fail with a
//! confusing error (extraction is tracked in #5 / #6).
//!
//! `prove` emits a structurally valid [`VerificationBundle`] whose Groth16 proof
//! bytes are still a placeholder; STARK→Groth16 compression is TODO(#11).

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use zkml_common::commitment::to_hex;
use zkml_common::error::ZkmlError;
use zkml_common::fixed_point::{FixedPoint, DEFAULT_SCALE};
use zkml_common::inference::try_run_inference;
use zkml_common::models::{Model, TreeNode};

use crate::model_io::import_json;
use crate::prover::{bundle_to_json, generate_proof, model_commitment};
use crate::quantization::{validate_model, QuantizationConfig};

/// Largest input magnitude that survives `FixedPoint::quantize` without the
/// `as i64` cast saturating: `i64::MAX / 2^16 ≈ 1.4e14`. Matches the bound in
/// `crates/zkml-common/tests/fixed_point_props.rs`.
const MAX_INPUT_MAGNITUDE: f64 = (i64::MAX as f64) / ((1i64 << DEFAULT_SCALE) as f64);

// ---------------------------------------------------------------------------
// Argument types
// ---------------------------------------------------------------------------

/// Top-level parsed command line.
#[derive(Debug, Parser)]
#[command(
    name = "zkml-prover",
    version,
    about = "Off-chain ML inference, commitments, and proof bundles for zkml-soroban"
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// The prover's subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Print the model commitment (the value registered on-chain at `initialize`).
    Commit {
        /// Path to a JSON model.
        model: PathBuf,
    },

    /// Run inference and print the commitment, dequantized output, and raw Q16.16 value.
    Infer {
        /// Path to a JSON model.
        model: PathBuf,
        /// Comma-separated feature vector, e.g. `"0.5,0.2,0.9,0.1"`.
        #[arg(short, long)]
        input: String,
    },

    /// Generate a verification bundle as JSON.
    ///
    /// The Groth16 proof bytes are a placeholder until issue #11; the bundle is
    /// structurally valid and round-trips, but must not be submitted on-chain.
    Prove {
        /// Path to a JSON model.
        model: PathBuf,
        /// Comma-separated feature vector.
        #[arg(short, long)]
        input: String,
        /// Write the bundle here instead of stdout.
        #[arg(short, long)]
        out: Option<PathBuf>,
    },

    /// Run the quantization validation passes and print the report.
    Validate {
        /// Path to a JSON model.
        model: PathBuf,
        /// Optional JSON validation dataset. Without it, only the range and
        /// overflow-bounds passes run.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Maximum absolute input magnitude assumed by overflow-bounds analysis.
        #[arg(long, default_value_t = QuantizationConfig::default().max_input_magnitude)]
        max_input_magnitude: f64,
        /// Minimum float/quantized agreement rate required by the accuracy pass.
        #[arg(long, default_value_t = QuantizationConfig::default().min_agreement)]
        min_agreement: f64,
    },

    /// Print a model's kind, feature count, structure, commitment, and validity.
    Inspect {
        /// Path to a JSON model.
        model: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failures surfaced by the CLI.
///
/// Hand-written `Display` to match [`ZkmlError`] and
/// [`crate::onnx::OnnxImportError`]; the workspace uses no `thiserror`/`anyhow`.
#[derive(Debug)]
pub enum CliError {
    /// `--input` was empty or all whitespace.
    EmptyInput,
    /// A comma-separated field was empty (1-based position).
    EmptyField { position: usize },
    /// A field did not parse as `f64` (1-based position).
    InvalidInput { position: usize, token: String },
    /// A field parsed but was `NaN` or infinite (1-based position).
    NonFiniteInput { position: usize, token: String },
    /// A field exceeded the Q16.16 representable range (1-based position).
    InputOutOfRange { position: usize, value: f64 },
    /// The input vector length disagreed with the model's feature count.
    FeatureCount {
        model: String,
        expected: usize,
        got: usize,
    },
    /// A file could not be read or written.
    Io { path: String, message: String },
    /// A model file could not be imported.
    ModelImport { path: String, message: String },
    /// A dataset file could not be parsed.
    DatasetParse { path: String, message: String },
    /// The model's own structural validation failed.
    InvalidModel { path: String, source: ZkmlError },
    /// Inference failed (overflow, topology, cycle).
    Inference(ZkmlError),
    /// Quantization validation failed.
    Validation(ZkmlError),
    /// A bundle could not be serialized.
    Serialization(String),
}

impl CliError {
    /// Process exit code for this error.
    ///
    /// `2` mirrors clap's own "bad invocation" code and covers malformed
    /// `--input`; everything else is a runtime failure and exits `1`.
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::EmptyInput
            | CliError::EmptyField { .. }
            | CliError::InvalidInput { .. }
            | CliError::NonFiniteInput { .. }
            | CliError::InputOutOfRange { .. } => 2,
            _ => 1,
        }
    }
}

impl core::fmt::Display for CliError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CliError::EmptyInput => write!(f, "--input is empty: expected a comma-separated feature vector"),
            CliError::EmptyField { position } => {
                write!(f, "--input field {position} is empty")
            }
            CliError::InvalidInput { position, token } => write!(
                f,
                "--input field {position} ('{token}') is not a number"
            ),
            CliError::NonFiniteInput { position, token } => write!(
                f,
                "--input field {position} ('{token}') is not finite"
            ),
            CliError::InputOutOfRange { position, value } => write!(
                f,
                "--input field {position} ({value:e}) exceeds the Q16.16 range of ±{MAX_INPUT_MAGNITUDE:e}"
            ),
            CliError::FeatureCount { model, expected, got } => write!(
                f,
                "{model} expects {expected} feature(s), got {got}"
            ),
            CliError::Io { path, message } => write!(f, "{path}: {message}"),
            CliError::ModelImport { path, message } => {
                write!(f, "failed to import model {path}: {message}")
            }
            CliError::DatasetParse { path, message } => {
                write!(f, "failed to parse dataset {path}: {message}")
            }
            CliError::InvalidModel { path, source } => {
                write!(f, "model {path} is structurally invalid: {source}")
            }
            CliError::Inference(e) => write!(f, "inference failed: {e}"),
            CliError::Validation(e) => write!(f, "validation failed: {e}"),
            CliError::Serialization(m) => write!(f, "failed to serialize bundle: {m}"),
        }
    }
}

impl std::error::Error for CliError {}

/// Map a `writeln!` / file-write failure onto [`CliError::Io`].
fn io_err(path: &str, e: std::io::Error) -> CliError {
    CliError::Io {
        path: path.to_string(),
        message: e.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Input parsing
// ---------------------------------------------------------------------------

/// Parse a comma-separated feature vector into fixed-point values.
///
/// Every field must be present and parse; unlike the previous `filter_map`
/// implementation nothing is silently dropped. `NaN`/`inf` are rejected
/// explicitly because `"nan".parse::<f64>()` *succeeds*, and out-of-range
/// values are rejected because `FixedPoint::quantize`'s `as i64` cast
/// saturates rather than failing.
pub fn parse_inputs(raw: &str) -> Result<Vec<FixedPoint>, CliError> {
    if raw.trim().is_empty() {
        return Err(CliError::EmptyInput);
    }

    raw.split(',')
        .enumerate()
        .map(|(i, field)| {
            let position = i + 1;
            let token = field.trim();
            if token.is_empty() {
                return Err(CliError::EmptyField { position });
            }
            let value = token.parse::<f64>().map_err(|_| CliError::InvalidInput {
                position,
                token: token.to_string(),
            })?;
            if !value.is_finite() {
                return Err(CliError::NonFiniteInput {
                    position,
                    token: token.to_string(),
                });
            }
            if value.abs() > MAX_INPUT_MAGNITUDE {
                return Err(CliError::InputOutOfRange { position, value });
            }
            Ok(FixedPoint::quantize(value))
        })
        .collect()
}

#[cfg(test)]
mod tests_parse_inputs {
    use super::*;

    #[test]
    fn accepts_well_formed_vectors() {
        let parsed = parse_inputs("0.5, -0.25 ,1e-3,0").unwrap();
        assert_eq!(parsed.len(), 4);
        assert!((parsed[1].dequantize() + 0.25).abs() < 1e-9);
    }

    #[test]
    fn rejects_unparseable_token_naming_it() {
        // The regression this whole issue exists for: `filter_map` used to drop
        // `o.2` and run inference on a 3-element vector.
        match parse_inputs("0.5,o.2,0.9,0.1") {
            Err(CliError::InvalidInput { position, token }) => {
                assert_eq!(position, 2);
                assert_eq!(token, "o.2");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_field() {
        match parse_inputs("0.5,,0.2") {
            Err(CliError::EmptyField { position }) => assert_eq!(position, 2),
            other => panic!("expected EmptyField, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_and_whitespace_input() {
        assert!(matches!(parse_inputs(""), Err(CliError::EmptyInput)));
        assert!(matches!(parse_inputs("   "), Err(CliError::EmptyInput)));
    }

    #[test]
    fn rejects_non_finite() {
        for raw in ["nan", "inf", "-inf", "0.5,NaN"] {
            assert!(
                matches!(parse_inputs(raw), Err(CliError::NonFiniteInput { .. })),
                "{raw} should be rejected as non-finite"
            );
        }
    }

    #[test]
    fn rejects_out_of_range() {
        match parse_inputs("1e30") {
            Err(CliError::InputOutOfRange { position, .. }) => assert_eq!(position, 1),
            other => panic!("expected InputOutOfRange, got {other:?}"),
        }
        assert!(matches!(
            parse_inputs("-1e30"),
            Err(CliError::InputOutOfRange { .. })
        ));
    }

    #[test]
    fn malformed_input_exits_two() {
        assert_eq!(parse_inputs("0.5,abc").unwrap_err().exit_code(), 2);
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Read and import a JSON model from disk.
pub fn load_model(path: &Path) -> Result<Model, CliError> {
    let display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| io_err(&display, e))?;
    import_json(&bytes).map_err(|message| CliError::ModelImport {
        path: display,
        message,
    })
}

/// One row of a `--dataset` file: features plus the **float** model's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSample {
    /// Raw (unquantized) feature vector.
    pub inputs: Vec<f64>,
    /// Output the original floating-point model produced for `inputs`.
    pub expected: f64,
}

/// Read a validation dataset and shape it for [`validate_model`].
pub fn load_dataset(path: &Path) -> Result<Vec<(Vec<f64>, f64)>, CliError> {
    let display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| io_err(&display, e))?;
    let samples: Vec<ValidationSample> =
        serde_json::from_slice(&bytes).map_err(|e| CliError::DatasetParse {
            path: display,
            message: e.to_string(),
        })?;
    Ok(samples
        .into_iter()
        .map(|s| (s.inputs, s.expected))
        .collect())
}

#[cfg(test)]
mod tests_dataset {
    use super::*;

    #[test]
    fn schema_round_trips() {
        let samples = vec![
            ValidationSample {
                inputs: vec![0.5, 0.2],
                expected: 0.31,
            },
            ValidationSample {
                inputs: vec![0.0, 0.0],
                expected: -0.2,
            },
        ];
        let json = serde_json::to_string(&samples).unwrap();
        let restored: Vec<ValidationSample> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].inputs, vec![0.5, 0.2]);
        assert_eq!(restored[1].expected, -0.2);
    }

    #[test]
    fn malformed_dataset_errors() {
        let dir = std::env::temp_dir().join("zkml_cli_tests_dataset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, b"{ not json }").unwrap();
        assert!(matches!(
            load_dataset(&path),
            Err(CliError::DatasetParse { .. })
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_dataset_is_io_error() {
        let path = Path::new("/nonexistent/zkml/dataset.json");
        assert!(matches!(load_dataset(path), Err(CliError::Io { .. })));
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Parse `raw` and check its length against the model's feature count.
///
/// Checked up front so the error names the model file and both counts, rather
/// than surfacing as a bare `FeatureCountMismatch` from deep inside inference.
fn inputs_for(model: &Model, path: &Path, raw: &str) -> Result<Vec<FixedPoint>, CliError> {
    let inputs = parse_inputs(raw)?;
    let expected = model.num_features();
    if inputs.len() != expected {
        return Err(CliError::FeatureCount {
            model: path.display().to_string(),
            expected,
            got: inputs.len(),
        });
    }
    Ok(inputs)
}

/// Run structural validation that `try_run_inference` does not perform.
///
/// `try_run_inference` validates `TinyMLP` topology but never walks a tree for
/// cycles or orphans, so `inspect`/`validate` call `DecisionTree::validate`
/// explicitly.
fn check_structure(model: &Model, path: &Path) -> Result<(), CliError> {
    let result = match model {
        Model::DecisionTree(tree) => tree.validate(),
        Model::TinyMLP(mlp) => mlp.validate(),
        Model::LogisticRegression(_) => Ok(()),
    };
    result.map_err(|source| CliError::InvalidModel {
        path: path.display().to_string(),
        source,
    })
}

/// Human-readable model kind, matching the JSON `kind` tags.
fn kind_of(model: &Model) -> &'static str {
    match model {
        Model::DecisionTree(_) => "decision_tree",
        Model::LogisticRegression(_) => "logistic_regression",
        Model::TinyMLP(_) => "tiny_mlp",
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Execute a parsed command line, writing normal output to `out`.
pub fn run(cli: &Cli, out: &mut impl Write) -> Result<(), CliError> {
    match &cli.command {
        Command::Commit { model } => cmd_commit(model, out),
        Command::Infer { model, input } => cmd_infer(model, input, out),
        Command::Prove {
            model,
            input,
            out: dest,
        } => cmd_prove(model, input, dest.as_deref(), out),
        Command::Validate {
            model,
            dataset,
            max_input_magnitude,
            min_agreement,
        } => cmd_validate(
            model,
            dataset.as_deref(),
            &QuantizationConfig {
                max_input_magnitude: *max_input_magnitude,
                min_agreement: *min_agreement,
            },
            out,
        ),
        Command::Inspect { model } => cmd_inspect(model, out),
    }
}

/// `commit <MODEL>` — print the model commitment as 64-char hex.
pub fn cmd_commit(model_path: &Path, out: &mut impl Write) -> Result<(), CliError> {
    let model = load_model(model_path)?;
    writeln!(out, "{}", to_hex(&model_commitment(&model))).map_err(|e| io_err("<stdout>", e))
}

/// `infer <MODEL> -i <CSV>` — commitment, dequantized output, raw Q16.16 value.
pub fn cmd_infer(model_path: &Path, raw: &str, out: &mut impl Write) -> Result<(), CliError> {
    let model = load_model(model_path)?;
    let inputs = inputs_for(&model, model_path, raw)?;
    let output = try_run_inference(&model, &inputs).map_err(CliError::Inference)?;

    writeln!(
        out,
        "model commitment: {}",
        to_hex(&model_commitment(&model))
    )
    .and_then(|()| writeln!(out, "output: {}", output.dequantize()))
    .and_then(|()| writeln!(out, "output (raw Q16.16): {}", output.value))
    .map_err(|e| io_err("<stdout>", e))
}

/// `prove <MODEL> -i <CSV> [-o <FILE>]` — emit a `VerificationBundle` as JSON.
pub fn cmd_prove(
    model_path: &Path,
    raw: &str,
    dest: Option<&Path>,
    out: &mut impl Write,
) -> Result<(), CliError> {
    let model = load_model(model_path)?;
    let inputs = inputs_for(&model, model_path, raw)?;

    let bundle = generate_proof(&model, &inputs)
        .map_err(|message| CliError::Inference(ZkmlError::InvalidModel(message)))?;
    let json = bundle_to_json(&bundle).map_err(CliError::Serialization)?;

    match dest {
        Some(path) => {
            let display = path.display().to_string();
            std::fs::write(path, json.as_bytes()).map_err(|e| io_err(&display, e))?;
            writeln!(out, "wrote verification bundle to {display}")
                .map_err(|e| io_err("<stdout>", e))
        }
        None => writeln!(out, "{json}").map_err(|e| io_err("<stdout>", e)),
    }
}

/// `validate <MODEL> [--dataset <FILE>]` — run the quantization passes.
pub fn cmd_validate(
    model_path: &Path,
    dataset_path: Option<&Path>,
    cfg: &QuantizationConfig,
    out: &mut impl Write,
) -> Result<(), CliError> {
    let model = load_model(model_path)?;
    check_structure(&model, model_path)?;

    let dataset = match dataset_path {
        Some(path) => load_dataset(path)?,
        None => Vec::new(),
    };

    let report = validate_model(&model, cfg, &dataset).map_err(CliError::Validation)?;

    writeln!(out, "model: {}", model_path.display())
        .and_then(|()| writeln!(out, "range check: ok"))
        .and_then(|()| {
            writeln!(
                out,
                "overflow bounds: ok (max input magnitude {})",
                cfg.max_input_magnitude
            )
        })
        .and_then(|()| {
            if dataset.is_empty() {
                // An empty dataset yields agreement = 1.0, which would read as a
                // stronger guarantee than was actually checked.
                writeln!(
                    out,
                    "accuracy: not checked (no --dataset; only range and overflow bounds ran)"
                )
            } else {
                writeln!(out, "accuracy: ok over {} sample(s)", report.count)
                    .and_then(|()| {
                        writeln!(
                            out,
                            "  agreement:      {:.2}% (threshold {:.2}%)",
                            report.agreement * 100.0,
                            cfg.min_agreement * 100.0
                        )
                    })
                    .and_then(|()| writeln!(out, "  max deviation:  {:e}", report.max_error))
                    .and_then(|()| writeln!(out, "  mean deviation: {:e}", report.mean_deviation))
            }
        })
        .map_err(|e| io_err("<stdout>", e))
}

/// `inspect <MODEL>` — kind, feature count, structure, commitment, validity.
pub fn cmd_inspect(model_path: &Path, out: &mut impl Write) -> Result<(), CliError> {
    let model = load_model(model_path)?;

    writeln!(out, "model: {}", model_path.display())
        .and_then(|()| writeln!(out, "kind: {}", kind_of(&model)))
        .and_then(|()| writeln!(out, "features: {}", model.num_features()))
        .map_err(|e| io_err("<stdout>", e))?;

    match &model {
        Model::DecisionTree(tree) => {
            let splits = tree
                .nodes
                .iter()
                .filter(|n| matches!(n, TreeNode::Split { .. }))
                .count();
            writeln!(
                out,
                "nodes: {} ({} split, {} leaf)",
                tree.nodes.len(),
                splits,
                tree.nodes.len() - splits
            )
        }
        Model::LogisticRegression(lr) => writeln!(
            out,
            "weights: {}, bias: {}",
            lr.weights.len(),
            lr.bias.dequantize()
        ),
        Model::TinyMLP(mlp) => {
            let shape: Vec<String> = mlp
                .layers
                .iter()
                .map(|l| format!("{}x{}", l.input_size, l.output_size))
                .collect();
            writeln!(out, "layers: {} [{}]", mlp.layers.len(), shape.join(", "))
        }
    }
    .map_err(|e| io_err("<stdout>", e))?;

    writeln!(out, "commitment: {}", to_hex(&model_commitment(&model)))
        .map_err(|e| io_err("<stdout>", e))?;

    // Report validity rather than failing: `inspect` is a diagnostic, and a
    // broken model is exactly when you want to run it.
    let structure = match check_structure(&model, model_path) {
        Ok(()) => "valid".to_string(),
        Err(e) => format!("INVALID ({e})"),
    };
    writeln!(out, "structure: {structure}").map_err(|e| io_err("<stdout>", e))
}

#[cfg(test)]
mod tests_commands {
    use super::*;

    const CREDIT: &str = "../../examples/models/credit_lr.json";
    const KYC: &str = "../../examples/models/kyc_tree.json";

    fn capture(f: impl FnOnce(&mut Vec<u8>) -> Result<(), CliError>) -> Result<String, CliError> {
        let mut buf = Vec::new();
        f(&mut buf)?;
        Ok(String::from_utf8(buf).expect("command output is utf-8"))
    }

    #[test]
    fn commit_prints_64_hex_chars() {
        let out = capture(|w| cmd_commit(Path::new(CREDIT), w)).unwrap();
        let hex = out.trim();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn infer_reports_commitment_and_both_output_forms() {
        let out = capture(|w| cmd_infer(Path::new(CREDIT), "0.5,0.2,0.9,0.1", w)).unwrap();
        assert!(out.contains("model commitment: "));
        assert!(out.contains("output: "));
        assert!(out.contains("output (raw Q16.16): "));
    }

    #[test]
    fn infer_feature_count_mismatch_errors_without_panicking() {
        match cmd_infer(Path::new(CREDIT), "0.5,0.2", &mut Vec::new()) {
            Err(CliError::FeatureCount { expected, got, .. }) => {
                assert_eq!((expected, got), (4, 2));
            }
            other => panic!("expected FeatureCount, got {other:?}"),
        }
    }

    #[test]
    fn prove_writes_bundle_json_to_the_writer() {
        let out = capture(|w| cmd_prove(Path::new(CREDIT), "0.5,0.2,0.9,0.1", None, w)).unwrap();
        let bundle = crate::prover::bundle_from_json(out.trim()).expect("bundle parses");
        assert_eq!(bundle.public_inputs.output.len(), 8);
    }

    #[test]
    fn prove_to_file_reports_the_destination() {
        let dir = std::env::temp_dir().join("zkml_cli_tests_commands");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bundle.json");
        let out = capture(|w| cmd_prove(Path::new(KYC), "0.6,0.1,0.0", Some(&path), w)).unwrap();
        assert!(out.contains("wrote verification bundle to"));
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(crate::prover::bundle_from_json(&written).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn validate_without_dataset_says_accuracy_was_not_checked() {
        let cfg = QuantizationConfig::default();
        let out = capture(|w| cmd_validate(Path::new(CREDIT), None, &cfg, w)).unwrap();
        assert!(out.contains("accuracy: not checked"));
        assert!(out.contains("only range and overflow bounds ran"));
    }

    #[test]
    fn inspect_describes_tree_structure() {
        let out = capture(|w| cmd_inspect(Path::new(KYC), w)).unwrap();
        assert!(out.contains("kind: decision_tree"));
        assert!(out.contains("features: 3"));
        assert!(out.contains("nodes: 5 (2 split, 3 leaf)"));
        assert!(out.contains("structure: valid"));
    }

    #[test]
    fn inspect_describes_logistic_regression() {
        let out = capture(|w| cmd_inspect(Path::new(CREDIT), w)).unwrap();
        assert!(out.contains("kind: logistic_regression"));
        assert!(out.contains("weights: 4"));
    }

    #[test]
    fn missing_model_file_is_an_io_error() {
        let err = cmd_commit(Path::new("/nonexistent/model.json"), &mut Vec::new()).unwrap_err();
        assert!(matches!(err, CliError::Io { .. }));
        assert_eq!(err.exit_code(), 1);
    }
}

#[cfg(test)]
mod tests_args {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn arg_definitions_are_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn validate_defaults_match_quantization_config() {
        let cli = Cli::try_parse_from(["zkml-prover", "validate", "model.json"]).unwrap();
        let defaults = QuantizationConfig::default();
        match cli.command {
            Command::Validate {
                max_input_magnitude,
                min_agreement,
                dataset,
                ..
            } => {
                assert_eq!(max_input_magnitude, defaults.max_input_magnitude);
                assert_eq!(min_agreement, defaults.min_agreement);
                assert!(dataset.is_none());
            }
            other => panic!("expected Validate, got {other:?}"),
        }
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        assert!(Cli::try_parse_from(["zkml-prover", "frobnicate"]).is_err());
    }

    #[test]
    fn infer_requires_input() {
        assert!(Cli::try_parse_from(["zkml-prover", "infer", "model.json"]).is_err());
    }
}
