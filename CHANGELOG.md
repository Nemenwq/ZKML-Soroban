# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `zkml-prover` CLI subcommands (`commit`, `infer`, `prove`, `validate`,
  `inspect`) built on clap, with strict `--input` parsing that rejects
  unparseable, empty, non-finite, and out-of-range fields instead of silently
  dropping them. Replaces the positional `zkml-prover <model> "<inputs>"` form.
- TinyMLP fixed-point inference with checked Q16.16 dense layers, ReLU on
  hidden layers only, topology validation via `TinyMLP::validate` /
  `try_run_inference`, and prover acceptance tests (hand-computed 2→2→1,
  ReLU, shape errors, golden float bound).
- ONNX importer foundation: protobuf parse, per-domain opset validation
  (core >= 17, ai.onnx.ml >= 1), operator allowlist, and typed
  `OnnxImportError` (parameter extraction deferred to #5/#6).
- `model_io::import_json` for the JSON exchange path used by the CLI and demos.
- RISC Zero guest program and host `generate_receipt` with journal
  cross-checks (dev-mode CI; real proving documented and `#[ignore]`d).
- Shared inference engine in `zkml-common` for native and guest paths.
- `commitment_hash` abstraction over model/input binding (Poseidon: #13).
- Minimal golden vectors under `crates/zkml-prover/tests/vectors/`.
- `PartialOrd` / `Ord` for `FixedPoint` (same-scale raw integer comparison).
- `Add`, `Sub`, and `Mul` operators as panicking wrappers over checked arithmetic.
- `Neg` for `FixedPoint` so negation uses the standard unary operator.
- `FixedPoint::abs`, `is_zero`, and `signum` helpers.
- `FixedPoint::clamp` for range saturation.
- `sum` and `mean` slice reductions for pooling layers.
- `max`, `min`, and `argmax` slice reductions for max-pooling and classification.
- ZK-friendly `leaky_relu` activation with a power-of-two slope.
- `relu6` bounded activation for quantized networks.
- `hard_sigmoid` and `hard_swish` piecewise-linear activations.
- `hardtanh` bounded activation clamping to `[-1, 1]`.
- Element-wise `relu6_vec`, `hard_sigmoid_vec`, and `hard_swish_vec` helpers.
- Initial workspace scaffold with `zkml-common`, `zkml-prover`, and
  `zkml-verifier` crates.

### Changed
- `prover::generate_proof` runs `try_run_inference` instead of the panicking
  `run_inference`, so feature-count and overflow failures surface as `Err`
  rather than aborting the caller.

[Unreleased]: https://github.com/diegoveme/ZKML-Soroban/compare/main...HEAD

## [0.2.0] - 2026-06-17

### Added
- Fixed-point checked/saturating arithmetic, division, and dot product.
- Quantized ReLU activation, `Tensor` type, and model validation helpers.
- Model and input commitments plus a Merkle tree over parameters.
- JSON model import, batch inference, and validated inference.
- A prover CLI binary printing the model commitment and output.
- Verification bundle assembly, JSON serialization, and bundle ids.
- Verifier contract: public input parsing, events, and query methods.
- Documentation set, CI with fmt/clippy, and contributor tooling.

[0.2.0]: https://github.com/diegoveme/ZKML-Soroban/releases/tag/v0.2.0
