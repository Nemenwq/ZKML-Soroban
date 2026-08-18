//! # zkml-prover
//!
//! Off-chain component responsible for:
//!
//! 1. Importing ML models from ONNX format.
//! 2. Quantizing floating-point weights into fixed-point representations.
//! 3. Executing model inference inside a ZK-provable environment.
//! 4. Generating Groth16 proofs that attest to correct inference.

pub mod cli;
pub mod inference;
pub mod model_io;
pub mod onnx;
pub mod prover;
pub mod quantization;
pub mod timing;
