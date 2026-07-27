//! Error types shared across the zkml-soroban crates.

use serde::{Deserialize, Serialize};

/// Errors that can occur during model handling, quantization, inference,
/// or proof preparation in the off-chain components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkmlError {
    /// The number of inputs did not match the model's expected feature count.
    FeatureCountMismatch { expected: usize, got: usize },
    /// A model definition was structurally invalid.
    InvalidModel(String),
    /// Parsing an external model representation failed.
    ParseError(String),
    /// A fixed-point operation overflowed.
    ArithmeticOverflow,
    /// Quantization validation failed.
    QuantizationError(String),
}

impl core::fmt::Display for ZkmlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZkmlError::FeatureCountMismatch { expected, got } => {
                write!(f, "feature count mismatch: expected {expected}, got {got}")
            }
            ZkmlError::InvalidModel(m) => write!(f, "invalid model: {m}"),
            ZkmlError::ParseError(m) => write!(f, "parse error: {m}"),
            ZkmlError::ArithmeticOverflow => write!(f, "arithmetic overflow"),
            ZkmlError::QuantizationError(m) => write!(f, "quantization error: {m}"),
        }
    }
}

impl std::error::Error for ZkmlError {}
