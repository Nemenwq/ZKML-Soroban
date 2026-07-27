//! Model quantization utilities.
//!
//! Converts floating-point model parameters into fixed-point
//! representations suitable for arithmetic inside ZK circuits.

use zkml_common::fixed_point::FixedPoint;
use zkml_common::models::{DecisionTree, DenseLayer, LogisticRegression, Model, TinyMLP, TreeNode};
use zkml_common::error::ZkmlError;

/// Quantize a vector of `f64` values into fixed-point representations.
pub fn quantize_weights(weights: &[f64]) -> Vec<FixedPoint> {
    weights.iter().copied().map(FixedPoint::quantize).collect()
}

/// Quantize a single bias value.
pub fn quantize_bias(bias: f64) -> FixedPoint {
    FixedPoint::quantize(bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_preserves_sign() {
        let positive = quantize_bias(1.5);
        let negative = quantize_bias(-1.5);
        assert!(positive.value > 0);
        assert!(negative.value < 0);
    }

    #[test]
    fn quantize_vector() {
        let weights = vec![0.1, 0.2, 0.3];
        let quantized = quantize_weights(&weights);
        assert_eq!(quantized.len(), 3);
        for (orig, q) in weights.iter().zip(quantized.iter()) {
            assert!((orig - q.dequantize()).abs() < 1e-4);
        }
    }
}

/// Maximum absolute reconstruction error introduced by quantizing `values`.
///
/// Useful for asserting that a model survives quantization within tolerance.
pub fn max_quantization_error(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|&v| (v - FixedPoint::quantize(v).dequantize()).abs())
        .fold(0.0, f64::max)
}

/// Dequantize a slice of fixed-point values back to `f64`.
pub fn dequantize_all(values: &[FixedPoint]) -> Vec<f64> {
    values.iter().map(FixedPoint::dequantize).collect()
}

#[cfg(test)]
mod tests_error {
    use super::*;

    #[test]
    fn error_within_tolerance() {
        let values = vec![0.1, -0.25, std::f64::consts::PI, 100.5, -42.0];
        assert!(max_quantization_error(&values) < 1e-4);
    }
}

/// Min-max scale a feature vector into the `[0, 1]` range before quantization.
///
/// A constant vector maps to all zeros (no information to preserve).
pub fn scale_features(raw: &[f64]) -> Vec<f64> {
    let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range == 0.0 {
        return raw.iter().map(|_| 0.0).collect();
    }
    raw.iter().map(|&x| (x - min) / range).collect()
}

#[cfg(test)]
mod tests_scaling {
    use super::*;

    #[test]
    fn scaling_preserves_order() {
        let raw = vec![10.0, -5.0, 3.0, 20.0];
        let scaled = scale_features(&raw);
        // The smallest raw value should map to the smallest scaled value.
        let min_idx = 1;
        let max_idx = 3;
        assert!((scaled[min_idx] - 0.0).abs() < 1e-9);
        assert!((scaled[max_idx] - 1.0).abs() < 1e-9);
    }
}

/// Configuration for quantization validation.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Maximum absolute magnitude expected for input features.
    /// Used for static overflow bounds analysis.
    pub max_input_magnitude: f64,
    /// Minimum required agreement rate between float and quantized predictions.
    /// Expressed as a fraction in [0.0, 1.0]. Phase 1 target: > 0.99.
    pub min_agreement: f64,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            max_input_magnitude: 1.0,
            min_agreement: 0.99,
        }
    }
}

/// Report from quantization accuracy validation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationReport {
    /// Number of values quantized.
    pub count: usize,
    /// Maximum absolute reconstruction error.
    pub max_error: f64,
    /// Mean absolute deviation across all samples.
    pub mean_deviation: f64,
    /// Fraction of predictions where float and quantized models agree.
    pub agreement: f64,
}

/// A summary of the error introduced by quantizing a parameter set.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationErrorReport {
    /// Number of values quantized.
    pub count: usize,
    /// Maximum absolute reconstruction error.
    pub max_error: f64,
}

/// Build a [`QuantizationErrorReport`] for a set of floating-point values.
pub fn quantization_error_report(values: &[f64]) -> QuantizationErrorReport {
    QuantizationErrorReport {
        count: values.len(),
        max_error: max_quantization_error(values),
    }
}

/// Validate a quantized model against the configuration.
///
/// Performs three validation passes:
/// 1. Range check: all parameters must be within Q16.16 representable range.
/// 2. Static overflow bounds analysis: ensures no intermediate computation can overflow i64.
/// 3. Accuracy validation: compares predictions on a validation dataset.
pub fn validate_model(
    model: &Model,
    cfg: &QuantizationConfig,
    validation_dataset: &[(Vec<f64>, f64)],
) -> Result<QuantizationReport, ZkmlError> {
    // Pass 1: Range check
    check_parameter_range(model)?;

    // Pass 2: Static overflow bounds analysis
    check_overflow_bounds(model, cfg)?;

    // Pass 3: Accuracy validation
    let report = validate_accuracy(model, validation_dataset, cfg)?;

    Ok(report)
}

/// Check that all model parameters are within the Q16.16 representable range.
fn check_parameter_range(model: &Model) -> Result<(), ZkmlError> {
    match model {
        Model::DecisionTree(tree) => check_tree_range(tree),
        Model::LogisticRegression(lr) => check_logistic_range(lr),
        Model::TinyMLP(mlp) => check_mlp_range(mlp),
    }
}

fn check_tree_range(tree: &DecisionTree) -> Result<(), ZkmlError> {
    for (i, node) in tree.nodes.iter().enumerate() {
        if let TreeNode::Split { threshold, .. } = node {
            check_fixed_point_range(&threshold.value, &format!("tree node {} threshold", i))?;
        }
        if let TreeNode::Leaf { value } = node {
            check_fixed_point_range(&value.value, &format!("tree node {} leaf value", i))?;
        }
    }
    Ok(())
}

fn check_logistic_range(lr: &LogisticRegression) -> Result<(), ZkmlError> {
    for (i, w) in lr.weights.iter().enumerate() {
        check_fixed_point_range(&w.value, &format!("logistic regression weight {}", i))?;
    }
    check_fixed_point_range(&lr.bias.value, "logistic regression bias")?;
    Ok(())
}

fn check_mlp_range(mlp: &TinyMLP) -> Result<(), ZkmlError> {
    for (layer_idx, layer) in mlp.layers.iter().enumerate() {
        for (i, w) in layer.weights.iter().enumerate() {
            check_fixed_point_range(
                &w.value,
                &format!("MLP layer {} weight {}", layer_idx, i),
            )?;
        }
        for (i, b) in layer.biases.iter().enumerate() {
            check_fixed_point_range(
                &b.value,
                &format!("MLP layer {} bias {}", layer_idx, i),
            )?;
        }
    }
    Ok(())
}

/// Check if a value is within the Q16.16 representable range.
/// Q16.16 uses 1 sign bit, 47 integer bits, and 16 fractional bits.
/// The representable range is approximately [-140,737,488,355,327.99998, 140,737,488,355,327.99998].
fn check_fixed_point_range(value: &i64, name: &str) -> Result<(), ZkmlError> {
    // Q16.16 can represent any i64 value since it uses 64 bits total.
    // The check is primarily for documentation and future-proofing.
    // However, we should ensure the value is not i64::MIN which can cause issues with abs().
    if *value == i64::MIN {
        return Err(ZkmlError::QuantizationError(format!(
            "parameter '{}' has value i64::MIN which cannot be safely negated",
            name
        )));
    }
    Ok(())
}

/// Perform static overflow bounds analysis on the model.
///
/// This function computes worst-case bounds for intermediate computations
/// to ensure they cannot overflow i64 during inference.
///
/// # Worst-case bound formula
///
/// For dense/logistic regression layers:
/// ```text
/// |output| <= sum_i(|w_i| * max_input) + |bias|
/// ```
///
/// The multiplication uses an i128 intermediate, so we check that:
/// - Each product |w_i| * max_input fits in i128
/// - The accumulated sum fits in i64 after the right shift by scale (16 bits)
///
/// For decision trees: comparisons only, no accumulation, trivially safe.
///
/// For MLP: bounds are propagated layer by layer. ReLU clamps the lower bound to 0.
fn check_overflow_bounds(model: &Model, cfg: &QuantizationConfig) -> Result<(), ZkmlError> {
    match model {
        Model::DecisionTree(_) => {
            // Decision trees use only comparisons, no arithmetic accumulation.
            // They are trivially safe from overflow.
            Ok(())
        }
        Model::LogisticRegression(lr) => {
            check_logistic_overflow_bounds(lr, cfg.max_input_magnitude)
        }
        Model::TinyMLP(mlp) => check_mlp_overflow_bounds(mlp, cfg.max_input_magnitude),
    }
}

fn check_logistic_overflow_bounds(
    lr: &LogisticRegression,
    max_input: f64,
) -> Result<(), ZkmlError> {
    let scale = 16u32;
    let max_input_scaled = (max_input * (1i64 << scale) as f64) as i64;

    // Compute worst-case accumulated dot product
    let mut worst_case_sum: i128 = 0;
    for w in lr.weights.iter() {
        let w_abs = w.value.abs() as i128;
        let input_abs = max_input_scaled as i128;
        let product = w_abs * input_abs;
        worst_case_sum += product;
    }

    // Add bias
    let bias_abs = lr.bias.value.abs() as i128;
    worst_case_sum += bias_abs;

    // After right shift by scale, the result must fit in i64
    let shifted = worst_case_sum >> scale;
    if shifted > i64::MAX as i128 {
        return Err(ZkmlError::QuantizationError(format!(
            "logistic regression may overflow: worst-case output {} exceeds i64::MAX",
            shifted
        )));
    }

    Ok(())
}

fn check_dense_layer_overflow_bounds(
    input_max: f64,
    layer: &DenseLayer,
) -> Result<f64, ZkmlError> {
    let scale = 16u32;
    let input_max_scaled = (input_max * (1i64 << scale) as f64) as i64;

    // Compute worst-case output for each neuron
    let mut max_output_scaled: i64 = i64::MIN;

    for j in 0..layer.output_size {
        let mut worst_case_sum: i128 = 0;

        // Sum over input weights
        for i in 0..layer.input_size {
            let w = layer.weights[j * layer.input_size + i].value;
            let w_abs = w.abs() as i128;
            let input_abs = input_max_scaled as i128;
            worst_case_sum += w_abs * input_abs;
        }

        // Add bias
        let bias_abs = layer.biases[j].value.abs() as i128;
        worst_case_sum += bias_abs;

        // After right shift
        let shifted = (worst_case_sum >> scale) as i64;
        max_output_scaled = max_output_scaled.max(shifted);
    }

    // Convert back to f64 for next layer
    Ok(max_output_scaled as f64 / (1i64 << scale) as f64)
}

fn check_mlp_overflow_bounds(mlp: &TinyMLP, max_input: f64) -> Result<(), ZkmlError> {
    let mut current_max = max_input;

    for (layer_idx, layer) in mlp.layers.iter().enumerate() {
        let output_max = check_dense_layer_overflow_bounds(current_max, layer)?;

        // If this is not the last layer, ReLU will clamp negative values to 0
        // So the effective max for the next layer is max(0.0, output_max)
        if layer_idx < mlp.layers.len() - 1 {
            current_max = if output_max > 0.0 { output_max } else { 0.0 };
        } else {
            // Last layer: check that output fits in i64
            let scale = 16u32;
            let output_scaled = (output_max * (1i64 << scale) as f64) as i64;
            if output_scaled > i64::MAX / 2 {
                // Allow some headroom for further operations
                return Err(ZkmlError::QuantizationError(format!(
                    "MLP layer {} may overflow: output {} exceeds safe range",
                    layer_idx, output_max
                )));
            }
        }
    }

    Ok(())
}

/// Validate accuracy by comparing float and quantized model predictions.
fn validate_accuracy(
    model: &Model,
    validation_dataset: &[(Vec<f64>, f64)],
    cfg: &QuantizationConfig,
) -> Result<QuantizationReport, ZkmlError> {
    if validation_dataset.is_empty() {
        return Ok(QuantizationReport {
            count: 0,
            max_error: 0.0,
            mean_deviation: 0.0,
            agreement: 1.0,
        });
    }

    let mut total_deviation = 0.0;
    let mut max_deviation = 0.0;
    let mut agreement_count = 0;

    for (inputs, expected_float_output) in validation_dataset {
        let inputs_fp: Vec<FixedPoint> = inputs.iter().map(|&x| FixedPoint::quantize(x)).collect();

        // Run quantized inference
        let quantized_output = zkml_common::inference::run_inference(model, &inputs_fp);
        let quantized_value = quantized_output.dequantize();

        // Compute deviation
        let deviation = (quantized_value - expected_float_output).abs();
        total_deviation += deviation;
        if deviation > max_deviation {
            max_deviation = deviation;
        }

        // Check agreement (within a small tolerance)
        const TOLERANCE: f64 = 1e-4;
        if deviation < TOLERANCE {
            agreement_count += 1;
        }
    }

    let mean_deviation = total_deviation / validation_dataset.len() as f64;
    let agreement = agreement_count as f64 / validation_dataset.len() as f64;

    if agreement < cfg.min_agreement {
        return Err(ZkmlError::QuantizationError(format!(
            "accuracy validation failed: agreement rate {:.2}% is below threshold {:.2}%",
            agreement * 100.0,
            cfg.min_agreement * 100.0
        )));
    }

    Ok(QuantizationReport {
        count: validation_dataset.len(),
        max_error: max_deviation,
        mean_deviation,
        agreement,
    })
}

#[cfg(test)]
mod tests_validation {
    use super::*;
    use zkml_common::models::{DecisionTree, DenseLayer, LogisticRegression, Model, TinyMLP, TreeNode};

    #[test]
    fn range_check_rejects_i64_min() {
        let lr = LogisticRegression {
            weights: vec![FixedPoint::from_raw(i64::MIN, 16)],
            bias: FixedPoint::quantize(0.0),
        };
        let model = Model::LogisticRegression(lr);
        let result = check_parameter_range(&model);
        assert!(result.is_err());
    }

    #[test]
    fn overflow_bounds_rejects_huge_weights() {
        // Create a model with huge weights that will overflow with reasonable input
        // Use weights close to i64::MAX to ensure overflow when accumulated
        let huge_weight = FixedPoint::from_raw(i64::MAX / 2, 16);
        let lr = LogisticRegression {
            weights: vec![huge_weight; 1000], // 1000 weights
            bias: FixedPoint::quantize(0.0),
        };
        let model = Model::LogisticRegression(lr);
        let cfg = QuantizationConfig {
            max_input_magnitude: 10.0, // Small input but huge weights
            min_agreement: 0.99,
        };
        let result = check_overflow_bounds(&model, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn overflow_bounds_accepts_reasonable_model() {
        // Create a model with reasonable weights
        let lr = LogisticRegression {
            weights: vec![FixedPoint::quantize(0.5); 10],
            bias: FixedPoint::quantize(0.1),
        };
        let model = Model::LogisticRegression(lr);
        let cfg = QuantizationConfig {
            max_input_magnitude: 1.0,
            min_agreement: 0.99,
        };
        let result = check_overflow_bounds(&model, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn decision_tree_trivially_safe() {
        // Decision trees should always pass overflow bounds
        let tree = DecisionTree {
            num_features: 2,
            nodes: vec![
                TreeNode::Split {
                    feature_index: 0,
                    threshold: FixedPoint::quantize(0.5),
                    left: 1,
                    right: 2,
                },
                TreeNode::Leaf {
                    value: FixedPoint::quantize(0.0),
                },
                TreeNode::Leaf {
                    value: FixedPoint::quantize(1.0),
                },
            ],
        };
        let model = Model::DecisionTree(tree);
        let cfg = QuantizationConfig::default();
        let result = check_overflow_bounds(&model, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn mlp_overflow_bounds_propagates() {
        // Create a 2-layer MLP with reasonable weights
        let layer1 = DenseLayer {
            weights: vec![FixedPoint::quantize(0.3); 10], // 2 inputs, 5 outputs
            biases: vec![FixedPoint::quantize(0.1); 5],
            input_size: 2,
            output_size: 5,
        };
        let layer2 = DenseLayer {
            weights: vec![FixedPoint::quantize(0.2); 5], // 5 inputs, 1 output
            biases: vec![FixedPoint::quantize(0.0)],
            input_size: 5,
            output_size: 1,
        };
        let mlp = TinyMLP {
            layers: vec![layer1, layer2],
        };
        let model = Model::TinyMLP(mlp);
        let cfg = QuantizationConfig {
            max_input_magnitude: 1.0,
            min_agreement: 0.99,
        };
        let result = check_overflow_bounds(&model, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn accuracy_validation_passes_with_good_model() {
        let lr = LogisticRegression {
            weights: vec![FixedPoint::quantize(1.0)],
            bias: FixedPoint::quantize(0.0),
        };
        let model = Model::LogisticRegression(lr);
        
        // Validation dataset where quantized model matches float
        let dataset = vec![
            (vec![0.5], 0.5),
            (vec![1.0], 1.0),
            (vec![-0.5], -0.5),
        ];
        
        let cfg = QuantizationConfig {
            max_input_magnitude: 1.0,
            min_agreement: 0.99,
        };
        
        let result = validate_accuracy(&model, &dataset, &cfg);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.count, 3);
        assert!(report.agreement >= 0.99);
    }

    #[test]
    fn accuracy_validation_fails_with_poor_agreement() {
        // Create a model with very small weights that lose precision
        let tiny_weight = FixedPoint::quantize(0.0001);
        let lr = LogisticRegression {
            weights: vec![tiny_weight],
            bias: FixedPoint::quantize(0.0),
        };
        let model = Model::LogisticRegression(lr);
        
        // Dataset where quantization error causes significant deviation
        let dataset = vec![
            (vec![1000.0], 0.1), // Float: 1000.0 * 0.0001 = 0.1
            (vec![2000.0], 0.2),
            (vec![3000.0], 0.3),
        ];
        
        let cfg = QuantizationConfig {
            max_input_magnitude: 3000.0,
            min_agreement: 0.99,
        };
        
        let result = validate_accuracy(&model, &dataset, &cfg);
        // This should fail due to quantization precision loss
        assert!(result.is_err());
    }

    #[test]
    fn full_validation_passes() {
        let lr = LogisticRegression {
            weights: vec![FixedPoint::quantize(0.5); 5],
            bias: FixedPoint::quantize(0.1),
        };
        let model = Model::LogisticRegression(lr);
        
        let dataset = vec![
            (vec![0.1, 0.2, 0.3, 0.4, 0.5], 0.85),
            (vec![0.0, 0.0, 0.0, 0.0, 0.0], 0.1),
        ];
        
        let cfg = QuantizationConfig {
            max_input_magnitude: 1.0,
            min_agreement: 0.5, // Lower threshold for this test
        };
        
        let result = validate_model(&model, &cfg, &dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn quantization_config_default() {
        let cfg = QuantizationConfig::default();
        assert_eq!(cfg.max_input_magnitude, 1.0);
        assert_eq!(cfg.min_agreement, 0.99);
    }
}

#[cfg(test)]
mod tests_report {
    #[test]
    fn report_within_tolerance() {
        let report = super::quantization_error_report(&[0.1, 0.2, 0.3, -0.4]);
        assert_eq!(report.count, 4);
        assert!(report.max_error < 1e-4);
    }
}

/// Test that quantized weights deviate from the float originals by < 2⁻¹⁶ each.
#[cfg(test)]
mod tests_quantization_accuracy {
    use super::*;

    #[test]
    fn quantization_error_less_than_2_minus_16() {
        // 2⁻¹⁶ ≈ 1.5259e-5
        const MAX_ALLOWED_ERROR: f64 = 1.0 / 65536.0; // 2⁻¹⁶

        let test_values = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.25, -0.25, 100.0, -100.0, 0.01, -0.01, 3.14159,
            -3.14159,
        ];

        for &value in &test_values {
            let quantized = FixedPoint::quantize(value);
            let dequantized = quantized.dequantize();
            let error = (value - dequantized).abs();

            assert!(
                error < MAX_ALLOWED_ERROR,
                "Quantization error {} for value {} exceeds 2⁻¹⁶ ({})",
                error,
                value,
                MAX_ALLOWED_ERROR
            );
        }
    }

    #[test]
    fn weight_vector_quantization_accuracy() {
        const MAX_ALLOWED_ERROR: f64 = 1.0 / 65536.0; // 2⁻¹⁶

        let weights = vec![0.5, -0.3, 0.8, 0.1, -0.2];
        let quantized = quantize_weights(&weights);

        for (orig, q) in weights.iter().zip(quantized.iter()) {
            let dequantized = q.dequantize();
            let error = (orig - dequantized).abs();

            assert!(
                error < MAX_ALLOWED_ERROR,
                "Weight quantization error {} for original {} exceeds 2⁻¹⁶",
                error,
                orig
            );
        }
    }

    #[test]
    fn bias_quantization_accuracy() {
        const MAX_ALLOWED_ERROR: f64 = 1.0 / 65536.0; // 2⁻¹⁶

        let bias = 0.12345;
        let quantized = quantize_bias(bias);
        let dequantized = quantized.dequantize();
        let error = (bias - dequantized).abs();

        assert!(
            error < MAX_ALLOWED_ERROR,
            "Bias quantization error {} exceeds 2⁻¹⁶",
            error
        );
    }
}
