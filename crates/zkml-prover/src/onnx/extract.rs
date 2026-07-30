//! Parameter extraction from ONNX operators into internal model types.

use super::error::OnnxImportError;
use super::proto::NodeProto;
use zkml_common::fixed_point::FixedPoint;
use zkml_common::models::LogisticRegression;

/// Extract a binary logistic regression model from a LinearClassifier node.
///
/// # Arguments
///
/// * `node` - The LinearClassifier node to extract from
///
/// # Errors
///
/// - Returns an error if the model is multi-class (more than 2 classes)
/// - Returns an error if required attributes (coefficients, intercepts) are missing
/// - Returns an error if post_transform is not NONE or LOGISTIC
pub fn extract_linear_classifier(node: &NodeProto) -> Result<LogisticRegression, OnnxImportError> {
    // Extract coefficients (weights)
    let coefficients = get_floats_attribute(node, "coefficients").ok_or_else(|| {
        OnnxImportError::MalformedModel("LinearClassifier missing 'coefficients' attribute".into())
    })?;

    // Extract intercepts (bias)
    let intercepts = get_floats_attribute(node, "intercepts").ok_or_else(|| {
        OnnxImportError::MalformedModel("LinearClassifier missing 'intercepts' attribute".into())
    })?;

    // Check for multi-class (more than 1 coefficient vector)
    // For binary classification, skl2onnx stores coefficients as [num_features]
    // For multi-class, it stores as [num_classes * num_features]

    // Check if this is multi-class by looking at classlabels attributes
    let classlabels_ints = get_ints_attribute(node, "classlabels_ints");
    let classlabels_strings = get_strings_attribute(node, "classlabels_strings");

    let num_classes = if let Some(labels) = classlabels_ints {
        labels.len()
    } else if let Some(labels) = classlabels_strings {
        labels.len()
    } else {
        // If no classlabels, infer from intercepts length
        intercepts.len()
    };

    // Reject multi-class models
    if num_classes > 2 {
        return Err(OnnxImportError::MalformedModel(format!(
            "Multi-class LinearClassifier ({} classes) is not supported yet. Binary classification only. See issue #6 for multi-class support.",
            num_classes
        )));
    }

    // Validate intercepts length (should be 1 for binary classification)
    if intercepts.len() != 1 {
        return Err(OnnxImportError::MalformedModel(format!(
            "Binary LinearClassifier should have exactly 1 intercept, found {}",
            intercepts.len()
        )));
    }

    // Check post_transform attribute
    if let Some(post_transform) = get_string_attribute(node, "post_transform") {
        if post_transform != "NONE" && post_transform != "LOGISTIC" {
            return Err(OnnxImportError::MalformedModel(format!(
                "Unsupported post_transform '{}'. Only NONE and LOGISTIC are supported (LOGISTIC is dropped in favor of thresholding the raw score).",
                post_transform
            )));
        }
        // Note: LOGISTIC is intentionally dropped because thresholding the raw score
        // is equivalent for binary decisions and sigmoid is not ZK-friendly
    }

    // Convert to f64 for quantization
    let weights_f64: Vec<f64> = coefficients.to_vec();
    let bias_f64 = intercepts[0] as f64;

    // Quantize using existing helpers
    let quantized_weights = weights_f64
        .iter()
        .map(|&w| FixedPoint::quantize(w))
        .collect();
    let quantized_bias = FixedPoint::quantize(bias_f64);

    Ok(LogisticRegression {
        weights: quantized_weights,
        bias: quantized_bias,
    })
}

/// Helper to get a list of floats from an attribute.
pub(crate) fn get_floats_attribute(node: &NodeProto, name: &str) -> Option<Vec<f64>> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .and_then(|attr| {
            if !attr.floats.is_empty() {
                Some(attr.floats.clone())
            } else if !attr.floats_f32.is_empty() {
                Some(attr.floats_f32.iter().map(|&x| x as f64).collect())
            } else {
                None
            }
        })
}

/// Helper to get a list of ints from an attribute.
fn get_ints_attribute(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .and_then(|attr| {
            if !attr.ints.is_empty() {
                Some(attr.ints.clone())
            } else {
                None
            }
        })
}

/// Helper to get a list of strings from an attribute.
fn get_strings_attribute(node: &NodeProto, name: &str) -> Option<Vec<String>> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .and_then(|attr| {
            if !attr.strings.is_empty() {
                Some(attr.strings.clone())
            } else if !attr.strings_bytes.is_empty() {
                Some(
                    attr.strings_bytes
                        .iter()
                        .filter_map(|bytes| String::from_utf8(bytes.clone()).ok())
                        .collect(),
                )
            } else if !attr.s.is_empty() {
                // Single value stored in 's' field
                String::from_utf8(attr.s.clone()).ok().map(|s| vec![s])
            } else {
                None
            }
        })
}

/// Helper to get a single string from an attribute.
fn get_string_attribute(node: &NodeProto, name: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .and_then(|attr| {
            if !attr.strings.is_empty() {
                attr.strings.first().cloned()
            } else if !attr.strings_bytes.is_empty() {
                attr.strings_bytes
                    .first()
                    .and_then(|bytes| String::from_utf8(bytes.clone()).ok())
            } else if !attr.s.is_empty() {
                String::from_utf8(attr.s.clone()).ok()
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::proto::AttributeProto;

    fn make_node_with_coefficients(coeffs: Vec<f32>, intercepts: Vec<f32>) -> NodeProto {
        NodeProto {
            name: "test".into(),
            op_type: "LinearClassifier".into(),
            domain: "ai.onnx.ml".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            attribute: vec![
                AttributeProto {
                    name: "coefficients".into(),
                    floats: coeffs.iter().map(|&x| x as f64).collect(),
                    f: 0.0,
                    i: 0,
                    ints: vec![],
                    s: vec![],
                    strings: vec![],
                    t: None,
                    g: None,
                    floats_f32: vec![],
                    ints_extra: vec![],
                    strings_bytes: vec![],
                    sparse_tensor: None,
                    r#type: 0,
                },
                AttributeProto {
                    name: "intercepts".into(),
                    floats: intercepts.iter().map(|&x| x as f64).collect(),
                    f: 0.0,
                    i: 0,
                    ints: vec![],
                    s: vec![],
                    strings: vec![],
                    t: None,
                    g: None,
                    floats_f32: vec![],
                    ints_extra: vec![],
                    strings_bytes: vec![],
                    sparse_tensor: None,
                    r#type: 0,
                },
            ],
        }
    }

    #[test]
    fn extract_binary_classifier_succeeds() {
        let node = make_node_with_coefficients(vec![0.5, -0.3, 0.8], vec![0.1]);
        let result = extract_linear_classifier(&node);
        assert!(result.is_ok());
        let lr = result.unwrap();
        assert_eq!(lr.weights.len(), 3);
    }

    #[test]
    fn multi_class_rejected() {
        let mut node = make_node_with_coefficients(vec![0.5, -0.3, 0.8], vec![0.1, 0.2]);
        node.attribute.push(AttributeProto {
            name: "classlabels_ints".into(),
            floats: vec![],
            f: 0.0,
            i: 0,
            ints: vec![0, 1, 2], // 3 classes
            s: vec![],
            strings: vec![],
            t: None,
            g: None,
            floats_f32: vec![],
            ints_extra: vec![],
            strings_bytes: vec![],
            sparse_tensor: None,
            r#type: 0,
        });
        let result = extract_linear_classifier(&node);
        assert!(result.is_err());
        match result.unwrap_err() {
            OnnxImportError::MalformedModel(msg) => {
                assert!(msg.contains("Multi-class"));
                assert!(msg.contains("3 classes"));
            }
            _ => panic!("Expected MalformedModel error"),
        }
    }
}
