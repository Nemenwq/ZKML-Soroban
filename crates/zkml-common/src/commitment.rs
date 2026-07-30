//! Commitment helpers binding models and inputs to a proof.
//!
//! The verifier checks that a proof corresponds to a specific model and input
//! by comparing Poseidon commitments. The on-chain side uses the Poseidon host
//! function from CAP-0075; off-chain we expose a stable hashing interface so
//! the prover can compute matching commitments.
//!
//! This implementation uses Poseidon with circomlib-compatible parameters over
//! the BN254 scalar field, matching the Soroban host function configuration.

use crate::models::{Model, TreeNode};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use light_poseidon::{Poseidon, PoseidonHasher};

/// A 32-byte commitment value.
pub type Commitment = [u8; 32];

/// Domain tag for model commitments (capacity element initialization).
const MODEL_DOMAIN: u64 = 1;

/// Domain tag for input commitments (capacity element initialization).
const INPUT_DOMAIN: u64 = 2;

/// State size for Poseidon (t=3: rate=2, capacity=1).
/// This matches the circomlib implementation and CAP-0075 defaults.
/// Note: light-poseidon's new_circom(n) creates width t = n + 1, so new_circom(2) creates t=3.
const STATE_SIZE: usize = 3;

/// Compute a Poseidon commitment over a sequence of i64 field elements.
///
/// This uses the circomlib-compatible Poseidon parameters over BN254:
/// - State size t=3 (rate=2, capacity=1)
/// - S-box degree x^5
/// - 8 full rounds, 57 partial rounds
///
/// Domain separation is achieved by prepending the domain tag to the input stream.
/// For inputs exceeding the rate, we use iterative hashing (Merkle-Damgard style):
/// each chunk is hashed with the previous hash as input, chaining the results.
///
/// # Arguments
///
/// * `elements` - Sequence of i64 values to hash
/// * `domain` - Domain tag (1 for model, 2 for input)
fn poseidon_commit(elements: &[i64], domain: u64) -> Commitment {
    let rate = STATE_SIZE - 1; // rate = 2 for t=3

    // Convert i64 elements to Fr field elements with sign preservation
    // For injective mapping: negative values map to (field_order - abs(value))
    // This ensures +w and -w produce different field elements
    let fr_elements: Vec<Fr> = elements
        .iter()
        .map(|&v| {
            if v >= 0 {
                Fr::from(v as u64)
            } else {
                // Map negative to field_order - abs(v) for injective mapping
                let abs_val = v.unsigned_abs();
                Fr::from(0u64) - Fr::from(abs_val)
            }
        })
        .collect();

    // Create Poseidon instance with t=3 (rate=2, capacity=1)
    // new_circom(2) creates width t = 2 + 1 = 3
    let mut poseidon = Poseidon::<Fr>::new_circom(2).unwrap();

    // Iterative hashing: chain hashes for inputs exceeding rate
    // Start with domain tag as initial hash
    let mut current_hash = Fr::from(domain);

    for (chunk_index, chunk) in fr_elements.chunks(rate).enumerate() {
        let mut inputs = chunk.to_vec();
        // Pad with zeros if chunk is smaller than rate
        while inputs.len() < rate {
            inputs.push(Fr::from(0u64));
        }
        // Mix previous hash into first element to preserve order
        inputs[0] = inputs[0] + current_hash + Fr::from(chunk_index as u64);
        let chunk_hash = poseidon.hash(&inputs).unwrap();
        current_hash = chunk_hash;
    }

    // Convert Fr to bytes
    let hash_bytes = current_hash.into_bigint().to_bytes_le();
    let mut result = [0u8; 32];
    result.copy_from_slice(&hash_bytes);
    result
}

/// Canonical commitment over little-endian `i64` field elements.
///
/// Host and zkVM guest must call this same function so journal public inputs
/// match native cross-checks.
///
/// This is implemented using Poseidon with circomlib-compatible parameters
/// over BN254, matching the Soroban host function configuration from CAP-0075.
/// For inputs exceeding the rate, iterative hashing (Merkle-Damgard style) is used.
///
/// # Arguments
///
/// * `elements` - Sequence of i64 values to hash
///
/// # Returns
///
/// 32-byte Poseidon commitment
pub fn commitment_hash(elements: &[i64]) -> Commitment {
    // Use domain 0 for generic commitment_hash (legacy compatibility)
    poseidon_commit(elements, 0)
}

/// Flatten model parameters into the element stream used by
/// [`commitment_hash`] for model binding.
///
/// Shared by the host prover and the zkVM guest so journal `model_hash`
/// cannot drift from native `model_commitment`.
pub fn model_elements(model: &Model) -> Vec<i64> {
    let mut out = Vec::new();
    match model {
        Model::LogisticRegression(lr) => {
            out.extend(lr.weights.iter().map(|w| w.value));
            out.push(lr.bias.value);
        }
        Model::DecisionTree(tree) => {
            out.push(tree.num_features as i64);
            for node in &tree.nodes {
                match node {
                    TreeNode::Split {
                        feature_index,
                        threshold,
                        left,
                        right,
                    } => {
                        out.push(*feature_index as i64);
                        out.push(threshold.value);
                        out.push(*left as i64);
                        out.push(*right as i64);
                    }
                    TreeNode::Leaf { value } => out.push(value.value),
                }
            }
        }
        Model::TinyMLP(mlp) => {
            for layer in &mlp.layers {
                out.extend(layer.weights.iter().map(|w| w.value));
                out.extend(layer.biases.iter().map(|b| b.value));
                out.push(layer.input_size as i64);
                out.push(layer.output_size as i64);
            }
        }
    }
    out
}

/// Compute a Poseidon commitment to a model.
///
/// This hashes all quantized model parameters using the serialization order
/// specified in docs/commitment-scheme.md, with domain separation tag 1.
///
/// # Arguments
///
/// * `model` - The model to commit to
///
/// # Returns
///
/// 32-byte Poseidon commitment
pub fn commit_model(model: &Model) -> Commitment {
    let elements = model_elements(model);
    poseidon_commit(&elements, MODEL_DOMAIN)
}

/// Compute a Poseidon commitment to input features.
///
/// This hashes the input feature vector with domain separation tag 2.
///
/// # Arguments
///
/// * `features` - Slice of FixedPoint input features
///
/// # Returns
///
/// 32-byte Poseidon commitment
pub fn commit_inputs(features: &[crate::fixed_point::FixedPoint]) -> Commitment {
    let elements: Vec<i64> = features.iter().map(|f| f.value).collect();
    poseidon_commit(&elements, INPUT_DOMAIN)
}

/// Fold a sequence of little-endian `i64` field elements into a commitment.
///
/// This is now implemented using Poseidon with circomlib-compatible parameters
/// over BN254, matching the Soroban host function configuration from CAP-0075.
///
/// # Arguments
///
/// * `elements` - Sequence of i64 values to hash
///
/// # Returns
///
/// 32-byte Poseidon commitment
pub fn commit_i64(elements: &[i64]) -> Commitment {
    poseidon_commit(elements, 0)
}

/// Encode a commitment as a 64-character lowercase hex string.
pub fn to_hex(c: &Commitment) -> String {
    let mut s = String::with_capacity(64);
    for b in c.iter() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Decode a 64-character hex string into a commitment, if well-formed.
pub fn from_hex(s: &str) -> Option<Commitment> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests_hex {
    use super::*;

    #[test]
    fn hex_round_trips() {
        let c = commit_i64(&[1, 2, 3, 4]);
        let encoded = to_hex(&c);
        assert_eq!(encoded.len(), 64);
        assert_eq!(from_hex(&encoded), Some(c));
    }

    #[test]
    fn from_hex_rejects_bad_length() {
        assert_eq!(from_hex("abcd"), None);
    }
}

#[cfg(test)]
mod tests_stability {
    use super::*;

    #[test]
    fn commitment_is_stable() {
        assert_eq!(commit_i64(&[1, 2, 3]), commit_i64(&[1, 2, 3]));
    }

    #[test]
    fn commitment_is_order_sensitive() {
        assert_ne!(commit_i64(&[1, 2, 3]), commit_i64(&[3, 2, 1]));
    }

    #[test]
    fn empty_commitment_is_deterministic() {
        let empty1 = commit_i64(&[]);
        let empty2 = commit_i64(&[]);
        assert_eq!(empty1, empty2);
    }

    #[test]
    fn domain_separation_produces_different_hashes() {
        let elements = vec![1, 2, 3];
        let hash_domain1 = poseidon_commit(&elements, 1);
        let hash_domain2 = poseidon_commit(&elements, 2);
        assert_ne!(hash_domain1, hash_domain2);
    }
}

#[cfg(test)]
mod tests_model_elements {
    use super::*;
    use crate::fixed_point::FixedPoint;
    use crate::models::{DecisionTree, LogisticRegression, Model, TreeNode};

    #[test]
    fn logistic_flattens_weights_then_bias() {
        let model = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::from_raw(1, 16), FixedPoint::from_raw(2, 16)],
            bias: FixedPoint::from_raw(3, 16),
        });
        assert_eq!(model_elements(&model), vec![1, 2, 3]);
    }

    #[test]
    fn tree_flattens_features_and_nodes() {
        let model = Model::DecisionTree(DecisionTree {
            num_features: 1,
            nodes: vec![
                TreeNode::Split {
                    feature_index: 0,
                    threshold: FixedPoint::from_raw(10, 16),
                    left: 1,
                    right: 2,
                },
                TreeNode::Leaf {
                    value: FixedPoint::from_raw(0, 16),
                },
                TreeNode::Leaf {
                    value: FixedPoint::from_raw(1, 16),
                },
            ],
        });
        assert_eq!(model_elements(&model), vec![1, 0, 10, 1, 2, 0, 1]);
    }

    #[test]
    fn commit_model_produces_deterministic_hash() {
        let model = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::from_raw(1, 16), FixedPoint::from_raw(2, 16)],
            bias: FixedPoint::from_raw(3, 16),
        });
        let hash1 = commit_model(&model);
        let hash2 = commit_model(&model);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn commit_inputs_produces_deterministic_hash() {
        let features = vec![
            FixedPoint::from_raw(1, 16),
            FixedPoint::from_raw(2, 16),
            FixedPoint::from_raw(3, 16),
        ];
        let hash1 = commit_inputs(&features);
        let hash2 = commit_inputs(&features);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn changing_model_parameter_changes_commitment() {
        let model1 = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::from_raw(1, 16), FixedPoint::from_raw(2, 16)],
            bias: FixedPoint::from_raw(3, 16),
        });
        let model2 = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::from_raw(1, 16), FixedPoint::from_raw(99, 16)], // Changed weight
            bias: FixedPoint::from_raw(3, 16),
        });
        assert_ne!(commit_model(&model1), commit_model(&model2));
    }

    #[test]
    fn changing_input_changes_commitment() {
        let features1 = vec![FixedPoint::from_raw(1, 16), FixedPoint::from_raw(2, 16)];
        let features2 = vec![FixedPoint::from_raw(1, 16), FixedPoint::from_raw(99, 16)]; // Changed input
        assert_ne!(commit_inputs(&features1), commit_inputs(&features2));
    }
}

#[cfg(test)]
mod tests_snapshot {
    use super::*;
    use crate::fixed_point::FixedPoint;
    use crate::models::{DecisionTree, DenseLayer, LogisticRegression, Model, TinyMLP, TreeNode};
    use insta::assert_debug_snapshot;

    #[test]
    fn snapshot_logistic_regression_commitment() {
        let model = Model::LogisticRegression(LogisticRegression {
            weights: vec![
                FixedPoint::from_raw(100, 16),
                FixedPoint::from_raw(200, 16),
                FixedPoint::from_raw(300, 16),
            ],
            bias: FixedPoint::from_raw(50, 16),
        });
        let hash = commit_model(&model);
        assert_debug_snapshot!(hash);
    }

    #[test]
    fn snapshot_decision_tree_commitment() {
        let model = Model::DecisionTree(DecisionTree {
            num_features: 2,
            nodes: vec![
                TreeNode::Split {
                    feature_index: 0,
                    threshold: FixedPoint::from_raw(150, 16),
                    left: 1,
                    right: 2,
                },
                TreeNode::Leaf {
                    value: FixedPoint::from_raw(0, 16),
                },
                TreeNode::Split {
                    feature_index: 1,
                    threshold: FixedPoint::from_raw(200, 16),
                    left: 3,
                    right: 4,
                },
                TreeNode::Leaf {
                    value: FixedPoint::from_raw(1, 16),
                },
                TreeNode::Leaf {
                    value: FixedPoint::from_raw(2, 16),
                },
            ],
        });
        let hash = commit_model(&model);
        assert_debug_snapshot!(hash);
    }

    #[test]
    fn snapshot_tiny_mlp_commitment() {
        let model = Model::TinyMLP(TinyMLP {
            layers: vec![DenseLayer {
                weights: vec![FixedPoint::from_raw(10, 16), FixedPoint::from_raw(20, 16)],
                biases: vec![FixedPoint::from_raw(5, 16)],
                input_size: 2,
                output_size: 1,
            }],
        });
        let hash = commit_model(&model);
        assert_debug_snapshot!(hash);
    }

    #[test]
    fn snapshot_input_commitment() {
        let features = vec![
            FixedPoint::from_raw(100, 16),
            FixedPoint::from_raw(200, 16),
            FixedPoint::from_raw(300, 16),
        ];
        let hash = commit_inputs(&features);
        assert_debug_snapshot!(hash);
    }

    #[test]
    fn snapshot_empty_input_commitment() {
        let features: Vec<FixedPoint> = vec![];
        let hash = commit_inputs(&features);
        assert_debug_snapshot!(hash);
    }
}

#[cfg(test)]
mod test_determinism {
    use super::*;

    /// Determinism test for Poseidon commitment
    ///
    /// This test verifies that the off-chain implementation produces consistent
    /// digests for the same input. This is a basic sanity check, not a cross-check
    /// against the on-chain CAP-0075 host function.
    ///
    /// TODO: To implement a true on-chain equivalence test, we need:
    /// 1. Access to Soroban testnet/devnet to run the CAP-0075 host function
    /// 2. OR integration with soroban-poseidon crate to compute reference digest
    /// 3. A known test vector: fixed input -> expected digest from on-chain source
    #[test]
    fn poseidon_commitment_is_deterministic() {
        let input = [1i64, 2i64, 3i64];
        let digest1 = poseidon_commit(&input, 1);
        let digest2 = poseidon_commit(&input, 1);
        assert_eq!(digest1, digest2, "commitment should be deterministic");
    }
}
