# Poseidon Commitment Scheme

This document specifies the consensus-critical commitment scheme used for model and input binding in zkml-soroban. The scheme uses Poseidon hash over the BN254 scalar field, matching the Soroban host functions from CAP-0075.

## Overview

The commitment scheme provides two cryptographic bindings:

1. **Model commitment**: A Poseidon hash of all quantized model parameters serves as the on-chain model identifier. This ensures a proof for model M cannot validate for model M'.
2. **Input commitment**: A Poseidon hash of the input feature vector is included as a public input, preventing the prover from swapping inputs after proving.

Both commitments use the same Poseidon parameters to ensure interoperability between off-chain prover computation and on-chain verification.

## Field and Parameters

### Field

- **Field**: BN254 Fr (scalar field of the BN254 elliptic curve)
- **Field order**: `r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001`
- **Host function parameter**: `field = 1` (BN254 Fr)

### Poseidon Parameters

The commitment scheme uses Poseidon with parameters matching the circomlib implementation for BN254, as provided by the `rs-soroban-poseidon` SDK:

- **State size (t)**: 3
- **Rate (r)**: 2 (number of input elements per permutation)
- **Capacity (c)**: 1 (domain separation element)
- **S-box degree (d)**: 5
- **Rounds full (rounds_f)**: 8 (must be even)
- **Rounds partial (rounds_p)**: 57
- **MDS matrix**: Standard Cauchy matrix from circomlib
- **Round constants**: Pre-generated constants from circomlib

These parameters are the default BN254 configuration in `rs-soroban-poseidon` and match the widely-used circomlib implementation, ensuring interoperability with existing ZK tooling.

### Domain Separation

Domain separation prevents cross-contamination between different commitment types:

- **Model commitment**: Capacity element initialized to `1` (domain tag for models)
- **Input commitment**: Capacity element initialized to `2` (domain tag for inputs)

The capacity element is set before absorbing any data and remains unchanged throughout the sponge computation.

## Serialization

### Model Parameter Serialization

Model parameters are serialized into field elements in the following order:

#### Logistic Regression

1. Weights: `weights[0].value, weights[1].value, ..., weights[n-1].value` (little-endian i64)
2. Bias: `bias.value` (little-endian i64)

#### Decision Tree

1. Number of features: `num_features` (as i64)
2. For each node in order (index 0 to n-1):
   - **Split node**: `feature_index` (as i64), `threshold.value` (little-endian i64), `left` (as i64), `right` (as i64)
   - **Leaf node**: `value.value` (little-endian i64)

#### TinyMLP

For each layer in order:
1. Weights: `weights[0].value, weights[1].value, ..., weights[m-1].value` (row-major order, little-endian i64)
2. Biases: `biases[0].value, biases[1].value, ..., biases[k-1].value` (little-endian i64)
3. Input size: `input_size` (as i64)
4. Output size: `output_size` (as i64)

All FixedPoint values are serialized as their raw i64 representation (the quantized integer value, not the dequantized float).

### Input Feature Serialization

Input features are serialized as a flat vector of FixedPoint values:

1. `features[0].value, features[1].value, ..., features[n-1].value` (little-endian i64)

The order matches the feature index order expected by the model.

## Iterative Hashing Construction

Since the Poseidon implementation has a fixed rate (2 elements for t=3), the commitment uses iterative hashing (Merkle-Damgard style) to handle arbitrary-length inputs:

### Hashing Process

1. Initialize with domain tag as the starting hash:
   - `current_hash = domain_tag` (1 for model, 2 for input)

2. For each chunk of rate-sized elements (2 elements for t=3):
   - Create input vector: `[current_hash, chunk[0], chunk[1]]` (or padded with zeros)
   - Apply Poseidon hash to get new hash: `current_hash = poseidon_hash(inputs)`
   - The previous hash carries state between iterations

3. If the final chunk is smaller than rate, pad with zeros before hashing

4. The final `current_hash` is the commitment digest

### Multi-Round Absorption

For inputs exceeding the rate (more than 2 elements), the iterative hashing performs multiple rounds:

```
current_hash = domain_tag
rate = 2  # rate = t - 1 = 2 for t=3
for chunk in elements.chunks(rate):
    inputs = [current_hash] + chunk + [0] * (rate - chunk.len())
    current_hash = poseidon_hash(inputs)
hash = current_hash
```

The key property is that each iteration's output becomes the next iteration's input, ensuring the hash is order-sensitive and non-malleable for arbitrary-length inputs.

## Field Element Conversion

### i64 to BN254 Fr

FixedPoint values are stored as i64 integers. To convert to BN254 Fr field elements with injective sign preservation:

1. For non-negative values: directly convert to Fr
   - `Fr::from(v as u64)` for `v >= 0`

2. For negative values: map to field_order - abs(value)
   - `Fr::from(0u64) - Fr::from(abs(v))` for `v < 0`
   - This ensures +w and -w produce different field elements
   - The mapping is injective because the field order is much larger than i64 range

**Implementation note**: This mapping preserves the sign information in the field element representation, ensuring that flipping the sign of any model parameter changes the commitment.

### BN254 Fr to Bytes

The output field element is converted to 32 bytes using little-endian encoding:

```
bytes = field_element.to_bytes_le()  // 32 bytes
```

## Off-Chain Implementation

The off-chain implementation uses the `light-poseidon` crate with iterative hashing (Merkle-Damgard style) to handle arbitrary-length inputs:

```rust
use light_poseidon::{Poseidon, PoseidonHasher};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};

// Create Poseidon instance with t=3 (rate=2, capacity=1)
// new_circom(2) creates width t = 2 + 1 = 3
let mut poseidon = Poseidon::<Fr>::new_circom(2).unwrap();

// Iterative hashing: chain hashes for inputs exceeding rate
let rate = 2; // rate = t - 1 = 2 for t=3
let mut current_hash = Fr::from(domain);

for chunk in fr_elements.chunks(rate) {
    let mut inputs = vec![current_hash];
    inputs.extend(chunk.iter());
    // Pad with zeros if chunk is smaller than rate
    while inputs.len() < rate {
        inputs.push(Fr::from(0u64));
    }
    current_hash = poseidon.hash(&inputs).unwrap();
}

// Convert Fr to bytes
let hash_bytes = current_hash.into_bigint().to_bytes_le();
```

**Note**: The `light-poseidon` crate provides the circomlib-compatible Poseidon permutation. `new_circom(2)` instantiates a width-3 permutation (t=3 with rate=2, capacity=1, 8 full rounds, 57 partial rounds). Since the `hash()` method is fixed-arity and resets state between calls, we use iterative hashing (Merkle-Damgard style) where each chunk is hashed with the previous hash as input, chaining the results together. This provides a valid approach for arbitrary-length inputs with fixed-rate hash functions.

## On-Chain Usage

The verifier contract uses the Poseidon host function from CAP-0075:

```rust
use soroban_sdk::Env;

// In verify_inference:
let stored_model_hash = env.storage().instance().get(&MODEL_HASH);
let public_model_hash = public_inputs.slice(0..32);

// Verify that the proof's model_hash matches the stored commitment
assert_eq!(stored_model_hash, public_model_hash);
```

The contract does not recompute the hash on-chain (to avoid gas costs). Instead, it verifies that the model_hash public input matches the pre-computed commitment stored during initialization.

### Public Input Byte Order

The flat public-input buffer passed to `verify_inference` is `model_hash (32 bytes) || input_hash (32 bytes) || output`, and every field is **little-endian**, matching this scheme (`to_bytes_le`) and the prover (`i64::to_le_bytes`). soroban's `U256` parses big-endian only, so the verifier's `bytes_to_fr` zero-extends each field to 32 bytes (low order first) and reverses it into big-endian before parsing. Any future circuit or prover integration (issue #11) MUST keep emitting little-endian public inputs so the on-chain scalars match.

### Groth16 Public Input Layout

The verifier constructs the Groth16 pairing check using the public input scalars and the verification key's IC (Input Commitment) points. The computation follows:

```
L = ic[0] + sum(scalar_i * ic[i]) for i from 1 to n
```

where `n` is the total number of public input scalars.

#### Extensible Layout

The public input buffer is parsed into scalars with the following extensible layout:

1. **model_hash**: 32 bytes (canonical Poseidon commitment)
2. **input_hash**: 32 bytes (canonical Poseidon commitment)
3. **output_scalars**: N × 8 bytes (each canonical i64 in little-endian)

**Canonical length requirements**:
- `model_hash` must be exactly 32 bytes
- `input_hash` must be exactly 32 bytes
- Each output scalar must be exactly 8 bytes (canonical i64)
- Total output length must be a multiple of 8 bytes

The verifier rejects any public input that violates these canonical length requirements with `InvalidPublicInputLength`.

#### Multi-Scalar Output

The output field supports multiple scalars for multi-class inference decisions:

- **Single-class output**: 8 bytes (one i64)
- **Multi-class output**: N × 8 bytes (N i64 values, e.g., class probabilities or logits)

For example, a 3-class classifier would have:
- Total public inputs: 32 + 32 + 24 = 88 bytes
- Parsed scalars: [model_hash, input_hash, output_0, output_1, output_2]
- Required IC points: 6 (ic[0], ic[1], ic[2], ic[3], ic[4], ic[5])

#### Verification Key Validation

The verification key must contain exactly `n + 1` IC points where `n` is the number of parsed public input scalars. The verifier validates this with `VerificationKeyLengthMismatch` if the lengths don't match.

This validation ensures that the verification key was generated for the exact public input structure being verified, preventing mismatched key/data pairs.

## Security Properties

### Collision Resistance

The Poseidon hash function with the specified parameters provides collision resistance up to the 128-bit security level of BN254. Any change to the serialized parameters will produce a different hash with overwhelming probability.

### Binding

- **Model binding**: The model commitment is stored on-chain during initialization. A proof for model M will only verify if the public input contains the hash of M. Changing any model parameter changes the hash, breaking the proof.
- **Input binding**: The input commitment is included in the public inputs. The prover cannot substitute a different input after generating the proof without invalidating the hash.

### Uniqueness

The domain separation ensures that model commitments and input commitments live in distinct hash spaces, preventing cross-type collisions.

## Test Vectors

Test vectors are provided in the `zkml-common` crate to ensure off-chain and on-chain implementations produce identical digests. See `crates/zkml-common/tests/commitment_cross_check.rs` for the cross-check test.

## References

- CAP-0075: Poseidon hash host functions (Stellar Protocol 25)
- rs-soroban-poseidon: https://github.com/stellar/rs-soroban-poseidon
- circomlib Poseidon implementation: https://github.com/iden3/circomlib/blob/master/circuits/poseidon.circom
- Poseidon paper: "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems" (Grassi et al., 2021)
