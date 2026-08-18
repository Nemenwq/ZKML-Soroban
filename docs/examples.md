# Examples

The `examples/models` directory contains ready-to-run models in the JSON
exchange format. See [the CLI reference](cli.md) for the full subcommand surface.

## Credit scoring (logistic regression)

```bash
cargo run -p zkml-prover -- infer examples/models/credit_lr.json -i "0.5,0.2,0.9,0.1"
```

A positive output suggests the applicant clears the risk threshold; the verifier
contract compares the raw linear output against a configured cutoff.

Export a verification bundle for the same evaluation:

```bash
cargo run -p zkml-prover -- prove examples/models/credit_lr.json \
  -i "0.5,0.2,0.9,0.1" -o bundle.json
```

(The Groth16 proof bytes in that bundle are a placeholder until issue #11.)

## KYC risk (decision tree)

```bash
cargo run -p zkml-prover -- infer examples/models/kyc_tree.json -i "0.6,0.1,0.0"
```

The tree returns a leaf value of `1.0` for the high-risk branch and `0.0`
otherwise.

Inspect its structure and commitment without running inference:

```bash
cargo run -p zkml-prover -- inspect examples/models/kyc_tree.json
```
