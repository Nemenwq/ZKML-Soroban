# Prover CLI

The `zkml-prover` binary exposes the off-chain pipeline — commitments,
inference, bundle export, and quantization validation — as subcommands.

> **Breaking change (issue #44).** The old positional form
> `zkml-prover <model.json> "<inputs>"` is gone; use `infer` (see below). The old
> form silently dropped any input field that failed to parse, so a typo in the
> feature vector ran inference against the wrong feature count instead of
> erroring.

## Build

```bash
cargo build -p zkml-prover
```

## Commands

```text
zkml-prover <COMMAND>

commit   <MODEL>                                Model commitment as 64-char hex
infer    <MODEL> -i <CSV>                       Commitment + dequantized output + raw Q16.16
prove    <MODEL> -i <CSV> [-o <FILE>]           VerificationBundle JSON (stdout or file)
validate <MODEL> [--dataset <FILE>]
                 [--max-input-magnitude <F>]    default 1.0
                 [--min-agreement <F>]          default 0.99
inspect  <MODEL>                                Kind, features, structure, commitment, validity
```

`<MODEL>` is a file in the [JSON exchange format](model-format.md). ONNX files
are **not** accepted: parameter extraction currently handles only single-node
`TreeEnsembleClassifier` / `LinearClassifier` graphs (issues #5 / #6), so most
real `.onnx` files would fail confusingly. Import them separately for now.

`-i/--input` is a single comma-separated string, e.g. `"0.5,0.2,0.9,0.1"`.

### `commit`

Prints the model commitment — the value registered on-chain at `initialize`.

```bash
cargo run -p zkml-prover -- commit examples/models/credit_lr.json
```

```text
73e881eda85b98eef6a08eec16e3210330c6bd53a182eab7944c87d9c4fce710
```

### `infer`

```bash
cargo run -p zkml-prover -- infer examples/models/credit_lr.json -i "0.5,0.2,0.9,0.1"
```

```text
model commitment: 73e881eda85b98eef6a08eec16e3210330c6bd53a182eab7944c87d9c4fce710
output: 0.5319976806640625
output (raw Q16.16): 34865
```

The raw value is `FixedPoint::value`, the integer that goes into the proof's
public inputs; the dequantized value is for humans.

### `prove`

Writes a `VerificationBundle` as JSON — to stdout by default, so it composes
with pipes, or to `-o <FILE>`.

```bash
cargo run -p zkml-prover -- prove examples/models/credit_lr.json \
  -i "0.5,0.2,0.9,0.1" -o bundle.json
```

The output round-trips through `zkml_prover::prover::bundle_from_json`, and its
`model_hash` equals `model_commitment(&model)`.

> **The proof bytes are a placeholder.** `Groth16Proof.data` is empty until
> STARK→Groth16 compression lands (issue #11). The bundle is structurally valid
> and useful for exercising the on-chain interface, but **do not submit one to a
> live contract** expecting it to verify.

### `validate`

Runs the quantization validation passes from `zkml_prover::quantization`:

1. **Range check** — no parameter is `i64::MIN` (which cannot be negated).
2. **Static overflow bounds** — worst-case intermediates fit in `i64`, assuming
   inputs bounded by `--max-input-magnitude`.
3. **Accuracy** — only with `--dataset`; compares quantized inference against
   the float model's recorded outputs at a `1e-4` tolerance and fails below
   `--min-agreement`.

```bash
cargo run -p zkml-prover -- validate examples/models/credit_lr.json --dataset dataset.json
```

```text
model: examples/models/credit_lr.json
range check: ok
overflow bounds: ok (max input magnitude 1)
accuracy: ok over 2 sample(s)
  agreement:      100.00% (threshold 99.00%)
  max deviation:  0e0
  mean deviation: 0e0
```

Without `--dataset` the accuracy pass is skipped and the report says so
explicitly — an empty dataset scores `agreement = 1.0`, which would otherwise
read as a stronger guarantee than was actually checked:

```text
accuracy: not checked (no --dataset; only range and overflow bounds ran)
```

#### `--dataset` schema

An array of samples. `inputs` is the raw (unquantized) feature vector; `expected`
is the output the **original floating-point model** produced for it.

```json
[
  { "inputs": [0.5, 0.2, 0.9, 0.1], "expected": 0.31 },
  { "inputs": [0.0, 0.0, 0.0, 0.0], "expected": -0.20 }
]
```

> **A mislabeled dataset looks like a quantization failure.** `expected` must be
> the float model's *output*, not the ground-truth label from your training set.
> Feed labels instead and `validate` reports a low agreement rate as though
> quantization were at fault. When agreement is surprisingly bad, check the
> dataset before touching the model.

### `inspect`

```bash
cargo run -p zkml-prover -- inspect examples/models/kyc_tree.json
```

```text
model: examples/models/kyc_tree.json
kind: decision_tree
features: 3
nodes: 5 (2 split, 3 leaf)
commitment: 55541e9a9593c81c1531f0067aa7534a8ee8e98e058afa99a0b190a5def23a2f
structure: valid
```

`structure` reports the result of `DecisionTree::validate` / `TinyMLP::validate`
rather than failing on it — a broken model is exactly when you want to run
`inspect`.

## Input validation

Every field of `--input` must be present and parse; nothing is silently dropped.
Rejected, with the 1-based position and the offending token in the message:

| Input | Error |
| ----- | ----- |
| `""` / `"   "` | empty input |
| `"0.5,,0.2"` | field 2 is empty |
| `"0.5,o.2,0.9"` | field 2 (`'o.2'`) is not a number |
| `"nan"`, `"inf"` | not finite — note `"nan".parse::<f64>()` *succeeds* in Rust |
| `"1e30"` | exceeds the Q16.16 range of ±1.4e14 — `quantize`'s `as i64` cast saturates |

The vector length is checked against the model's feature count before inference,
so the error names the model and both counts:

```text
error: examples/models/credit_lr.json expects 4 feature(s), got 2
```

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | Success |
| 2 | Bad invocation: clap argument errors, malformed `--input` |
| 1 | Everything else: file IO, model import, validation failure, inference error |

Errors go to stderr prefixed with `error: `; normal output goes to stdout.
