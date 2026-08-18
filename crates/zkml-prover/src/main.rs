//! Command-line entrypoint for the zkml prover.
//!
//! Usage:
//!
//! ```text
//! zkml-prover <COMMAND>
//!
//! commit   <MODEL>                       Model commitment as 64-char hex
//! infer    <MODEL> -i <CSV>              Commitment + dequantized output + raw Q16.16
//! prove    <MODEL> -i <CSV> [-o <FILE>]  VerificationBundle JSON (stdout or file)
//! validate <MODEL> [--dataset <FILE>]    Quantization validation report
//! inspect  <MODEL>                       Kind, features, structure, commitment, validity
//! ```
//!
//! Everything of substance lives in [`zkml_prover::cli`] so the command bodies
//! are testable without spawning a process; this binary only dispatches and
//! maps [`CliError`] onto an exit code. See `docs/cli.md`.

use std::io::Write;
use std::process::exit;

use clap::Parser;
use zkml_prover::cli::{run, Cli, CliError};

fn main() {
    let cli = Cli::parse(); // clap exits 2 on its own for bad invocations
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if let Err(e) = run(&cli, &mut out) {
        // Flush whatever the command already emitted before the error text, so
        // partial output and the diagnostic do not interleave.
        let _ = out.flush();
        eprintln!("error: {e}");
        exit(e.exit_code());
    }

    if let Err(e) = out.flush() {
        eprintln!("error: failed to flush stdout: {e}");
        exit(
            CliError::Io {
                path: "<stdout>".into(),
                message: e.to_string(),
            }
            .exit_code(),
        );
    }
}
