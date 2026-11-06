#![forbid(unsafe_code)]

mod cli;

use clap::Parser;
use cli::{BenchArgs, Cli, Command, VerifyArgs};
use std::process::ExitCode;
use wz::bench::{self, BenchConfig, report};
use wz::generate::input_matrix;
use wz::verify::{residual, tolerance};
use wz::{Factorizer, Matrix, Real, WzError};

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Bench(args) => run_bench(args),
        Command::Verify(args) => run_verify(args),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("błąd: {e}");
            ExitCode::FAILURE
        }
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn run_bench(args: BenchArgs) -> Result<bool, Box<dyn std::error::Error>> {
    let threads = if args.threads.is_empty() {
        vec![default_threads()]
    } else {
        args.threads
    };
    let cfg = BenchConfig {
        sizes: args.sizes.0,
        strategies: args.strategies,
        kernels: args.kernels,
        threads,
        reps: args.reps,
        warmup: args.warmup,
        seed: args.seed,
    };
    let mut csv = match &args.csv {
        Some(path) => Some(
            std::fs::File::create(path)
                .map_err(|e| format!("nie można utworzyć pliku {}: {e}", path.display()))?,
        ),
        None => None,
    };
    let mut records = Vec::new();
    if args.precision.includes_f64() {
        records.extend(bench::run::<f64>(&cfg)?);
    }
    if args.precision.includes_f32() {
        records.extend(bench::run::<f32>(&cfg)?);
    }
    report::print_table(&records);
    if let (Some(file), Some(path)) = (csv.as_mut(), &args.csv) {
        report::write_csv(file, &records)?;
        eprintln!("zapisano {}", path.display());
    }
    Ok(true)
}

fn run_verify(args: VerifyArgs) -> Result<bool, Box<dyn std::error::Error>> {
    let threads = args.threads.unwrap_or_else(default_threads);
    let mut all_ok = true;
    if args.precision.includes_f64() {
        all_ok &= verify_precision::<f64>(&args, threads)?;
    }
    if args.precision.includes_f32() {
        all_ok &= verify_precision::<f32>(&args, threads)?;
    }
    Ok(all_ok)
}

fn verify_precision<T: Real>(args: &VerifyArgs, threads: usize) -> Result<bool, WzError> {
    let tol = tolerance::<T>();
    let mut all_ok = true;
    for &n in &args.sizes.0 {
        let a = input_matrix(n, args.seed).convert::<T>();
        for &strategy in &args.strategies {
            for &kernel in &args.kernels {
                let mut z = a.clone();
                let mut w = Matrix::<T>::identity(n);
                Factorizer::new(strategy, kernel, threads).factorize(&mut z, &mut w)?;
                let res = residual(&a, &z, &w);
                let ok = res < tol;
                all_ok &= ok;
                println!(
                    "{} n={n} {strategy}/{kernel} residuum={res:.3e} (tol {tol:.1e}) {}",
                    T::NAME,
                    if ok { "OK" } else { "FAIL" }
                );
            }
        }
    }
    Ok(all_ok)
}
