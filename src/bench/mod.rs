pub mod report;
pub mod stats;

use crate::generate::input_matrix;
use crate::{Factorizer, Kernel, Matrix, Real, Strategy, WzError};
use stats::Summary;
use std::time::Instant;

pub struct BenchConfig {
    pub sizes: Vec<usize>,
    pub strategies: Vec<Strategy>,
    pub kernels: Vec<Kernel>,
    pub threads: Vec<usize>,
    pub reps: usize,
    pub warmup: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub precision: &'static str,
    pub strategy: Strategy,
    pub kernel: Kernel,
    pub threads: usize,
    pub n: usize,
    pub reps: usize,
    pub summary: Summary,
    pub gflops: f64,
    pub speedup: Option<f64>,
}

pub fn wz_flops(n: usize) -> f64 {
    (0..n.saturating_sub(1) / 2)
        .map(|k| {
            let s = (n - 2 - 2 * k) as f64;
            3.0 + s * (8.0 + 4.0 * s)
        })
        .sum()
}

pub fn run<T: Real>(cfg: &BenchConfig) -> Result<Vec<Record>, WzError> {
    let mut records = Vec::new();
    for &n in &cfg.sizes {
        let a = input_matrix(n, cfg.seed).convert::<T>();
        let identity = Matrix::<T>::identity(n);
        let flops = wz_flops(n);
        for &strategy in &cfg.strategies {
            let thread_counts = if strategy == Strategy::Seq {
                &[1][..]
            } else {
                &cfg.threads[..]
            };
            for &kernel in &cfg.kernels {
                for &threads in thread_counts {
                    let factorizer = Factorizer::new(strategy, kernel, threads);
                    let mut z = Matrix::<T>::zeros(n);
                    let mut w = Matrix::<T>::zeros(n);
                    let mut samples = Vec::with_capacity(cfg.reps);
                    for i in 0..cfg.warmup + cfg.reps {
                        z.copy_from(&a);
                        w.copy_from(&identity);
                        let start = Instant::now();
                        factorizer.factorize(&mut z, &mut w)?;
                        let elapsed = start.elapsed().as_secs_f64();
                        if i >= cfg.warmup {
                            samples.push(elapsed);
                        }
                    }
                    let summary = Summary::from_samples(&samples);
                    let gflops = if summary.median > 0.0 {
                        flops / summary.median / 1e9
                    } else {
                        0.0
                    };
                    eprintln!(
                        "{} n={n} {strategy}/{kernel} t={threads} mediana={:.3} ms",
                        T::NAME,
                        summary.median * 1e3
                    );
                    records.push(Record {
                        precision: T::NAME,
                        strategy,
                        kernel,
                        threads,
                        n,
                        reps: cfg.reps,
                        summary,
                        gflops,
                        speedup: None,
                    });
                }
            }
        }
    }
    fill_speedups(&mut records);
    Ok(records)
}

fn fill_speedups(records: &mut [Record]) {
    let baselines: Vec<Option<f64>> = records
        .iter()
        .map(|r| {
            records
                .iter()
                .find(|b| {
                    b.strategy == Strategy::Seq
                        && b.precision == r.precision
                        && b.n == r.n
                        && b.kernel == r.kernel
                })
                .filter(|b| b.summary.median > 0.0 && r.summary.median > 0.0)
                .map(|b| b.summary.median / r.summary.median)
        })
        .collect();
    for (r, s) in records.iter_mut().zip(baselines) {
        r.speedup = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    fn small(sizes: &[usize], strategies: &[Strategy]) -> BenchConfig {
        BenchConfig {
            sizes: sizes.to_vec(),
            strategies: strategies.to_vec(),
            kernels: Kernel::value_variants().to_vec(),
            threads: vec![1, 2],
            reps: 3,
            warmup: 1,
            seed: 1,
        }
    }

    #[test]
    fn flops() {
        assert_eq!(wz_flops(1), 0.0);
        assert_eq!(wz_flops(2), 0.0);
        assert_eq!(wz_flops(3), 15.0);
        assert_eq!(wz_flops(4), 35.0);
    }

    #[test]
    fn record_count() {
        let records = run::<f64>(&small(&[16], Strategy::value_variants())).unwrap();
        assert_eq!(records.len(), 10);
        let count = |s| records.iter().filter(|r| r.strategy == s).count();
        assert_eq!(count(Strategy::Seq), 2);
        assert_eq!(count(Strategy::Rayon), 4);
        assert_eq!(count(Strategy::Threads), 4);
        for r in &records {
            assert!(r.summary.min <= r.summary.median && r.summary.median <= r.summary.max);
            if r.strategy == Strategy::Seq {
                assert_eq!(r.speedup, Some(1.0));
            }
        }
    }

    #[test]
    fn tiny_sizes() {
        let records = run::<f64>(&small(&[1, 2], Strategy::value_variants())).unwrap();
        assert!(records.iter().all(|r| r.gflops == 0.0));
    }

    #[test]
    fn no_seq() {
        let records = run::<f64>(&small(&[8], &[Strategy::Rayon])).unwrap();
        assert!(records.iter().all(|r| r.speedup.is_none()));
    }
}
