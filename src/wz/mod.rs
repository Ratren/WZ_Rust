pub mod kernel;
mod par_rayon;
mod par_threads;
mod seq;
mod step;

use crate::matrix::Matrix;
use crate::real::Real;
use clap::ValueEnum;
use std::fmt;

pub use kernel::Kernel;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Strategy {
    Seq,
    Rayon,
    Threads,
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_possible_value().unwrap().get_name())
    }
}

pub struct Factorizer {
    strategy: Strategy,
    kernel: Kernel,
    threads: usize,
    pool: Option<rayon::ThreadPool>,
}

impl Factorizer {
    pub fn new(strategy: Strategy, kernel: Kernel, threads: usize) -> Self {
        assert!(threads >= 1);
        let pool = (strategy == Strategy::Rayon).then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        });
        Self {
            strategy,
            kernel,
            threads,
            pool,
        }
    }

    /// w na wejściu = I, a po rozkładzie zawiera Z (bez zerowania)
    pub fn factorize<T: Real>(&self, a: &mut Matrix<T>, w: &mut Matrix<T>) -> Result<(), WzError> {
        assert_eq!(a.n(), w.n());
        match self.strategy {
            Strategy::Seq => seq::factorize(a, w, self.kernel),
            Strategy::Rayon => self
                .pool
                .as_ref()
                .unwrap()
                .install(|| par_rayon::factorize(a, w, self.kernel)),
            Strategy::Threads => par_threads::factorize(a, w, self.kernel, self.threads),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WzError {
    SingularPivot { step: usize },
}

impl fmt::Display for WzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WzError::SingularPivot { step } => {
                write!(f, "zerowy wyznacznik w kroku {step} (potrzebny pivoting)")
            }
        }
    }
}

impl std::error::Error for WzError {}
