use crate::real::Real;
use clap::ValueEnum;
use std::fmt;
use std::sync::atomic::{Ordering, compiler_fence};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Kernel {
    Simd,
    Scalar,
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_possible_value().unwrap().get_name())
    }
}

impl Kernel {
    pub fn update<T: Real>(self, row: &mut [T], pivot_k: &[T], pivot_k2: &[T], w_k: T, w_k2: T) {
        match self {
            Kernel::Simd => update_simd(row, pivot_k, pivot_k2, w_k, w_k2),
            Kernel::Scalar => update_scalar(row, pivot_k, pivot_k2, w_k, w_k2),
        }
    }
}

fn update_simd<T: Real>(row: &mut [T], pivot_k: &[T], pivot_k2: &[T], w_k: T, w_k2: T) {
    for ((a, &p), &q) in row.iter_mut().zip(pivot_k).zip(pivot_k2) {
        *a = *a - w_k * p - w_k2 * q;
    }
}

fn update_scalar<T: Real>(row: &mut [T], pivot_k: &[T], pivot_k2: &[T], w_k: T, w_k2: T) {
    for ((a, &p), &q) in row.iter_mut().zip(pivot_k).zip(pivot_k2) {
        *a = *a - w_k * p - w_k2 * q;
        // żeby kompilator nie zwektoryzował tej pętli
        compiler_fence(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_kernels() {
        let p: Vec<f64> = (0..7).map(|j| j as f64 * 0.5).collect();
        let q: Vec<f64> = (0..7).map(|j| 3.0 - j as f64).collect();
        let expected: Vec<f64> = (0..7).map(|j| 1.0 - 0.25 * p[j] - 2.0 * q[j]).collect();
        for kernel in [Kernel::Simd, Kernel::Scalar] {
            let mut row = vec![1.0; 7];
            kernel.update(&mut row, &p, &q, 0.25, 2.0);
            assert_eq!(row, expected, "{kernel}");
        }
    }
}
