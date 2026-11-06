use crate::Matrix;
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Liczby z [0, 1). Można to zaimplementować we własnym generatorze i podmienić w input_matrix.
pub trait RandomSource {
    fn next_unit(&mut self) -> f64;
}

pub struct StdRandom(StdRng);

impl StdRandom {
    pub fn new(seed: u64) -> Self {
        Self(StdRng::seed_from_u64(seed))
    }
}

impl RandomSource for StdRandom {
    fn next_unit(&mut self) -> f64 {
        self.0.random()
    }
}

// to samo co PobierzMacierzLos z macierze.h
pub fn diagonally_dominant(n: usize, rng: &mut impl RandomSource) -> Matrix<f64> {
    let mut a = Matrix::zeros(n);
    for i in 0..n {
        a[(i, i)] = 0.0;
        // w C pierwszy wiersz losuje też przekątną, tu pomijamy ją zawsze
        for j in (0..n).filter(|&j| j != i) {
            let x = rng.next_unit() * 20.0 - 10.0;
            a[(i, j)] = x;
            a[(i, i)] += x.abs();
            a[(j, j)] += x.abs();
        }
        a[(i, i)] += rng.next_unit() * 9.0 + 1.0;
    }
    a
}

// tu można podmienić generator
pub fn input_matrix(n: usize, seed: u64) -> Matrix<f64> {
    diagonally_dominant(n, &mut StdRandom::new(seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_strictly_row_dominant(a: &Matrix<f64>) {
        for i in 0..a.n() {
            let off: f64 = (0..a.n())
                .filter(|&j| j != i)
                .map(|j| a[(i, j)].abs())
                .sum();
            assert!(a[(i, i)] > off, "wiersz {i}");
        }
    }

    #[test]
    fn dominant_diagonal() {
        for n in [1, 5, 50] {
            assert_strictly_row_dominant(&input_matrix(n, 7));
        }
    }

    #[test]
    fn same_seed() {
        assert_eq!(input_matrix(20, 1), input_matrix(20, 1));
        assert_ne!(input_matrix(20, 1), input_matrix(20, 2));
    }
}
