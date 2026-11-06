use crate::real::Real;
use std::ops::{Index, IndexMut, Range};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    n: usize,
    data: Vec<T>,
}

impl<T: Real> Matrix<T> {
    pub fn zeros(n: usize) -> Self {
        assert!(n > 0);
        Self {
            n,
            data: vec![T::ZERO; n * n],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m[(i, i)] = T::ONE;
        }
        m
    }

    pub fn from_vec(n: usize, data: Vec<T>) -> Self {
        assert!(n > 0);
        assert_eq!(data.len(), n * n);
        Self { n, data }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    pub fn copy_from(&mut self, other: &Matrix<T>) {
        assert_eq!(self.n, other.n);
        self.data.copy_from_slice(&other.data);
    }

    pub fn convert<U: Real>(&self) -> Matrix<U> {
        Matrix {
            n: self.n,
            data: self.data.iter().map(|&v| U::from_f64(v.to_f64())).collect(),
        }
    }

    pub fn rows_mut(&mut self) -> std::slice::ChunksMut<'_, T> {
        self.data.chunks_mut(self.n)
    }

    pub fn row_range_mut(&mut self, rows: Range<usize>) -> &mut [T] {
        &mut self.data[rows.start * self.n..rows.end * self.n]
    }

    // wiersz k, wiersze k+1..k2-1 jednym kawałkiem, wiersz k2
    pub fn split_step(&mut self, k: usize, k2: usize) -> (&[T], &mut [T], &[T]) {
        let n = self.n;
        let (head, rest) = self.data.split_at_mut((k + 1) * n);
        let (middle, tail) = rest.split_at_mut((k2 - k - 1) * n);
        (&head[k * n..], middle, &tail[..n])
    }
}

impl<T: Real> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &T {
        debug_assert!(i < self.n && j < self.n);
        &self.data[i * self.n + j]
    }
}

impl<T: Real> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        debug_assert!(i < self.n && j < self.n);
        &mut self.data[i * self.n + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        let mut m = Matrix::from_vec(4, (0..16).map(f64::from).collect());
        let (row_k, middle, row_k2) = m.split_step(0, 3);
        assert_eq!(row_k, &[0., 1., 2., 3.]);
        assert_eq!(middle, &[4., 5., 6., 7., 8., 9., 10., 11.]);
        assert_eq!(row_k2, &[12., 13., 14., 15.]);
    }
}
