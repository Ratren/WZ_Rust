use super::{Kernel, WzError};
use crate::real::Real;

pub(crate) fn steps(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n.saturating_sub(1) / 2).map(move |k| (k, n - 1 - k))
}

pub(crate) struct Pivots<'a, T> {
    pub(crate) k: usize,
    pub(crate) k2: usize,
    pub(crate) row_k: &'a [T],
    pub(crate) row_k2: &'a [T],
    pub(crate) det: T,
}

impl<'a, T: Real> Pivots<'a, T> {
    pub(crate) fn new(
        row_k: &'a [T],
        row_k2: &'a [T],
        k: usize,
        k2: usize,
    ) -> Result<Self, WzError> {
        let det = row_k2[k] * row_k[k2] - row_k[k] * row_k2[k2];
        if det == T::ZERO || !det.to_f64().is_finite() {
            return Err(WzError::SingularPivot { step: k });
        }
        Ok(Self {
            k,
            k2,
            row_k,
            row_k2,
            det,
        })
    }

    pub(crate) fn coefficients(&self, a_ik: T, a_ik2: T) -> (T, T) {
        let (k, k2) = (self.k, self.k2);
        let w_k = (self.row_k2[k] * a_ik2 - self.row_k2[k2] * a_ik) / self.det;
        let w_k2 = (self.row_k[k2] * a_ik - self.row_k[k] * a_ik2) / self.det;
        (w_k, w_k2)
    }
}

pub(crate) fn eliminate_row<T: Real>(
    a_row: &mut [T],
    w_row: &mut [T],
    pivots: &Pivots<T>,
    kernel: Kernel,
) {
    let (k, k2) = (pivots.k, pivots.k2);
    let (w_k, w_k2) = pivots.coefficients(a_row[k], a_row[k2]);
    w_row[k] = w_k;
    w_row[k2] = w_k2;
    let cols = k + 1..k2;
    kernel.update(
        &mut a_row[cols.clone()],
        &pivots.row_k[cols.clone()],
        &pivots.row_k2[cols],
        w_k,
        w_k2,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_count() {
        assert_eq!(steps(1).count(), 0);
        assert_eq!(steps(2).count(), 0);
        assert_eq!(steps(3).collect::<Vec<_>>(), [(0, 2)]);
        assert_eq!(steps(5).collect::<Vec<_>>(), [(0, 4), (1, 3)]);
        assert_eq!(steps(6).collect::<Vec<_>>(), [(0, 5), (1, 4)]);
    }

    #[test]
    fn coeffs() {
        let row_k = [4.0_f64, 1.0, 2.0];
        let row_k2 = [1.0, 2.0, 5.0];
        let p = Pivots::new(&row_k, &row_k2, 0, 2).unwrap();
        let (a_ik, a_ik2) = (3.0, 7.0);
        let (w_k, w_k2) = p.coefficients(a_ik, a_ik2);
        assert!((w_k * 4.0 + w_k2 * 1.0 - a_ik).abs() < 1e-12);
        assert!((w_k * 2.0 + w_k2 * 5.0 - a_ik2).abs() < 1e-12);
    }

    #[test]
    fn zero_det() {
        let ones = [1.0; 4];
        assert_eq!(
            Pivots::new(&ones, &ones, 1, 2).err(),
            Some(WzError::SingularPivot { step: 1 })
        );
    }
}
