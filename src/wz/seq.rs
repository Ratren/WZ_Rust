use super::step::{Pivots, eliminate_row, steps};
use super::{Kernel, WzError};
use crate::matrix::Matrix;
use crate::real::Real;

pub(crate) fn factorize<T: Real>(
    a: &mut Matrix<T>,
    w: &mut Matrix<T>,
    kernel: Kernel,
) -> Result<(), WzError> {
    let n = a.n();
    for (k, k2) in steps(n) {
        let (row_k, middle, row_k2) = a.split_step(k, k2);
        let pivots = Pivots::new(row_k, row_k2, k, k2)?;
        let w_rows = w.row_range_mut(k + 1..k2);
        for (a_row, w_row) in middle.chunks_mut(n).zip(w_rows.chunks_mut(n)) {
            eliminate_row(a_row, w_row, &pivots, kernel);
        }
    }
    Ok(())
}
