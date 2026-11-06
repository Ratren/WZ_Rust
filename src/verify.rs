use crate::matrix::Matrix;
use crate::real::Real;

/// ||A - W*Z|| / ||A|| (Frobenius), w z_raw zerujemy wyeliminowane pozycje
pub fn residual<T: Real>(a: &Matrix<T>, z_raw: &Matrix<T>, w: &Matrix<T>) -> f64 {
    let n = a.n();
    let z = |i: usize, j: usize| {
        let eliminated = (i > j && i < n - 1 - j) || (i < j && i > n - 1 - j);
        if eliminated {
            0.0
        } else {
            z_raw[(i, j)].to_f64()
        }
    };
    let (mut diff, mut norm) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            let wz: f64 = (0..n).map(|l| w[(i, l)].to_f64() * z(l, j)).sum();
            let a_ij = a[(i, j)].to_f64();
            diff += (a_ij - wz).powi(2);
            norm += a_ij.powi(2);
        }
    }
    (diff / norm).sqrt()
}

pub fn tolerance<T: Real>() -> f64 {
    1e4 * T::EPSILON
}
