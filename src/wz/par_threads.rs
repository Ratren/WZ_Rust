use super::step::{Pivots, eliminate_row, steps};
use super::{Kernel, WzError};
use crate::matrix::Matrix;
use crate::real::Real;
use std::array::from_fn;
use std::panic::resume_unwind;
use std::sync::{Barrier, RwLock};
use std::thread;

type Rows<'a, T> = Vec<(usize, &'a mut [T], &'a mut [T])>;
type Buffers<T> = [[RwLock<Vec<T>>; 2]; 2];

pub(crate) fn factorize<T: Real>(
    a: &mut Matrix<T>,
    w: &mut Matrix<T>,
    kernel: Kernel,
    threads: usize,
) -> Result<(), WzError> {
    let n = a.n();
    let mut owned: Vec<Rows<T>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, (a_row, w_row)) in a.rows_mut().zip(w.rows_mut()).enumerate() {
        owned[i % threads].push((i, a_row, w_row));
    }
    let buffers: Buffers<T> = from_fn(|_| from_fn(|_| RwLock::new(vec![T::ZERO; n])));
    let barrier = Barrier::new(threads);
    let (buffers, barrier) = (&buffers, &barrier);
    thread::scope(|s| {
        let handles: Vec<_> = owned
            .into_iter()
            .map(|rows| s.spawn(move || worker(rows, n, kernel, buffers, barrier)))
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| resume_unwind(e)))
            .collect();
        results.into_iter().collect()
    })
}

fn worker<T: Real>(
    mut rows: Rows<'_, T>,
    n: usize,
    kernel: Kernel,
    buffers: &Buffers<T>,
    barrier: &Barrier,
) -> Result<(), WzError> {
    for (k, k2) in steps(n) {
        let [buf_k, buf_k2] = &buffers[k % 2];
        for (buf, pivot) in [(buf_k, k), (buf_k2, k2)] {
            if let Some((_, a_row, _)) = rows.iter().find(|r| r.0 == pivot) {
                buf.write().unwrap().copy_from_slice(a_row);
            }
        }
        // bufory są podwójne (k % 2), więc wystarczy jedna bariera na krok
        barrier.wait();
        let (row_k, row_k2) = (buf_k.read().unwrap(), buf_k2.read().unwrap());
        let pivots = Pivots::new(&row_k, &row_k2, k, k2)?;
        for (i, a_row, w_row) in &mut rows {
            if k < *i && *i < k2 {
                eliminate_row(a_row, w_row, &pivots, kernel);
            }
        }
    }
    Ok(())
}
