use clap::ValueEnum;
use wz::generate::{RandomSource, diagonally_dominant, input_matrix};
use wz::verify::{residual, tolerance};
use wz::{Factorizer, Kernel, Matrix, Real, Strategy, WzError};

struct Counter(u64);
impl RandomSource for Counter {
    fn next_unit(&mut self) -> f64 {
        self.0 += 1;
        (self.0 % 97) as f64 / 97.0
    }
}

#[test]
fn custom_generator() {
    let a = diagonally_dominant(6, &mut Counter(0));
    assert_eq!(a, diagonally_dominant(6, &mut Counter(0)));
    for i in 0..6 {
        let off: f64 = (0..6).filter(|&j| j != i).map(|j| a[(i, j)].abs()).sum();
        assert!(a[(i, i)] > off);
    }
}

const THREADS: usize = 3;

fn factorize_copy<T: Real>(
    strategy: Strategy,
    kernel: Kernel,
    threads: usize,
    a: &Matrix<T>,
) -> Result<(Matrix<T>, Matrix<T>), WzError> {
    let (mut z, mut w) = (a.clone(), Matrix::identity(a.n()));
    Factorizer::new(strategy, kernel, threads).factorize(&mut z, &mut w)?;
    Ok((z, w))
}

fn variants() -> impl Iterator<Item = (Strategy, Kernel)> {
    Strategy::value_variants()
        .iter()
        .flat_map(|&s| Kernel::value_variants().iter().map(move |&k| (s, k)))
}

fn check_residuals<T: Real>(a: &Matrix<T>, label: &str) {
    for (s, k) in variants() {
        let (z, w) = factorize_copy(s, k, THREADS, a).unwrap();
        let r = residual(a, &z, &w);
        assert!(r < tolerance::<T>(), "{label} {s} {k}: residuum {r}");
    }
}

#[test]
fn matrices_from_c() {
    let known: [(usize, Vec<f64>); 4] = [
        (
            4,
            vec![
                3., 2., 3., -4., 4., 5., 7., -4., 9., 12., 1., 11., 1., -2., -3., -4.,
            ],
        ),
        (
            5,
            vec![
                -3., 2., 3., -4., 1., 4., 5., 7., -4., 2., 9., 12., 1., 11., 3., 1., -2., -3., -4.,
                -5., 3., 2., -1., 1., 1.,
            ],
        ),
        (
            6,
            vec![
                -3., 2., 3., -4., 1., 0., 4., 5., 7., -4., 2., 0., 9., 12., 1., 11., 3., 0., 1.,
                -2., -3., -4., -5., 0., 3., 2., -1., 1., 1., 0., 0., 0., 0., 0., 0., 1.,
            ],
        ),
        (
            8,
            vec![
                1., 2., 1., 0., 1., 0., 3., 4., 5., 0., 0., 1., 0., 1., 7., 8., 1., 0., 1., 0., 2.,
                3., 0., 0., 2., 3., -1., 1., 4., 5., 1., 0., 4., 5., -1., -1., 3., 2., 1., 1., 6.,
                7., -2., 0., 0., 1., 0., 1., 9., 0., 3., 8., 1., 9., 11., 12., -3., 1., 2., 9., 7.,
                6., 0., 6.,
            ],
        ),
    ];
    for (n, data) in known {
        check_residuals(&Matrix::from_vec(n, data), &format!("n={n}"));
    }
}

const SIZES: [usize; 9] = [1, 2, 3, 4, 5, 7, 16, 33, 100];

#[test]
fn random() {
    for n in SIZES {
        let a = input_matrix(n, 42);
        check_residuals(&a, &format!("f64 n={n}"));
        check_residuals(&a.convert::<f32>(), &format!("f32 n={n}"));
    }
}

#[test]
fn odd_n() {
    let a = input_matrix(5, 42);
    for (s, k) in variants() {
        let (_, w) = factorize_copy(s, k, THREADS, &a).unwrap();
        assert!(w[(2, 1)] != 0.0 && w[(2, 3)] != 0.0, "{s} {k}");
    }
}

#[test]
fn singular() {
    let a = Matrix::from_vec(4, vec![1.0_f64; 16]);
    for (s, k) in variants() {
        let err = factorize_copy(s, k, THREADS, &a).unwrap_err();
        assert_eq!(err, WzError::SingularPivot { step: 0 });
    }
}

#[test]
fn singular_later_step() {
    // krok 0 przechodzi, a w kroku 1 blok 2x2 to same jedynki
    let mut a = Matrix::<f64>::identity(6);
    a[(1, 4)] = 1.0;
    a[(4, 1)] = 1.0;
    for threads in [1, 2, 3, 8] {
        for (s, k) in variants() {
            let err = factorize_copy(s, k, threads, &a).unwrap_err();
            assert_eq!(
                err,
                WzError::SingularPivot { step: 1 },
                "{s}/{k} t={threads}"
            );
        }
    }
}

fn bits<T: Real>((z, w): &(Matrix<T>, Matrix<T>)) -> Vec<u64> {
    (0..z.n())
        .flat_map(|i| z.row(i).iter().chain(w.row(i)))
        .map(|x| x.to_f64().to_bits())
        .collect()
}

fn check_same_as_seq<T: Real>() {
    for n in [2, 7, 64, 101] {
        let a = input_matrix(n, 3).convert::<T>();
        let reference = factorize_copy(Strategy::Seq, Kernel::Simd, 1, &a).unwrap();
        for threads in [1, 2, 3, 8] {
            for (s, k) in variants() {
                assert_eq!(
                    bits(&factorize_copy(s, k, threads, &a).unwrap()),
                    bits(&reference),
                    "{s}/{k} t={threads} n={n}"
                );
            }
        }
    }
}

#[test]
fn same_as_seq() {
    check_same_as_seq::<f64>();
    check_same_as_seq::<f32>();
}
