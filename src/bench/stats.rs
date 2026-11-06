#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub median: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

impl Summary {
    pub fn from_samples(samples: &[f64]) -> Self {
        assert!(!samples.is_empty());
        let n = samples.len();
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        let mean = samples.iter().sum::<f64>() / n as f64;
        let std_dev = if n > 1 {
            let ss: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum();
            (ss / (n - 1) as f64).sqrt()
        } else {
            0.0
        };
        Self {
            median,
            mean,
            std_dev,
            min: sorted[0],
            max: sorted[n - 1],
        }
    }

    pub fn cv_pct(&self) -> f64 {
        if self.mean == 0.0 {
            0.0
        } else {
            100.0 * self.std_dev / self.mean
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median() {
        assert_eq!(Summary::from_samples(&[3.0, 1.0, 2.0]).median, 2.0);
        assert_eq!(Summary::from_samples(&[4.0, 1.0, 3.0, 2.0]).median, 2.5);
    }

    #[test]
    fn mean_and_std_dev() {
        let s = Summary::from_samples(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!((s.mean, s.min, s.max), (5.0, 2.0, 9.0));
        assert!((s.std_dev - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
    }
}
