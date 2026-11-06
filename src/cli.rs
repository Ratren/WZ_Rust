use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use wz::{Kernel, Strategy};

/// rozkład WZ - pomiary i sprawdzanie wyników
#[derive(Parser)]
#[command(name = "wz", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// pomiary czasu
    Bench(BenchArgs),
    /// sprawdza ||A - WZ|| / ||A||
    Verify(VerifyArgs),
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Precision {
    F32,
    F64,
    Both,
}

impl Precision {
    pub fn includes_f32(self) -> bool {
        matches!(self, Precision::F32 | Precision::Both)
    }

    pub fn includes_f64(self) -> bool {
        matches!(self, Precision::F64 | Precision::Both)
    }
}

#[derive(Clone, Debug)]
pub struct Sizes(pub Vec<usize>);

#[derive(Args)]
pub struct BenchArgs {
    /// np. 500,1000 albo 1000:4000:500
    #[arg(long, value_parser = parse_sizes, default_value = "500,1000,1500,2000")]
    pub sizes: Sizes,
    #[arg(long, value_delimiter = ',', default_values_t = Strategy::value_variants().to_vec())]
    pub strategies: Vec<Strategy>,
    #[arg(long, value_delimiter = ',', default_values_t = Kernel::value_variants().to_vec())]
    pub kernels: Vec<Kernel>,
    /// np. 1,2,4,8 (domyślnie wszystkie procesory)
    #[arg(long, value_delimiter = ',', value_parser = positive)]
    pub threads: Vec<usize>,
    #[arg(long, value_parser = positive, default_value_t = 10)]
    pub reps: usize,
    /// ile powtórzeń na rozgrzewkę (nie liczą się)
    #[arg(long, default_value_t = 2)]
    pub warmup: usize,
    #[arg(long, value_enum, default_value_t = Precision::F64)]
    pub precision: Precision,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long)]
    pub csv: Option<PathBuf>,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// np. 5,8,101 albo 100:500:100
    #[arg(long, value_parser = parse_sizes, default_value = "5,6,7,8,101,500")]
    pub sizes: Sizes,
    #[arg(long, value_delimiter = ',', default_values_t = Strategy::value_variants().to_vec())]
    pub strategies: Vec<Strategy>,
    #[arg(long, value_delimiter = ',', default_values_t = Kernel::value_variants().to_vec())]
    pub kernels: Vec<Kernel>,
    /// liczba wątków (domyślnie wszystkie procesory)
    #[arg(long, value_parser = positive)]
    pub threads: Option<usize>,
    #[arg(long, value_enum, default_value_t = Precision::F64)]
    pub precision: Precision,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

pub fn positive(s: &str) -> Result<usize, String> {
    match s.trim().parse::<usize>() {
        Ok(0) => Err("musi być > 0".into()),
        Ok(v) => Ok(v),
        Err(_) => Err(format!("zła liczba: {s}")),
    }
}

pub fn parse_sizes(s: &str) -> Result<Sizes, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 3 {
        let (from, to, step) = (
            positive(parts[0])?,
            positive(parts[1])?,
            positive(parts[2])?,
        );
        if from > to {
            return Err("zły zakres".into());
        }
        return Ok(Sizes((from..=to).step_by(step).collect()));
    }
    s.split(',')
        .map(positive)
        .collect::<Result<_, _>>()
        .map(Sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_sizes("500,1000").unwrap().0, [500, 1000]);
        assert_eq!(
            parse_sizes("1000:4000:1000").unwrap().0,
            [1000, 2000, 3000, 4000]
        );
        assert_eq!(parse_sizes("7").unwrap().0, [7]);
    }

    #[test]
    fn bad_sizes() {
        for bad in ["0", "10:5:1", "1:10:0", "abc", ""] {
            assert!(parse_sizes(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn zero_not_allowed() {
        assert!(Cli::try_parse_from(["wz", "bench", "--threads", "0"]).is_err());
        assert!(Cli::try_parse_from(["wz", "bench", "--reps", "0"]).is_err());
    }
}
