use super::Record;
use std::io::Write;

pub const CSV_HEADER: &str = "precision,strategy,kernel,threads,n,reps,median_s,mean_s,std_dev_s,min_s,max_s,cv_pct,gflops,speedup";

pub fn write_csv(out: &mut impl Write, records: &[Record]) -> std::io::Result<()> {
    writeln!(out, "{CSV_HEADER}")?;
    for r in records {
        let s = &r.summary;
        let speedup = r.speedup.map(|x| x.to_string()).unwrap_or_default();
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.precision,
            r.strategy,
            r.kernel,
            r.threads,
            r.n,
            r.reps,
            s.median,
            s.mean,
            s.std_dev,
            s.min,
            s.max,
            s.cv_pct(),
            r.gflops,
            speedup
        )?;
    }
    Ok(())
}

pub fn print_table(records: &[Record]) {
    println!(
        "{:<5} {:<8} {:<7} {:>3} {:>6} {:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>7} {:>8} {:>8}",
        "prec",
        "strategy",
        "kernel",
        "thr",
        "n",
        "reps",
        "median[ms]",
        "mean[ms]",
        "stddev[ms]",
        "min[ms]",
        "max[ms]",
        "cv[%]",
        "GFLOPS",
        "speedup"
    );
    for r in records {
        let s = &r.summary;
        let speedup = r.speedup.map_or("-".to_string(), |x| format!("{x:.2}"));
        println!(
            "{:<5} {:<8} {:<7} {:>3} {:>6} {:>4} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>7.2} {:>8.3} {:>8}",
            r.precision,
            r.strategy.to_string(),
            r.kernel.to_string(),
            r.threads,
            r.n,
            r.reps,
            s.median * 1e3,
            s.mean * 1e3,
            s.std_dev * 1e3,
            s.min * 1e3,
            s.max * 1e3,
            s.cv_pct(),
            r.gflops,
            speedup
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{BenchConfig, run};
    use crate::{Kernel, Strategy};
    use clap::ValueEnum;

    #[test]
    fn csv() {
        let cfg = BenchConfig {
            sizes: vec![8],
            strategies: Strategy::value_variants().to_vec(),
            kernels: Kernel::value_variants().to_vec(),
            threads: vec![1, 2],
            reps: 3,
            warmup: 1,
            seed: 1,
        };
        let records = run::<f64>(&cfg).unwrap();
        let mut buf = Vec::new();
        write_csv(&mut buf, &records).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(lines.len(), 1 + records.len());
        assert!(lines.iter().all(|l| l.split(',').count() == 14));
    }
}
