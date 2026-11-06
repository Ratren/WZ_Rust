use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

pub trait Real:
    Copy
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
    const EPSILON: f64;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;

    fn from_f64(v: f64) -> Self {
        v
    }

    fn to_f64(self) -> f64 {
        self
    }
}
