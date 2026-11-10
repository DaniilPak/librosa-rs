#![allow(dead_code)]
//! Shared helpers: load librosa reference fixtures and compare with tolerances.

use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};
use ndarray_npy::read_npy;
use num_complex::Complex64;
use std::path::PathBuf;

pub const SR: f64 = 22050.0;

pub fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub fn load_d(name: &str) -> ArrayD<f64> {
    let p = path(&format!("{name}.npy"));
    read_npy(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

pub fn load1(name: &str) -> Array1<f64> {
    let a = load_d(name);
    if a.ndim() == 0 {
        return Array1::from(vec![a.iter().next().copied().unwrap()]);
    }
    a.into_dimensionality::<Ix1>().expect("1-D fixture")
}

pub fn load2(name: &str) -> Array2<f64> {
    load_d(name)
        .into_dimensionality::<Ix2>()
        .expect("2-D fixture")
}

pub fn load2c(name: &str) -> Array2<Complex64> {
    let re = load2(&format!("{name}_re"));
    let im = load2(&format!("{name}_im"));
    ndarray::Zip::from(&re)
        .and(&im)
        .map_collect(|&r, &i| Complex64::new(r, i))
}

pub fn sig(name: &str) -> Array1<f64> {
    load1(&format!("sig_{name}"))
}

pub fn meta() -> serde_json::Value {
    let s = std::fs::read_to_string(path("meta.json")).unwrap();
    serde_json::from_str(&s).unwrap()
}

/// Assert `|a - e| <= atol + rtol * |e|` elementwise; NaNs must match.
pub fn assert_close<'a>(
    what: &str,
    actual: impl IntoIterator<Item = &'a f64>,
    expected: impl IntoIterator<Item = &'a f64>,
    rtol: f64,
    atol: f64,
) {
    let a: Vec<f64> = actual.into_iter().copied().collect();
    let e: Vec<f64> = expected.into_iter().copied().collect();
    assert_eq!(
        a.len(),
        e.len(),
        "{what}: length mismatch {} vs {}",
        a.len(),
        e.len()
    );
    let mut worst = (0usize, 0.0f64);
    let mut bad = 0usize;
    for (i, (&x, &y)) in a.iter().zip(&e).enumerate() {
        if x.is_nan() || y.is_nan() {
            assert!(
                x.is_nan() && y.is_nan(),
                "{what}: NaN mismatch at {i}: {x} vs {y}"
            );
            continue;
        }
        if x == y {
            continue;
        }
        let err = (x - y).abs();
        let tol = atol + rtol * y.abs();
        if err > tol {
            bad += 1;
            if err - tol > worst.1 {
                worst = (i, err - tol);
            }
        }
    }
    assert!(
        bad == 0,
        "{what}: {bad}/{} elements out of tolerance (rtol={rtol}, atol={atol}); worst at {}: {} vs {}",
        a.len(),
        worst.0,
        a[worst.0],
        e[worst.0]
    );
}

pub fn assert_close2(what: &str, a: &Array2<f64>, e: &Array2<f64>, rtol: f64, atol: f64) {
    assert_eq!(a.dim(), e.dim(), "{what}: shape mismatch");
    assert_close(what, a.iter(), e.iter(), rtol, atol);
}

pub fn assert_close2c(
    what: &str,
    a: &Array2<Complex64>,
    e: &Array2<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(a.dim(), e.dim(), "{what}: shape mismatch");
    let ar: Vec<f64> = a.iter().flat_map(|z| [z.re, z.im]).collect();
    let er: Vec<f64> = e.iter().flat_map(|z| [z.re, z.im]).collect();
    assert_close(what, ar.iter(), er.iter(), rtol, atol);
}

/// Max absolute value, used to scale absolute tolerances.
pub fn max_abs<'a>(x: impl IntoIterator<Item = &'a f64>) -> f64 {
    x.into_iter().fold(0.0, |m, v| m.max(v.abs()))
}

pub fn to_usize(a: &Array1<f64>) -> Vec<usize> {
    a.iter().map(|&v| v as usize).collect()
}
