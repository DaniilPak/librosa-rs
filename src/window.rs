//! Window functions (the subset of `scipy.signal.get_window` that librosa uses).

use crate::error::{Result, param};
use ndarray::Array1;
use std::f64::consts::PI;

/// A window specification, mirroring the strings accepted by `librosa.filters.get_window`.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Window {
    /// Hann window (`"hann"`), librosa's default.
    #[default]
    Hann,
    /// Hamming window (`"hamming"`).
    Hamming,
    /// Blackman window (`"blackman"`).
    Blackman,
    /// Rectangular window (`"ones"` / `"boxcar"`).
    Ones,
    /// Triangular window (`"triangle"` / `"triang"`).
    Triangle,
    /// Kaiser window with shape parameter beta (`("kaiser", beta)`).
    Kaiser(f64),
    /// A user-supplied window; its length must equal the requested length.
    Custom(Vec<f64>),
}

fn general_cosine(m: usize, a: &[f64]) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    (0..m)
        .map(|k| {
            // scipy: fac = linspace(-pi, pi, M); w = sum_j a_j cos(j fac)
            let fac = -PI + 2.0 * PI * k as f64 / (m - 1) as f64;
            a.iter()
                .enumerate()
                .map(|(j, &aj)| aj * (j as f64 * fac).cos())
                .sum::<f64>()
        })
        .collect()
}

fn triang_sym(m: usize) -> Vec<f64> {
    let half = m.div_ceil(2);
    let n: Vec<f64> = (1..=half).map(|v| v as f64).collect();
    if m.is_multiple_of(2) {
        let w: Vec<f64> = n.iter().map(|&k| (2.0 * k - 1.0) / m as f64).collect();
        let mut out = w.clone();
        out.extend(w.iter().rev());
        out
    } else {
        let w: Vec<f64> = n.iter().map(|&k| 2.0 * k / (m as f64 + 1.0)).collect();
        let mut out = w.clone();
        out.extend(w.iter().rev().skip(1));
        out
    }
}

/// Modified Bessel function of the first kind, order zero.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    let mut k = 1.0;
    loop {
        term *= q / (k * k);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
        k += 1.0;
    }
    sum
}

fn kaiser_sym(m: usize, beta: f64) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let alpha = (m as f64 - 1.0) / 2.0;
    let denom = bessel_i0(beta);
    (0..m)
        .map(|n| {
            let r = (n as f64 - alpha) / alpha;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Compute a window of length `n`.
///
/// `fftbins = true` gives the periodic window (DFT-even), as used for spectral analysis;
/// `false` gives the symmetric window, as for filter design. Same as
/// `librosa.filters.get_window(window, n, fftbins=fftbins)`.
pub fn get_window(window: &Window, n: usize, fftbins: bool) -> Result<Array1<f64>> {
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    if let Window::Custom(w) = window {
        if w.len() != n {
            return param(format!("Window size mismatch: {} != {}", w.len(), n));
        }
        return Ok(Array1::from(w.clone()));
    }
    let m = if fftbins { n + 1 } else { n };
    let mut w = match window {
        Window::Hann => general_cosine(m, &[0.5, 0.5]),
        Window::Hamming => general_cosine(m, &[0.54, 0.46]),
        Window::Blackman => general_cosine(m, &[0.42, 0.50, 0.08]),
        Window::Ones => vec![1.0; m],
        Window::Triangle => triang_sym(m),
        Window::Kaiser(beta) => kaiser_sym(m, *beta),
        Window::Custom(_) => unreachable!(),
    };
    w.truncate(n);
    Ok(Array1::from(w))
}

/// Equivalent noise bandwidth of a window, as in `librosa.filters.window_bandwidth`.
pub fn window_bandwidth(window: &Window) -> f64 {
    match window {
        Window::Hann => 1.50018310546875,
        Window::Hamming => 1.3629455320350348,
        Window::Blackman => 1.7269681554262326,
        Window::Ones => 1.0,
        Window::Triangle => 1.333170652355585,
        _ => {
            let n = 1000;
            let w = get_window(window, n, true).unwrap_or_else(|_| Array1::ones(n));
            let s: f64 = w.sum();
            n as f64 * w.mapv(|v| v * v).sum() / (s * s + f64::MIN_POSITIVE)
        }
    }
}
