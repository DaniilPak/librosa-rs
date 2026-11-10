//! Ports of the scipy routines librosa depends on (resampling, FIR design, Savitzky-Golay,
//! a couple of distribution functions). Internal.

use crate::fft::Rfft;
use crate::window::{Window, get_window};
use num_complex::Complex64;
use std::f64::consts::PI;

/// `scipy.signal.resample(x, num)` for real input (Fourier method).
pub(crate) fn resample_fft(x: &[f64], num: usize) -> Vec<f64> {
    let n_x = x.len();
    if num == 0 || n_x == 0 {
        return vec![0.0; num];
    }
    let s_fac = n_x as f64 / num as f64;
    let m = num.min(n_x);
    let m2 = m / 2 + 1;
    let mut fwd = Rfft::new(n_x);
    let mut spec = vec![Complex64::new(0.0, 0.0); n_x / 2 + 1];
    fwd.forward(x, &mut spec);
    spec.truncate(m2);
    if m.is_multiple_of(2) && num != n_x {
        let f = if num < n_x { 2.0 } else { 0.5 };
        spec[m / 2] *= f;
    }
    spec.iter_mut().for_each(|z| *z /= s_fac);
    let mut inv = Rfft::new(num);
    let mut out = vec![0.0; num];
    inv.inverse(&spec, &mut out);
    out
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// `scipy.signal.firwin(numtaps, cutoff, window)` for a low-pass filter (cutoff relative to Nyquist).
pub(crate) fn firwin_lowpass(numtaps: usize, cutoff: f64, window: &Window) -> Vec<f64> {
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let win = get_window(window, numtaps, false).expect("valid window");
    let mut h: Vec<f64> = (0..numtaps)
        .map(|k| {
            let m = k as f64 - alpha;
            cutoff * sinc(cutoff * m) * win[k]
        })
        .collect();
    let s: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= s);
    h
}

fn upfirdn_output_len(len_h: usize, n_in: usize, up: usize, down: usize) -> usize {
    ((n_in - 1) * up + len_h - 1) / down + 1
}

/// `scipy.signal.upfirdn(h, x, up, down)` with zero boundary.
fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let n_out = upfirdn_output_len(h.len(), x.len(), up, down);
    let mut y = vec![0.0; n_out];
    for (k, yk) in y.iter_mut().enumerate() {
        let t = k * down;
        // i such that 0 <= t - i*up < len(h)
        let i_max = (t / up).min(x.len() - 1);
        let i_min = if t + 1 > h.len() {
            (t + 1 - h.len()).div_ceil(up)
        } else {
            0
        };
        let mut acc = 0.0;
        if i_min <= i_max {
            for i in i_min..=i_max {
                acc += x[i] * h[t - i * up];
            }
        }
        *yk = acc;
    }
    y
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `scipy.signal.resample_poly(x, up, down)` with the default Kaiser(5.0) window.
pub(crate) fn resample_poly(x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    if n_in == 0 {
        return Vec::new();
    }
    let n_out = (n_in * up).div_ceil(down);
    let max_rate = up.max(down);
    let f_c = 1.0 / max_rate as f64;
    let half_len = 10 * max_rate;
    let mut h = firwin_lowpass(2 * half_len + 1, f_c, &Window::Kaiser(5.0));
    h.iter_mut().for_each(|v| *v *= up as f64);
    let n_pre_pad = down - half_len % down;
    let mut n_post_pad = 0;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    while upfirdn_output_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down)
        < n_out + n_pre_remove
    {
        n_post_pad += 1;
    }
    let mut hp = vec![0.0; n_pre_pad];
    hp.extend_from_slice(&h);
    hp.extend(std::iter::repeat_n(0.0, n_post_pad));
    let y = upfirdn(&hp, x, up, down);
    y[n_pre_remove..n_pre_remove + n_out].to_vec()
}

/// Solve a small dense linear system with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| a[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    x
}

/// Least-squares polynomial fit; returns coefficients from highest to lowest degree.
fn polyfit(x: &[f64], y: &[f64], deg: usize) -> Vec<f64> {
    let m = deg + 1;
    // Normal equations on powers (lowest first).
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];
    for (&xi, &yi) in x.iter().zip(y) {
        let pw: Vec<f64> = (0..m).map(|p| xi.powi(p as i32)).collect();
        for r in 0..m {
            aty[r] += pw[r] * yi;
            for c in 0..m {
                ata[r][c] += pw[r] * pw[c];
            }
        }
    }
    let mut c = solve(ata, aty);
    c.reverse();
    c
}

fn polyder(p: &[f64], m: usize) -> Vec<f64> {
    let mut p = p.to_vec();
    for _ in 0..m {
        let n = p.len();
        if n <= 1 {
            return vec![0.0];
        }
        p = (0..n - 1).map(|i| p[i] * (n - 1 - i) as f64).collect();
    }
    p
}

fn polyval(p: &[f64], x: f64) -> f64 {
    p.iter().fold(0.0, |acc, &c| acc * x + c)
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|v| v as f64).product()
}

/// Savitzky-Golay coefficients for use as a correlation ("dot") kernel centered on the window.
fn savgol_dot_coeffs(window_length: usize, polyorder: usize, deriv: usize) -> Vec<f64> {
    if deriv > polyorder {
        return vec![0.0; window_length];
    }
    let pos = (window_length / 2) as f64;
    let xs: Vec<f64> = (0..window_length).map(|k| k as f64 - pos).collect();
    let m = polyorder + 1;
    // Minimum-norm solution of A c = y with A[p][k] = xs[k]^p: c = A^T (A A^T)^-1 y.
    let a: Vec<Vec<f64>> = (0..m)
        .map(|p| xs.iter().map(|x| x.powi(p as i32)).collect())
        .collect();
    let aat: Vec<Vec<f64>> = (0..m)
        .map(|r| {
            (0..m)
                .map(|c| (0..window_length).map(|k| a[r][k] * a[c][k]).sum())
                .collect()
        })
        .collect();
    let mut y = vec![0.0; m];
    y[deriv] = factorial(deriv);
    let z = solve(aat, y);
    (0..window_length)
        .map(|k| (0..m).map(|p| a[p][k] * z[p]).sum())
        .collect()
}

/// `scipy.signal.savgol_filter(x, window_length, polyorder, deriv, mode="interp")` on one lane.
pub(crate) fn savgol_interp(
    x: &[f64],
    window_length: usize,
    polyorder: usize,
    deriv: usize,
) -> Vec<f64> {
    let n = x.len();
    let c = savgol_dot_coeffs(window_length, polyorder, deriv);
    let half = window_length / 2;
    let mut y = vec![0.0; n];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, ck) in c.iter().enumerate() {
            let j = i as isize + k as isize - half as isize;
            if j >= 0 && (j as usize) < n {
                acc += ck * x[j as usize];
            }
        }
        *yi = acc;
    }
    // Fit polynomials to the edges.
    let fit_edge = |y: &mut [f64], ws: usize, we: usize, is: usize, ie: usize| {
        let xs: Vec<f64> = (0..we - ws).map(|v| v as f64).collect();
        let p = polyfit(&xs, &x[ws..we], polyorder);
        let p = polyder(&p, deriv);
        for i in is..ie {
            y[i] = polyval(&p, (i - ws) as f64);
        }
    };
    fit_edge(&mut y, 0, window_length, 0, half);
    fit_edge(&mut y, n - window_length, n, n - half, n);
    y
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9).
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-300 {
        d = 1e-300;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..1000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// CDF of the Beta(a, b) distribution (regularized incomplete beta function).
pub(crate) fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a.fract() == 0.0 && b.fract() == 0.0 && a + b < 170.0 {
        // Exact binomial expansion for integer parameters.
        let n = (a + b - 1.0) as usize;
        let a = a as usize;
        let mut sum = 0.0;
        for j in a..=n {
            // exact binomial coefficient C(n, j)
            let mut c = 1.0;
            for i in 0..j {
                c = c * (n - i) as f64 / (i + 1) as f64;
            }
            sum += c * x.powi(j as i32) * (1.0 - x).powi((n - j) as i32);
        }
        return sum.min(1.0);
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// PMF of the truncated discrete exponential (`scipy.stats.boltzmann.pmf(k, lambda_, n)`).
pub(crate) fn boltzmann_pmf(k: i64, lambda: f64, n: i64) -> f64 {
    if k < 0 || k >= n {
        return 0.0;
    }
    let fact = (1.0 - (-lambda).exp()) / (1.0 - (-lambda * n as f64).exp());
    fact * (-lambda * k as f64).exp()
}

/// Zero-phase band-limited resampler (Kaiser-windowed sinc, 64 zero crossings per side,
/// beta = 12, cutoff 0.97 x Nyquist of the lower rate). `phases = Some((up, down))` means the
/// ratio is exactly `up / down` and the kernel is evaluated exactly for each of the `up`
/// output phases; otherwise a finely sampled kernel table is linearly interpolated.
pub(crate) fn resample_bandlimited(
    x: &[f64],
    ratio: f64,
    phases: Option<(usize, usize)>,
) -> Vec<f64> {
    const ZEROS: f64 = 64.0;
    const BETA: f64 = 12.0;
    const OVERSAMPLE: usize = 4096;
    let n_in = x.len();
    let n_out = (n_in as f64 * ratio).ceil() as usize;
    let cutoff = 0.97 * ratio.min(1.0);
    let half_width = ZEROS / cutoff;
    let i0b = crate::window::bessel_i0(BETA);
    // kernel as a function of r = d / half_width in [-1, 1]
    let kernel = |r: f64| -> f64 {
        if r.abs() >= 1.0 {
            return 0.0;
        }
        cutoff * sinc(ZEROS * r) * crate::window::bessel_i0(BETA * (1.0 - r * r).sqrt()) / i0b
    };
    let mut out = vec![0.0; n_out];
    if n_in == 0 {
        return out;
    }
    let taps = |t: f64| -> (usize, usize) {
        let lo = (t - half_width).ceil().max(0.0) as usize;
        let hi = ((t + half_width).floor().max(0.0) as usize).min(n_in - 1);
        (lo, hi)
    };
    if let Some((up, down)) = phases.filter(|(u, _)| *u <= 2048) {
        // exact kernels per output phase: t_k = k * down / up = m + j / up
        let width = (2.0 * half_width).ceil() as usize + 2;
        let base = half_width.ceil() as isize;
        let tables: Vec<Vec<f64>> = (0..up)
            .map(|j| {
                let frac = j as f64 / up as f64;
                (0..width)
                    .map(|q| {
                        let n_off = q as isize - base; // input index relative to floor(t)
                        kernel((frac - n_off as f64) / half_width)
                    })
                    .collect()
            })
            .collect();
        for (k, o) in out.iter_mut().enumerate() {
            let num = k * down;
            let m = (num / up) as isize;
            let j = num % up;
            let tab = &tables[j];
            let mut acc = 0.0;
            for (q, &h) in tab.iter().enumerate() {
                let n = m + q as isize - base;
                if n >= 0 && (n as usize) < n_in {
                    acc += x[n as usize] * h;
                }
            }
            *o = acc;
        }
        return out;
    }
    let tsize = OVERSAMPLE * ZEROS as usize;
    let table: Vec<f64> = (0..=tsize + 1)
        .map(|i| kernel(i as f64 / tsize as f64))
        .collect();
    for (k, o) in out.iter_mut().enumerate() {
        let t = k as f64 / ratio;
        let (lo, hi) = taps(t);
        let mut acc = 0.0;
        for (n, xn) in x.iter().enumerate().take(hi + 1).skip(lo) {
            let pos = ((t - n as f64).abs() / half_width) * tsize as f64;
            let i = pos as usize;
            if i >= tsize {
                continue;
            }
            let f = pos - i as f64;
            acc += xn * (table[i] + f * (table[i + 1] - table[i]));
        }
        *o = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_path_matches_exact_phase_path() {
        let x: Vec<f64> = (0..4000)
            .map(|n| (n as f64 * 0.05).sin() + 0.3 * (n as f64 * 0.31).cos())
            .collect();
        let a = resample_bandlimited(&x, 320.0 / 441.0, Some((320, 441)));
        let b = resample_bandlimited(&x, 320.0 / 441.0, None);
        assert_eq!(a.len(), b.len());
        let worst = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-6, "worst {worst}");
    }

    #[test]
    fn beta_cdf_integer_and_general_agree() {
        for &x in &[0.01, 0.1, 0.3, 0.7] {
            let exact = beta_cdf(x, 2.0, 18.0);
            let cf = {
                // force the continued-fraction path with a non-integer perturbation
                beta_cdf(x, 2.0 + 1e-12, 18.0)
            };
            assert!((exact - cf).abs() < 1e-9, "{x}: {exact} vs {cf}");
        }
    }

    #[test]
    fn savgol_first_derivative_of_line_is_slope() {
        let x: Vec<f64> = (0..20).map(|n| 3.0 * n as f64 + 1.0).collect();
        let d = savgol_interp(&x, 9, 1, 1);
        assert!(d.iter().all(|v| (v - 3.0).abs() < 1e-10));
    }
}
