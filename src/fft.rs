//! Thin wrappers around `realfft` / `rustfft` with numpy/scipy conventions.

use num_complex::Complex64;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

/// Real FFT helper that keeps planned transforms for one size.
pub(crate) struct Rfft {
    n: usize,
    fwd: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    inv: std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
    scratch_in: Vec<f64>,
    scratch_c: Vec<Complex64>,
}

impl Rfft {
    pub(crate) fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        Rfft {
            n,
            fwd,
            inv,
            scratch_in: vec![0.0; n],
            scratch_c: vec![Complex64::new(0.0, 0.0); n / 2 + 1],
        }
    }

    /// `numpy.fft.rfft(x, n)`: `x` is zero-padded or truncated to `n`.
    pub(crate) fn forward(&mut self, x: &[f64], out: &mut [Complex64]) {
        let m = x.len().min(self.n);
        self.scratch_in[..m].copy_from_slice(&x[..m]);
        self.scratch_in[m..].iter_mut().for_each(|v| *v = 0.0);
        self.fwd
            .process(&mut self.scratch_in, out)
            .expect("rfft sizes are consistent");
    }

    /// `numpy.fft.irfft(X, n)` (normalized by `1/n`). Imaginary parts of the DC and
    /// Nyquist bins are ignored, as in numpy/scipy.
    pub(crate) fn inverse(&mut self, spec: &[Complex64], out: &mut [f64]) {
        let half = self.n / 2 + 1;
        let m = spec.len().min(half);
        self.scratch_c[..m].copy_from_slice(&spec[..m]);
        self.scratch_c[m..]
            .iter_mut()
            .for_each(|v| *v = Complex64::new(0.0, 0.0));
        self.scratch_c[0].im = 0.0;
        if self.n.is_multiple_of(2) {
            self.scratch_c[half - 1].im = 0.0;
        }
        // realfft returns an error only for the imaginary-part check we already handled.
        let _ = self.inv.process(&mut self.scratch_c, out);
        let scale = 1.0 / self.n as f64;
        out.iter_mut().for_each(|v| *v *= scale);
    }
}

/// Full complex FFT of `x` (zero-padded to `n`).
pub(crate) fn cfft(x: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    let m = x.len().min(n);
    buf[..m].copy_from_slice(&x[..m]);
    fft.process(&mut buf);
    buf
}
