//! Constant-Q and variable-Q transforms (`librosa.cqt`, `librosa.vqt`).

use crate::audio::{ResType, resample};
use crate::convert::{interval_frequencies, note_to_hz};
use crate::error::{Result, param};
use crate::fft::cfft;
use crate::filters::{relative_bandwidth, wavelet, wavelet_lengths};
use crate::spectrum::{StftParams, stft};
use crate::util::{Norm, PadMode};
use crate::window::Window;
use ndarray::{Array1, Array2, ArrayView1, s};
use num_complex::Complex64;

/// Parameters for [`vqt`] / [`cqt`], with librosa's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct CqtParams {
    /// Hop length (must be divisible by `2^(n_octaves - 1)` for multirate processing).
    pub hop_length: usize,
    /// Lowest frequency; `None` means C1 (~32.7 Hz).
    pub fmin: Option<f64>,
    /// Number of frequency bins.
    pub n_bins: usize,
    /// Bins per octave.
    pub bins_per_octave: usize,
    /// Tuning offset in fractions of a bin; `None` estimates it from the signal.
    pub tuning: Option<f64>,
    /// Filter scale factor (larger = longer filters).
    pub filter_scale: f64,
    /// Normalization of each basis filter.
    pub norm: Option<Norm>,
    /// Fraction of each filter's energy that may be discarded (sparsification).
    pub sparsity: f64,
    /// Window applied to each wavelet.
    pub window: Window,
    /// Scale responses by `1/sqrt(filter_length)` (librosa's `scale=True`).
    pub scale: bool,
    /// Padding mode for the internal STFTs.
    pub pad_mode: PadMode,
    /// Resampler used for the multirate decomposition.
    pub res_type: ResType,
}

impl Default for CqtParams {
    fn default() -> Self {
        CqtParams {
            hop_length: 512,
            fmin: None,
            n_bins: 84,
            bins_per_octave: 12,
            tuning: Some(0.0),
            filter_scale: 1.0,
            norm: Some(Norm::P(1.0)),
            sparsity: 0.01,
            window: Window::Hann,
            scale: true,
            pad_mode: PadMode::Constant,
            res_type: ResType::Sinc,
        }
    }
}

fn num_two_factors(mut x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    let mut n = 0;
    while x.is_multiple_of(2) {
        n += 1;
        x /= 2;
    }
    n
}

/// Zero out the smallest-magnitude entries of each row whose cumulative share of the row's
/// L1 mass is below `quantile` (`librosa.util.sparsify_rows`).
fn sparsify_rows(x: &mut Array2<Complex64>, quantile: f64) {
    for mut row in x.rows_mut() {
        let mags: Vec<f64> = row.iter().map(|z| z.norm()).collect();
        let norm: f64 = mags.iter().sum();
        let mut sorted = mags.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mut cum = 0.0;
        let mut idx = 0;
        for (i, m) in sorted.iter().enumerate() {
            cum += m / norm;
            if cum >= quantile {
                idx = i;
                break;
            }
        }
        let thresh = sorted[idx];
        for (z, &m) in row.iter_mut().zip(mags.iter()) {
            if m < thresh {
                *z = Complex64::new(0.0, 0.0);
            }
        }
    }
}

/// Build the FFT-domain filter basis for one octave. Returns `(basis, n_fft)`.
#[allow(clippy::too_many_arguments)]
fn vqt_filter_fft(
    sr: f64,
    freqs: &[f64],
    filter_scale: f64,
    norm: Option<Norm>,
    sparsity: f64,
    window: &Window,
    gamma: Option<f64>,
    alpha: &[f64],
) -> Result<(Array2<Complex64>, usize)> {
    let (basis, lengths) = wavelet(
        freqs,
        sr,
        window,
        filter_scale,
        true,
        norm,
        gamma,
        Some(alpha),
    )?;
    let n_fft = basis.ncols();
    let mut fft_basis = Array2::<Complex64>::zeros((freqs.len(), n_fft / 2 + 1));
    for (i, row) in basis.rows().into_iter().enumerate() {
        let scaled: Vec<Complex64> = row
            .iter()
            .map(|z| z * (lengths[i] / n_fft as f64))
            .collect();
        let f = cfft(&scaled, n_fft);
        for k in 0..=n_fft / 2 {
            fft_basis[[i, k]] = f[k];
        }
    }
    sparsify_rows(&mut fft_basis, sparsity);
    Ok((fft_basis, n_fft))
}

fn cqt_response(
    y: ArrayView1<f64>,
    n_fft: usize,
    hop: usize,
    basis: &Array2<Complex64>,
    pad_mode: PadMode,
) -> Result<Array2<Complex64>> {
    let p = StftParams {
        n_fft,
        hop_length: Some(hop),
        win_length: None,
        window: Window::Ones,
        center: true,
        pad_mode,
    };
    let d = stft(y, &p)?;
    // Sparse-aware product basis (n_filters x n_bins) . D (n_bins x n_frames)
    let (nf, nb) = basis.dim();
    let nt = d.ncols();
    let mut out = Array2::<Complex64>::zeros((nf, nt));
    for i in 0..nf {
        let nz: Vec<(usize, Complex64)> = (0..nb)
            .filter_map(|k| {
                let b = basis[[i, k]];
                (b.re != 0.0 || b.im != 0.0).then_some((k, b))
            })
            .collect();
        for t in 0..nt {
            let mut acc = Complex64::new(0.0, 0.0);
            for &(k, b) in &nz {
                acc += b * d[[k, t]];
            }
            out[[i, t]] = acc;
        }
    }
    Ok(out)
}

/// Variable-Q transform (`librosa.vqt`). `gamma = None` uses the ERB-derived default
/// bandwidth offset; `gamma = Some(0.0)` gives the constant-Q transform.
///
/// Output shape: `(n_bins, n_frames)`.
pub fn vqt(
    y: ArrayView1<f64>,
    sr: f64,
    gamma: Option<f64>,
    p: &CqtParams,
) -> Result<Array2<Complex64>> {
    let bpo = p.bins_per_octave;
    let fmin = match p.fmin {
        Some(f) => f,
        None => note_to_hz("C1")?,
    };
    let tuning = match p.tuning {
        Some(t) => t,
        None => crate::pitch::estimate_tuning(
            y,
            sr,
            &crate::pitch::TuningParams {
                bins_per_octave: bpo,
                ..Default::default()
            },
        )?,
    };
    let fmin = fmin * 2f64.powf(tuning / bpo as f64);
    if fmin >= sr / 2.0 {
        return param(format!("fmin={fmin} must be less than sr/2={}", sr / 2.0));
    }
    let n_bins = p.n_bins;
    let freqs = interval_frequencies(n_bins, fmin, bpo).to_vec();
    let alpha = if n_bins == 1 {
        // __et_relative_bw
        let r = 2f64.powf(1.0 / bpo as f64);
        vec![(r * r - 1.0) / (r * r + 1.0)]
    } else {
        relative_bandwidth(&freqs)?
    };
    let (_, filter_cutoff) =
        wavelet_lengths(&freqs, sr, &p.window, p.filter_scale, gamma, Some(&alpha))?;
    let nyquist = sr / 2.0;
    if filter_cutoff > nyquist {
        return param(format!(
            "Wavelet basis with max frequency would exceed the Nyquist frequency={nyquist}. Try reducing the number of frequency bins."
        ));
    }
    let n_octaves = n_bins.div_ceil(bpo);
    let n_filters = bpo.min(n_bins);

    // Early downsampling
    let mut y_cur: Array1<f64> = y.to_owned();
    let mut sr_cur = sr;
    let mut hop = p.hop_length;
    let c1 = (((nyquist / filter_cutoff).log2().ceil() as i64) - 1 - 1).max(0) as usize;
    let c2 = (num_two_factors(hop) as i64 - n_octaves as i64 + 1).max(0) as usize;
    let count = c1.min(c2);
    if count > 0 {
        let factor = 1usize << count;
        hop /= factor;
        if y_cur.len() < factor {
            return param(format!(
                "Input signal length={} is too short for {n_octaves}-octave CQT",
                y.len()
            ));
        }
        y_cur = resample(y_cur.view(), factor as f64, 1.0, p.res_type, true, true)?;
        if !p.scale {
            let s = (factor as f64).sqrt();
            y_cur.mapv_inplace(|v| v * s);
        }
        sr_cur = sr / factor as f64;
    }
    let sr_base = sr_cur;

    let mut responses: Vec<Array2<Complex64>> = Vec::new();
    let mut my_y = y_cur;
    let mut my_sr = sr_cur;
    let mut my_hop = hop;
    for i in 0..n_octaves {
        let (lo, hi) = if i == 0 {
            (n_bins - n_filters, n_bins)
        } else {
            (
                n_bins.saturating_sub(n_filters * (i + 1)),
                n_bins - n_filters * i,
            )
        };
        let freqs_oct = &freqs[lo..hi];
        let alpha_oct = &alpha[lo..hi];
        let (mut basis, n_fft) = vqt_filter_fft(
            my_sr,
            freqs_oct,
            p.filter_scale,
            p.norm,
            p.sparsity,
            &p.window,
            gamma,
            alpha_oct,
        )?;
        let sc = (sr_base / my_sr).sqrt();
        basis.mapv_inplace(|z| z * sc);
        responses.push(cqt_response(
            my_y.view(),
            n_fft,
            my_hop,
            &basis,
            p.pad_mode,
        )?);
        if i < n_octaves - 1 {
            // f_max_next = freqs[sl.start - 1]
            let start = n_bins as i64 - (n_filters * (i + 1)) as i64;
            let f_max_next = freqs[(start - 1).max(0) as usize];
            if my_hop.is_multiple_of(2) && f_max_next <= my_sr / 5.0 {
                my_hop /= 2;
                my_sr /= 2.0;
                my_y = resample(my_y.view(), 2.0, 1.0, p.res_type, true, true)?;
            }
        }
    }
    // Trim and stack (highest octave first in `responses`)
    let max_col = responses.iter().map(|r| r.ncols()).min().unwrap_or(0);
    let mut out = Array2::<Complex64>::zeros((n_bins, max_col));
    let mut end = n_bins;
    for r in &responses {
        let n_oct = r.nrows();
        if end < n_oct {
            out.slice_mut(s![..end, ..])
                .assign(&r.slice(s![n_oct - end.., ..max_col]));
            end = 0;
        } else {
            out.slice_mut(s![end - n_oct..end, ..])
                .assign(&r.slice(s![.., ..max_col]));
            end -= n_oct;
        }
    }
    if p.scale {
        let (lengths, _) = wavelet_lengths(
            &freqs,
            sr_base,
            &p.window,
            p.filter_scale,
            gamma,
            Some(&alpha),
        )?;
        for (i, mut row) in out.rows_mut().into_iter().enumerate() {
            let s = lengths[i].sqrt();
            row.mapv_inplace(|z| z / s);
        }
    }
    Ok(out)
}

/// Constant-Q transform (`librosa.cqt`): the VQT with `gamma = 0`.
///
/// Output shape: `(n_bins, n_frames)`.
pub fn cqt(y: ArrayView1<f64>, sr: f64, p: &CqtParams) -> Result<Array2<Complex64>> {
    vqt(y, sr, Some(0.0), p)
}
