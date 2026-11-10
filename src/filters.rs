//! Filter banks (`librosa.filters`): mel, chroma, constant-Q wavelets and window envelopes.

use crate::convert::{fft_frequencies, hz_to_midi, hz_to_octs, mel_frequencies, note_to_hz};
use crate::error::{Result, param};
use crate::util::{self, Norm};
use crate::window::{Window, get_window, window_bandwidth};
use ndarray::{Array1, Array2, s};
use num_complex::Complex64;
use std::f64::consts::PI;

pub use crate::window::{get_window as window, window_bandwidth as bandwidth};

/// Normalization of mel filters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MelNorm {
    /// Divide each triangle by its bandwidth (`norm="slaney"`, the default).
    #[default]
    Slaney,
    /// Normalize each filter to unit p-norm (`norm=p`).
    P(f64),
    /// Leave the triangles with peak 1 (`norm=None`).
    None,
}

/// Mel filter bank parameters (librosa defaults: 128 bands, `fmin=0`, `fmax=sr/2`, Slaney).
#[derive(Debug, Clone, PartialEq)]
pub struct MelParams {
    /// Number of mel bands.
    pub n_mels: usize,
    /// Lowest frequency (Hz).
    pub fmin: f64,
    /// Highest frequency (Hz); `None` means `sr / 2`.
    pub fmax: Option<f64>,
    /// Use the HTK formula instead of Slaney's.
    pub htk: bool,
    /// Filter normalization.
    pub norm: MelNorm,
}

impl Default for MelParams {
    fn default() -> Self {
        MelParams {
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
            htk: false,
            norm: MelNorm::Slaney,
        }
    }
}

/// Mel filter bank of shape `(n_mels, 1 + n_fft/2)` (`librosa.filters.mel`).
pub fn mel(sr: f64, n_fft: usize, p: &MelParams) -> Result<Array2<f64>> {
    let fmax = p.fmax.unwrap_or(sr / 2.0);
    let n_mels = p.n_mels;
    let n_bins = 1 + n_fft / 2;
    let fftfreqs = fft_frequencies(sr, n_fft);
    let mel_f = mel_frequencies(n_mels + 2, p.fmin, fmax, p.htk);
    let fdiff: Vec<f64> = (0..n_mels + 1).map(|i| mel_f[i + 1] - mel_f[i]).collect();
    let mut weights = Array2::<f64>::zeros((n_mels, n_bins));
    for i in 0..n_mels {
        for k in 0..n_bins {
            let lower = -(mel_f[i] - fftfreqs[k]) / fdiff[i];
            let upper = (mel_f[i + 2] - fftfreqs[k]) / fdiff[i + 1];
            weights[[i, k]] = 0f64.max(lower.min(upper));
        }
    }
    match p.norm {
        MelNorm::Slaney => {
            for i in 0..n_mels {
                let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
                weights.row_mut(i).mapv_inplace(|w| w * enorm);
            }
        }
        MelNorm::P(pn) => {
            weights = util::normalize(weights.view(), Some(Norm::P(pn)), 1, None, None)?;
        }
        MelNorm::None => {}
    }
    Ok(weights)
}

/// Chroma filter bank parameters (librosa defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaParams {
    /// Number of chroma bins.
    pub n_chroma: usize,
    /// Tuning deviation from A440 in fractions of a chroma bin.
    pub tuning: f64,
    /// Center of the Gaussian octave weighting (in octaves above `A440/16`).
    pub ctroct: f64,
    /// Width of the Gaussian octave weighting; `None` disables it.
    pub octwidth: Option<f64>,
    /// Column normalization of the filters; `None` disables it.
    pub norm: Option<Norm>,
    /// Start the filter bank at C (otherwise at A).
    pub base_c: bool,
}

impl Default for ChromaParams {
    fn default() -> Self {
        ChromaParams {
            n_chroma: 12,
            tuning: 0.0,
            ctroct: 5.0,
            octwidth: Some(2.0),
            norm: Some(Norm::P(2.0)),
            base_c: true,
        }
    }
}

/// Chroma filter bank of shape `(n_chroma, 1 + n_fft/2)` (`librosa.filters.chroma`).
pub fn chroma(sr: f64, n_fft: usize, p: &ChromaParams) -> Result<Array2<f64>> {
    let n_chroma = p.n_chroma;
    let nc = n_chroma as f64;
    // frequencies = linspace(0, sr, n_fft, endpoint=False)[1:]
    let step = sr / n_fft as f64;
    let mut frqbins: Vec<f64> = (1..n_fft)
        .map(|k| nc * hz_to_octs(k as f64 * step, p.tuning, n_chroma))
        .collect();
    frqbins.insert(0, frqbins[0] - 1.5 * nc);
    let mut binwidth: Vec<f64> = (0..n_fft - 1)
        .map(|k| (frqbins[k + 1] - frqbins[k]).max(1.0))
        .collect();
    binwidth.push(1.0);
    let n_chroma2 = (nc / 2.0).round_ties_even();
    let mut wts = Array2::<f64>::zeros((n_chroma, n_fft));
    for c in 0..n_chroma {
        for k in 0..n_fft {
            let d = frqbins[k] - c as f64;
            let d = (d + n_chroma2 + 10.0 * nc).rem_euclid(nc) - n_chroma2;
            wts[[c, k]] = (-0.5 * (2.0 * d / binwidth[k]).powi(2)).exp();
        }
    }
    let mut wts = util::normalize(wts.view(), p.norm, 0, None, None)?;
    if let Some(ow) = p.octwidth {
        for k in 0..n_fft {
            let g = (-0.5 * ((frqbins[k] / nc - p.ctroct) / ow).powi(2)).exp();
            wts.column_mut(k).mapv_inplace(|v| v * g);
        }
    }
    if p.base_c {
        let shift = 3 * (n_chroma / 12);
        let rolled =
            Array2::from_shape_fn((n_chroma, n_fft), |(c, k)| wts[[(c + shift) % n_chroma, k]]);
        wts = rolled;
    }
    Ok(wts.slice(s![.., ..1 + n_fft / 2]).to_owned())
}

/// Relative bandwidths of a set of (log-spaced) center frequencies.
pub(crate) fn relative_bandwidth(freqs: &[f64]) -> Result<Vec<f64>> {
    let n = freqs.len();
    if n <= 1 {
        return param("2 or more frequencies are required to compute bandwidths.");
    }
    let logf: Vec<f64> = freqs.iter().map(|f| f.log2()).collect();
    let mut bpo = vec![0.0; n];
    bpo[0] = 1.0 / (logf[1] - logf[0]);
    bpo[n - 1] = 1.0 / (logf[n - 1] - logf[n - 2]);
    for i in 1..n - 1 {
        bpo[i] = 2.0 / (logf[i + 1] - logf[i - 1]);
    }
    Ok(bpo
        .iter()
        .map(|b| (2f64.powf(2.0 / b) - 1.0) / (2f64.powf(2.0 / b) + 1.0))
        .collect())
}

/// Filter lengths (in samples) and the highest cutoff frequency of a wavelet basis
/// (`librosa.filters.wavelet_lengths`). `gamma = None` uses the ERB-based default of VQT.
pub fn wavelet_lengths(
    freqs: &[f64],
    sr: f64,
    window: &Window,
    filter_scale: f64,
    gamma: Option<f64>,
    alpha: Option<&[f64]>,
) -> Result<(Vec<f64>, f64)> {
    if filter_scale <= 0.0 {
        return param(format!("filter_scale={filter_scale} must be positive"));
    }
    if let Some(g) = gamma
        && g < 0.0
    {
        return param(format!("gamma={g} must be non-negative"));
    }
    if freqs.iter().any(|&f| f <= 0.0) {
        return param("frequencies must be strictly positive");
    }
    if freqs.windows(2).any(|w| w[0] > w[1]) {
        return param("Frequency array must be in strictly ascending order");
    }
    let alpha: Vec<f64> = match alpha {
        Some(a) => a.to_vec(),
        None => relative_bandwidth(freqs)?,
    };
    let gammas: Vec<f64> = match gamma {
        Some(g) => vec![g; freqs.len()],
        None => alpha.iter().map(|a| a * 24.7 / 0.108).collect(),
    };
    let bw = window_bandwidth(window);
    let mut f_cutoff = f64::NEG_INFINITY;
    let mut lengths = Vec::with_capacity(freqs.len());
    for i in 0..freqs.len() {
        let q = filter_scale / alpha[i];
        f_cutoff = f_cutoff.max(freqs[i] * (1.0 + 0.5 * bw / q) + 0.5 * gammas[i]);
        lengths.push(q * sr / (freqs[i] + gammas[i] / alpha[i]));
    }
    Ok((lengths, f_cutoff))
}

/// Complex wavelet (constant-Q / variable-Q) filters, each centered in a common buffer
/// (`librosa.filters.wavelet`). Returns `(filters, lengths)`; filters have shape
/// `(n_freqs, max_len)` with `max_len` rounded up to a power of two when `pad_fft`.
#[allow(clippy::too_many_arguments)]
pub fn wavelet(
    freqs: &[f64],
    sr: f64,
    window: &Window,
    filter_scale: f64,
    pad_fft: bool,
    norm: Option<Norm>,
    gamma: Option<f64>,
    alpha: Option<&[f64]>,
) -> Result<(Array2<Complex64>, Vec<f64>)> {
    let (lengths, _) = wavelet_lengths(freqs, sr, window, filter_scale, gamma, alpha)?;
    let mut filters: Vec<Vec<Complex64>> = Vec::with_capacity(freqs.len());
    for (&ilen, &freq) in lengths.iter().zip(freqs) {
        let start = (-ilen / 2.0).floor();
        let stop = (ilen / 2.0).floor();
        let n = (stop - start).ceil().max(0.0) as usize;
        let win = float_window(window, n as f64)?;
        let mut sig: Vec<Complex64> = (0..n)
            .map(|k| {
                let a = (start + k as f64) * 2.0 * PI * freq / sr;
                Complex64::new(a.cos(), a.sin()) * win[k]
            })
            .collect();
        if let Some(nm) = norm {
            let mags: Vec<f64> = sig.iter().map(|z| z.norm()).collect();
            let length = match nm {
                Norm::Inf => mags.iter().cloned().fold(0.0, f64::max),
                Norm::NegInf => mags.iter().cloned().fold(f64::INFINITY, f64::min),
                Norm::Zero => mags.iter().filter(|&&m| m > 0.0).count() as f64,
                Norm::P(p) => mags.iter().map(|m| m.powf(p)).sum::<f64>().powf(1.0 / p),
            };
            if length >= util::TINY {
                sig.iter_mut().for_each(|z| *z /= length);
            }
        }
        filters.push(sig);
    }
    let max_len = lengths.iter().cloned().fold(0.0, f64::max);
    let max_len = if pad_fft {
        2f64.powf(max_len.log2().ceil()) as usize
    } else {
        max_len.ceil() as usize
    };
    let mut out = Array2::<Complex64>::zeros((freqs.len(), max_len));
    for (i, f) in filters.iter().enumerate() {
        let lpad = (max_len - f.len()) / 2;
        for (k, z) in f.iter().enumerate() {
            out[[i, lpad + k]] = *z;
        }
    }
    Ok((out, lengths))
}

/// librosa's `__float_window`: a window of `floor(n)` samples padded to `ceil(n)`.
fn float_window(window: &Window, n: f64) -> Result<Vec<f64>> {
    let n_min = n.floor() as usize;
    let n_max = n.ceil() as usize;
    let mut w = get_window(window, n_min, true)?.to_vec();
    w.resize(n_max.max(n_min), 0.0);
    Ok(w)
}

/// Map constant-Q bins onto chroma bins (`librosa.filters.cq_to_chroma`).
///
/// `fmin` defaults to C1. Returns a matrix of shape `(n_chroma, n_input)`.
pub fn cq_to_chroma(
    n_input: usize,
    bins_per_octave: usize,
    n_chroma: usize,
    fmin: Option<f64>,
    base_c: bool,
) -> Result<Array2<f64>> {
    let n_merge = bins_per_octave as f64 / n_chroma as f64;
    let fmin = match fmin {
        Some(f) => f,
        None => note_to_hz("C1")?,
    };
    if n_merge.fract() != 0.0 {
        return param(
            "Incompatible CQ merge: input bins must be an integer multiple of output bins.",
        );
    }
    let n_merge = n_merge as usize;
    // eye(n_chroma) repeated n_merge times along columns, rolled left by n_merge // 2
    let width = n_chroma * n_merge;
    let base = Array2::from_shape_fn((n_chroma, width), |(r, c)| {
        let src = (c + n_merge / 2) % width;
        if src / n_merge == r { 1.0 } else { 0.0 }
    });
    let n_octaves = n_input.div_ceil(bins_per_octave);
    let mut tiled = Array2::<f64>::zeros((n_chroma, n_octaves * width));
    for o in 0..n_octaves {
        tiled
            .slice_mut(s![.., o * width..(o + 1) * width])
            .assign(&base);
    }
    let tiled = tiled.slice(s![.., ..n_input]).to_owned();
    let midi_0 = hz_to_midi(fmin).rem_euclid(12.0);
    let roll = if base_c { midi_0 } else { midi_0 - 9.0 };
    let roll = (roll * (n_chroma as f64 / 12.0)).round_ties_even() as i64;
    let out = Array2::from_shape_fn((n_chroma, n_input), |(r, c)| {
        let src = (r as i64 - roll).rem_euclid(n_chroma as i64) as usize;
        tiled[[src, c]]
    });
    Ok(out)
}

/// Sum-squared envelope of a window applied at `n_frames` hops
/// (`librosa.filters.window_sumsquare`), length `n_fft + hop_length * (n_frames - 1)`.
pub fn window_sumsquare(
    window: &Window,
    n_frames: usize,
    hop_length: usize,
    win_length: Option<usize>,
    n_fft: usize,
    norm: Option<Norm>,
) -> Result<Array1<f64>> {
    let win_length = win_length.unwrap_or(n_fft);
    let n = n_fft + hop_length * n_frames.saturating_sub(1);
    let mut x = Array1::<f64>::zeros(n);
    let w = get_window(window, win_length, true)?;
    let w = util::normalize_1d(w.view(), norm, None, None)?.mapv(|v| v * v);
    let w = util::pad_center(w.view(), n_fft)?;
    for i in 0..n_frames {
        let sample = i * hop_length;
        let end = n.min(sample + n_fft);
        for k in sample..end {
            x[k] += w[k - sample];
        }
    }
    Ok(x)
}

/// Orthonormal DCT-II matrix of shape `(n_out, n_in)` (`scipy.fft.dct(type=2, norm="ortho")`).
pub fn dct_ortho(n_out: usize, n_in: usize) -> Array2<f64> {
    let n = n_in as f64;
    Array2::from_shape_fn((n_out, n_in), |(k, i)| {
        let scale = if k == 0 {
            (1.0 / (4.0 * n)).sqrt()
        } else {
            (1.0 / (2.0 * n)).sqrt()
        };
        2.0 * scale * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n)).cos()
    })
}
