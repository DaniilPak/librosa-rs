//! Pitch and tuning estimation (`librosa.core.pitch`): `piptrack`, `estimate_tuning`,
//! `yin` and `pyin`.

use crate::audio::autocorrelate;
use crate::convert::{fft_frequencies, hz_to_octs};
use crate::error::{Result, param};
use crate::sequence::{transition_local, transition_loop, viterbi};
use crate::signal::{beta_cdf, boltzmann_pmf};
use crate::spectrum::{StftParams, spectrogram};
use crate::util::{self, PadMode, TINY, localmin};
use crate::window::Window;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

/// Parameters for [`piptrack`] and [`estimate_tuning`].
#[derive(Debug, Clone, PartialEq)]
pub struct TuningParams {
    /// STFT parameters used when the input is audio (`hop_length` defaults to `n_fft / 4`).
    pub stft: StftParams,
    /// Lowest frequency considered (Hz).
    pub fmin: f64,
    /// Highest frequency considered (Hz).
    pub fmax: f64,
    /// Peaks below `threshold * max(column)` are ignored.
    pub threshold: f64,
    /// Histogram resolution, in fractions of a bin (for tuning estimation).
    pub resolution: f64,
    /// Bins per octave (for tuning estimation).
    pub bins_per_octave: usize,
}

impl Default for TuningParams {
    fn default() -> Self {
        TuningParams {
            stft: StftParams::default(),
            fmin: 150.0,
            fmax: 4000.0,
            threshold: 0.1,
            resolution: 0.01,
            bins_per_octave: 12,
        }
    }
}

/// Pitch tracking on thresholded parabolically-interpolated STFT peaks (`librosa.piptrack`)
/// from a magnitude spectrogram. Returns `(pitches, magnitudes)`, both shaped like `s`.
pub fn piptrack_spec(
    s: ArrayView2<f64>,
    sr: f64,
    n_fft: usize,
    p: &TuningParams,
) -> (Array2<f64>, Array2<f64>) {
    let s = s.mapv(f64::abs);
    let (n_bins, n_t) = s.dim();
    let fmin = p.fmin.max(0.0);
    let fmax = p.fmax.min(sr / 2.0);
    let freqs = fft_frequencies(sr, n_fft);
    let mut pitches = Array2::zeros((n_bins, n_t));
    let mut mags = Array2::zeros((n_bins, n_t));
    for t in 0..n_t {
        let col = s.column(t);
        let ref_value = p.threshold * col.fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let masked: Array1<f64> = col.mapv(|v| if v > ref_value { v } else { 0.0 });
        let lm = util::localmax(masked.view());
        for f in 0..n_bins {
            if !(lm[f] && fmin <= freqs[f] && freqs[f] < fmax) {
                continue;
            }
            // np.gradient along frequency (edge_order=1)
            let avg = if f == 0 {
                col[1] - col[0]
            } else if f == n_bins - 1 {
                col[f] - col[f - 1]
            } else {
                (col[f + 1] - col[f - 1]) / 2.0
            };
            let shift = if f == 0 || f == n_bins - 1 {
                0.0
            } else {
                pi_stencil(col[f - 1], col[f], col[f + 1])
            };
            let dskew = 0.5 * avg * shift;
            pitches[[f, t]] = (f as f64 + shift) * sr / n_fft as f64;
            mags[[f, t]] = col[f] + dskew;
        }
    }
    (pitches, mags)
}

/// [`piptrack_spec`] computed from audio.
pub fn piptrack(
    y: ArrayView1<f64>,
    sr: f64,
    p: &TuningParams,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let s = spectrogram(y, &p.stft, 1.0)?;
    Ok(piptrack_spec(s.view(), sr, p.stft.n_fft, p))
}

fn pi_stencil(xm1: f64, x0: f64, xp1: f64) -> f64 {
    let a = xp1 + xm1 - 2.0 * x0;
    let b = (xp1 - xm1) / 2.0;
    if b.abs() >= a.abs() { 0.0 } else { -b / a }
}

/// Estimate the tuning deviation of a set of frequencies (`librosa.pitch_tuning`),
/// in fractions of a bin in `[-0.5, 0.5)`.
pub fn pitch_tuning(frequencies: &[f64], resolution: f64, bins_per_octave: usize) -> f64 {
    let freqs: Vec<f64> = frequencies.iter().cloned().filter(|&f| f > 0.0).collect();
    if freqs.is_empty() {
        return 0.0;
    }
    let residual: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            let r = (bins_per_octave as f64 * hz_to_octs(f, 0.0, 12)).rem_euclid(1.0);
            if r >= 0.5 { r - 1.0 } else { r }
        })
        .collect();
    let n_edges = (1.0 / resolution).ceil() as usize + 1;
    let edges = crate::convert::linspace(-0.5, 0.5, n_edges);
    let counts = histogram(&residual, edges.as_slice().expect("contiguous"));
    let mut best = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c > counts[best] {
            best = i;
        }
    }
    edges[best]
}

/// `numpy.histogram` with uniform bin edges.
fn histogram(values: &[f64], edges: &[f64]) -> Vec<usize> {
    let n_bins = edges.len() - 1;
    let (lo, hi) = (edges[0], edges[n_bins]);
    let mut counts = vec![0usize; n_bins];
    let norm = n_bins as f64 / (hi - lo);
    for &v in values {
        if v < lo || v > hi {
            continue;
        }
        let mut i = ((v - lo) * norm) as usize;
        if i >= n_bins {
            i = n_bins - 1;
        }
        if v < edges[i] {
            i = i.saturating_sub(1);
        } else if i + 1 < n_bins && v >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1;
    }
    counts
}

/// Estimate tuning from a magnitude spectrogram (`librosa.estimate_tuning(S=...)`).
pub fn estimate_tuning_spec(s: ArrayView2<f64>, sr: f64, n_fft: usize, p: &TuningParams) -> f64 {
    let (pitch, mag) = piptrack_spec(s, sr, n_fft, p);
    let mut sel: Vec<f64> = pitch
        .iter()
        .zip(mag.iter())
        .filter(|(p, _)| **p > 0.0)
        .map(|(_, m)| *m)
        .collect();
    let threshold = if sel.is_empty() {
        0.0
    } else {
        util::median(&mut sel)
    };
    let freqs: Vec<f64> = pitch
        .iter()
        .zip(mag.iter())
        .filter(|(p, m)| **p > 0.0 && **m >= threshold)
        .map(|(p, _)| *p)
        .collect();
    pitch_tuning(&freqs, p.resolution, p.bins_per_octave)
}

/// Estimate tuning from audio (`librosa.estimate_tuning(y=...)`).
pub fn estimate_tuning(y: ArrayView1<f64>, sr: f64, p: &TuningParams) -> Result<f64> {
    let s = spectrogram(y, &p.stft, 1.0)?;
    Ok(estimate_tuning_spec(s.view(), sr, p.stft.n_fft, p))
}

/// Parameters shared by [`yin`] and [`pyin`].
#[derive(Debug, Clone, PartialEq)]
pub struct YinParams {
    /// Frame length in samples.
    pub frame_length: usize,
    /// Hop length; `None` means `frame_length / 4`.
    pub hop_length: Option<usize>,
    /// Pad so that frames are centered.
    pub center: bool,
    /// Padding mode when centering.
    pub pad_mode: PadMode,
    /// YIN only: absolute threshold for trough detection.
    pub trough_threshold: f64,
}

impl Default for YinParams {
    fn default() -> Self {
        YinParams {
            frame_length: 2048,
            hop_length: None,
            center: true,
            pad_mode: PadMode::Constant,
            trough_threshold: 0.1,
        }
    }
}

fn check_yin_params(sr: f64, fmax: f64, fmin: f64, frame_length: usize) -> Result<()> {
    if fmax > sr / 2.0 {
        return param(format!(
            "fmax={fmax:.3} cannot exceed Nyquist frequency {}",
            sr / 2.0
        ));
    }
    if fmin >= fmax {
        return param(format!("fmin={fmin:.3} must be less than fmax={fmax:.3}"));
    }
    if fmin <= 0.0 {
        return param(format!("fmin={fmin:.3} must be strictly positive"));
    }
    if sr / fmin >= frame_length as f64 - 1.0 {
        return param(format!(
            "fmin={fmin:.3} is too small for frame_length={frame_length} and sr={sr}."
        ));
    }
    Ok(())
}

/// Cumulative mean normalized difference function per frame: shape `(max_p - min_p + 1, n_frames)`.
fn cmndf(frames: &Array2<f64>, min_period: usize, max_period: usize) -> Array2<f64> {
    let (fl, n_t) = frames.dim();
    let n_out = max_period - min_period + 1;
    let mut out = Array2::zeros((n_out, n_t));
    let mut d = vec![0.0; max_period + 1];
    for t in 0..n_t {
        let col = frames.column(t);
        let acf = autocorrelate(col, Some(max_period + 1));
        // energy cumsum
        let mut cum = vec![0.0; fl];
        let mut acc = 0.0;
        for i in 0..fl {
            acc += col[i] * col[i];
            cum[i] = acc;
        }
        d[0] = 0.0;
        for k in 1..=max_period {
            d[k] = 2.0 * (acf[0] - acf[k]) - cum[k - 1];
        }
        // cumulative mean over k = 1..max_period
        let mut cm = vec![0.0; max_period];
        let mut s = 0.0;
        for k in 1..=max_period {
            s += d[k];
            cm[k - 1] = s / k as f64;
        }
        for (r, p) in (min_period..=max_period).enumerate() {
            let den = cm[p - 1];
            out[[r, t]] = d[p] / (den + TINY);
        }
    }
    out
}

fn parabolic_shifts(x: &Array2<f64>) -> Array2<f64> {
    let (n, n_t) = x.dim();
    let mut out = Array2::zeros((n, n_t));
    for t in 0..n_t {
        for i in 1..n.saturating_sub(1) {
            out[[i, t]] = pi_stencil(x[[i - 1, t]], x[[i, t]], x[[i + 1, t]]);
        }
    }
    out
}

fn frames_for_yin(y: ArrayView1<f64>, p: &YinParams) -> Result<(Array2<f64>, usize)> {
    crate::spectrum::valid_audio(y)?;
    let hop = p.hop_length.unwrap_or(p.frame_length / 4);
    let padded;
    let yy = if p.center {
        padded = util::pad(y, p.frame_length / 2, p.frame_length / 2, p.pad_mode);
        padded.view()
    } else {
        y
    };
    Ok((util::frame(yy, p.frame_length, hop)?, hop))
}

/// YIN fundamental frequency estimator (`librosa.yin`). Returns f0 (Hz) per frame.
pub fn yin(
    y: ArrayView1<f64>,
    fmin: f64,
    fmax: f64,
    sr: f64,
    p: &YinParams,
) -> Result<Array1<f64>> {
    check_yin_params(sr, fmax, fmin, p.frame_length)?;
    let (frames, _) = frames_for_yin(y, p)?;
    let min_period = (sr / fmax).floor() as usize;
    let max_period = ((sr / fmin).ceil() as usize).min(p.frame_length - 1);
    let yf = cmndf(&frames, min_period, max_period);
    let shifts = parabolic_shifts(&yf);
    let (n, n_t) = yf.dim();
    let mut f0 = Array1::zeros(n_t);
    for t in 0..n_t {
        let col = yf.column(t);
        let mut is_trough = localmin(col);
        if n > 1 {
            is_trough[0] = col[0] < col[1];
        }
        let mut period = None;
        for i in 0..n {
            if is_trough[i] && col[i] < p.trough_threshold {
                period = Some(i);
                break;
            }
        }
        let period = period.unwrap_or_else(|| {
            let mut best = 0;
            for i in 0..n {
                if col[i] < col[best] {
                    best = i;
                }
            }
            best
        });
        let yp = min_period as f64 + period as f64 + shifts[[period, t]];
        f0[t] = sr / yp;
    }
    Ok(f0)
}

/// Parameters for [`pyin`] (librosa defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct PyinParams {
    /// Framing parameters (`trough_threshold` is ignored).
    pub frames: YinParams,
    /// Number of thresholds for peak estimation.
    pub n_thresholds: usize,
    /// Shape parameters of the Beta prior over thresholds.
    pub beta_parameters: (f64, f64),
    /// Shape parameter of the Boltzmann prior over troughs.
    pub boltzmann_parameter: f64,
    /// Pitch resolution in semitones (`0.1` = 10 bins per semitone).
    pub resolution: f64,
    /// Maximum pitch transition rate in octaves per second.
    pub max_transition_rate: f64,
    /// Probability of switching between voiced and unvoiced.
    pub switch_prob: f64,
    /// Maximum probability to add to the global minimum if no trough is below threshold.
    pub no_trough_prob: f64,
    /// Value used for unvoiced frames (librosa: NaN).
    pub fill_na: f64,
    /// Minimum transition probability considered by Viterbi.
    pub transition_min_prob: Option<f64>,
}

impl Default for PyinParams {
    fn default() -> Self {
        PyinParams {
            frames: YinParams::default(),
            n_thresholds: 100,
            beta_parameters: (2.0, 18.0),
            boltzmann_parameter: 2.0,
            resolution: 0.1,
            max_transition_rate: 35.92,
            switch_prob: 0.01,
            no_trough_prob: 0.01,
            fill_na: f64::NAN,
            transition_min_prob: Some(1e-4),
        }
    }
}

/// Output of [`pyin`].
#[derive(Debug, Clone, PartialEq)]
pub struct PyinResult {
    /// Fundamental frequency per frame (`fill_na` where unvoiced).
    pub f0: Array1<f64>,
    /// Voicing decision per frame.
    pub voiced_flag: Vec<bool>,
    /// Probability that each frame is voiced.
    pub voiced_prob: Array1<f64>,
}

/// Probabilistic YIN (`librosa.pyin`): f0, voicing flags and voicing probabilities.
pub fn pyin(
    y: ArrayView1<f64>,
    fmin: f64,
    fmax: f64,
    sr: f64,
    p: &PyinParams,
) -> Result<PyinResult> {
    let fl = p.frames.frame_length;
    check_yin_params(sr, fmax, fmin, fl)?;
    let (frames, hop) = frames_for_yin(y, &p.frames)?;
    let min_period = (sr / fmax).floor() as usize;
    let max_period = ((sr / fmin).ceil() as usize).min(fl - 1);
    let yf = cmndf(&frames, min_period, max_period);
    let shifts = parabolic_shifts(&yf);
    let (n, n_t) = yf.dim();

    let thresholds = crate::convert::linspace(0.0, 1.0, p.n_thresholds + 1);
    let cdf: Vec<f64> = thresholds
        .iter()
        .map(|&x| beta_cdf(x, p.beta_parameters.0, p.beta_parameters.1))
        .collect();
    // clamp: in the far tail the CDF difference is below rounding noise
    let beta_probs: Vec<f64> = cdf.windows(2).map(|w| (w[1] - w[0]).max(0.0)).collect();
    let n_bins_per_semitone = (1.0 / p.resolution).ceil() as usize;
    let n_pitch_bins =
        (12.0 * n_bins_per_semitone as f64 * (fmax / fmin).log2()).floor() as usize + 1;

    let mut obs = Array2::<f64>::zeros((2 * n_pitch_bins, n_t));
    let mut voiced_prob = Array1::<f64>::zeros(n_t);
    let n_thr = p.n_thresholds;
    for t in 0..n_t {
        let col = yf.column(t);
        let mut is_trough = localmin(col);
        if n > 1 {
            is_trough[0] = col[0] < col[1];
        }
        let trough_index: Vec<usize> = (0..n).filter(|&i| is_trough[i]).collect();
        let mut yin_probs = vec![0.0; n];
        if !trough_index.is_empty() {
            let heights: Vec<f64> = trough_index.iter().map(|&i| col[i]).collect();
            let nt = heights.len();
            // below[i][j] = heights[i] < thresholds[j+1]
            let below: Vec<Vec<bool>> = heights
                .iter()
                .map(|&h| (0..n_thr).map(|j| h < thresholds[j + 1]).collect())
                .collect();
            let mut n_troughs = vec![0i64; n_thr];
            let mut positions = vec![vec![0i64; n_thr]; nt];
            for j in 0..n_thr {
                let mut c = 0i64;
                for i in 0..nt {
                    if below[i][j] {
                        c += 1;
                    }
                    positions[i][j] = c - 1;
                }
                n_troughs[j] = c;
            }
            let mut probs = vec![0.0; nt];
            for i in 0..nt {
                let mut acc = 0.0;
                for j in 0..n_thr {
                    if below[i][j] {
                        acc += boltzmann_pmf(positions[i][j], p.boltzmann_parameter, n_troughs[j])
                            * beta_probs[j];
                    }
                }
                probs[i] = acc;
            }
            let mut gmin = 0;
            for i in 0..nt {
                if heights[i] < heights[gmin] {
                    gmin = i;
                }
            }
            let n_below_min = (0..n_thr).filter(|&j| !below[gmin][j]).count();
            probs[gmin] += p.no_trough_prob * beta_probs[..n_below_min].iter().sum::<f64>();
            for (k, &i) in trough_index.iter().enumerate() {
                yin_probs[i] = probs[k];
            }
        }
        for (period, &prob) in yin_probs.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }
            let cand = min_period as f64 + period as f64 + shifts[[period, t]];
            let f0 = sr / cand;
            let bin = 12.0 * n_bins_per_semitone as f64 * (f0 / fmin).log2();
            let bin = bin.round_ties_even().clamp(0.0, n_pitch_bins as f64) as usize;
            obs[[bin, t]] = prob;
        }
        let vp = obs.slice(s![..n_pitch_bins, t]).sum().clamp(0.0, 1.0);
        voiced_prob[t] = vp;
        obs.slice_mut(s![n_pitch_bins.., t])
            .fill((1.0 - vp) / n_pitch_bins as f64);
    }

    let max_semitones_per_frame =
        (p.max_transition_rate * 12.0 * hop as f64 / sr).round_ties_even() as usize;
    let transition_width = max_semitones_per_frame * n_bins_per_semitone + 1;
    let local = transition_local(n_pitch_bins, transition_width, &Window::Triangle, false)?;
    let t_switch = transition_loop(2, 1.0 - p.switch_prob)?;
    let mut transition = Array2::<f64>::zeros((2 * n_pitch_bins, 2 * n_pitch_bins));
    for a in 0..2 {
        for b in 0..2 {
            let w = t_switch[[a, b]];
            transition
                .slice_mut(s![
                    a * n_pitch_bins..(a + 1) * n_pitch_bins,
                    b * n_pitch_bins..(b + 1) * n_pitch_bins
                ])
                .assign(&local.mapv(|v| v * w));
        }
    }
    let p_init = Array1::from_elem(2 * n_pitch_bins, 1.0 / (2 * n_pitch_bins) as f64);
    let states = viterbi(
        obs.view(),
        transition.view(),
        Some(p_init.view()),
        p.transition_min_prob,
    )?;
    let mut f0 = Array1::zeros(n_t);
    let mut voiced = vec![false; n_t];
    for t in 0..n_t {
        let st = states[t];
        let fbin = st % n_pitch_bins;
        f0[t] = fmin * 2f64.powf(fbin as f64 / (12 * n_bins_per_semitone) as f64);
        voiced[t] = st < n_pitch_bins;
        if !voiced[t] {
            f0[t] = p.fill_na;
        }
    }
    Ok(PyinResult {
        f0,
        voiced_flag: voiced,
        voiced_prob,
    })
}
