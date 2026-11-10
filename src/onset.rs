//! Onset detection (`librosa.onset`).

use crate::convert::time_to_frames;
use crate::error::{Result, param};
use crate::feature::{MelSpectrogramParams, melspectrogram};
use crate::filters::MelParams;
use crate::spectrum::{DbRef, StftParams, power_to_db};
use crate::util::{self, PeakPick, TINY};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// How to aggregate onset differences across frequency bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregate {
    /// Mean across bins (librosa default for `onset_strength`).
    #[default]
    Mean,
    /// Median across bins (used by `beat_track`).
    Median,
}

/// Parameters for [`onset_strength`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetStrengthParams {
    /// FFT size of the mel spectrogram.
    pub n_fft: usize,
    /// Hop length.
    pub hop_length: usize,
    /// Time lag (frames) for computing differences.
    pub lag: usize,
    /// Size (bins) of the local max filter applied to the reference spectrum.
    pub max_size: usize,
    /// Remove the DC component with a high-pass filter.
    pub detrend: bool,
    /// Shift the envelope to compensate for centered frames.
    pub center: bool,
    /// Aggregation across frequency.
    pub aggregate: Aggregate,
    /// Mel filter bank parameters (`fmax` defaults to `sr/2`).
    pub mel: MelParams,
}

impl Default for OnsetStrengthParams {
    fn default() -> Self {
        OnsetStrengthParams {
            n_fft: 2048,
            hop_length: 512,
            lag: 1,
            max_size: 1,
            detrend: false,
            center: true,
            aggregate: Aggregate::Mean,
            mel: MelParams::default(),
        }
    }
}

/// Spectral-flux onset strength envelope (`librosa.onset.onset_strength`) from audio.
pub fn onset_strength(y: ArrayView1<f64>, sr: f64, p: &OnsetStrengthParams) -> Result<Array1<f64>> {
    let mel = MelSpectrogramParams {
        stft: StftParams::new(p.n_fft, p.hop_length),
        power: Some(2.0),
        mel: MelParams {
            fmax: Some(p.mel.fmax.unwrap_or(0.5 * sr)),
            ..p.mel.clone()
        },
    };
    let s = melspectrogram(y, sr, &mel)?;
    let s = power_to_db(s.view(), DbRef::Value(1.0), 1e-10, Some(80.0))?;
    onset_strength_spec(s.view(), p)
}

/// Onset strength from a precomputed (dB-scaled) spectrogram (`onset_strength(S=...)`).
pub fn onset_strength_spec(s: ArrayView2<f64>, p: &OnsetStrengthParams) -> Result<Array1<f64>> {
    if p.lag == 0 {
        return param(format!("lag={} must be a positive integer", p.lag));
    }
    if p.max_size == 0 {
        return param(format!(
            "max_size={} must be a positive integer",
            p.max_size
        ));
    }
    let (n_f, n_t) = s.dim();
    let reference = if p.max_size == 1 {
        s.to_owned()
    } else {
        let mut r = s.to_owned();
        for t in 0..n_t {
            let col = s.column(t).to_vec();
            let m = util::maximum_filter_reflect(&col, p.max_size);
            for f in 0..n_f {
                r[[f, t]] = m[f];
            }
        }
        r
    };
    let n_diff = n_t.saturating_sub(p.lag);
    let mut env = Vec::with_capacity(n_diff);
    let mut buf = vec![0.0; n_f];
    for t in 0..n_diff {
        for f in 0..n_f {
            buf[f] = (s[[f, t + p.lag]] - reference[[f, t]]).max(0.0);
        }
        env.push(match p.aggregate {
            Aggregate::Mean => buf.iter().sum::<f64>() / n_f as f64,
            Aggregate::Median => util::median(&mut buf.clone()),
        });
    }
    let mut pad_width = p.lag;
    if p.center {
        pad_width += p.n_fft / (2 * p.hop_length);
    }
    let mut out = vec![0.0; pad_width];
    out.extend(env);
    if p.detrend {
        // lfilter([1, -1], [1, -0.99])
        let mut prev_x = 0.0;
        let mut prev_y = 0.0;
        for v in out.iter_mut() {
            let x = *v;
            let yv = x - prev_x + 0.99 * prev_y;
            prev_x = x;
            prev_y = yv;
            *v = yv;
        }
    }
    if p.center {
        out.truncate(n_t);
    }
    Ok(Array1::from(out))
}

/// Parameters for [`onset_detect`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OnsetDetectParams {
    /// Hop length of the envelope.
    pub hop_length: Option<usize>,
    /// Roll back each onset to the preceding local minimum of the envelope.
    pub backtrack: bool,
    /// Normalize the envelope to `[0, 1]` first.
    pub normalize: Option<bool>,
    /// Peak-picking parameters; `None` uses librosa's tuned defaults for `sr` and `hop_length`.
    pub peak_pick: Option<PeakPick>,
}

fn default_peak_pick(sr: f64, hop: usize) -> PeakPick {
    let h = hop as f64;
    let fd = |x: f64| (x / h).floor();
    PeakPick {
        pre_max: fd(0.03 * sr).ceil() as usize,
        post_max: (fd(0.00 * sr) + 1.0).ceil() as usize,
        pre_avg: fd(0.10 * sr).ceil() as usize,
        post_avg: (fd(0.10 * sr) + 1.0).ceil() as usize,
        wait: fd(0.03 * sr).ceil() as usize,
        delta: 0.07,
    }
}

/// Detect onsets (frame indices) from audio (`librosa.onset.onset_detect`).
///
/// Convert to seconds with [`frames_to_time`](crate::frames_to_time).
pub fn onset_detect(y: ArrayView1<f64>, sr: f64, p: &OnsetDetectParams) -> Result<Vec<usize>> {
    let hop = p.hop_length.unwrap_or(512);
    let env = onset_strength(
        y,
        sr,
        &OnsetStrengthParams {
            hop_length: hop,
            ..Default::default()
        },
    )?;
    onset_detect_env(env.view(), sr, p)
}

/// Detect onsets from a precomputed onset envelope.
pub fn onset_detect_env(
    onset_envelope: ArrayView1<f64>,
    sr: f64,
    p: &OnsetDetectParams,
) -> Result<Vec<usize>> {
    let hop = p.hop_length.unwrap_or(512);
    let mut env = onset_envelope.to_owned();
    if p.normalize.unwrap_or(true) {
        let mn = env.fold(f64::INFINITY, |m, &v| m.min(v));
        env.mapv_inplace(|v| v - mn);
        let mx = env.fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        env.mapv_inplace(|v| v / (mx + TINY));
    }
    if !env.iter().any(|&v| v != 0.0) || !env.iter().all(|v| v.is_finite()) {
        return Ok(Vec::new());
    }
    let pp = p.peak_pick.unwrap_or_else(|| default_peak_pick(sr, hop));
    let mut onsets = util::peak_pick(env.view(), &pp)?;
    if p.backtrack {
        onsets = onset_backtrack(&onsets, env.view());
    }
    Ok(onsets)
}

/// Move each detected onset back to the nearest preceding local minimum of `energy`
/// (`librosa.onset.onset_backtrack`).
pub fn onset_backtrack(events: &[usize], energy: ArrayView1<f64>) -> Vec<usize> {
    let n = energy.len();
    let mut minima = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if energy[i] <= energy[i - 1] && energy[i] < energy[i + 1] {
            minima.push(i);
        }
    }
    let minima = util::fix_frames(&minima, Some(0), None, true);
    util::match_events_left(events, &minima)
        .into_iter()
        .map(|i| minima[i])
        .collect()
}

/// Number of frames librosa's `tempo` uses for an `ac_size`-second autocorrelation window.
pub(crate) fn ac_frames(ac_size: f64, sr: f64, hop: usize) -> usize {
    time_to_frames(&[ac_size], sr, hop, None)[0].max(1) as usize
}
