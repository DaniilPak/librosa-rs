use crate::audio::autocorrelate;
use crate::convert::tempo_frequencies;
use crate::error::{Result, param};
use crate::onset::{OnsetStrengthParams, ac_frames, onset_strength};
use crate::util::{self, Norm, PadMode};
use crate::window::{Window, get_window};
use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Parameters for [`tempogram`].
#[derive(Debug, Clone, PartialEq)]
pub struct TempogramParams {
    /// Hop length of the onset envelope.
    pub hop_length: usize,
    /// Autocorrelation window length in frames.
    pub win_length: usize,
    /// Center the windows (linear-ramp padding).
    pub center: bool,
    /// Window applied before autocorrelation.
    pub window: Window,
    /// Per-frame normalization.
    pub norm: Option<Norm>,
}

impl Default for TempogramParams {
    fn default() -> Self {
        TempogramParams {
            hop_length: 512,
            win_length: 384,
            center: true,
            window: Window::Hann,
            norm: Some(Norm::Inf),
        }
    }
}

/// Autocorrelation tempogram from an onset envelope, shape `(win_length, n_frames)`
/// (`librosa.feature.tempogram(onset_envelope=...)`).
pub fn tempogram_env(onset_envelope: ArrayView1<f64>, p: &TempogramParams) -> Result<Array2<f64>> {
    if p.win_length < 1 {
        return param("win_length must be a positive integer");
    }
    let w = get_window(&p.window, p.win_length, true)?;
    let n = onset_envelope.len();
    let padded = if p.center {
        util::pad(
            onset_envelope,
            p.win_length / 2,
            p.win_length / 2,
            PadMode::LinearRamp,
        )
    } else {
        onset_envelope.to_owned()
    };
    let mut frames = util::frame(padded.view(), p.win_length, 1)?;
    if p.center && frames.ncols() > n {
        frames = frames.slice(ndarray::s![.., ..n]).to_owned();
    }
    let mut ac = Array2::zeros(frames.dim());
    for (t, col) in frames.axis_iter(Axis(1)).enumerate() {
        let x = &col * &w;
        ac.column_mut(t).assign(&autocorrelate(x.view(), None));
    }
    util::normalize(ac.view(), p.norm, 0, None, None)
}

/// Autocorrelation tempogram computed from audio (`librosa.feature.tempogram(y=...)`).
pub fn tempogram(y: ArrayView1<f64>, sr: f64, p: &TempogramParams) -> Result<Array2<f64>> {
    let env = onset_strength(
        y,
        sr,
        &OnsetStrengthParams {
            hop_length: p.hop_length,
            ..Default::default()
        },
    )?;
    tempogram_env(env.view(), p)
}

/// Parameters for [`tempo`].
#[derive(Debug, Clone, PartialEq)]
pub struct TempoParams {
    /// Hop length of the onset envelope.
    pub hop_length: usize,
    /// Prior mean tempo (BPM).
    pub start_bpm: f64,
    /// Prior standard deviation in octaves.
    pub std_bpm: f64,
    /// Autocorrelation window length in seconds.
    pub ac_size: f64,
    /// Upper limit on tempo (BPM); `None` disables it.
    pub max_tempo: Option<f64>,
    /// Average the tempogram over time (global tempo). `false` gives one estimate per frame.
    pub aggregate: bool,
}

impl Default for TempoParams {
    fn default() -> Self {
        TempoParams {
            hop_length: 512,
            start_bpm: 120.0,
            std_bpm: 1.0,
            ac_size: 8.0,
            max_tempo: Some(320.0),
            aggregate: true,
        }
    }
}

/// Estimate tempo (BPM) from an onset envelope (`librosa.feature.tempo(onset_envelope=...)`).
/// Returns one value, or one per frame when `aggregate` is false.
pub fn tempo_env(onset_envelope: ArrayView1<f64>, sr: f64, p: &TempoParams) -> Result<Vec<f64>> {
    if p.start_bpm <= 0.0 {
        return param("start_bpm must be strictly positive");
    }
    let win_length = ac_frames(p.ac_size, sr, p.hop_length);
    let tg = tempogram_env(
        onset_envelope,
        &TempogramParams {
            hop_length: p.hop_length,
            win_length,
            ..Default::default()
        },
    )?;
    let tg = if p.aggregate {
        tg.mean_axis(Axis(1))
            .expect("non-empty")
            .insert_axis(Axis(1))
    } else {
        tg
    };
    let bpms = tempo_frequencies(win_length, p.hop_length, sr);
    let mut logprior: Array1<f64> =
        bpms.mapv(|b| -0.5 * ((b.log2() - p.start_bpm.log2()) / p.std_bpm).powi(2));
    if let Some(mt) = p.max_tempo {
        let max_idx = bpms.iter().position(|&b| b < mt).unwrap_or(0);
        logprior
            .slice_mut(ndarray::s![..max_idx])
            .fill(f64::NEG_INFINITY);
    }
    let mut out = Vec::with_capacity(tg.ncols());
    for col in tg.axis_iter(Axis(1)) {
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for (i, (&v, &lp)) in col.iter().zip(logprior.iter()).enumerate() {
            let score = (1e6 * v).ln_1p() + lp;
            if score > best_v {
                best_v = score;
                best = i;
            }
        }
        out.push(bpms[best]);
    }
    Ok(out)
}

/// Estimate tempo (BPM) from audio (`librosa.feature.tempo(y=...)`).
pub fn tempo(y: ArrayView1<f64>, sr: f64, p: &TempoParams) -> Result<Vec<f64>> {
    let env = onset_strength(
        y,
        sr,
        &OnsetStrengthParams {
            hop_length: p.hop_length,
            ..Default::default()
        },
    )?;
    tempo_env(env.view(), sr, p)
}
