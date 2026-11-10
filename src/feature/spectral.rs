use super::{Input, get_spec};
use crate::audio::{ResType, zero_crossings};
use crate::constantq::{CqtParams, cqt};
use crate::convert::fft_frequencies;
use crate::error::{Result, param};
use crate::filters::{self, ChromaParams, MelParams, cq_to_chroma, dct_ortho};
use crate::pitch::{TuningParams, estimate_tuning_spec};
use crate::spectrum::{DbRef, StftParams, power_to_db};
use crate::util::{self, Norm, PadMode};
use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

fn check_nonneg(s: &Array2<f64>, what: &str) -> Result<()> {
    if s.iter().any(|&v| v < 0.0) {
        return param(format!("{what} is only defined with non-negative energies"));
    }
    Ok(())
}

/// Spectral centroid per frame, shape `(1, n_frames)` (`librosa.feature.spectral_centroid`).
///
/// From audio, `|STFT|` is used. `freq` overrides the bin center frequencies.
pub fn spectral_centroid<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    stft: &StftParams,
    freq: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let (s, n_fft) = get_spec(input.into(), stft, 1.0)?;
    check_nonneg(&s, "Spectral centroid")?;
    let f = freq.cloned().unwrap_or_else(|| fft_frequencies(sr, n_fft));
    let sn = util::normalize(s.view(), Some(Norm::P(1.0)), 0, None, None)?;
    let mut out = Array2::zeros((1, s.ncols()));
    for t in 0..s.ncols() {
        out[[0, t]] = sn.column(t).iter().zip(f.iter()).map(|(a, b)| a * b).sum();
    }
    Ok(out)
}

/// Parameters for [`spectral_bandwidth`].
#[derive(Debug, Clone, PartialEq)]
pub struct BandwidthParams {
    /// STFT parameters when the input is audio.
    pub stft: StftParams,
    /// Normalize each frame to unit L1 norm first.
    pub norm: bool,
    /// Order of the bandwidth (2 = standard deviation around the centroid).
    pub p: f64,
}

impl Default for BandwidthParams {
    fn default() -> Self {
        BandwidthParams {
            stft: StftParams::default(),
            norm: true,
            p: 2.0,
        }
    }
}

/// p-th order spectral bandwidth, shape `(1, n_frames)` (`librosa.feature.spectral_bandwidth`).
pub fn spectral_bandwidth<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    bp: &BandwidthParams,
) -> Result<Array2<f64>> {
    let (s, n_fft) = get_spec(input.into(), &bp.stft, 1.0)?;
    check_nonneg(&s, "Spectral bandwidth")?;
    let centroid = spectral_centroid(s.view(), sr, &bp.stft, None)?;
    let f = fft_frequencies(sr, n_fft);
    let sn = if bp.norm {
        util::normalize(s.view(), Some(Norm::P(1.0)), 0, None, None)?
    } else {
        s.clone()
    };
    let mut out = Array2::zeros((1, s.ncols()));
    for t in 0..s.ncols() {
        let c = centroid[[0, t]];
        let acc: f64 = sn
            .column(t)
            .iter()
            .zip(f.iter())
            .map(|(a, fr)| a * (fr - c).abs().powf(bp.p))
            .sum();
        out[[0, t]] = acc.powf(1.0 / bp.p);
    }
    Ok(out)
}

/// Parameters for [`spectral_contrast`].
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastParams {
    /// STFT parameters when the input is audio.
    pub stft: StftParams,
    /// Upper frequency of the first (lowest) band.
    pub fmin: f64,
    /// Number of octave bands (the output has `n_bands + 1` rows).
    pub n_bands: usize,
    /// Quantile defining peaks and valleys.
    pub quantile: f64,
    /// Return the linear difference instead of the dB difference.
    pub linear: bool,
}

impl Default for ContrastParams {
    fn default() -> Self {
        ContrastParams {
            stft: StftParams::default(),
            fmin: 200.0,
            n_bands: 6,
            quantile: 0.02,
            linear: false,
        }
    }
}

/// Spectral contrast, shape `(n_bands + 1, n_frames)` (`librosa.feature.spectral_contrast`).
pub fn spectral_contrast<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    cp: &ContrastParams,
) -> Result<Array2<f64>> {
    let (s, n_fft) = get_spec(input.into(), &cp.stft, 1.0)?;
    let freq = fft_frequencies(sr, n_fft);
    if freq.len() != s.nrows() {
        return param(format!("freq.shape mismatch: expected ({},)", s.nrows()));
    }
    if cp.n_bands < 1 {
        return param("n_bands must be a positive integer");
    }
    if !(0.0 < cp.quantile && cp.quantile < 1.0) {
        return param("quantile must lie in the range (0, 1)");
    }
    if cp.fmin <= 0.0 {
        return param("fmin must be a positive number");
    }
    let mut octa = vec![0.0; cp.n_bands + 2];
    for (k, o) in octa.iter_mut().enumerate().skip(1) {
        *o = cp.fmin * 2f64.powi(k as i32 - 1);
    }
    if octa[..octa.len() - 1].iter().any(|&o| o >= 0.5 * sr) {
        return param("Frequency band exceeds Nyquist. Reduce either fmin or n_bands.");
    }
    let n_t = s.ncols();
    let mut valley = Array2::<f64>::zeros((cp.n_bands + 1, n_t));
    let mut peak = Array2::<f64>::zeros((cp.n_bands + 1, n_t));
    for k in 0..=cp.n_bands {
        let (f_low, f_high) = (octa[k], octa[k + 1]);
        let mut band: Vec<bool> = freq.iter().map(|&f| f >= f_low && f <= f_high).collect();
        let idx: Vec<usize> = (0..band.len()).filter(|&i| band[i]).collect();
        if k > 0
            && let Some(&first) = idx.first()
            && first > 0
        {
            band[first - 1] = true;
        }
        if k == cp.n_bands
            && let Some(&last) = idx.last()
        {
            band[last + 1..].iter_mut().for_each(|b| *b = true);
        }
        let mut rows: Vec<usize> = (0..band.len()).filter(|&i| band[i]).collect();
        let count = rows.len();
        if k < cp.n_bands {
            rows.pop();
        }
        let q = ((cp.quantile * count as f64).round_ties_even() as usize).max(1);
        for t in 0..n_t {
            let mut vals: Vec<f64> = rows.iter().map(|&r| s[[r, t]]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let m = vals.len();
            let lo = q.min(m);
            valley[[k, t]] = vals[..lo].iter().sum::<f64>() / lo as f64;
            peak[[k, t]] = vals[m - lo..].iter().sum::<f64>() / lo as f64;
        }
    }
    if cp.linear {
        Ok(peak - valley)
    } else {
        let pdb = power_to_db(peak.view(), DbRef::Value(1.0), 1e-10, Some(80.0))?;
        let vdb = power_to_db(valley.view(), DbRef::Value(1.0), 1e-10, Some(80.0))?;
        Ok(pdb - vdb)
    }
}

/// Roll-off frequency: the lowest bin frequency below which `roll_percent` of the energy lies.
/// Shape `(1, n_frames)` (`librosa.feature.spectral_rolloff`).
pub fn spectral_rolloff<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    stft: &StftParams,
    roll_percent: f64,
) -> Result<Array2<f64>> {
    if !(0.0 < roll_percent && roll_percent < 1.0) {
        return param("roll_percent must lie in the range (0, 1)");
    }
    let (s, n_fft) = get_spec(input.into(), stft, 1.0)?;
    check_nonneg(&s, "Spectral rolloff")?;
    let freq = fft_frequencies(sr, n_fft);
    let mut out = Array2::zeros((1, s.ncols()));
    for t in 0..s.ncols() {
        let col = s.column(t);
        let mut cum = Vec::with_capacity(col.len());
        let mut acc = 0.0;
        for &v in col.iter() {
            acc += v;
            cum.push(acc);
        }
        let th = roll_percent * acc;
        let mut best = f64::NAN;
        for (i, &c) in cum.iter().enumerate() {
            if c >= th {
                let v = freq[i];
                if best.is_nan() || v < best {
                    best = v;
                }
            }
        }
        out[[0, t]] = best;
    }
    Ok(out)
}

/// Spectral flatness (Wiener entropy), shape `(1, n_frames)` (`librosa.feature.spectral_flatness`).
pub fn spectral_flatness<'a>(
    input: impl Into<Input<'a>>,
    stft: &StftParams,
    amin: f64,
    power: f64,
) -> Result<Array2<f64>> {
    if amin <= 0.0 {
        return param("amin must be strictly positive");
    }
    let (s, _) = get_spec(input.into(), stft, 1.0)?;
    check_nonneg(&s, "Spectral flatness")?;
    let n = s.nrows() as f64;
    let mut out = Array2::zeros((1, s.ncols()));
    for t in 0..s.ncols() {
        let th: Vec<f64> = s
            .column(t)
            .iter()
            .map(|&v| amin.max(v.powf(power)))
            .collect();
        let g = (th.iter().map(|v| v.ln()).sum::<f64>() / n).exp();
        let a = th.iter().sum::<f64>() / n;
        out[[0, t]] = g / a;
    }
    Ok(out)
}

/// Parameters for [`rms`] and [`zero_crossing_rate`].
#[derive(Debug, Clone, PartialEq)]
pub struct FrameParams {
    /// Frame length in samples.
    pub frame_length: usize,
    /// Hop length in samples.
    pub hop_length: usize,
    /// Center frames by padding `frame_length / 2` on both sides.
    pub center: bool,
    /// Padding mode for [`rms`] (zero-crossing rate always pads with `Edge`, as librosa does).
    pub pad_mode: PadMode,
}

impl Default for FrameParams {
    fn default() -> Self {
        FrameParams {
            frame_length: 2048,
            hop_length: 512,
            center: true,
            pad_mode: PadMode::Constant,
        }
    }
}

/// Root-mean-square energy per frame, shape `(1, n_frames)` (`librosa.feature.rms`).
///
/// From a spectrogram input, `frame_length` must equal `2 * (n_bins - 1)` (or `+1`).
pub fn rms<'a>(input: impl Into<Input<'a>>, fp: &FrameParams) -> Result<Array2<f64>> {
    match input.into() {
        Input::Audio(y) => {
            crate::spectrum::valid_audio(y)?;
            let padded;
            let yy = if fp.center {
                padded = util::pad(y, fp.frame_length / 2, fp.frame_length / 2, fp.pad_mode);
                padded.view()
            } else {
                y
            };
            let x = util::frame(yy, fp.frame_length, fp.hop_length)?;
            let p = x
                .mapv(|v| v * v)
                .mean_axis(Axis(0))
                .expect("non-empty frames");
            Ok(p.mapv(f64::sqrt).insert_axis(Axis(0)))
        }
        Input::Spectrogram(s) => {
            if s.nrows() != fp.frame_length / 2 + 1 {
                return param(format!(
                    "Since S.shape[-2] is {}, frame_length is expected to be {} or {}; found {}",
                    s.nrows(),
                    s.nrows() * 2 - 2,
                    s.nrows() * 2 - 1,
                    fp.frame_length
                ));
            }
            let mut x = s.mapv(|v| v * v);
            x.row_mut(0).mapv_inplace(|v| v * 0.5);
            if fp.frame_length.is_multiple_of(2) {
                let last = x.nrows() - 1;
                x.row_mut(last).mapv_inplace(|v| v * 0.5);
            }
            let fl2 = (fp.frame_length * fp.frame_length) as f64;
            let p = x.sum_axis(Axis(0)).mapv(|v| (2.0 * v / fl2).sqrt());
            Ok(p.insert_axis(Axis(0)))
        }
    }
}

/// Zero-crossing rate per frame, shape `(1, n_frames)` (`librosa.feature.zero_crossing_rate`).
pub fn zero_crossing_rate(
    y: ndarray::ArrayView1<f64>,
    fp: &FrameParams,
    threshold: f64,
    zero_pos: bool,
) -> Result<Array2<f64>> {
    crate::spectrum::valid_audio(y)?;
    let padded;
    let yy = if fp.center {
        padded = util::pad(y, fp.frame_length / 2, fp.frame_length / 2, PadMode::Edge);
        padded.view()
    } else {
        y
    };
    let x = util::frame(yy, fp.frame_length, fp.hop_length)?;
    let mut out = Array2::zeros((1, x.ncols()));
    for t in 0..x.ncols() {
        let z = zero_crossings(x.column(t), threshold, false, zero_pos);
        out[[0, t]] = z.iter().filter(|&&b| b).count() as f64 / x.nrows() as f64;
    }
    Ok(out)
}

/// Parameters for [`melspectrogram`] and [`mfcc`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MelSpectrogramParams {
    /// STFT parameters when the input is audio.
    pub stft: StftParams,
    /// Exponent of the magnitude (2 = power, 1 = energy). librosa default: 2.
    pub power: Option<f64>,
    /// Mel filter bank parameters.
    pub mel: MelParams,
}

impl MelSpectrogramParams {
    fn power(&self) -> f64 {
        self.power.unwrap_or(2.0)
    }
}

/// Mel-scaled spectrogram, shape `(n_mels, n_frames)` (`librosa.feature.melspectrogram`).
///
/// ```
/// use librosa::feature::{melspectrogram, MelSpectrogramParams};
/// let y = librosa::tone(440.0, 22050.0, 22050);
/// let m = melspectrogram(&y, 22050.0, &MelSpectrogramParams::default()).unwrap();
/// assert_eq!(m.dim(), (128, 44));
/// ```
pub fn melspectrogram<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    mp: &MelSpectrogramParams,
) -> Result<Array2<f64>> {
    let (s, n_fft) = get_spec(input.into(), &mp.stft, mp.power())?;
    let basis = filters::mel(sr, n_fft, &mp.mel)?;
    Ok(basis.dot(&s))
}

/// Parameters for [`mfcc`].
#[derive(Debug, Clone, PartialEq)]
pub struct MfccParams {
    /// Number of coefficients.
    pub n_mfcc: usize,
    /// Liftering coefficient (0 disables liftering).
    pub lifter: f64,
    /// Mel spectrogram parameters (used when the input is audio).
    pub mel: MelSpectrogramParams,
}

impl Default for MfccParams {
    fn default() -> Self {
        MfccParams {
            n_mfcc: 20,
            lifter: 0.0,
            mel: MelSpectrogramParams::default(),
        }
    }
}

/// Mel-frequency cepstral coefficients, shape `(n_mfcc, n_frames)` (`librosa.feature.mfcc`),
/// using an orthonormal DCT-II. A spectrogram input is taken as a log-power mel spectrogram.
pub fn mfcc<'a>(input: impl Into<Input<'a>>, sr: f64, p: &MfccParams) -> Result<Array2<f64>> {
    let s = match input.into() {
        Input::Audio(y) => {
            let m = melspectrogram(y, sr, &p.mel)?;
            power_to_db(m.view(), DbRef::Value(1.0), 1e-10, Some(80.0))?
        }
        Input::Spectrogram(s) => s.to_owned(),
    };
    if p.lifter < 0.0 {
        return param(format!(
            "MFCC lifter={} must be a non-negative number",
            p.lifter
        ));
    }
    let n = p.n_mfcc.min(s.nrows());
    let d = dct_ortho(n, s.nrows());
    let mut m = d.dot(&s);
    if p.lifter > 0.0 {
        for k in 0..n {
            let li = (PI * (k + 1) as f64 / p.lifter).sin();
            m.row_mut(k)
                .mapv_inplace(|v| v * (1.0 + (p.lifter / 2.0) * li));
        }
    }
    Ok(m)
}

/// Parameters for [`chroma_stft`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaStftParams {
    /// STFT parameters when the input is audio.
    pub stft: StftParams,
    /// Per-frame normalization of the chromagram (`None` disables it).
    pub norm: Option<Norm>,
    /// Tuning deviation; `None` estimates it from the spectrogram (librosa default).
    pub tuning: Option<f64>,
    /// Chroma filter bank parameters (`tuning` here is overridden by the field above).
    pub chroma: ChromaParams,
}

impl Default for ChromaStftParams {
    fn default() -> Self {
        ChromaStftParams {
            stft: StftParams::default(),
            norm: Some(Norm::Inf),
            tuning: None,
            chroma: ChromaParams::default(),
        }
    }
}

/// Chromagram from a power spectrogram, shape `(n_chroma, n_frames)` (`librosa.feature.chroma_stft`).
pub fn chroma_stft<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    p: &ChromaStftParams,
) -> Result<Array2<f64>> {
    let (s, n_fft) = get_spec(input.into(), &p.stft, 2.0)?;
    let tuning = match p.tuning {
        Some(t) => t,
        None => estimate_tuning_spec(
            s.view(),
            sr,
            n_fft,
            &TuningParams {
                bins_per_octave: p.chroma.n_chroma,
                ..Default::default()
            },
        ),
    };
    let fb = filters::chroma(
        sr,
        n_fft,
        &ChromaParams {
            tuning,
            ..p.chroma.clone()
        },
    )?;
    let raw = fb.dot(&s);
    util::normalize(raw.view(), p.norm, 0, None, None)
}

/// Parameters for [`chroma_cqt`] and [`tonnetz`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaCqtParams {
    /// Hop length.
    pub hop_length: usize,
    /// Lowest CQT frequency; `None` means C1.
    pub fmin: Option<f64>,
    /// Per-frame normalization (`None` disables it).
    pub norm: Option<Norm>,
    /// Values below this are set to zero before normalizing.
    pub threshold: f64,
    /// Tuning deviation; `None` estimates it from the signal.
    pub tuning: Option<f64>,
    /// Number of chroma bins.
    pub n_chroma: usize,
    /// Number of CQT octaves.
    pub n_octaves: usize,
    /// CQT bins per octave (must be a multiple of `n_chroma`).
    pub bins_per_octave: usize,
    /// Resampler for the CQT's multirate decomposition.
    pub res_type: ResType,
}

impl Default for ChromaCqtParams {
    fn default() -> Self {
        ChromaCqtParams {
            hop_length: 512,
            fmin: None,
            norm: Some(Norm::Inf),
            threshold: 0.0,
            tuning: None,
            n_chroma: 12,
            n_octaves: 7,
            bins_per_octave: 36,
            res_type: ResType::Sinc,
        }
    }
}

/// Constant-Q chromagram, shape `(n_chroma, n_frames)` (`librosa.feature.chroma_cqt`).
/// A spectrogram input is taken as a CQT magnitude (`C=`).
pub fn chroma_cqt<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    p: &ChromaCqtParams,
) -> Result<Array2<f64>> {
    if !p.bins_per_octave.is_multiple_of(p.n_chroma) {
        return param(format!(
            "bins_per_octave={} must be an integer multiple of n_chroma={}",
            p.bins_per_octave, p.n_chroma
        ));
    }
    let c = match input.into() {
        Input::Audio(y) => cqt(
            y,
            sr,
            &CqtParams {
                hop_length: p.hop_length,
                fmin: p.fmin,
                n_bins: p.n_octaves * p.bins_per_octave,
                bins_per_octave: p.bins_per_octave,
                tuning: p.tuning,
                res_type: p.res_type,
                ..Default::default()
            },
        )?
        .mapv(|z| z.norm()),
        Input::Spectrogram(c) => c.to_owned(),
    };
    let map = cq_to_chroma(c.nrows(), p.bins_per_octave, p.n_chroma, p.fmin, true)?;
    let mut chroma = map.dot(&c);
    chroma.mapv_inplace(|v| if v < p.threshold { 0.0 } else { v });
    util::normalize(chroma.view(), p.norm, 0, None, None)
}

/// Tonal centroid features, shape `(6, n_frames)` (`librosa.feature.tonnetz`). A spectrogram
/// input is taken as a chromagram; audio is converted with [`chroma_cqt`].
pub fn tonnetz<'a>(
    input: impl Into<Input<'a>>,
    sr: f64,
    p: &ChromaCqtParams,
) -> Result<Array2<f64>> {
    let chroma = match input.into() {
        Input::Audio(y) => chroma_cqt(y, sr, p)?,
        Input::Spectrogram(c) => c.to_owned(),
    };
    let n = chroma.nrows();
    let dim_map: Vec<f64> = (0..n).map(|k| 12.0 * k as f64 / n as f64).collect();
    let scale = [
        7.0 / 6.0,
        7.0 / 6.0,
        3.0 / 2.0,
        3.0 / 2.0,
        2.0 / 3.0,
        2.0 / 3.0,
    ];
    let r = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let phi = Array2::from_shape_fn((6, n), |(i, c)| {
        let mut v = scale[i] * dim_map[c];
        if i % 2 == 0 {
            v -= 0.5;
        }
        r[i] * (PI * v).cos()
    });
    let cn = util::normalize(chroma.view(), Some(Norm::P(1.0)), 0, None, None)?;
    Ok(phi.dot(&cn))
}
