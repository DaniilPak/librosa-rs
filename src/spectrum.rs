//! Spectral representations (`librosa.core.spectrum`): STFT, inverse STFT, dB scaling,
//! magnitude/phase separation and the phase vocoder.

use crate::error::{Result, param};
use crate::fft::Rfft;
use crate::filters::window_sumsquare;
use crate::util::{self, PadMode, TINY};
use crate::window::{Window, get_window};
use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Dimension};
use num_complex::Complex64;

/// STFT parameters, with librosa's defaults (`n_fft=2048`, `hop_length=win_length/4`,
/// `win_length=n_fft`, `window="hann"`, `center=True`, `pad_mode="constant"`).
#[derive(Debug, Clone, PartialEq)]
pub struct StftParams {
    /// FFT size.
    pub n_fft: usize,
    /// Hop between frames; `None` means `win_length / 4`.
    pub hop_length: Option<usize>,
    /// Window length; `None` means `n_fft`. Shorter windows are zero-padded to `n_fft`.
    pub win_length: Option<usize>,
    /// Window function.
    pub window: Window,
    /// Pad the signal so that frame `t` is centered at sample `t * hop_length`.
    pub center: bool,
    /// How to pad when `center` is true.
    pub pad_mode: PadMode,
}

impl Default for StftParams {
    fn default() -> Self {
        StftParams {
            n_fft: 2048,
            hop_length: None,
            win_length: None,
            window: Window::Hann,
            center: true,
            pad_mode: PadMode::Constant,
        }
    }
}

impl StftParams {
    /// Parameters with the given FFT size and hop length, other fields default.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        StftParams {
            n_fft,
            hop_length: Some(hop_length),
            ..Default::default()
        }
    }

    pub(crate) fn win_length(&self) -> usize {
        self.win_length.unwrap_or(self.n_fft)
    }

    pub(crate) fn hop(&self) -> usize {
        self.hop_length.unwrap_or(self.win_length() / 4)
    }
}

pub(crate) fn valid_audio(y: ArrayView1<f64>) -> Result<()> {
    if !y.iter().all(|v| v.is_finite()) {
        return param("Audio buffer is not finite everywhere");
    }
    Ok(())
}

/// Short-time Fourier transform. Returns a complex matrix of shape `(1 + n_fft/2, n_frames)`.
///
/// ```
/// use librosa::{stft, StftParams};
/// let y = librosa::tone(440.0, 22050.0, 22050);
/// let d = stft(y.view(), &StftParams::default()).unwrap();
/// assert_eq!(d.dim(), (1025, 44));
/// ```
pub fn stft(y: ArrayView1<f64>, p: &StftParams) -> Result<Array2<Complex64>> {
    let n_fft = p.n_fft;
    let win_length = p.win_length();
    let hop = p.hop();
    if hop == 0 {
        return param(format!("hop_length={hop} must be a positive integer"));
    }
    if win_length > n_fft {
        return param(format!("win_length={win_length} must be <= n_fft={n_fft}"));
    }
    valid_audio(y)?;
    let window = util::pad_center(get_window(&p.window, win_length, true)?.view(), n_fft)?;
    let padded;
    let x = if p.center {
        if matches!(p.pad_mode, PadMode::LinearRamp) {
            return param("pad_mode='linear_ramp' is not supported by stft");
        }
        padded = util::pad(y, n_fft / 2, n_fft / 2, p.pad_mode);
        padded.view()
    } else {
        if n_fft > y.len() {
            return param(format!(
                "n_fft={n_fft} is too large for uncentered analysis of input signal of length={}",
                y.len()
            ));
        }
        y
    };
    if x.len() < n_fft {
        return param(format!(
            "Input is too short (n={}) for frame_length={n_fft}",
            x.len()
        ));
    }
    let n_frames = 1 + (x.len() - n_fft) / hop;
    let n_bins = 1 + n_fft / 2;
    let mut out = Array2::<Complex64>::zeros((n_bins, n_frames));
    let mut fft = Rfft::new(n_fft);
    let mut buf = vec![0.0; n_fft];
    let mut spec = vec![Complex64::new(0.0, 0.0); n_bins];
    let xs = x
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| x.to_vec());
    for t in 0..n_frames {
        let start = t * hop;
        for (k, b) in buf.iter_mut().enumerate() {
            *b = window[k] * xs[start + k];
        }
        fft.forward(&buf, &mut spec);
        out.column_mut(t).assign(&ArrayView1::from(&spec[..]));
    }
    Ok(out)
}

/// Parameters for [`istft`]. `n_fft` defaults to `2 * (n_bins - 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct IstftParams {
    /// Hop length; `None` means `win_length / 4`.
    pub hop_length: Option<usize>,
    /// Window length; `None` means `n_fft`.
    pub win_length: Option<usize>,
    /// FFT size; `None` infers it from the number of rows.
    pub n_fft: Option<usize>,
    /// Window function (must match the analysis window).
    pub window: Window,
    /// Whether the STFT was computed with `center=True`.
    pub center: bool,
    /// Exact output length; `None` returns everything the frames cover.
    pub length: Option<usize>,
}

impl Default for IstftParams {
    fn default() -> Self {
        IstftParams {
            hop_length: None,
            win_length: None,
            n_fft: None,
            window: Window::Hann,
            center: true,
            length: None,
        }
    }
}

impl From<&StftParams> for IstftParams {
    fn from(p: &StftParams) -> Self {
        IstftParams {
            hop_length: p.hop_length,
            win_length: p.win_length,
            n_fft: Some(p.n_fft),
            window: p.window.clone(),
            center: p.center,
            length: None,
        }
    }
}

/// Inverse STFT by weighted overlap-add (`librosa.istft`).
pub fn istft(d: ArrayView2<Complex64>, p: &IstftParams) -> Result<Array1<f64>> {
    let n_fft = p.n_fft.unwrap_or(2 * (d.nrows().saturating_sub(1)));
    let win_length = p.win_length.unwrap_or(n_fft);
    let hop = p.hop_length.unwrap_or(win_length / 4);
    if hop == 0 || n_fft == 0 {
        return param("hop_length and n_fft must be positive");
    }
    let window = util::pad_center(get_window(&p.window, win_length, true)?.view(), n_fft)?;
    let mut n_frames = d.ncols();
    if let Some(len) = p.length
        && len > 0
    {
        let padded = if p.center { len + 2 * (n_fft / 2) } else { len };
        n_frames = n_frames.min(padded.div_ceil(hop));
    }
    let full_len = n_fft + hop * n_frames.saturating_sub(1);
    let mut y = vec![0.0; if n_frames == 0 { 0 } else { full_len }];
    let mut fft = Rfft::new(n_fft);
    let mut frame = vec![0.0; n_fft];
    let mut col = vec![Complex64::new(0.0, 0.0); d.nrows()];
    for t in 0..n_frames {
        for (c, v) in col.iter_mut().zip(d.column(t).iter()) {
            *c = *v;
        }
        fft.inverse(&col, &mut frame);
        let start = t * hop;
        for k in 0..n_fft {
            y[start + k] += window[k] * frame[k];
        }
    }
    let wss = window_sumsquare(&p.window, n_frames, hop, Some(win_length), n_fft, None)?;
    for (v, w) in y.iter_mut().zip(wss.iter()) {
        if *w > TINY {
            *v /= *w;
        }
    }
    let start = if p.center { n_fft / 2 } else { 0 };
    let expected = match p.length {
        Some(l) if l > 0 => l,
        _ => {
            if p.center {
                full_len.saturating_sub(2 * (n_fft / 2))
            } else {
                full_len
            }
        }
    };
    let mut out = Array1::zeros(expected);
    for i in 0..expected {
        if let Some(v) = y.get(start + i) {
            out[i] = *v;
        }
    }
    Ok(out)
}

/// Separate a complex spectrogram into magnitude (raised to `power`) and unit phase.
///
/// Zero entries get magnitude 0 and phase `1 + 0j`.
pub fn magphase(d: ArrayView2<Complex64>, power: f64) -> (Array2<f64>, Array2<Complex64>) {
    let mag = d.mapv(|z| z.norm());
    let phase = ndarray::Zip::from(&d).and(&mag).map_collect(|z, &m| {
        let zero = if m == 0.0 { 1.0 } else { 0.0 };
        let mn = m + zero;
        Complex64::new(z.re / mn + zero, z.im / mn)
    });
    (mag.mapv(|m| m.powf(power)), phase)
}

/// Reference value for dB conversion: a fixed value, or the maximum of the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DbRef {
    /// Use a fixed reference (librosa's `ref=1.0` default).
    Value(f64),
    /// Use the maximum of the input (`ref=np.max`).
    Max,
}

impl Default for DbRef {
    fn default() -> Self {
        DbRef::Value(1.0)
    }
}

/// Power spectrogram to decibels: `10 * log10(max(amin, S) / ref)`, floored at
/// `max - top_db` when `top_db` is set. librosa defaults: `ref=1.0, amin=1e-10, top_db=80`.
pub fn power_to_db<D: Dimension>(
    s: ArrayView<f64, D>,
    reference: DbRef,
    amin: f64,
    top_db: Option<f64>,
) -> Result<Array<f64, D>> {
    if amin <= 0.0 {
        return param("amin must be strictly positive");
    }
    let ref_value = match reference {
        DbRef::Value(v) => v.abs(),
        DbRef::Max => s.fold(f64::NEG_INFINITY, |m, &v| m.max(v)),
    };
    let r = 10.0 * amin.max(ref_value).log10();
    let mut log_spec = s.mapv(|v| 10.0 * amin.max(v).log10() - r);
    if let Some(top) = top_db {
        if top < 0.0 {
            return param("top_db must be non-negative");
        }
        let mx = log_spec.fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        log_spec.mapv_inplace(|v| v.max(mx - top));
    }
    Ok(log_spec)
}

/// Amplitude spectrogram to decibels (`librosa.amplitude_to_db`; default `amin=1e-5`).
pub fn amplitude_to_db<D: Dimension>(
    s: ArrayView<f64, D>,
    reference: DbRef,
    amin: f64,
    top_db: Option<f64>,
) -> Result<Array<f64, D>> {
    let mag = s.mapv(f64::abs);
    let ref_value = match reference {
        DbRef::Value(v) => v.abs(),
        DbRef::Max => mag.fold(f64::NEG_INFINITY, |m, &v| m.max(v)),
    };
    let power = mag.mapv(|v| v * v);
    power_to_db(
        power.view(),
        DbRef::Value(ref_value * ref_value),
        amin * amin,
        top_db,
    )
}

/// Decibels to power: `ref * 10^(dB / 10)`.
pub fn db_to_power<D: Dimension>(s_db: ArrayView<f64, D>, reference: f64) -> Array<f64, D> {
    s_db.mapv(|v| reference * 10f64.powf(v * 0.1))
}

/// Decibels to amplitude: `sqrt(db_to_power(S_db, ref^2))`.
pub fn db_to_amplitude<D: Dimension>(s_db: ArrayView<f64, D>, reference: f64) -> Array<f64, D> {
    db_to_power(s_db, reference * reference).mapv(f64::sqrt)
}

/// Phase vocoder: time-stretch an STFT matrix by `rate` (> 1 speeds up), as implemented
/// in librosa 1.0 (`librosa.phase_vocoder(D, rate=rate)`, linear magnitude interpolation).
pub fn phase_vocoder(d: ArrayView2<Complex64>, rate: f64) -> Result<Array2<Complex64>> {
    if rate <= 0.0 {
        return param(format!("rate={rate} must be a positive number"));
    }
    let (n_bins, n_frames) = d.dim();
    if n_frames < 2 {
        return param("phase_vocoder needs at least two frames");
    }
    let n_out = ((n_frames as f64) / rate).ceil() as usize;
    let t_out: Vec<f64> = (0..n_out).map(|k| k as f64 * rate).collect();
    let ph = d.mapv(|z| z.arg());
    let mag = d.mapv(|z| z.norm());
    let mut out = Array2::<Complex64>::zeros((n_bins, n_out));
    for f in 0..n_bins {
        let mut acc = 0.0;
        for (k, &t) in t_out.iter().enumerate() {
            if k == 0 {
                acc = ph[[f, t.floor() as usize]];
            } else {
                let tp = t_out[k - 1];
                let i0 = tp.floor() as usize;
                let i1 = (i0 + 1).min(n_frames - 1);
                acc += ph[[f, i1]] - ph[[f, i0]];
            }
            // scipy interp1d(kind="linear", fill_value="extrapolate")
            let idx = (t.ceil() as usize).clamp(1, n_frames - 1);
            let lo = idx - 1;
            let (ylo, yhi) = (mag[[f, lo]], mag[[f, idx]]);
            let m = (yhi - ylo) / 1.0 * (t - lo as f64) + ylo;
            out[[f, k]] = Complex64::new(acc.cos() * m, acc.sin() * m);
        }
    }
    Ok(out)
}

/// Magnitude spectrogram `|STFT(y)|^power` (librosa's internal `_spectrogram`).
pub(crate) fn spectrogram(y: ArrayView1<f64>, p: &StftParams, power: f64) -> Result<Array2<f64>> {
    let d = stft(y, p)?;
    Ok(if power == 1.0 {
        d.mapv(|z| z.norm())
    } else if power == 2.0 {
        d.mapv(|z| z.norm_sqr())
    } else {
        d.mapv(|z| z.norm().powf(power))
    })
}
