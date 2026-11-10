//! Audio loading and time-domain processing (`librosa.core.audio`).

use crate::error::{Error, Result, param};
use crate::fft::Rfft;
use crate::signal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::path::Path;

/// Resampling algorithm (`res_type` in librosa).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResType {
    /// High-quality zero-phase band-limited sinc interpolation (Kaiser window, 64 zero
    /// crossings). This is the default and plays the role of librosa's default `"soxr_hq"`:
    /// it agrees with soxr to about -80 dB, not bit-for-bit.
    #[default]
    Sinc,
    /// `rubato`'s asynchronous sinc resampler (requires the `rubato` feature). Fast and
    /// streaming-friendly, but its startup-delay trimming is integer-only, so the output can
    /// be offset by a fraction of a sample relative to librosa.
    #[cfg(feature = "rubato")]
    Rubato,
    /// Fourier-domain resampling, identical to librosa's `res_type="fft"` (`scipy.signal.resample`).
    Fft,
    /// Polyphase FIR resampling, identical to librosa's `res_type="polyphase"`
    /// (`scipy.signal.resample_poly`). Requires integer sample rates.
    Polyphase,
}

/// Change the sample rate of a signal (`librosa.resample`).
///
/// With `fix = true` the output has exactly `ceil(len * target_sr / orig_sr)` samples.
/// With `scale = true` the output is divided by `sqrt(target_sr / orig_sr)` (energy-preserving).
pub fn resample(
    y: ArrayView1<f64>,
    orig_sr: f64,
    target_sr: f64,
    res_type: ResType,
    fix: bool,
    scale: bool,
) -> Result<Array1<f64>> {
    crate::spectrum::valid_audio(y)?;
    if orig_sr <= 0.0 || target_sr <= 0.0 {
        return param("sample rates must be positive");
    }
    if orig_sr == target_sr {
        return Ok(y.to_owned());
    }
    let ratio = target_sr / orig_sr;
    let n_samples = (y.len() as f64 * ratio).ceil() as usize;
    let x = y.to_vec();
    let mut out = match res_type {
        ResType::Fft => signal::resample_fft(&x, n_samples),
        ResType::Polyphase => {
            if orig_sr.fract() != 0.0 || target_sr.fract() != 0.0 {
                return param(
                    "polyphase resampling is only supported for integer-valued sampling rates.",
                );
            }
            let (o, t) = (orig_sr as usize, target_sr as usize);
            let g = gcd(o, t);
            signal::resample_poly(&x, t / g, o / g)
        }
        ResType::Sinc => resample_sinc(&x, orig_sr, target_sr),
        #[cfg(feature = "rubato")]
        ResType::Rubato => resample_rubato(&x, ratio)?,
    };
    if fix {
        out.resize(n_samples, 0.0);
    }
    let mut out = Array1::from(out);
    if scale {
        let s = ratio.sqrt();
        out.mapv_inplace(|v| v / s);
    }
    Ok(out)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn resample_sinc(x: &[f64], orig_sr: f64, target_sr: f64) -> Vec<f64> {
    let ratio = target_sr / orig_sr;
    let phases = if orig_sr.fract() == 0.0 && target_sr.fract() == 0.0 {
        let (o, t) = (orig_sr as usize, target_sr as usize);
        let g = gcd(o, t);
        Some((t / g, o / g))
    } else {
        None
    };
    signal::resample_bandlimited(x, ratio, phases)
}

#[cfg(feature = "rubato")]
fn resample_rubato(x: &[f64], ratio: f64) -> Result<Vec<f64>> {
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{
        Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction, calculate_cutoff,
    };
    let err = |e: &dyn std::fmt::Display| Error::Resample(e.to_string());
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let sinc_len = 256;
    let window = WindowFunction::BlackmanHarris2;
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff: Some(calculate_cutoff(sinc_len, window)),
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window,
    };
    let mut rs = Async::<f64>::new_sinc(ratio, 1.0, &params, 1024, 1, FixedAsync::Input)
        .map_err(|e| err(&e))?;
    let data = vec![x.to_vec()];
    let input = SequentialSliceOfVecs::new(&data, 1, x.len()).map_err(|e| err(&e))?;
    let out_len = rs.process_all_needed_output_len(x.len());
    let mut out_data = vec![vec![0.0f64; out_len]; 1];
    let mut output =
        SequentialSliceOfVecs::new_mut(&mut out_data, 1, out_len).map_err(|e| err(&e))?;
    let (_, n_out) = rs
        .process_all_into_buffer(&input, &mut output, x.len(), None)
        .map_err(|e| err(&e))?;
    let mut v = out_data.swap_remove(0);
    v.truncate(n_out);
    Ok(v)
}

/// Options for [`load`]. Defaults follow `librosa.load`: resample to 22050 Hz, mono.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    /// Target sample rate; `None` keeps the file's native rate.
    pub sr: Option<f64>,
    /// Mix down to mono by averaging channels.
    pub mono: bool,
    /// Start reading after this many seconds.
    pub offset: f64,
    /// Only read up to this many seconds.
    pub duration: Option<f64>,
    /// Resampling algorithm.
    pub res_type: ResType,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            sr: Some(22050.0),
            mono: true,
            offset: 0.0,
            duration: None,
            res_type: ResType::Sinc,
        }
    }
}

/// Decoded audio: samples of shape `(channels, frames)` and the native sample rate.
fn decode_file(path: &Path) -> Result<(Array2<f64>, f64)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymErr;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| Error::Decode(e.to_string()))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| Error::Decode("no audio track".into()))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|c| c.audio())
        .ok_or_else(|| Error::Decode("track has no audio codec parameters".into()))?
        .clone();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&codec_params, &AudioDecoderOptions::default())
        .map_err(|e| Error::Decode(e.to_string()))?;
    let mut interleaved: Vec<f64> = Vec::new();
    let mut channels = codec_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(0);
    let mut sr = codec_params.sample_rate.map(f64::from).unwrap_or(0.0);
    let mut buf: Vec<f64> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymErr::ResetRequired) => break,
            Err(SymErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::Decode(e.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio) => {
                channels = audio.spec().channels().count();
                sr = f64::from(audio.spec().rate());
                buf.resize(audio.samples_interleaved(), 0.0);
                audio.copy_to_slice_interleaved(&mut buf);
                interleaved.extend_from_slice(&buf);
            }
            Err(SymErr::DecodeError(_)) => continue,
            Err(e) => return Err(Error::Decode(e.to_string())),
        }
    }
    if channels == 0 || sr <= 0.0 {
        return Err(Error::Decode(
            "could not determine channels / sample rate".into(),
        ));
    }
    let frames = interleaved.len() / channels;
    let arr = Array2::from_shape_vec((frames, channels), interleaved)
        .map_err(|e| Error::Decode(e.to_string()))?
        .reversed_axes()
        .to_owned();
    Ok((arr, sr))
}

/// Load an audio file (WAV, FLAC, MP3, OGG/Vorbis) as floating point samples in `[-1, 1]`.
///
/// Returns the samples with shape `(channels, frames)` and the sample rate. With
/// `opts.mono` the result has a single row. This mirrors `librosa.load(path, sr=..., mono=...)`.
pub fn load_multichannel(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<(Array2<f64>, f64)> {
    let (mut y, sr_native) = decode_file(path.as_ref())?;
    let n = y.ncols();
    let start = ((opts.offset * sr_native) as usize).min(n);
    let end = match opts.duration {
        Some(d) => (start + (d * sr_native) as usize).min(n),
        None => n,
    };
    y = y.slice(ndarray::s![.., start..end]).to_owned();
    if opts.mono && y.nrows() > 1 {
        let m = to_mono(y.view());
        y = m.insert_axis(Axis(0));
    }
    let mut sr = sr_native;
    if let Some(target) = opts.sr
        && target != sr_native
    {
        let rows: Result<Vec<Array1<f64>>> = y
            .axis_iter(Axis(0))
            .map(|row| resample(row, sr_native, target, opts.res_type, true, false))
            .collect();
        let rows = rows?;
        let len = rows.first().map_or(0, |r| r.len());
        let mut out = Array2::zeros((rows.len(), len));
        for (i, r) in rows.iter().enumerate() {
            out.row_mut(i).assign(r);
        }
        y = out;
        sr = target;
    }
    Ok((y, sr))
}

/// Load an audio file as a mono signal (`librosa.load(path)`): returns `(y, sr)`.
///
/// ```no_run
/// let (y, sr) = librosa::load("song.mp3", &librosa::LoadOptions::default()).unwrap();
/// ```
pub fn load(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<(Array1<f64>, f64)> {
    let opts = LoadOptions {
        mono: true,
        ..opts.clone()
    };
    let (y, sr) = load_multichannel(path, &opts)?;
    Ok((y.row(0).to_owned(), sr))
}

/// Native sample rate of an audio file.
pub fn get_samplerate(path: impl AsRef<Path>) -> Result<f64> {
    Ok(decode_file(path.as_ref())?.1)
}

/// Duration (seconds) of an audio file.
pub fn get_duration_path(path: impl AsRef<Path>) -> Result<f64> {
    let (y, sr) = decode_file(path.as_ref())?;
    Ok(y.ncols() as f64 / sr)
}

/// Duration (seconds) of a signal with `n_samples` samples at rate `sr`.
pub fn get_duration(n_samples: usize, sr: f64) -> f64 {
    n_samples as f64 / sr
}

/// Duration (seconds) of the signal behind a spectrogram with `n_frames` columns.
pub fn get_duration_frames(
    n_frames: usize,
    sr: f64,
    n_fft: usize,
    hop_length: usize,
    center: bool,
) -> f64 {
    let mut n = n_fft + hop_length * n_frames.saturating_sub(1);
    if center {
        n = n.saturating_sub(2 * (n_fft / 2));
    }
    n as f64 / sr
}

/// Average the channels of a `(channels, frames)` array (`librosa.to_mono`).
pub fn to_mono(y: ArrayView2<f64>) -> Array1<f64> {
    y.mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(y.ncols()))
}

/// Autocorrelation of a signal up to lag `max_size - 1` (`librosa.autocorrelate`).
pub fn autocorrelate(y: ArrayView1<f64>, max_size: Option<usize>) -> Array1<f64> {
    let n = y.len();
    let max_size = max_size.unwrap_or(n).min(n);
    if n == 0 {
        return Array1::zeros(0);
    }
    let n_pad = (2 * n - 1).next_power_of_two();
    let mut fft = Rfft::new(n_pad);
    let mut spec = vec![Complex64::new(0.0, 0.0); n_pad / 2 + 1];
    let x = y.to_vec();
    fft.forward(&x, &mut spec);
    let pow: Vec<Complex64> = spec
        .iter()
        .map(|z| Complex64::new(z.norm_sqr(), 0.0))
        .collect();
    let mut out = vec![0.0; n_pad];
    fft.inverse(&pow, &mut out);
    Array1::from(out[..max_size].to_vec())
}

/// Zero-crossing indicators (`librosa.zero_crossings`). Values with `|y| <= threshold`
/// count as zero; `zero_pos` treats zero as positive. `pad` sets the first element.
pub fn zero_crossings(y: ArrayView1<f64>, threshold: f64, pad: bool, zero_pos: bool) -> Vec<bool> {
    let n = y.len();
    let clip = |v: f64| if v.abs() <= threshold { 0.0 } else { v };
    let mut out = vec![false; n];
    for i in 1..n {
        let (x0, x1) = (clip(y[i]), clip(y[i - 1]));
        out[i] = if zero_pos {
            x0.is_sign_negative() != x1.is_sign_negative()
        } else {
            sign(x0) != sign(x1)
        };
    }
    if n > 0 {
        out[0] = pad;
    }
    out
}

fn sign(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// A pure tone `cos(2π f n / sr - π/2)` (`librosa.tone` with the default phase).
pub fn tone(frequency: f64, sr: f64, length: usize) -> Array1<f64> {
    Array1::from_iter((0..length).map(|n| (2.0 * PI * frequency * n as f64 / sr - 0.5 * PI).cos()))
}

/// A chirp sweeping `fmin` to `fmax` over `length` samples (`librosa.chirp`),
/// exponential by default or linear with `linear = true`.
pub fn chirp(fmin: f64, fmax: f64, sr: f64, length: usize, linear: bool) -> Array1<f64> {
    let duration = length as f64 / sr;
    let n = (duration * sr) as usize;
    let phi = -0.5 * PI;
    Array1::from_iter((0..n).map(|k| {
        let t = k as f64 / sr;
        let phase = if linear {
            let beta = (fmax - fmin) / duration;
            2.0 * PI * (fmin * t + 0.5 * beta * t * t)
        } else if fmin == fmax {
            2.0 * PI * fmin * t
        } else {
            let beta = duration / (fmax / fmin).ln();
            2.0 * PI * beta * fmin * ((fmax / fmin).powf(t / duration) - 1.0)
        };
        (phase + phi).cos()
    }))
}

/// Pre-emphasis filter `y[n] - coef * y[n-1]` (`librosa.effects.preemphasis`), with the
/// initial state chosen for linear extrapolation as in librosa.
pub fn preemphasis(y: ArrayView1<f64>, coef: f64) -> Array1<f64> {
    let n = y.len();
    let mut out = Array1::zeros(n);
    if n == 0 {
        return out;
    }
    // librosa initializes the lfilter state as zi = 2*y[0] - y[1], so out[0] = y[0] + zi.
    let zi = if n > 1 { 2.0 * y[0] - y[1] } else { y[0] };
    out[0] = y[0] + zi;
    for i in 1..n {
        out[i] = y[i] - coef * y[i - 1];
    }
    out
}
