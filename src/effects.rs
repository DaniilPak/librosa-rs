//! Time-domain effects (`librosa.effects`): HPSS, time-stretch, pitch-shift, trimming and splitting.

use crate::audio::{ResType, resample};
use crate::decompose::{self, HpssParams};
use crate::error::{Result, param};
use crate::feature::{FrameParams, rms};
use crate::spectrum::{
    DbRef, IstftParams, StftParams, amplitude_to_db, istft, phase_vocoder, stft,
};
use crate::util::fix_length;
use ndarray::{Array1, ArrayView1};

pub use crate::audio::preemphasis;

fn istft_like(p: &StftParams, length: usize) -> IstftParams {
    IstftParams {
        length: Some(length),
        ..IstftParams::from(p)
    }
}

/// Separate audio into harmonic and percussive components (`librosa.effects.hpss`).
pub fn hpss(
    y: ArrayView1<f64>,
    hp: &HpssParams,
    sp: &StftParams,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let d = stft(y, sp)?;
    let (h, p) = decompose::hpss(
        d.view(),
        &HpssParams {
            mask: false,
            ..hp.clone()
        },
    )?;
    let ip = istft_like(sp, y.len());
    Ok((istft(h.view(), &ip)?, istft(p.view(), &ip)?))
}

/// Harmonic component only (`librosa.effects.harmonic`).
pub fn harmonic(y: ArrayView1<f64>, hp: &HpssParams, sp: &StftParams) -> Result<Array1<f64>> {
    Ok(hpss(y, hp, sp)?.0)
}

/// Percussive component only (`librosa.effects.percussive`).
pub fn percussive(y: ArrayView1<f64>, hp: &HpssParams, sp: &StftParams) -> Result<Array1<f64>> {
    Ok(hpss(y, hp, sp)?.1)
}

/// Change speed without changing pitch, via the phase vocoder (`librosa.effects.time_stretch`).
/// `rate > 1` speeds up. The output has `round(len / rate)` samples.
pub fn time_stretch(y: ArrayView1<f64>, rate: f64, sp: &StftParams) -> Result<Array1<f64>> {
    if rate <= 0.0 {
        return param("rate must be a positive number");
    }
    let d = stft(y, sp)?;
    let ds = phase_vocoder(d.view(), rate)?;
    let len = (y.len() as f64 / rate).round_ties_even() as usize;
    istft(ds.view(), &istft_like(sp, len))
}

/// Parameters for [`pitch_shift`].
#[derive(Debug, Clone, PartialEq)]
pub struct PitchShiftParams {
    /// Steps per octave (12 = semitones).
    pub bins_per_octave: usize,
    /// Resampler (librosa default `soxr_hq`; here [`ResType::Sinc`]).
    pub res_type: ResType,
    /// Energy-preserving resampling.
    pub scale: bool,
    /// STFT parameters for the time stretch.
    pub stft: StftParams,
}

impl Default for PitchShiftParams {
    fn default() -> Self {
        PitchShiftParams {
            bins_per_octave: 12,
            res_type: ResType::Sinc,
            scale: false,
            stft: StftParams::default(),
        }
    }
}

/// Shift pitch by `n_steps` without changing duration (`librosa.effects.pitch_shift`):
/// time-stretch by `2^(-n_steps/bins_per_octave)`, then resample back.
pub fn pitch_shift(
    y: ArrayView1<f64>,
    sr: f64,
    n_steps: f64,
    p: &PitchShiftParams,
) -> Result<Array1<f64>> {
    if p.bins_per_octave == 0 {
        return param("bins_per_octave must be a positive integer.");
    }
    let rate = 2f64.powf(-n_steps / p.bins_per_octave as f64);
    let stretched = time_stretch(y, rate, &p.stft)?;
    let shifted = resample(stretched.view(), sr / rate, sr, p.res_type, true, p.scale)?;
    Ok(fix_length(shifted.view(), y.len()))
}

/// Parameters for [`trim`] and [`split`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrimParams {
    /// Threshold (dB below the reference) under which frames count as silent.
    pub top_db: f64,
    /// Reference power (default: the maximum RMS).
    pub reference: DbRef,
    /// Frame length for the RMS analysis.
    pub frame_length: usize,
    /// Hop length for the RMS analysis.
    pub hop_length: usize,
}

impl Default for TrimParams {
    fn default() -> Self {
        TrimParams {
            top_db: 60.0,
            reference: DbRef::Max,
            frame_length: 2048,
            hop_length: 512,
        }
    }
}

fn nonsilent(y: ArrayView1<f64>, p: &TrimParams) -> Result<Vec<bool>> {
    let mse = rms(
        y,
        &FrameParams {
            frame_length: p.frame_length,
            hop_length: p.hop_length,
            ..Default::default()
        },
    )?;
    let db = amplitude_to_db(mse.row(0), p.reference, 1e-5, None)?;
    Ok(db.iter().map(|&v| v > -p.top_db).collect())
}

/// Trim leading and trailing silence (`librosa.effects.trim`). Returns the trimmed signal
/// and the `(start, end)` sample interval it came from.
pub fn trim(y: ArrayView1<f64>, p: &TrimParams) -> Result<(Array1<f64>, (usize, usize))> {
    let ns = nonsilent(y, p)?;
    let idx: Vec<usize> = (0..ns.len()).filter(|&i| ns[i]).collect();
    let (start, end) = match (idx.first(), idx.last()) {
        (Some(&a), Some(&b)) => (a * p.hop_length, y.len().min((b + 1) * p.hop_length)),
        _ => (0, 0),
    };
    Ok((y.slice(ndarray::s![start..end]).to_owned(), (start, end)))
}

/// Split a signal into non-silent intervals (`librosa.effects.split`), as `(start, end)`
/// sample pairs.
pub fn split(y: ArrayView1<f64>, p: &TrimParams) -> Result<Vec<(usize, usize)>> {
    let ns = nonsilent(y, p)?;
    let mut edges: Vec<usize> = Vec::new();
    if ns.first() == Some(&true) {
        edges.push(0);
    }
    for i in 1..ns.len() {
        if ns[i] != ns[i - 1] {
            edges.push(i);
        }
    }
    if ns.last() == Some(&true) {
        edges.push(ns.len());
    }
    let edges: Vec<usize> = edges
        .iter()
        .map(|&f| (f * p.hop_length).min(y.len()))
        .collect();
    Ok(edges
        .chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| (c[0], c[1]))
        .collect())
}
