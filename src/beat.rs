//! Beat tracking (`librosa.beat`): Ellis' (2007) dynamic-programming beat tracker.

use crate::error::{Result, param};
use crate::feature::{TempoParams, tempo_env};
use crate::onset::{Aggregate, OnsetStrengthParams, onset_strength};
use crate::util::{self, TINY};
use ndarray::{Array1, ArrayView1};

/// Parameters for [`beat_track`] (librosa defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct BeatTrackParams {
    /// Hop length of the onset envelope.
    pub hop_length: usize,
    /// Initial tempo guess (BPM) for the tempo estimator.
    pub start_bpm: f64,
    /// How closely the tracker follows the tempo (larger = stricter).
    pub tightness: f64,
    /// Drop weak leading/trailing beats.
    pub trim: bool,
    /// Use this tempo instead of estimating one.
    pub bpm: Option<f64>,
}

impl Default for BeatTrackParams {
    fn default() -> Self {
        BeatTrackParams {
            hop_length: 512,
            start_bpm: 120.0,
            tightness: 100.0,
            trim: true,
            bpm: None,
        }
    }
}

/// Track beats in audio. Returns `(tempo_bpm, beat_frames)` (`librosa.beat.beat_track`).
///
/// The onset envelope uses median aggregation across mel bands, as librosa does.
/// Convert frames to seconds with [`frames_to_time`](crate::frames_to_time).
pub fn beat_track(y: ArrayView1<f64>, sr: f64, p: &BeatTrackParams) -> Result<(f64, Vec<usize>)> {
    let env = onset_strength(
        y,
        sr,
        &OnsetStrengthParams {
            hop_length: p.hop_length,
            aggregate: Aggregate::Median,
            ..Default::default()
        },
    )?;
    beat_track_env(env.view(), sr, p)
}

/// Track beats from a precomputed onset envelope.
pub fn beat_track_env(
    onset_envelope: ArrayView1<f64>,
    sr: f64,
    p: &BeatTrackParams,
) -> Result<(f64, Vec<usize>)> {
    if !onset_envelope.iter().any(|&v| v != 0.0) {
        return Ok((0.0, Vec::new()));
    }
    let bpm = match p.bpm {
        Some(b) => b,
        None => tempo_env(
            onset_envelope,
            sr,
            &TempoParams {
                hop_length: p.hop_length,
                start_bpm: p.start_bpm,
                ..Default::default()
            },
        )?[0],
    };
    if bpm <= 0.0 {
        return param(format!("bpm={bpm} must be strictly positive"));
    }
    if p.tightness <= 0.0 {
        return param("tightness must be strictly positive");
    }
    let frame_rate = sr / p.hop_length as f64;
    let fpb = (frame_rate * 60.0 / bpm).round_ties_even();
    let beats = beat_tracker(onset_envelope, fpb, p.tightness, p.trim);
    Ok((bpm, (0..beats.len()).filter(|&i| beats[i]).collect()))
}

fn beat_tracker(
    onset_envelope: ArrayView1<f64>,
    fpb: f64,
    tightness: f64,
    trim: bool,
) -> Vec<bool> {
    let n = onset_envelope.len();
    // normalize by standard deviation (ddof=1)
    let mean = onset_envelope.sum() / n as f64;
    let var = onset_envelope
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / (n as f64 - 1.0);
    let norm = var.sqrt() + TINY;
    let onsets: Vec<f64> = onset_envelope.iter().map(|v| v / norm).collect();

    // local score: same-mode convolution with a Gaussian window
    let fpb_i = fpb as i64;
    let window: Vec<f64> = (-fpb_i..=fpb_i)
        .map(|k| (-0.5 * (k as f64 * 32.0 / fpb).powi(2)).exp())
        .collect();
    let kk = window.len() as i64;
    let nn = n as i64;
    let mut localscore = vec![0.0; n];
    for (i, ls) in localscore.iter_mut().enumerate() {
        let i = i as i64;
        let lo = (i + kk / 2 - nn + 1).max(0);
        let hi = (i + kk / 2).min(kk);
        let mut acc = 0.0;
        for k in lo..hi {
            acc += window[k as usize] * onsets[(i + kk / 2 - k) as usize];
        }
        *ls = acc;
    }

    // dynamic programming
    let score_thresh = 0.01 * localscore.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut backlink = vec![-1i64; n];
    let mut cumscore = vec![0.0; n];
    cumscore[0] = localscore[0];
    let mut first_beat = true;
    let half = (fpb / 2.0).round_ties_even() as i64;
    let stop = (-2.0 * fpb - 1.0) as i64; // int(i - 2*fpb - 1) == i + int(-2*fpb - 1) for integer fpb
    let ln_fpb = fpb.ln();
    for i in 0..n {
        let ii = i as i64;
        let mut best_score = f64::NEG_INFINITY;
        let mut beat_location = -1i64;
        let mut loc = ii - half;
        while loc > ii + stop {
            if loc < 0 {
                break;
            }
            let score =
                cumscore[loc as usize] - tightness * (((ii - loc) as f64).ln() - ln_fpb).powi(2);
            if score > best_score {
                best_score = score;
                beat_location = loc;
            }
            loc -= 1;
        }
        cumscore[i] = if beat_location >= 0 {
            localscore[i] + best_score
        } else {
            localscore[i]
        };
        if first_beat && localscore[i] < score_thresh {
            backlink[i] = -1;
        } else {
            backlink[i] = beat_location;
            first_beat = false;
        }
    }

    // last beat: last local max of cumscore above half the median of local maxima
    let cs = Array1::from(cumscore.clone());
    let lmax = util::localmax(cs.view());
    let mut peaks: Vec<f64> = (0..n).filter(|&i| lmax[i]).map(|i| cumscore[i]).collect();
    let threshold = 0.5 * util::median(&mut peaks);
    let mut tail = n as i64 - 1;
    let mut k = n as i64 - 1;
    while k >= 0 {
        if lmax[k as usize] && cumscore[k as usize] >= threshold {
            tail = k;
            break;
        }
        k -= 1;
    }
    let mut beats = vec![false; n];
    let mut b = tail;
    while b >= 0 {
        beats[b as usize] = true;
        b = backlink[b as usize];
    }

    // trim weak leading/trailing beats
    let mut trimmed = beats.clone();
    let threshold = if trim {
        let w = [0.0, 0.5, 1.0, 0.5, 0.0]; // np.hanning(5)
        let sel: Vec<f64> = (0..n)
            .filter(|&i| beats[i])
            .map(|i| localscore[i])
            .collect();
        let full_len = sel.len() + w.len() - 1;
        let conv: Vec<f64> = (0..full_len)
            .map(|j| {
                let mut acc = 0.0;
                for (m, wm) in w.iter().enumerate() {
                    if j >= m && j - m < sel.len() {
                        acc += sel[j - m] * wm;
                    }
                }
                acc
            })
            .collect();
        let start = w.len() / 2;
        let end = (n + w.len() / 2).min(conv.len());
        let smooth: Vec<f64> = if start < end {
            conv[start..end].to_vec()
        } else {
            Vec::new()
        };
        let ms = smooth.iter().map(|v| v * v).sum::<f64>() / smooth.len().max(1) as f64;
        0.5 * ms.sqrt()
    } else {
        0.0
    };
    let mut i = 0;
    while i < n && localscore[i] <= threshold {
        trimmed[i] = false;
        i += 1;
    }
    let mut j = n as i64 - 1;
    while j >= 0 && localscore[j as usize] <= threshold {
        trimmed[j as usize] = false;
        j -= 1;
    }
    trimmed
}
