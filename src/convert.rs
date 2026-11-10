//! Unit conversions (`librosa.core.convert`): frames, samples, time, Hz, mel, MIDI and notes.

use crate::error::{Result, param};
use ndarray::Array1;

/// Frame indices to sample indices: `frames * hop_length (+ n_fft / 2)`.
pub fn frames_to_samples(frames: &[i64], hop_length: usize, n_fft: Option<usize>) -> Vec<i64> {
    let offset = n_fft.map_or(0, |n| (n / 2) as i64);
    frames
        .iter()
        .map(|&f| f * hop_length as i64 + offset)
        .collect()
}

/// Sample indices to frame indices (floor division, as in librosa).
pub fn samples_to_frames(samples: &[i64], hop_length: usize, n_fft: Option<usize>) -> Vec<i64> {
    let offset = n_fft.map_or(0, |n| (n / 2) as i64);
    samples
        .iter()
        .map(|&s| (s - offset).div_euclid(hop_length as i64))
        .collect()
}

/// Frame indices to time in seconds.
pub fn frames_to_time(
    frames: &[i64],
    sr: f64,
    hop_length: usize,
    n_fft: Option<usize>,
) -> Vec<f64> {
    samples_to_time(&frames_to_samples(frames, hop_length, n_fft), sr)
}

/// Time in seconds to frame indices.
pub fn time_to_frames(times: &[f64], sr: f64, hop_length: usize, n_fft: Option<usize>) -> Vec<i64> {
    samples_to_frames(&time_to_samples(times, sr), hop_length, n_fft)
}

/// Time in seconds to sample indices (truncation toward zero, like `astype(int)`).
pub fn time_to_samples(times: &[f64], sr: f64) -> Vec<i64> {
    times.iter().map(|&t| (t * sr) as i64).collect()
}

/// Sample indices to time in seconds.
pub fn samples_to_time(samples: &[i64], sr: f64) -> Vec<f64> {
    samples.iter().map(|&s| s as f64 / sr).collect()
}

/// Hz to mel. Slaney's auditory-toolbox formula by default, HTK's with `htk = true`.
pub fn hz_to_mel(frequency: f64, htk: bool) -> f64 {
    if htk {
        return 2595.0 * (1.0 + frequency / 700.0).log10();
    }
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let mut mels = (frequency - f_min) / f_sp;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if frequency >= min_log_hz {
        mels = min_log_mel + (frequency / min_log_hz).ln() / logstep;
    }
    mels
}

/// Mel to Hz (inverse of [`hz_to_mel`]).
pub fn mel_to_hz(mel: f64, htk: bool) -> f64 {
    if htk {
        return 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    }
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let mut freq = f_min + f_sp * mel;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        freq = min_log_hz * (logstep * (mel - min_log_mel)).exp();
    }
    freq
}

/// Hz to (fractional) MIDI note number.
pub fn hz_to_midi(frequency: f64) -> f64 {
    12.0 * (frequency.log2() - 440f64.log2()) + 69.0
}

/// MIDI note number to Hz.
pub fn midi_to_hz(note: f64) -> f64 {
    440.0 * 2f64.powf((note - 69.0) / 12.0)
}

/// Hz to octave number (`A440 / 16` = octave 0), with optional tuning deviation.
pub fn hz_to_octs(frequency: f64, tuning: f64, bins_per_octave: usize) -> f64 {
    let a440 = 440.0 * 2f64.powf(tuning / bins_per_octave as f64);
    (frequency / (a440 / 16.0)).log2()
}

/// Octave number to Hz (inverse of [`hz_to_octs`]).
pub fn octs_to_hz(octs: f64, tuning: f64, bins_per_octave: usize) -> f64 {
    let a440 = 440.0 * 2f64.powf(tuning / bins_per_octave as f64);
    (a440 / 16.0) * 2f64.powf(octs)
}

/// Reference A4 frequency to tuning deviation (in fractions of a bin).
pub fn a4_to_tuning(a4: f64, bins_per_octave: usize) -> f64 {
    bins_per_octave as f64 * (a4.log2() - 440f64.log2())
}

/// Tuning deviation to reference A4 frequency.
pub fn tuning_to_a4(tuning: f64, bins_per_octave: usize) -> f64 {
    440.0 * 2f64.powf(tuning / bins_per_octave as f64)
}

/// Parse a note name such as `"C4"`, `"A#3"`, `"Eb-1"`, `"C♯4+25"` into a MIDI number.
///
/// With `round_midi = false`, cents are kept as a fractional part.
pub fn note_to_midi(note: &str, round_midi: bool) -> Result<f64> {
    let mut chars = note.chars().peekable();
    let pitch = match chars.next().map(|c| c.to_ascii_uppercase()) {
        Some('C') => 0,
        Some('D') => 2,
        Some('E') => 4,
        Some('F') => 5,
        Some('G') => 7,
        Some('A') => 9,
        Some('B') => 11,
        _ => return param(format!("Improper note format: {note}")),
    };
    let mut offset = 0i64;
    while let Some(&c) = chars.peek() {
        let d = match c {
            '#' | '♯' => 1,
            'b' | '!' | '♭' => -1,
            '𝄪' => 2,
            '𝄫' => -2,
            '♮' | 'n' => 0,
            _ => break,
        };
        offset += d;
        chars.next();
    }
    let rest: String = chars.collect();
    // octave: [+-]?\d+ ; cents: [+-]\d+
    let bytes = rest.as_bytes();
    let mut i = 0;
    let mut octave = 0i64;
    if i < bytes.len() {
        let start = i;
        if bytes[i] == b'+' || bytes[i] == b'-' {
            i += 1;
        }
        let ds = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > ds {
            octave = rest[start..i]
                .parse()
                .map_err(|_| crate::Error::Parameter(format!("Improper note format: {note}")))?;
        } else {
            i = start;
        }
    }
    let mut cents = 0.0;
    if i < bytes.len() {
        let c = &rest[i..];
        let valid = c.len() > 1
            && (c.starts_with('+') || c.starts_with('-'))
            && c[1..].bytes().all(|b| b.is_ascii_digit());
        if !valid {
            return param(format!("Improper note format: {note}"));
        }
        cents = c.parse::<i64>().unwrap_or(0) as f64 * 1e-2;
    }
    let value = (12 * (octave + 1) + pitch + offset) as f64 + cents;
    Ok(if round_midi {
        value.round_ties_even()
    } else {
        value
    })
}

/// Note name to Hz, e.g. `note_to_hz("A4") == 440.0`.
pub fn note_to_hz(note: &str) -> Result<f64> {
    Ok(midi_to_hz(note_to_midi(note, false)?))
}

const SHARP_NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];
const SHARP_ASCII: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// MIDI number to note name in C major spelling (`"A4"`, `"C♯5"`; ASCII `#` if `unicode = false`).
///
/// With `cents = true` the deviation is appended (`"A4+12"`). Non-finite input gives `""`.
pub fn midi_to_note(midi: f64, octave: bool, cents: bool, unicode: bool) -> Result<String> {
    if cents && !octave {
        return param("Cannot encode cents without octave information.");
    }
    if !midi.is_finite() {
        return Ok(String::new());
    }
    let note_num = midi.round_ties_even() as i64;
    let note_cents = (100.0 * ((midi - note_num as f64) * 100.0).round_ties_even() / 100.0) as i64;
    let names = if unicode { &SHARP_NAMES } else { &SHARP_ASCII };
    let mut note = names[note_num.rem_euclid(12) as usize].to_string();
    if octave {
        // Python's int() truncates toward zero
        note.push_str(&format!("{}", (note_num as f64 / 12.0).trunc() as i64 - 1));
    }
    if cents {
        note.push_str(&format!("{note_cents:+}"));
    }
    Ok(note)
}

/// Hz to note name (see [`midi_to_note`]).
pub fn hz_to_note(frequency: f64, octave: bool, cents: bool, unicode: bool) -> Result<String> {
    midi_to_note(hz_to_midi(frequency), octave, cents, unicode)
}

/// Center frequencies of the STFT bins: `k * sr / n_fft` for `k = 0..=n_fft/2`.
pub fn fft_frequencies(sr: f64, n_fft: usize) -> Array1<f64> {
    // numpy.fft.rfftfreq: arange(0, n//2 + 1) / (n * d)
    let val = 1.0 / (n_fft as f64 * (1.0 / sr));
    Array1::from_iter((0..=n_fft / 2).map(|k| k as f64 * val))
}

/// `n_mels` frequencies evenly spaced on the mel scale between `fmin` and `fmax`.
pub fn mel_frequencies(n_mels: usize, fmin: f64, fmax: f64, htk: bool) -> Array1<f64> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);
    linspace(min_mel, max_mel, n_mels).mapv(|m| mel_to_hz(m, htk))
}

/// Center frequencies of constant-Q bins.
pub fn cqt_frequencies(
    n_bins: usize,
    fmin: f64,
    bins_per_octave: usize,
    tuning: f64,
) -> Array1<f64> {
    let correction = 2f64.powf(tuning / bins_per_octave as f64);
    Array1::from_iter(
        (0..n_bins).map(|k| correction * fmin * 2f64.powf(k as f64 / bins_per_octave as f64)),
    )
}

/// Equal-tempered interval frequencies, computed the way `librosa.interval_frequencies`
/// does for `intervals="equal"` (one octave of ratios, tiled by powers of two).
pub fn interval_frequencies(n_bins: usize, fmin: f64, bins_per_octave: usize) -> Array1<f64> {
    let ratios: Vec<f64> = (0..bins_per_octave)
        .map(|k| 2f64.powf(k as f64 / bins_per_octave as f64))
        .collect();
    let n_oct = n_bins.div_ceil(bins_per_octave);
    let mut all: Vec<f64> = Vec::with_capacity(n_oct * bins_per_octave);
    for o in 0..n_oct {
        let p = 2f64.powi(o as i32);
        all.extend(ratios.iter().map(|r| p * r));
    }
    all.truncate(n_bins);
    all.sort_by(|a, b| a.total_cmp(b));
    Array1::from_iter(all.into_iter().map(|r| r * fmin))
}

/// BPM associated with each autocorrelation lag of a tempogram (lag 0 is `inf`).
pub fn tempo_frequencies(n_bins: usize, hop_length: usize, sr: f64) -> Array1<f64> {
    let mut out = Array1::zeros(n_bins);
    if n_bins > 0 {
        out[0] = f64::INFINITY;
    }
    for k in 1..n_bins {
        out[k] = 60.0 * sr / (hop_length as f64 * k as f64);
    }
    out
}

/// BPM of each bin of a Fourier tempogram.
pub fn fourier_tempo_frequencies(sr: f64, win_length: usize, hop_length: usize) -> Array1<f64> {
    fft_frequencies(sr * 60.0 / hop_length as f64, win_length)
}

/// `numpy.linspace(start, stop, num)` (endpoint included).
pub(crate) fn linspace(start: f64, stop: f64, num: usize) -> Array1<f64> {
    if num == 0 {
        return Array1::zeros(0);
    }
    if num == 1 {
        return Array1::from(vec![start]);
    }
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut out = Array1::from_iter((0..num).map(|k| start + k as f64 * step));
    out[num - 1] = stop;
    out
}
