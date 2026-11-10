//! Analyze an audio file: `cargo run --release --example analyze -- path/to/file.wav`
//!
//! Without an argument it synthesizes a short test signal.

use librosa::beat::{BeatTrackParams, beat_track};
use librosa::feature::{ChromaStftParams, MfccParams, chroma_stft, mfcc, spectral_centroid};
use librosa::onset::{OnsetDetectParams, onset_detect};
use librosa::{LoadOptions, PyinParams, StftParams, frames_to_time, load, pyin};
use ndarray::Axis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (y, sr) = match std::env::args().nth(1) {
        Some(path) => load(path, &LoadOptions::default())?,
        None => {
            let sr = 22050.0;
            let mut y = librosa::tone(220.0, sr, 3 * 22050);
            for k in 0..6 {
                let start = (0.25 + 0.5 * k as f64) * sr;
                for i in 0..400 {
                    y[start as usize + i] +=
                        (-(i as f64) / 60.0).exp() * if i % 2 == 0 { 1.0 } else { -1.0 };
                }
            }
            (y, sr)
        }
    };
    println!(
        "duration: {:.2} s at {sr} Hz",
        librosa::get_duration(y.len(), sr)
    );

    let (tempo, beats) = beat_track(y.view(), sr, &BeatTrackParams::default())?;
    let beat_frames: Vec<i64> = beats.iter().map(|&b| b as i64).collect();
    println!(
        "tempo: {tempo:.1} BPM, beats at {:.2?} s",
        frames_to_time(&beat_frames, sr, 512, None)
    );

    let onsets: Vec<i64> = onset_detect(y.view(), sr, &OnsetDetectParams::default())?
        .into_iter()
        .map(|o| o as i64)
        .collect();
    println!("onsets at {:.2?} s", frames_to_time(&onsets, sr, 512, None));

    let m = mfcc(&y, sr, &MfccParams::default())?;
    println!("MFCC means: {:.1}", m.mean_axis(Axis(1)).unwrap());

    let c = chroma_stft(&y, sr, &ChromaStftParams::default())?;
    println!("chroma means: {:.2}", c.mean_axis(Axis(1)).unwrap());

    let cent = spectral_centroid(&y, sr, &StftParams::default(), None)?;
    println!("mean spectral centroid: {:.0} Hz", cent.mean().unwrap());

    let p = pyin(y.view(), 65.0, 1000.0, sr, &PyinParams::default())?;
    let mut voiced: Vec<f64> = p.f0.iter().cloned().filter(|f| f.is_finite()).collect();
    voiced.sort_by(|a, b| a.total_cmp(b));
    if !voiced.is_empty() {
        println!("median f0 (pYIN): {:.1} Hz", voiced[voiced.len() / 2]);
    }
    Ok(())
}
