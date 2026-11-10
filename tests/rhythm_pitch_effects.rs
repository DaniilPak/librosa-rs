//! Parity tests for onsets, tempo, beat tracking, pitch tracking, effects and sequence models.

mod common;
use common::*;

use librosa::beat::{BeatTrackParams, beat_track};
use librosa::decompose::{self, HpssParams};
use librosa::effects::{self, PitchShiftParams, TrimParams};
use librosa::feature::{TempoParams, TempogramParams, tempo, tempogram};
use librosa::onset::{
    Aggregate, OnsetDetectParams, OnsetStrengthParams, onset_detect, onset_strength,
};
use librosa::sequence::{transition_local, transition_loop, viterbi};
use librosa::*;

#[test]
fn onset_envelopes() {
    let y = sig("beats");
    let e = onset_strength(y.view(), SR, &OnsetStrengthParams::default()).unwrap();
    assert_close("onset env", &e, &load1("onset_env_beats"), 1e-7, 1e-7);
    let p = OnsetStrengthParams {
        aggregate: Aggregate::Median,
        ..Default::default()
    };
    assert_close(
        "onset env median",
        &onset_strength(y.view(), SR, &p).unwrap(),
        &load1("onset_env_beats_median"),
        1e-7,
        1e-7,
    );
    let p = OnsetStrengthParams {
        lag: 2,
        max_size: 3,
        ..Default::default()
    };
    assert_close(
        "onset env lag2",
        &onset_strength(y.view(), SR, &p).unwrap(),
        &load1("onset_env_beats_lag2_max3"),
        1e-7,
        1e-7,
    );
    let p = OnsetStrengthParams {
        detrend: true,
        ..Default::default()
    };
    assert_close(
        "onset env detrend",
        &onset_strength(y.view(), SR, &p).unwrap(),
        &load1("onset_env_beats_detrend"),
        1e-7,
        1e-7,
    );
}

#[test]
fn onset_detection() {
    let y = sig("beats");
    let o = onset_detect(y.view(), SR, &OnsetDetectParams::default()).unwrap();
    assert_eq!(o, to_usize(&load1("onset_detect_beats")));
    let o = onset_detect(
        y.view(),
        SR,
        &OnsetDetectParams {
            backtrack: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(o, to_usize(&load1("onset_detect_beats_bt")));
    let o = onset_detect(sig("chirp").view(), SR, &OnsetDetectParams::default()).unwrap();
    assert_eq!(o, to_usize(&load1("onset_detect_chirp")));
}

#[test]
fn tempo_and_beats() {
    let y = sig("beats");
    let tg = tempogram(y.view(), SR, &TempogramParams::default()).unwrap();
    assert_close2("tempogram", &tg, &load2("tempogram_beats"), 1e-6, 1e-7);
    let t = tempo(y.view(), SR, &TempoParams::default()).unwrap();
    assert_close("tempo", &t, &load1("tempo_beats"), 1e-12, 0.0);
    let t = tempo(
        y.view(),
        SR,
        &TempoParams {
            aggregate: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_close("tempo tv", &t, &load1("tempo_beats_tv"), 1e-12, 0.0);

    let (bpm, beats) = beat_track(y.view(), SR, &BeatTrackParams::default()).unwrap();
    assert_close("beat tempo", &[bpm], &load1("beat_track_tempo"), 1e-12, 0.0);
    assert_eq!(beats, to_usize(&load1("beat_track_frames")));
    let (_, beats) = beat_track(
        y.view(),
        SR,
        &BeatTrackParams {
            bpm: Some(100.0),
            trim: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(beats, to_usize(&load1("beat_track_frames_bpm100")));
}

#[test]
fn yin_pyin() {
    let v = sig("voice");
    let f0 = yin(v.view(), 80.0, 800.0, SR, &YinParams::default()).unwrap();
    assert_close("yin voice", &f0, &load1("yin_voice"), 1e-7, 1e-9);
    let f0 = yin(
        sig("sine").view(),
        100.0,
        1000.0,
        SR,
        &YinParams {
            frame_length: 1024,
            ..Default::default()
        },
    )
    .unwrap();
    assert_close("yin sine", &f0, &load1("yin_sine"), 1e-7, 1e-9);

    let r = pyin(v.view(), 80.0, 800.0, SR, &PyinParams::default()).unwrap();
    assert_close("pyin f0", &r.f0, &load1("pyin_f0"), 1e-9, 0.0);
    let vf: Vec<f64> = r.voiced_flag.iter().map(|&b| b as u8 as f64).collect();
    assert_close("pyin voiced", &vf, &load1("pyin_voiced"), 0.0, 0.0);
    assert_close(
        "pyin vprob",
        &r.voiced_prob,
        &load1("pyin_vprob"),
        1e-6,
        1e-9,
    );
}

#[test]
fn pyin_noisy_input() {
    // Regression: far-tail Beta-CDF differences must not go negative.
    let y = &sig("noise") + &(sig("sine") * 0.2);
    let r = pyin(y.view(), 65.0, 1000.0, SR, &PyinParams::default()).unwrap();
    assert_close("pyin noisy f0", &r.f0, &load1("pyin_noisy_f0"), 1e-9, 0.0);
    assert_close(
        "pyin noisy vprob",
        &r.voiced_prob,
        &load1("pyin_noisy_vprob"),
        1e-6,
        1e-9,
    );
}

#[test]
fn piptrack_and_tuning() {
    let (p, m) = piptrack(sig("chord").view(), SR, &TuningParams::default()).unwrap();
    assert_close2("piptrack pitch", &p, &load2("piptrack_pitch"), 1e-9, 1e-9);
    assert_close2("piptrack mag", &m, &load2("piptrack_mag"), 1e-9, 1e-9);
    let t = pitch_tuning(&[440.0, 446.0, 880.0, 452.0, 449.0, 449.0], 0.01, 12);
    assert_eq!(t, meta()["pitch_tuning"].as_f64().unwrap());
}

#[test]
fn hpss_effects() {
    let y = sig("mix");
    let sp = StftParams::default();
    let (h, p) = effects::hpss(y.view(), &HpssParams::default(), &sp).unwrap();
    assert_close("hpss h", &h, &load1("hpss_h"), 1e-7, 1e-9);
    assert_close("hpss p", &p, &load1("hpss_p"), 1e-7, 1e-9);
    let hp = HpssParams {
        margin: (2.0, 3.0),
        kernel_size: (17, 31),
        ..Default::default()
    };
    let (h, p) = effects::hpss(y.view(), &hp, &sp).unwrap();
    assert_close("hpss h margin", &h, &load1("hpss_h_margin"), 1e-7, 1e-9);
    assert_close("hpss p margin", &p, &load1("hpss_p_margin"), 1e-7, 1e-9);
    let s = stft(y.view(), &sp).unwrap().mapv(|z| z.norm());
    let (mh, mp) = decompose::hpss_masks(s.view(), &HpssParams::default()).unwrap();
    // Masks are ratios of medians, so in near-silent bins (|S| ~ 1e-12 of the peak) the
    // STFT's last-bit rounding shows up; compare only bins above -120 dB of the peak.
    let peak = max_abs(s.iter());
    let keep: Vec<usize> = (0..s.len())
        .filter(|&i| s.as_slice().unwrap()[i] > 1e-6 * peak)
        .collect();
    let pick = |a: &ndarray::Array2<f64>| {
        keep.iter()
            .map(|&i| a.as_slice().unwrap()[i])
            .collect::<Vec<f64>>()
    };
    assert_close(
        "hpss mask h",
        &pick(&mh),
        &pick(&load2("hpss_mask_h")),
        1e-7,
        1e-9,
    );
    assert_close(
        "hpss mask p",
        &pick(&mp),
        &pick(&load2("hpss_mask_p")),
        1e-7,
        1e-9,
    );
}

#[test]
fn stretch_and_shift() {
    let y = sig("chord");
    let sp = StftParams::default();
    assert_close(
        "time_stretch 2",
        &effects::time_stretch(y.view(), 2.0, &sp).unwrap(),
        &load1("time_stretch_2"),
        1e-6,
        1e-8,
    );
    assert_close(
        "time_stretch 0.8",
        &effects::time_stretch(y.view(), 0.8, &sp).unwrap(),
        &load1("time_stretch_08"),
        1e-6,
        1e-8,
    );
    let p = PitchShiftParams {
        res_type: ResType::Fft,
        ..Default::default()
    };
    assert_close(
        "pitch_shift fft",
        &effects::pitch_shift(y.view(), SR, 4.0, &p).unwrap(),
        &load1("pitch_shift_fft"),
        1e-6,
        1e-8,
    );
    // default resampler vs soxr_hq: compare by SNR
    let ours = effects::pitch_shift(y.view(), SR, -3.0, &PitchShiftParams::default()).unwrap();
    let e = load1("pitch_shift_soxr");
    assert_eq!(ours.len(), e.len());
    let r = 2048..ours.len() - 2048;
    let err: f64 = r.clone().map(|i| (ours[i] - e[i]).powi(2)).sum();
    let pow: f64 = r.map(|i| e[i].powi(2)).sum();
    let snr = 10.0 * (pow / err).log10();
    assert!(snr > 50.0, "pitch_shift sinc vs soxr SNR {snr:.1} dB");
}

#[test]
fn trim_and_split() {
    let y = sig("gaps");
    let (yt, (a, b)) = effects::trim(y.view(), &TrimParams::default()).unwrap();
    assert_eq!(vec![a, b], to_usize(&load1("trim_idx")));
    assert_eq!(yt.len(), b - a);
    let (_, (a, b)) = effects::trim(
        y.view(),
        &TrimParams {
            top_db: 20.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(vec![a, b], to_usize(&load1("trim_idx_20")));
    let flat = |v: Vec<(usize, usize)>| v.into_iter().flat_map(|(a, b)| [a, b]).collect::<Vec<_>>();
    let s = effects::split(y.view(), &TrimParams::default()).unwrap();
    assert_eq!(
        flat(s),
        to_usize(&ndarray::Array1::from_iter(
            load2("split_idx").iter().copied()
        ))
    );
    let s = effects::split(
        y.view(),
        &TrimParams {
            top_db: 10.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        flat(s),
        to_usize(&ndarray::Array1::from_iter(
            load2("split_idx_10").iter().copied()
        ))
    );
}

#[test]
fn sequence_models() {
    let t = transition_local(10, 5, &Window::Triangle, false).unwrap();
    assert_close2(
        "transition_local",
        &t,
        &load2("transition_local"),
        1e-12,
        1e-15,
    );
    let t = transition_loop(3, 0.9).unwrap();
    assert_close2(
        "transition_loop",
        &t,
        &load2("transition_loop"),
        1e-12,
        1e-15,
    );
    let p = load2("viterbi_prob");
    let tr = transition_loop(4, 0.7).unwrap();
    let s = viterbi(p.view(), tr.view(), None, None).unwrap();
    assert_eq!(s, to_usize(&load1("viterbi_states")));
}
