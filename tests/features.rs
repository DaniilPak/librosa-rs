//! Parity tests for `librosa.feature` and the constant-Q transform.
//!
//! "strict" fixtures were generated with `dtype=np.float64` filter banks, so both sides
//! compute in double precision. "_f32" fixtures use librosa's defaults, where the mel
//! filter bank is stored as float32; those are compared at `rtol = 1e-5`.

mod common;
use common::*;

use librosa::feature::*;
use librosa::util::Norm;
use librosa::*;
use ndarray::Array2;

const NAMES: [&str; 3] = ["chirp", "noise", "chord"];

#[test]
fn melspectrogram_and_mfcc() {
    for name in NAMES {
        let y = sig(name);
        let m = melspectrogram(&y, SR, &MelSpectrogramParams::default()).unwrap();
        let e = load2(&format!("melspec_{name}"));
        assert_close2(
            &format!("melspec {name}"),
            &m,
            &e,
            1e-8,
            1e-12 * max_abs(e.iter()),
        );
        let e = load2(&format!("melspec_{name}_f32"));
        assert_close2(
            &format!("melspec f32 {name}"),
            &m,
            &e,
            1e-5,
            1e-7 * max_abs(e.iter()),
        );

        let c = mfcc(&y, SR, &MfccParams::default()).unwrap();
        assert_close2(
            &format!("mfcc {name}"),
            &c,
            &load2(&format!("mfcc_{name}")),
            1e-7,
            1e-7,
        );
        assert_close2(
            &format!("mfcc f32 {name}"),
            &c,
            &load2(&format!("mfcc_{name}_f32")),
            1e-4,
            1e-3,
        );
    }
    let y = sig("chirp");
    let c = mfcc(
        &y,
        SR,
        &MfccParams {
            n_mfcc: 13,
            lifter: 22.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_close2("mfcc lifter", &c, &load2("mfcc_chirp_lifter"), 1e-7, 1e-7);
    let p = MfccParams {
        n_mfcc: 13,
        mel: MelSpectrogramParams {
            mel: librosa::filters::MelParams {
                n_mels: 40,
                htk: true,
                fmin: 20.0,
                fmax: Some(8000.0),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    assert_close2(
        "mfcc htk",
        &mfcc(&y, SR, &p).unwrap(),
        &load2("mfcc_chirp_htk"),
        1e-7,
        1e-7,
    );
    // from a log-mel spectrogram
    let s = power_to_db(
        melspectrogram(&y, SR, &Default::default()).unwrap().view(),
        DbRef::Value(1.0),
        1e-10,
        Some(80.0),
    )
    .unwrap();
    assert_close2(
        "mfcc from S",
        &mfcc(&s, SR, &MfccParams::default()).unwrap(),
        &load2("mfcc_chirp"),
        1e-7,
        1e-7,
    );
}

#[test]
fn spectral_shape_features() {
    let sp = StftParams::default();
    for name in NAMES {
        let y = sig(name);
        let c = spectral_centroid(&y, SR, &sp, None).unwrap();
        assert_close2(
            &format!("centroid {name}"),
            &c,
            &load2(&format!("centroid_{name}")),
            1e-9,
            1e-9,
        );
        let b = spectral_bandwidth(&y, SR, &BandwidthParams::default()).unwrap();
        assert_close2(
            &format!("bandwidth {name}"),
            &b,
            &load2(&format!("bandwidth_{name}")),
            1e-9,
            1e-9,
        );
        let k = spectral_contrast(&y, SR, &ContrastParams::default()).unwrap();
        assert_close2(
            &format!("contrast {name}"),
            &k,
            &load2(&format!("contrast_{name}")),
            1e-8,
            1e-8,
        );
        let r = spectral_rolloff(&y, SR, &sp, 0.85).unwrap();
        assert_close2(
            &format!("rolloff {name}"),
            &r,
            &load2(&format!("rolloff_{name}")),
            1e-12,
            1e-9,
        );
        let f = spectral_flatness(&y, &sp, 1e-10, 2.0).unwrap();
        assert_close2(
            &format!("flatness {name}"),
            &f,
            &load2(&format!("flatness_{name}")),
            1e-8,
            1e-12,
        );
        let e = rms(&y, &FrameParams::default()).unwrap();
        assert_close2(
            &format!("rms {name}"),
            &e,
            &load2(&format!("rms_{name}")),
            1e-10,
            1e-14,
        );
        let z = zero_crossing_rate(y.view(), &FrameParams::default(), 1e-10, true).unwrap();
        assert_close2(
            &format!("zcr {name}"),
            &z,
            &load2(&format!("zcr_{name}")),
            1e-12,
            0.0,
        );
    }
    let y = sig("chirp");
    let s = stft(y.view(), &sp).unwrap().mapv(|z| z.norm());
    assert_close2(
        "rms from S",
        &rms(&s, &FrameParams::default()).unwrap(),
        &load2("rms_chirp_S"),
        1e-9,
        1e-14,
    );
}

#[test]
fn chroma_features() {
    let meta = meta();
    for name in NAMES {
        let y = sig(name);
        let t = estimate_tuning(y.view(), SR, &TuningParams::default()).unwrap();
        let te = meta[format!("tuning_{name}")].as_f64().unwrap();
        assert!((t - te).abs() < 1e-9, "tuning {name}: {t} vs {te}");
        let c = chroma_stft(&y, SR, &ChromaStftParams::default()).unwrap();
        assert_close2(
            &format!("chroma_stft {name}"),
            &c,
            &load2(&format!("chroma_stft_{name}")),
            1e-8,
            1e-10,
        );
    }
    let y = sig("chord");
    let c = chroma_stft(
        &y,
        SR,
        &ChromaStftParams {
            tuning: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_close2(
        "chroma_stft t0",
        &c,
        &load2("chroma_stft_chord_t0"),
        1e-8,
        1e-10,
    );
}

#[test]
fn constant_q() {
    let y = sig("chord");
    // CQT filters are complex64 inside librosa, so compare at ~1e-5 of the peak.
    let check = |what: &str,
                 ours: &Array2<num_complex::Complex64>,
                 e: &Array2<num_complex::Complex64>,
                 rel: f64| {
        assert_eq!(ours.dim(), e.dim(), "{what} shape");
        let peak = e.iter().fold(0.0f64, |m, z| m.max(z.norm()));
        let worst = ours
            .iter()
            .zip(e.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(
            worst <= rel * peak,
            "{what}: max |diff| {worst:.3e} vs peak {peak:.3e}"
        );
    };
    let p = CqtParams {
        res_type: ResType::Polyphase,
        ..Default::default()
    };
    check(
        "cqt poly",
        &cqt(y.view(), SR, &p).unwrap(),
        &load2c("cqt_chord_poly"),
        1e-5,
    );
    let p = CqtParams {
        res_type: ResType::Fft,
        n_bins: 60,
        fmin: Some(note_to_hz("C2").unwrap()),
        ..Default::default()
    };
    check(
        "cqt fft",
        &cqt(y.view(), SR, &p).unwrap(),
        &load2c("cqt_chord_fft"),
        1e-5,
    );
    // Different resamplers (ours vs soxr) differ near each octave's Nyquist: ~0.6% of peak.
    check(
        "cqt sinc~soxr",
        &cqt(y.view(), SR, &CqtParams::default()).unwrap(),
        &load2c("cqt_chord_soxr"),
        1e-2,
    );
    let p = CqtParams {
        res_type: ResType::Polyphase,
        ..Default::default()
    };
    check(
        "vqt poly",
        &vqt(y.view(), SR, None, &p).unwrap(),
        &load2c("vqt_chord_poly"),
        1e-5,
    );
    let p = CqtParams {
        res_type: ResType::Polyphase,
        n_bins: 7 * 36,
        bins_per_octave: 36,
        hop_length: 256,
        ..Default::default()
    };
    check(
        "cqt chirp 36",
        &cqt(sig("chirp").view(), SR, &p).unwrap(),
        &load2c("cqt_chirp_36_poly"),
        1e-5,
    );

    // chroma_cqt / tonnetz from a precomputed CQT magnitude
    let p = CqtParams {
        res_type: ResType::Polyphase,
        n_bins: 252,
        bins_per_octave: 36,
        tuning: Some(0.0),
        ..Default::default()
    };
    let c = cqt(y.view(), SR, &p).unwrap().mapv(|z| z.norm());
    let ch = chroma_cqt(&c, SR, &ChromaCqtParams::default()).unwrap();
    assert_close2("chroma_cqt", &ch, &load2("chroma_cqt_from_C"), 1e-4, 1e-5);
    let tn = tonnetz(&ch, SR, &ChromaCqtParams::default()).unwrap();
    assert_close2("tonnetz", &tn, &load2("tonnetz_from_chroma"), 1e-4, 1e-5);
    let ch = chroma_cqt(
        &y,
        SR,
        &ChromaCqtParams {
            tuning: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    // default resampler vs soxr: chroma values (max-normalized) agree to ~2e-2
    assert_close2(
        "chroma_cqt audio",
        &ch,
        &load2("chroma_cqt_chord_soxr"),
        0.0,
        2e-2,
    );
}

#[test]
fn deltas_and_stacking() {
    let m = load2("mfcc_chirp");
    assert_close2(
        "delta",
        &delta(m.view(), 9, 1, 1).unwrap(),
        &load2("delta_mfcc"),
        1e-9,
        1e-9,
    );
    assert_close2(
        "delta2",
        &delta(m.view(), 5, 2, 1).unwrap(),
        &load2("delta2_mfcc"),
        1e-9,
        1e-9,
    );
    let sm = stack_memory(m.slice(ndarray::s![..3, ..]), 3, 2).unwrap();
    assert_close2("stack_memory", &sm, &load2("stack_memory"), 0.0, 0.0);
    let _ = Norm::Inf;
}
