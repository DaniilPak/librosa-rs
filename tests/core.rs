//! Parity tests for conversions, windows, filters, spectrum and audio I/O.
//!
//! Tolerances: functions computed in float64 by both libraries use `rtol = 1e-9`
//! (differences come only from FFT / libm rounding). See README for the full table.

mod common;
use common::*;

use librosa::filters::{self, ChromaParams, MelNorm, MelParams};
use librosa::util::Norm;
use librosa::*;
use ndarray::{Array1, Array2, Axis};

#[test]
fn conversions() {
    let freqs = [20.0, 100.0, 440.0, 999.0, 1000.0, 1001.0, 4000.0, 11025.0];
    let m: Vec<f64> = freqs.iter().map(|&f| hz_to_mel(f, false)).collect();
    assert_close("hz_to_mel", &m, &load1("conv_hz_to_mel"), 1e-12, 0.0);
    let m: Vec<f64> = freqs.iter().map(|&f| hz_to_mel(f, true)).collect();
    assert_close(
        "hz_to_mel_htk",
        &m,
        &load1("conv_hz_to_mel_htk"),
        1e-12,
        0.0,
    );
    let mels = librosa::convert::fft_frequencies(1.0, 2); // placeholder to keep import used
    assert_eq!(mels.len(), 2);
    let mels: Vec<f64> = (0..10).map(|k| 45.0 * k as f64 / 9.0).collect();
    let h: Vec<f64> = mels.iter().map(|&m| mel_to_hz(m, false)).collect();
    assert_close("mel_to_hz", &h, &load1("conv_mel_to_hz"), 1e-12, 1e-12);
    let h: Vec<f64> = mels.iter().map(|&m| mel_to_hz(m, true)).collect();
    assert_close(
        "mel_to_hz_htk",
        &h,
        &load1("conv_mel_to_hz_htk"),
        1e-12,
        1e-12,
    );
    let mi: Vec<f64> = freqs.iter().map(|&f| hz_to_midi(f)).collect();
    assert_close("hz_to_midi", &mi, &load1("conv_hz_to_midi"), 1e-12, 0.0);
    let hz: Vec<f64> = (0..18).map(|k| midi_to_hz(7.5 * k as f64)).collect();
    assert_close("midi_to_hz", &hz, &load1("conv_midi_to_hz"), 1e-12, 0.0);
    let o: Vec<f64> = freqs.iter().map(|&f| hz_to_octs(f, 0.1, 12)).collect();
    assert_close("hz_to_octs", &o, &load1("conv_hz_to_octs"), 1e-12, 1e-12);

    let meta = meta();
    let notes: Vec<String> = meta["notes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let hz: Vec<f64> = notes.iter().map(|n| note_to_hz(n).unwrap()).collect();
    assert_close("note_to_hz", &hz, &load1("conv_note_to_hz"), 1e-12, 0.0);
    let names: Vec<String> = [0.0, 21.0, 60.0, 61.0, 69.4, 127.0]
        .iter()
        .map(|&m| midi_to_note(m, true, false, true).unwrap())
        .collect();
    let exp: Vec<String> = meta["midi_to_note"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, exp);
    let names: Vec<String> = [60.25, 69.4, 70.6]
        .iter()
        .map(|&m| midi_to_note(m, true, true, true).unwrap())
        .collect();
    let exp: Vec<String> = meta["midi_to_note_cents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, exp);
    assert_eq!(hz_to_note(440.0, true, false, false).unwrap(), "A4");

    assert_close(
        "fft_frequencies",
        &fft_frequencies(SR, 2048),
        &load1("conv_fft_frequencies"),
        1e-14,
        0.0,
    );
    assert_close(
        "mel_frequencies",
        &mel_frequencies(40, 30.0, 8000.0, false),
        &load1("conv_mel_frequencies"),
        1e-12,
        0.0,
    );
    assert_close(
        "cqt_frequencies",
        &cqt_frequencies(36, 32.7, 12, 0.2),
        &load1("conv_cqt_frequencies"),
        1e-12,
        0.0,
    );
    let tf = tempo_frequencies(10, 512, SR);
    assert_close(
        "tempo_frequencies",
        tf.iter().skip(1),
        &load1("conv_tempo_frequencies"),
        1e-12,
        0.0,
    );
    let frames: Vec<i64> = (0..10).collect();
    assert_close(
        "frames_to_time",
        &frames_to_time(&frames, SR, 512, None),
        &load1("conv_frames_to_time"),
        1e-14,
        0.0,
    );
    let tf: Vec<f64> = time_to_frames(&[0.0, 0.1, 1.0, 2.5], SR, 512, None)
        .iter()
        .map(|&v| v as f64)
        .collect();
    assert_close(
        "time_to_frames",
        &tf,
        &load1("conv_time_to_frames"),
        0.0,
        0.0,
    );
    assert_eq!(samples_to_frames(&[-1], 512, None), vec![-1]);
}

#[test]
fn windows() {
    for (w, name) in [
        (Window::Hann, "hann"),
        (Window::Hamming, "hamming"),
        (Window::Blackman, "blackman"),
        (Window::Triangle, "triang"),
    ] {
        let p = get_window(&w, 64, true).unwrap();
        assert_close(
            name,
            &p,
            &load1(&format!("win_{name}_periodic")),
            1e-12,
            1e-15,
        );
        let s = get_window(&w, 63, false).unwrap();
        assert_close(name, &s, &load1(&format!("win_{name}_sym")), 1e-12, 1e-15);
    }
    let k = get_window(&Window::Kaiser(5.0), 41, false).unwrap();
    assert_close("kaiser", &k, &load1("win_kaiser_sym"), 1e-12, 1e-15);
}

#[test]
fn filter_banks() {
    let m = filters::mel(SR, 2048, &MelParams::default()).unwrap();
    assert_close2("mel slaney", &m, &load2("mel_slaney"), 1e-10, 1e-14);
    let m = filters::mel(
        SR,
        1024,
        &MelParams {
            n_mels: 40,
            htk: true,
            norm: MelNorm::None,
            fmin: 50.0,
            fmax: Some(8000.0),
        },
    )
    .unwrap();
    assert_close2("mel htk", &m, &load2("mel_htk"), 1e-10, 1e-14);
    let m = filters::mel(
        16000.0,
        512,
        &MelParams {
            n_mels: 40,
            norm: MelNorm::P(2.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_close2("mel l2", &m, &load2("mel_l2"), 1e-10, 1e-14);

    let c = filters::chroma(SR, 2048, &ChromaParams::default()).unwrap();
    assert_close2("chroma fb", &c, &load2("chroma_fb"), 1e-10, 1e-14);
    let c = filters::chroma(
        SR,
        4096,
        &ChromaParams {
            tuning: 0.25,
            n_chroma: 24,
            octwidth: None,
            base_c: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_close2(
        "chroma fb tuned",
        &c,
        &load2("chroma_fb_tuned"),
        1e-10,
        1e-14,
    );
    let c = filters::cq_to_chroma(84, 36, 12, None, true).unwrap();
    assert_close2("cq_to_chroma", &c, &load2("cq_to_chroma"), 0.0, 0.0);
    let w = filters::window_sumsquare(&Window::Hann, 10, 256, None, 1024, None).unwrap();
    assert_close(
        "window_sumsquare",
        &w,
        &load1("window_sumsquare"),
        1e-12,
        1e-14,
    );
    let _ = Norm::Inf;
}

#[test]
fn stft_variants() {
    let y = sig("chirp");
    let d = stft(y.view(), &StftParams::default()).unwrap();
    let e = load2c("stft_chirp");
    let scale = max_abs(e.iter().map(|z| &z.re));
    assert_close2c("stft", &d, &e, 1e-9, 1e-11 * scale);

    let p = StftParams {
        window: Window::Hamming,
        ..StftParams::new(512, 128)
    };
    assert_close2c(
        "stft hamming",
        &stft(y.view(), &p).unwrap(),
        &load2c("stft_chirp_n512_h128_ham"),
        1e-9,
        1e-10,
    );
    let p = StftParams {
        win_length: Some(400),
        ..StftParams::new(512, 160)
    };
    assert_close2c(
        "stft win400",
        &stft(y.view(), &p).unwrap(),
        &load2c("stft_chirp_win400"),
        1e-9,
        1e-10,
    );
    let p = StftParams {
        n_fft: 1024,
        pad_mode: PadMode::Reflect,
        ..Default::default()
    };
    assert_close2c(
        "stft reflect",
        &stft(y.view(), &p).unwrap(),
        &load2c("stft_chirp_reflect"),
        1e-9,
        1e-10,
    );
    let p = StftParams {
        center: false,
        ..StftParams::new(1024, 256)
    };
    assert_close2c(
        "stft nocenter",
        &stft(y.view(), &p).unwrap(),
        &load2c("stft_chirp_nocenter"),
        1e-9,
        1e-10,
    );
}

#[test]
fn istft_roundtrip() {
    let y = sig("chirp");
    let d = load2c("stft_chirp");
    let r = istft(d.view(), &IstftParams::default()).unwrap();
    assert_close("istft", &r, &load1("istft_chirp"), 1e-9, 1e-12);
    let r = istft(
        d.view(),
        &IstftParams {
            length: Some(y.len()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_close("istft len", &r, &load1("istft_chirp_len"), 1e-9, 1e-12);
    assert_close("istft ~ y", &r, &y, 0.0, 1e-9);
    let d2 = stft(y.view(), &StftParams::new(512, 128)).unwrap();
    let r = istft(
        d2.view(),
        &IstftParams {
            hop_length: Some(128),
            length: Some(y.len()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_close("istft n512", &r, &load1("istft_n512"), 1e-9, 1e-12);
}

#[test]
fn db_scaling_and_magphase() {
    let d = load2c("stft_chirp");
    let s = d.mapv(|z| z.norm());
    let a = amplitude_to_db(s.view(), DbRef::Value(1.0), 1e-5, Some(80.0)).unwrap();
    assert_close2(
        "amplitude_to_db",
        &a,
        &load2("amplitude_to_db_chirp"),
        1e-10,
        1e-9,
    );
    let a = amplitude_to_db(s.view(), DbRef::Max, 1e-5, Some(80.0)).unwrap();
    assert_close2(
        "amplitude_to_db max",
        &a,
        &load2("amplitude_to_db_chirp_refmax"),
        1e-10,
        1e-9,
    );
    let p = s.mapv(|v| v * v);
    let a = power_to_db(p.view(), DbRef::Value(1.0), 1e-10, None).unwrap();
    assert_close2("power_to_db", &a, &load2("power_to_db_chirp"), 1e-10, 1e-9);
    let a = power_to_db(p.view(), DbRef::Max, 1e-10, Some(80.0)).unwrap();
    assert_close2(
        "power_to_db max",
        &a,
        &load2("power_to_db_chirp_refmax"),
        1e-10,
        1e-9,
    );
    let back = db_to_power(
        power_to_db(p.view(), DbRef::Value(1.0), 1e-30, None)
            .unwrap()
            .view(),
        1.0,
    );
    assert_close2("db_to_power", &back, &p, 1e-9, 1e-25);
    let back = db_to_amplitude(
        amplitude_to_db(s.view(), DbRef::Value(1.0), 1e-15, None)
            .unwrap()
            .view(),
        1.0,
    );
    assert_close2("db_to_amplitude", &back, &s, 1e-9, 1e-14);

    let (mag, ph) = magphase(d.view(), 1.0);
    assert_close2("magphase mag", &mag, &load2("magphase_mag"), 1e-12, 1e-14);
    assert_close2c("magphase phase", &ph, &load2c("magphase_phase"), 1e-9, 1e-9);
}

#[test]
fn phase_vocoder_matches() {
    let d = load2c("stft_chirp");
    for (rate, name) in [(1.5, "phase_vocoder_15"), (0.7, "phase_vocoder_07")] {
        let out = phase_vocoder(d.view(), rate).unwrap();
        let e = load2c(name);
        // phase accumulates rounding over frames; compare with a magnitude-scaled atol
        let scale = max_abs(e.iter().map(|z| &z.re));
        assert_close2c(name, &out, &e, 1e-7, 1e-9 * scale);
    }
}

#[test]
fn resampling() {
    let noise = sig("noise");
    let chirp = sig("chirp");
    let r = resample(noise.view(), SR, 44100.0, ResType::Fft, true, false).unwrap();
    assert_close(
        "resample fft up",
        &r,
        &load1("resample_fft_up"),
        1e-9,
        1e-12,
    );
    let r = resample(chirp.view(), SR, 16000.0, ResType::Fft, true, false).unwrap();
    assert_close(
        "resample fft down",
        &r,
        &load1("resample_fft_down"),
        1e-9,
        1e-12,
    );
    let r = resample(chirp.view(), SR, 16000.0, ResType::Polyphase, true, false).unwrap();
    assert_close(
        "resample poly",
        &r,
        &load1("resample_poly_down"),
        1e-9,
        1e-12,
    );
    let r = resample(noise.view(), 2.0, 1.0, ResType::Polyphase, true, true).unwrap();
    assert_close(
        "resample poly half",
        &r,
        &load1("resample_poly_half"),
        1e-9,
        1e-12,
    );
    // Default sinc resampler vs soxr_hq: not bit-identical; require < -60 dB error in-band.
    let r = resample(chirp.view(), SR, 16000.0, ResType::Sinc, true, false).unwrap();
    let e = load1("resample_soxr_down");
    assert_eq!(r.len(), e.len());
    let (a, b) = (200, r.len() - 200); // ignore edge transients
    let err: f64 = (a..b).map(|i| (r[i] - e[i]).powi(2)).sum::<f64>();
    let pow: f64 = (a..b).map(|i| e[i].powi(2)).sum::<f64>();
    let snr = 10.0 * (pow / err).log10();
    assert!(snr > 60.0, "sinc vs soxr SNR {snr:.1} dB");
}

#[test]
fn audio_helpers() {
    let n = sig("noise");
    let ac = autocorrelate(n.slice(ndarray::s![..1000]), Some(300));
    assert_close(
        "autocorrelate",
        &ac,
        &load1("autocorrelate_noise"),
        1e-9,
        1e-12,
    );
    let s = sig("sine");
    let z: Vec<f64> = zero_crossings(s.slice(ndarray::s![..500]), 1e-10, true, true)
        .iter()
        .map(|&b| b as u8 as f64)
        .collect();
    assert_close(
        "zero_crossings",
        &z,
        &load1("zero_crossings_sine"),
        0.0,
        0.0,
    );
    assert_close("tone", &tone(440.0, SR, 1000), &load1("tone"), 1e-9, 1e-12);
    assert_close(
        "chirp lin",
        &chirp(100.0, 2000.0, SR, 3000, true),
        &load1("chirp_lin"),
        1e-9,
        1e-9,
    );
    assert_close(
        "chirp log",
        &chirp(100.0, 2000.0, SR, 3000, false),
        &load1("chirp_log"),
        1e-9,
        1e-9,
    );
    let c = sig("chirp");
    assert_close(
        "preemphasis",
        &librosa::effects::preemphasis(c.view(), 0.97),
        &load1("preemphasis_chirp"),
        1e-12,
        1e-14,
    );
    let st = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]).unwrap();
    assert_eq!(to_mono(st.view()), Array1::from(vec![2.0, 3.0, 4.0]));
    assert_eq!(get_duration(22050, SR), 1.0);
    assert!((get_duration_frames(44, SR, 2048, 512, true) - 22016.0 / SR).abs() < 1e-12);
}

#[test]
fn load_files() {
    let native = load_multichannel(
        path("stereo_44k.wav"),
        &LoadOptions {
            sr: None,
            mono: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(native.1, 44100.0);
    assert_close2(
        "load stereo",
        &native.0,
        &load2("load_stereo_native"),
        0.0,
        1e-12,
    );

    let (y, sr) = load(
        path("stereo_44k.wav"),
        &LoadOptions {
            res_type: ResType::Fft,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(sr, 22050.0);
    assert_close(
        "load mono fft",
        &y,
        &load1("load_stereo_mono_fft"),
        1e-9,
        1e-12,
    );

    let (y, _) = load(path("stereo_44k.wav"), &LoadOptions::default()).unwrap();
    let e = load1("load_stereo_mono_soxr");
    assert_eq!(y.len(), e.len());
    let mid = 500..y.len() - 500;
    let err: f64 = mid.clone().map(|i| (y[i] - e[i]).powi(2)).sum();
    let pow: f64 = mid.map(|i| e[i].powi(2)).sum();
    // The right channel is white noise up to Nyquist, so the two resamplers' different
    // transition bands (last ~3% below Nyquist) bound the agreement here.
    let snr = 10.0 * (pow / err).log10();
    assert!(snr > 20.0, "load sinc vs soxr SNR {snr:.1} dB");

    let (y, _) = load(
        path("stereo_44k.wav"),
        &LoadOptions {
            sr: None,
            offset: 0.25,
            duration: Some(0.5),
            ..Default::default()
        },
    )
    .unwrap();
    assert_close("load offset", &y, &load1("load_stereo_offset"), 0.0, 1e-12);

    let (y, sr) = load(
        path("mono_22k.flac"),
        &LoadOptions {
            sr: None,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(sr, 22050.0);
    assert_close("load flac", &y, &load1("load_flac"), 0.0, 1e-12);
    let d = get_duration_path(path("mono_22k.flac")).unwrap();
    assert!((d - meta()["duration_flac"].as_f64().unwrap()).abs() < 1e-9);
    assert_eq!(get_samplerate(path("mono_22k.flac")).unwrap(), 22050.0);

    // Lossy codecs: decoders may differ in the last bits; compare loosely.
    let (y, _) = load(
        path("mono_22k.ogg"),
        &LoadOptions {
            sr: None,
            ..Default::default()
        },
    )
    .unwrap();
    let e = load1("load_ogg");
    assert_eq!(y.len(), e.len(), "ogg length");
    assert_close("load ogg", &y, &e, 0.0, 1e-4);
    let (y, _) = load(
        path("mono_22k.mp3"),
        &LoadOptions {
            sr: None,
            ..Default::default()
        },
    )
    .unwrap();
    let e = load1("load_mp3");
    let n = y.len().min(e.len());
    assert!(
        (y.len() as i64 - e.len() as i64).abs() <= 1152,
        "mp3 length {} vs {}",
        y.len(),
        e.len()
    );
    // MP3 decoders may differ in priming-sample handling; compare energy
    let ey: f64 = y.iter().take(n).map(|v| v * v).sum();
    let ee: f64 = e.iter().take(n).map(|v| v * v).sum();
    assert!((ey / ee - 1.0).abs() < 0.05, "mp3 energy ratio {}", ey / ee);
    let _ = y.sum_axis(Axis(0));
}
