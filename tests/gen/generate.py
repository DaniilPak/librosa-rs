"""Generate reference fixtures for librosa-rs from Python librosa.

Usage:  python tests/gen/generate.py  (needs `pip install librosa soundfile`)

All test signals are synthesized here and dedicated to the public domain (CC0).
Inputs and outputs are float64 .npy files under tests/fixtures/.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
os.makedirs(OUT, exist_ok=True)
SR = 22050
meta = {"librosa": librosa.__version__, "numpy": np.__version__}


def save(name, arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        np.save(os.path.join(OUT, name + "_re.npy"), np.ascontiguousarray(arr.real, dtype=np.float64))
        np.save(os.path.join(OUT, name + "_im.npy"), np.ascontiguousarray(arr.imag, dtype=np.float64))
    else:
        np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(arr, dtype=np.float64))


rng = np.random.default_rng(0)
t1 = np.arange(SR) / SR

# ---- signals (CC0) ----
signals = {}
signals["sine"] = 0.5 * np.sin(2 * np.pi * 440.0 * t1)
signals["chirp"] = librosa.chirp(fmin=110, fmax=3520, sr=SR, duration=1.5) * 0.6
signals["noise"] = 0.1 * rng.standard_normal(SR)
# C major triad with a little noise
t2 = np.arange(2 * SR) / SR
signals["chord"] = sum(0.3 * np.sin(2 * np.pi * librosa.note_to_hz(n) * t2) for n in ["C4", "E4", "G4"]) + 0.003 * rng.standard_normal(2 * SR)
# drum-like clicks at 120 BPM over a quiet tone, 4 s
t4 = np.arange(4 * SR) / SR
beats = np.zeros_like(t4)
for k in range(8):
    start = int((0.25 + 0.5 * k) * SR)
    n = int(0.08 * SR)
    env = np.exp(-np.arange(n) / (0.01 * SR))
    burst = env * (rng.standard_normal(n) * 0.5 + np.sin(2 * np.pi * 180 * np.arange(n) / SR))
    beats[start:start + n] += burst
beats += 0.05 * np.sin(2 * np.pi * 330 * t4)
signals["beats"] = beats
# vibrato voice-like harmonic tone for pitch tracking (1.5 s), with a silent gap
t15 = np.arange(int(1.5 * SR)) / SR
f0 = 220 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t15))
phase = 2 * np.pi * np.cumsum(f0) / SR
voice = sum((0.5 / h) * np.sin(h * phase) for h in (1, 2, 3))
voice[int(0.6 * SR):int(0.8 * SR)] = 0.0
signals["voice"] = voice
# silence-padded tone for trim / split
sil = np.zeros(int(0.5 * SR))
tone = 0.4 * np.sin(2 * np.pi * 523.25 * np.arange(int(0.7 * SR)) / SR)
signals["gaps"] = np.concatenate([sil, tone, sil, tone * 0.5, sil])

for k, v in signals.items():
    save(f"sig_{k}", v)

# ---- conversions ----
freqs = np.array([20.0, 100.0, 440.0, 999.0, 1000.0, 1001.0, 4000.0, 11025.0])
save("conv_hz_to_mel", librosa.hz_to_mel(freqs))
save("conv_hz_to_mel_htk", librosa.hz_to_mel(freqs, htk=True))
mels = np.linspace(0, 45, 10)
save("conv_mel_to_hz", librosa.mel_to_hz(mels))
save("conv_mel_to_hz_htk", librosa.mel_to_hz(mels, htk=True))
save("conv_hz_to_midi", librosa.hz_to_midi(freqs))
save("conv_midi_to_hz", librosa.midi_to_hz(np.arange(0, 128, 7.5)))
save("conv_hz_to_octs", librosa.hz_to_octs(freqs, tuning=0.1))
notes = ["C4", "A4", "C#3", "Db5", "B-1", "G♯2", "E♭6", "A4+25", "Cbb3"]
save("conv_note_to_hz", librosa.note_to_hz(notes))
meta["notes"] = notes
meta["midi_to_note"] = [librosa.midi_to_note(m) for m in [0, 21, 60, 61, 69.4, 127]]
meta["midi_to_note_cents"] = [librosa.midi_to_note(m, cents=True) for m in [60.25, 69.4, 70.6]]
save("conv_fft_frequencies", librosa.fft_frequencies(sr=SR, n_fft=2048))
save("conv_mel_frequencies", librosa.mel_frequencies(n_mels=40, fmin=30, fmax=8000))
save("conv_cqt_frequencies", librosa.cqt_frequencies(n_bins=36, fmin=32.7, bins_per_octave=12, tuning=0.2))
save("conv_tempo_frequencies", librosa.tempo_frequencies(10)[1:])
save("conv_frames_to_time", librosa.frames_to_time(np.arange(10), sr=SR, hop_length=512))
save("conv_time_to_frames", librosa.time_to_frames([0.0, 0.1, 1.0, 2.5], sr=SR, hop_length=512))

# ---- windows / filters ----
for w in ["hann", "hamming", "blackman", "triang"]:
    save(f"win_{w}_periodic", librosa.filters.get_window(w, 64, fftbins=True))
    save(f"win_{w}_sym", librosa.filters.get_window(w, 63, fftbins=False))
save("win_kaiser_sym", librosa.filters.get_window(("kaiser", 5.0), 41, fftbins=False))
save("mel_slaney", librosa.filters.mel(sr=SR, n_fft=2048, dtype=np.float64))
save("mel_htk", librosa.filters.mel(sr=SR, n_fft=1024, n_mels=40, htk=True, norm=None, fmin=50, fmax=8000, dtype=np.float64))
save("mel_l2", librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, norm=2.0, dtype=np.float64))
save("chroma_fb", librosa.filters.chroma(sr=SR, n_fft=2048, dtype=np.float64))
save("chroma_fb_tuned", librosa.filters.chroma(sr=SR, n_fft=4096, tuning=0.25, n_chroma=24, octwidth=None, base_c=False, dtype=np.float64))
save("cq_to_chroma", librosa.filters.cq_to_chroma(84, bins_per_octave=36, n_chroma=12, dtype=np.float64))
save("window_sumsquare", librosa.filters.window_sumsquare(window="hann", n_frames=10, hop_length=256, n_fft=1024, dtype=np.float64))

# ---- spectrum ----
y = signals["chirp"]
D = librosa.stft(y)
save("stft_chirp", D)
save("stft_chirp_n512_h128_ham", librosa.stft(y, n_fft=512, hop_length=128, window="hamming"))
save("stft_chirp_win400", librosa.stft(y, n_fft=512, hop_length=160, win_length=400))
save("stft_chirp_reflect", librosa.stft(y, n_fft=1024, pad_mode="reflect"))
save("stft_chirp_nocenter", librosa.stft(y, n_fft=1024, hop_length=256, center=False))
save("istft_chirp", librosa.istft(D))
save("istft_chirp_len", librosa.istft(D, length=len(y)))
save("istft_n512", librosa.istft(librosa.stft(y, n_fft=512, hop_length=128), hop_length=128, length=len(y)))
S = np.abs(D)
save("amplitude_to_db_chirp", librosa.amplitude_to_db(S))
save("amplitude_to_db_chirp_refmax", librosa.amplitude_to_db(S, ref=np.max))
save("power_to_db_chirp", librosa.power_to_db(S ** 2, top_db=None))
save("power_to_db_chirp_refmax", librosa.power_to_db(S ** 2, ref=np.max))
mag, ph = librosa.magphase(D)
save("magphase_mag", mag)
save("magphase_phase", ph)
save("phase_vocoder_15", librosa.phase_vocoder(D, rate=1.5))
save("phase_vocoder_07", librosa.phase_vocoder(D, rate=0.7))

# ---- audio ----
save("resample_fft_up", librosa.resample(signals["noise"], orig_sr=SR, target_sr=44100, res_type="fft"))
save("resample_fft_down", librosa.resample(signals["chirp"], orig_sr=SR, target_sr=16000, res_type="fft"))
save("resample_poly_down", librosa.resample(signals["chirp"], orig_sr=SR, target_sr=16000, res_type="polyphase"))
save("resample_poly_half", librosa.resample(signals["noise"], orig_sr=2, target_sr=1, res_type="polyphase", scale=True))
save("resample_soxr_down", librosa.resample(signals["chirp"], orig_sr=SR, target_sr=16000))
save("autocorrelate_noise", librosa.autocorrelate(signals["noise"][:1000], max_size=300))
save("zero_crossings_sine", librosa.zero_crossings(signals["sine"][:500]).astype(float))
save("tone", librosa.tone(440, sr=SR, length=1000))
save("chirp_lin", librosa.chirp(fmin=100, fmax=2000, sr=SR, length=3000, linear=True))
save("chirp_log", librosa.chirp(fmin=100, fmax=2000, sr=SR, length=3000))
save("preemphasis_chirp", librosa.effects.preemphasis(signals["chirp"]))

# audio files (written by soundfile / libsndfile)
st = np.stack([signals["chord"][:SR], signals["noise"]])
sf.write(os.path.join(OUT, "stereo_44k.wav"), np.stack([librosa.resample(c, orig_sr=SR, target_sr=44100, res_type="fft") for c in st]).T * 0.8, 44100, subtype="PCM_16")
sf.write(os.path.join(OUT, "mono_22k.flac"), signals["chirp"], SR, subtype="PCM_16")
sf.write(os.path.join(OUT, "mono_22k.ogg"), signals["chord"][:SR], SR, format="OGG", subtype="VORBIS")
sf.write(os.path.join(OUT, "mono_22k.mp3"), signals["chord"][:SR], SR, format="MP3", subtype="MPEG_LAYER_III")
wav_native, sr_native = librosa.load(os.path.join(OUT, "stereo_44k.wav"), sr=None, mono=False, dtype=np.float64)
save("load_stereo_native", wav_native)
meta["load_stereo_sr"] = sr_native
save("load_stereo_mono_fft", librosa.load(os.path.join(OUT, "stereo_44k.wav"), sr=SR, res_type="fft", dtype=np.float64)[0])
save("load_stereo_mono_soxr", librosa.load(os.path.join(OUT, "stereo_44k.wav"), sr=SR, dtype=np.float64)[0])
save("load_stereo_offset", librosa.load(os.path.join(OUT, "stereo_44k.wav"), sr=None, offset=0.25, duration=0.5, dtype=np.float64)[0])
save("load_flac", librosa.load(os.path.join(OUT, "mono_22k.flac"), sr=None, dtype=np.float64)[0])
meta["duration_flac"] = librosa.get_duration(path=os.path.join(OUT, "mono_22k.flac"))
save("load_ogg", librosa.load(os.path.join(OUT, "mono_22k.ogg"), sr=None, dtype=np.float64)[0])
save("load_mp3", librosa.load(os.path.join(OUT, "mono_22k.mp3"), sr=None, dtype=np.float64)[0])

# ---- features ----
for name in ["chirp", "noise", "chord"]:
    y = signals[name]
    save(f"melspec_{name}", librosa.feature.melspectrogram(y=y, sr=SR, dtype=np.float64))
    save(f"melspec_{name}_f32", librosa.feature.melspectrogram(y=y, sr=SR))
    save(f"mfcc_{name}", librosa.feature.mfcc(y=y, sr=SR, dtype=np.float64))
    save(f"mfcc_{name}_f32", librosa.feature.mfcc(y=y, sr=SR))
    save(f"centroid_{name}", librosa.feature.spectral_centroid(y=y, sr=SR))
    save(f"bandwidth_{name}", librosa.feature.spectral_bandwidth(y=y, sr=SR))
    save(f"contrast_{name}", librosa.feature.spectral_contrast(y=y, sr=SR))
    save(f"rolloff_{name}", librosa.feature.spectral_rolloff(y=y, sr=SR))
    save(f"flatness_{name}", librosa.feature.spectral_flatness(y=y))
    save(f"rms_{name}", librosa.feature.rms(y=y, dtype=np.float64))
    save(f"zcr_{name}", librosa.feature.zero_crossing_rate(y))
    save(f"chroma_stft_{name}", librosa.feature.chroma_stft(y=y, sr=SR, dtype=np.float64))
    meta[f"tuning_{name}"] = float(librosa.estimate_tuning(y=y, sr=SR))
save("mfcc_chirp_lifter", librosa.feature.mfcc(y=signals["chirp"], sr=SR, n_mfcc=13, lifter=22, dtype=np.float64))
save("mfcc_chirp_htk", librosa.feature.mfcc(y=signals["chirp"], sr=SR, n_mfcc=13, n_mels=40, htk=True, fmin=20, fmax=8000, dtype=np.float64))
save("chroma_stft_chord_t0", librosa.feature.chroma_stft(y=signals["chord"], sr=SR, tuning=0.0, dtype=np.float64))
save("rms_chirp_S", librosa.feature.rms(S=np.abs(librosa.stft(signals["chirp"])), dtype=np.float64))
mfcc_c = librosa.feature.mfcc(y=signals["chirp"], sr=SR, dtype=np.float64)
save("delta_mfcc", librosa.feature.delta(mfcc_c))
save("delta2_mfcc", librosa.feature.delta(mfcc_c, order=2, width=5))
save("stack_memory", librosa.feature.stack_memory(mfcc_c[:3], n_steps=3, delay=2))

# ---- constant-Q ----
yc = signals["chord"]
save("cqt_chord_poly", librosa.cqt(yc, sr=SR, res_type="polyphase"))
save("cqt_chord_fft", librosa.cqt(yc, sr=SR, res_type="fft", n_bins=60, bins_per_octave=12, fmin=librosa.note_to_hz("C2")))
save("cqt_chord_soxr", librosa.cqt(yc, sr=SR))
save("vqt_chord_poly", librosa.vqt(yc, sr=SR, res_type="polyphase"))
save("cqt_chirp_36_poly", librosa.cqt(signals["chirp"], sr=SR, n_bins=7 * 36, bins_per_octave=36, res_type="polyphase", hop_length=256))
save("chroma_cqt_chord_poly", np.asarray(0))  # placeholder replaced below
C = np.abs(librosa.cqt(yc, sr=SR, n_bins=252, bins_per_octave=36, res_type="polyphase", tuning=0.0))
save("chroma_cqt_from_C", librosa.feature.chroma_cqt(C=C, sr=SR))
save("tonnetz_from_chroma", librosa.feature.tonnetz(chroma=librosa.feature.chroma_cqt(C=C, sr=SR)))
save("chroma_cqt_chord_soxr", librosa.feature.chroma_cqt(y=yc, sr=SR, tuning=0.0))
os.remove(os.path.join(OUT, "chroma_cqt_chord_poly.npy"))

# ---- onset / rhythm / beat ----
yb = signals["beats"]
save("onset_env_beats", librosa.onset.onset_strength(y=yb, sr=SR))
save("onset_env_beats_median", librosa.onset.onset_strength(y=yb, sr=SR, aggregate=np.median))
save("onset_env_beats_lag2_max3", librosa.onset.onset_strength(y=yb, sr=SR, lag=2, max_size=3))
save("onset_env_beats_detrend", librosa.onset.onset_strength(y=yb, sr=SR, detrend=True))
save("onset_detect_beats", librosa.onset.onset_detect(y=yb, sr=SR))
save("onset_detect_beats_bt", librosa.onset.onset_detect(y=yb, sr=SR, backtrack=True))
save("onset_detect_chirp", librosa.onset.onset_detect(y=signals["chirp"], sr=SR))
save("tempogram_beats", librosa.feature.tempogram(y=yb, sr=SR))
save("tempo_beats", librosa.feature.tempo(y=yb, sr=SR))
save("tempo_beats_tv", librosa.feature.tempo(y=yb, sr=SR, aggregate=None))
bt, bf = librosa.beat.beat_track(y=yb, sr=SR)
save("beat_track_tempo", np.atleast_1d(bt))
save("beat_track_frames", bf)
bt2, bf2 = librosa.beat.beat_track(y=yb, sr=SR, bpm=100.0, trim=False)
save("beat_track_frames_bpm100", bf2)

# ---- pitch ----
yv = signals["voice"]
save("yin_voice", librosa.yin(yv, fmin=80, fmax=800, sr=SR))
save("yin_sine", librosa.yin(signals["sine"], fmin=100, fmax=1000, sr=SR, frame_length=1024))
f0, vf, vp = librosa.pyin(yv, fmin=80, fmax=800, sr=SR)
save("pyin_f0", f0)
save("pyin_voiced", vf.astype(float))
save("pyin_vprob", vp)
f0n, vfn, vpn = librosa.pyin(signals["noise"] + 0.2 * signals["sine"], fmin=65, fmax=1000, sr=SR)
save("pyin_noisy_f0", f0n)
save("pyin_noisy_vprob", vpn)
pt, mg = librosa.piptrack(y=signals["chord"], sr=SR)
save("piptrack_pitch", pt)
save("piptrack_mag", mg)
meta["pitch_tuning"] = float(librosa.pitch_tuning(np.array([440.0, 446.0, 880.0, 452.0, 449.0, 449.0])))

# ---- effects / decompose ----
ym = signals["beats"][: 2 * SR] + 0.3 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR)
save("sig_mix", ym)
h, p = librosa.effects.hpss(ym)
save("hpss_h", h)
save("hpss_p", p)
h2, p2 = librosa.effects.hpss(ym, margin=(2.0, 3.0), kernel_size=(17, 31))
save("hpss_h_margin", h2)
save("hpss_p_margin", p2)
Dm = librosa.stft(ym)
mh, mp = librosa.decompose.hpss(np.abs(Dm), mask=True)
save("hpss_mask_h", mh)
save("hpss_mask_p", mp)
save("time_stretch_2", librosa.effects.time_stretch(signals["chord"], rate=2.0))
save("time_stretch_08", librosa.effects.time_stretch(signals["chord"], rate=0.8))
save("pitch_shift_fft", librosa.effects.pitch_shift(signals["chord"], sr=SR, n_steps=4, res_type="fft"))
save("pitch_shift_soxr", librosa.effects.pitch_shift(signals["chord"], sr=SR, n_steps=-3))
yt, idx = librosa.effects.trim(signals["gaps"])
save("trim_idx", idx)
save("trim_idx_20", librosa.effects.trim(signals["gaps"], top_db=20)[1])
save("split_idx", librosa.effects.split(signals["gaps"]))
save("split_idx_10", librosa.effects.split(signals["gaps"], top_db=10))

# ---- sequence ----
save("transition_local", librosa.sequence.transition_local(10, 5, window="triangle"))
save("transition_loop", librosa.sequence.transition_loop(3, 0.9))
P = rng.random((4, 12))
P /= P.sum(axis=0, keepdims=True)
T = librosa.sequence.transition_loop(4, 0.7)
save("viterbi_prob", P)
save("viterbi_states", librosa.sequence.viterbi(P, T))

with open(os.path.join(OUT, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1, ensure_ascii=False)
print("ok", len(os.listdir(OUT)), "files")
