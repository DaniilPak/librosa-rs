//! Feature extraction (`librosa.feature`): spectral features, chroma, MFCC, rhythm
//! features and deltas.

mod rhythm;
mod spectral;
mod utils;

pub use rhythm::*;
pub use spectral::*;
pub use utils::*;

use crate::error::Result;
use crate::spectrum::{StftParams, spectrogram};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Input to a feature: a time series, or a precomputed spectrogram-like matrix.
///
/// Anything that accepts `impl Into<Input>` takes `&Array1<f64>` / `ArrayView1` (audio) or
/// `&Array2<f64>` / `ArrayView2` (spectrogram), the same way librosa takes `y=` or `S=`.
#[derive(Debug, Clone, Copy)]
pub enum Input<'a> {
    /// Audio samples (librosa's `y=`).
    Audio(ArrayView1<'a, f64>),
    /// A precomputed (non-negative) spectrogram or feature matrix (librosa's `S=`, `C=`,
    /// `chroma=`, or `onset_envelope=` reshaped as one row).
    Spectrogram(ArrayView2<'a, f64>),
}

impl<'a> From<ArrayView1<'a, f64>> for Input<'a> {
    fn from(v: ArrayView1<'a, f64>) -> Self {
        Input::Audio(v)
    }
}

impl<'a> From<&'a Array1<f64>> for Input<'a> {
    fn from(v: &'a Array1<f64>) -> Self {
        Input::Audio(v.view())
    }
}

impl<'a> From<ArrayView2<'a, f64>> for Input<'a> {
    fn from(v: ArrayView2<'a, f64>) -> Self {
        Input::Spectrogram(v)
    }
}

impl<'a> From<&'a Array2<f64>> for Input<'a> {
    fn from(v: &'a Array2<f64>) -> Self {
        Input::Spectrogram(v.view())
    }
}

/// librosa's `_spectrogram`: compute `|STFT|^power` from audio, or pass a spectrogram
/// through, inferring `n_fft` from its height when it does not match `p.n_fft`.
pub(crate) fn get_spec(
    input: Input<'_>,
    p: &StftParams,
    power: f64,
) -> Result<(Array2<f64>, usize)> {
    match input {
        Input::Audio(y) => Ok((spectrogram(y, p, power)?, p.n_fft)),
        Input::Spectrogram(s) => {
            let n_fft = if p.n_fft / 2 + 1 != s.nrows() {
                2 * s.nrows().saturating_sub(1)
            } else {
                p.n_fft
            };
            Ok((s.to_owned(), n_fft))
        }
    }
}
