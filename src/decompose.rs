//! Spectrogram decomposition (`librosa.decompose`): harmonic/percussive separation.

use crate::error::{Result, param};
use crate::spectrum::magphase;
use crate::util::{median_filter_reflect, softmask};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

/// Parameters for [`hpss`] (librosa defaults: kernel 31, power 2, margin 1).
#[derive(Debug, Clone, PartialEq)]
pub struct HpssParams {
    /// Median filter sizes `(harmonic, percussive)`.
    pub kernel_size: (usize, usize),
    /// Exponent for the soft mask (`f64::INFINITY` gives a hard mask).
    pub power: f64,
    /// Return the masks instead of the masked spectrograms.
    pub mask: bool,
    /// Margins `(harmonic, percussive)`, each >= 1.
    pub margin: (f64, f64),
}

impl Default for HpssParams {
    fn default() -> Self {
        HpssParams {
            kernel_size: (31, 31),
            power: 2.0,
            mask: false,
            margin: (1.0, 1.0),
        }
    }
}

/// Median-filter a magnitude spectrogram and build the harmonic / percussive soft masks.
pub fn hpss_masks(s: ArrayView2<f64>, p: &HpssParams) -> Result<(Array2<f64>, Array2<f64>)> {
    if p.margin.0 < 1.0 || p.margin.1 < 1.0 {
        return param("Margins must be >= 1.0. A typical range is between 1 and 10.");
    }
    let (nf, nt) = s.dim();
    let mut harm = Array2::zeros((nf, nt));
    for f in 0..nf {
        let row = s.row(f).to_vec();
        let m = median_filter_reflect(&row, p.kernel_size.0);
        for t in 0..nt {
            harm[[f, t]] = m[t];
        }
    }
    let mut perc = Array2::zeros((nf, nt));
    for t in 0..nt {
        let col = s.column(t).to_vec();
        let m = median_filter_reflect(&col, p.kernel_size.1);
        for f in 0..nf {
            perc[[f, t]] = m[f];
        }
    }
    let split_zeros = p.margin.0 == 1.0 && p.margin.1 == 1.0;
    let mh = softmask(
        harm.view(),
        (&perc * p.margin.0).view(),
        p.power,
        split_zeros,
    )?;
    let mp = softmask(
        perc.view(),
        (&harm * p.margin.1).view(),
        p.power,
        split_zeros,
    )?;
    Ok((mh, mp))
}

/// Harmonic/percussive source separation of a complex STFT (`librosa.decompose.hpss`).
///
/// Returns `(harmonic, percussive)` complex spectrograms (or the masks, as complex numbers
/// with zero imaginary part, when `p.mask` is set).
pub fn hpss(
    d: ArrayView2<Complex64>,
    p: &HpssParams,
) -> Result<(Array2<Complex64>, Array2<Complex64>)> {
    let (mag, phase) = magphase(d, 1.0);
    let (mh, mp) = hpss_masks(mag.view(), p)?;
    if p.mask {
        return Ok((
            mh.mapv(|v| Complex64::new(v, 0.0)),
            mp.mapv(|v| Complex64::new(v, 0.0)),
        ));
    }
    let h = ndarray::Zip::from(&mag)
        .and(&mh)
        .and(&phase)
        .map_collect(|&s, &m, &ph| ph * (s * m));
    let pc = ndarray::Zip::from(&mag)
        .and(&mp)
        .and(&phase)
        .map_collect(|&s, &m, &ph| ph * (s * m));
    Ok((h, pc))
}
