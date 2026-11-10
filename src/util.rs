//! Array utilities (`librosa.util`): framing, padding, normalization, peak picking and masks.

use crate::error::{Result, param};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

/// Smallest positive normal `f64` (librosa's `util.tiny` for float64 input).
pub const TINY: f64 = f64::MIN_POSITIVE;

/// Padding modes, matching the `mode` argument of `numpy.pad`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PadMode {
    /// Pad with zeros (`"constant"`), librosa's default for STFT since 0.10.
    #[default]
    Constant,
    /// Repeat the edge value (`"edge"`).
    Edge,
    /// Mirror without repeating the edge sample (`"reflect"`).
    Reflect,
    /// Mirror including the edge sample (`"symmetric"`).
    Symmetric,
    /// Linear ramp from the edge value down to zero (`"linear_ramp"` with `end_values=0`).
    LinearRamp,
}

fn reflect_index(i: isize, n: isize, symmetric: bool) -> usize {
    if n == 1 {
        return 0;
    }
    let period = if symmetric { 2 * n } else { 2 * (n - 1) };
    let mut k = i.rem_euclid(period);
    if symmetric {
        if k >= n {
            k = period - 1 - k;
        }
    } else if k >= n {
        k = period - k;
    }
    k as usize
}

/// Pad a 1-D signal with `left` samples before and `right` samples after (like `numpy.pad`).
pub fn pad(x: ArrayView1<f64>, left: usize, right: usize, mode: PadMode) -> Array1<f64> {
    let n = x.len();
    let total = n + left + right;
    let mut out = Array1::zeros(total);
    out.slice_mut(s![left..left + n]).assign(&x);
    if n == 0 {
        return out;
    }
    match mode {
        PadMode::Constant => {}
        PadMode::Edge => {
            let (a, b) = (x[0], x[n - 1]);
            out.slice_mut(s![..left]).fill(a);
            out.slice_mut(s![left + n..]).fill(b);
        }
        PadMode::Reflect | PadMode::Symmetric => {
            let sym = mode == PadMode::Symmetric;
            for i in 0..left {
                let src = i as isize - left as isize;
                out[i] = x[reflect_index(src, n as isize, sym)];
            }
            for i in 0..right {
                let src = (n + i) as isize;
                out[left + n + i] = x[reflect_index(src, n as isize, sym)];
            }
        }
        PadMode::LinearRamp => {
            // numpy: linspace(end, edge, width, endpoint=False) on each side
            let a = x[0];
            for i in 0..left {
                out[i] = a * i as f64 / left as f64;
            }
            let b = x[n - 1];
            for i in 0..right {
                out[total - 1 - i] = b * i as f64 / right as f64;
            }
        }
    }
    out
}

/// Slice a signal into overlapping frames: output shape is `(frame_length, n_frames)`.
pub fn frame(x: ArrayView1<f64>, frame_length: usize, hop_length: usize) -> Result<Array2<f64>> {
    if x.len() < frame_length {
        return param(format!(
            "Input is too short (n={}) for frame_length={frame_length}",
            x.len()
        ));
    }
    if hop_length < 1 {
        return param(format!("Invalid hop_length: {hop_length}"));
    }
    let n_frames = 1 + (x.len() - frame_length) / hop_length;
    let mut out = Array2::zeros((frame_length, n_frames));
    for t in 0..n_frames {
        out.column_mut(t)
            .assign(&x.slice(s![t * hop_length..t * hop_length + frame_length]));
    }
    Ok(out)
}

/// Center `data` in a zero-padded array of length `size`.
pub fn pad_center(data: ArrayView1<f64>, size: usize) -> Result<Array1<f64>> {
    let n = data.len();
    if size < n {
        return param(format!(
            "Target size ({size}) must be at least input size ({n})"
        ));
    }
    let lpad = (size - n) / 2;
    Ok(pad(data, lpad, size - n - lpad, PadMode::Constant))
}

/// Trim or zero-pad `data` to exactly `size` samples.
pub fn fix_length(data: ArrayView1<f64>, size: usize) -> Array1<f64> {
    let n = data.len();
    if n >= size {
        data.slice(s![..size]).to_owned()
    } else {
        pad(data, 0, size - n, PadMode::Constant)
    }
}

/// Trim or zero-pad the last axis of a 2-D array to `size` columns.
pub fn fix_length_2d(data: ArrayView2<f64>, size: usize) -> Array2<f64> {
    let (r, c) = data.dim();
    let mut out = Array2::zeros((r, size));
    let m = c.min(size);
    out.slice_mut(s![.., ..m]).assign(&data.slice(s![.., ..m]));
    out
}

/// Normalization norms accepted by [`normalize`] (`norm=` in librosa).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Norm {
    /// Maximum absolute value (`np.inf`).
    Inf,
    /// Minimum absolute value (`-np.inf`).
    NegInf,
    /// Number of non-zero entries (`0`).
    Zero,
    /// The p-norm for `p > 0` (`1`, `2`, ...).
    P(f64),
}

fn vector_norm(mag: &[f64], norm: Norm) -> Result<(f64, f64)> {
    let n = mag.len();
    Ok(match norm {
        Norm::Inf => (mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max), 1.0),
        Norm::NegInf => (mag.iter().cloned().fold(f64::INFINITY, f64::min), 1.0),
        Norm::Zero => (mag.iter().filter(|&&v| v > 0.0).count() as f64, 1.0),
        Norm::P(p) if p > 0.0 => (
            mag.iter().map(|v| v.powf(p)).sum::<f64>().powf(1.0 / p),
            (n as f64).powf(-1.0 / p),
        ),
        Norm::P(p) => return param(format!("Unsupported norm: {p}")),
    })
}

fn normalize_slice(v: &mut [f64], norm: Norm, threshold: f64, fill: Option<bool>) -> Result<()> {
    let mag: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    let (length, fill_norm) = vector_norm(&mag, norm)?;
    if length < threshold {
        match fill {
            None => {}
            Some(true) => v.iter_mut().for_each(|x| *x = fill_norm),
            Some(false) => v.iter_mut().for_each(|x| *x = 0.0),
        }
    } else {
        v.iter_mut().for_each(|x| *x /= length);
    }
    Ok(())
}

/// Normalize a 1-D vector (`librosa.util.normalize` with `axis=0` on a vector).
///
/// `norm = None` returns the input unchanged. `threshold` defaults to [`TINY`].
/// `fill`: `None` leaves small vectors alone, `Some(true)` fills them with the unit-norm
/// constant, `Some(false)` zeroes them.
pub fn normalize_1d(
    x: ArrayView1<f64>,
    norm: Option<Norm>,
    threshold: Option<f64>,
    fill: Option<bool>,
) -> Result<Array1<f64>> {
    let mut out = x.to_owned();
    if !x.iter().all(|v| v.is_finite()) {
        return param("Input must be finite");
    }
    if let Some(norm) = norm {
        let th = threshold.unwrap_or(TINY);
        if th <= 0.0 {
            return param(format!("threshold={th} must be strictly positive"));
        }
        let mut v = out.to_vec();
        normalize_slice(&mut v, norm, th, fill)?;
        out = Array1::from(v);
    }
    Ok(out)
}

/// Normalize a 2-D array along `axis` (0 = normalize each column, 1 = each row).
pub fn normalize(
    s: ArrayView2<f64>,
    norm: Option<Norm>,
    axis: usize,
    threshold: Option<f64>,
    fill: Option<bool>,
) -> Result<Array2<f64>> {
    if !s.iter().all(|v| v.is_finite()) {
        return param("Input must be finite");
    }
    let mut out = s.to_owned();
    let Some(norm) = norm else {
        return Ok(out);
    };
    let th = threshold.unwrap_or(TINY);
    if th <= 0.0 {
        return param(format!("threshold={th} must be strictly positive"));
    }
    let lanes_axis = if axis == 0 { Axis(1) } else { Axis(0) };
    for mut lane in out.axis_iter_mut(lanes_axis) {
        let mut v = lane.to_vec();
        normalize_slice(&mut v, norm, th, fill)?;
        lane.assign(&Array1::from(v));
    }
    Ok(out)
}

/// Local maxima of a 1-D array: `x[i] > x[i-1]` and `x[i] >= x[i+1]`; the last element only
/// needs `x[-1] > x[-2]`, and the first element is never a maximum.
pub fn localmax(x: ArrayView1<f64>) -> Vec<bool> {
    let n = x.len();
    let mut out = vec![false; n];
    for i in 1..n.saturating_sub(1) {
        out[i] = x[i] > x[i - 1] && x[i] >= x[i + 1];
    }
    if n >= 2 {
        out[n - 1] = x[n - 1] > x[n - 2];
    }
    out
}

/// Local minima of a 1-D array (mirror image of [`localmax`]).
pub fn localmin(x: ArrayView1<f64>) -> Vec<bool> {
    let n = x.len();
    let mut out = vec![false; n];
    for i in 1..n.saturating_sub(1) {
        out[i] = x[i] < x[i - 1] && x[i] <= x[i + 1];
    }
    if n >= 2 {
        out[n - 1] = x[n - 1] < x[n - 2];
    }
    out
}

/// Parameters for [`peak_pick`] (all in frames).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakPick {
    /// Frames before `n` over which the max is computed.
    pub pre_max: usize,
    /// Frames after `n` over which the max is computed (must be > 0).
    pub post_max: usize,
    /// Frames before `n` over which the mean is computed.
    pub pre_avg: usize,
    /// Frames after `n` over which the mean is computed (must be > 0).
    pub post_avg: usize,
    /// Threshold offset for the mean.
    pub delta: f64,
    /// Frames to wait after picking a peak.
    pub wait: usize,
}

/// Greedy peak picking (`librosa.util.peak_pick(..., method="greedy")`); returns peak indices.
pub fn peak_pick(x: ArrayView1<f64>, p: &PeakPick) -> Result<Vec<usize>> {
    if p.delta < 0.0 {
        return param("delta must be non-negative");
    }
    if p.post_max == 0 {
        return param("post_max must be positive");
    }
    if p.post_avg == 0 {
        return param("post_avg must be positive");
    }
    let n = x.len();
    let mut peaks = Vec::new();
    if n == 0 {
        return Ok(peaks);
    }
    let max_of = |a: usize, b: usize| x.slice(s![a..b]).fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let mean_of = |a: usize, b: usize| x.slice(s![a..b]).sum() / (b - a) as f64;
    let mut first = x[0] >= max_of(0, p.post_max.min(n));
    first &= x[0] >= mean_of(0, p.post_avg.min(n)) + p.delta;
    let mut i = if first {
        peaks.push(0);
        p.wait + 1
    } else {
        1
    };
    while i < n {
        let maxn = max_of(i.saturating_sub(p.pre_max), (i + p.post_max).min(n));
        if x[i] != maxn {
            i += 1;
            continue;
        }
        let avgn = mean_of(i.saturating_sub(p.pre_avg), (i + p.post_avg).min(n));
        if x[i] < avgn + p.delta {
            i += 1;
            continue;
        }
        peaks.push(i);
        i += p.wait + 1;
    }
    Ok(peaks)
}

/// Soft mask `X^p / (X^p + X_ref^p)` (`librosa.util.softmask`); `power = inf` gives a hard mask.
pub fn softmask(
    x: ArrayView2<f64>,
    x_ref: ArrayView2<f64>,
    power: f64,
    split_zeros: bool,
) -> Result<Array2<f64>> {
    if x.dim() != x_ref.dim() {
        return param(format!("Shape mismatch: {:?}!={:?}", x.dim(), x_ref.dim()));
    }
    if x.iter().chain(x_ref.iter()).any(|&v| v < 0.0) {
        return param("X and X_ref must be non-negative");
    }
    if power <= 0.0 {
        return param("power must be strictly positive");
    }
    let mut out = Array2::zeros(x.dim());
    ndarray::Zip::from(&mut out)
        .and(&x)
        .and(&x_ref)
        .for_each(|o, &a, &b| {
            if power.is_infinite() {
                *o = if a > b { 1.0 } else { 0.0 };
                return;
            }
            let z = a.max(b);
            if z < TINY {
                *o = if split_zeros { 0.5 } else { 0.0 };
            } else {
                let m = (a / z).powf(power);
                let r = (b / z).powf(power);
                *o = m / (m + r);
            }
        });
    Ok(out)
}

/// Median of a slice (average of the two middle values for even lengths, like `np.median`).
pub(crate) fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// 1-D median filter with `scipy.ndimage` "reflect" boundary handling (`d c b a | a b c d`).
pub(crate) fn median_filter_reflect(x: &[f64], size: usize) -> Vec<f64> {
    let n = x.len();
    if size <= 1 || n == 0 {
        return x.to_vec();
    }
    let half = (size / 2) as isize;
    let mut buf = vec![0.0; size];
    (0..n)
        .map(|i| {
            for (k, b) in buf.iter_mut().enumerate() {
                let j = i as isize + k as isize - half;
                *b = x[reflect_index(j, n as isize, true)];
            }
            // scipy uses selection; for odd sizes this is the middle order statistic.
            buf.sort_by(|a, b| a.total_cmp(b));
            buf[size / 2]
        })
        .collect()
}

/// 1-D maximum filter (`scipy.ndimage.maximum_filter1d`, mode "reflect").
pub(crate) fn maximum_filter_reflect(x: &[f64], size: usize) -> Vec<f64> {
    let n = x.len();
    if size <= 1 || n == 0 {
        return x.to_vec();
    }
    let half = (size / 2) as isize;
    (0..n)
        .map(|i| {
            (0..size)
                .map(|k| x[reflect_index(i as isize + k as isize - half, n as isize, true)])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

/// Sorted, de-duplicated frame indices padded with `x_min` / `x_max` (`librosa.util.fix_frames`).
pub fn fix_frames(
    frames: &[usize],
    x_min: Option<usize>,
    x_max: Option<usize>,
    pad: bool,
) -> Vec<usize> {
    let mut f: Vec<usize> = frames.to_vec();
    if pad {
        if let Some(lo) = x_min {
            f.iter_mut().for_each(|v| *v = (*v).max(lo));
            f.push(lo);
        }
        if let Some(hi) = x_max {
            f.iter_mut().for_each(|v| *v = (*v).min(hi));
            f.push(hi);
        }
    }
    if let Some(lo) = x_min {
        f.retain(|&v| v >= lo);
    }
    if let Some(hi) = x_max {
        f.retain(|&v| v <= hi);
    }
    f.sort_unstable();
    f.dedup();
    f
}

/// For each event, the index (into `targets`) of the nearest target at or before it
/// (`librosa.util.match_events(..., right=False)`). `targets` must be sorted.
pub(crate) fn match_events_left(events: &[usize], targets: &[usize]) -> Vec<usize> {
    events
        .iter()
        .map(|&e| {
            let pos = targets.partition_point(|&t| t <= e);
            pos.saturating_sub(1)
        })
        .collect()
}
