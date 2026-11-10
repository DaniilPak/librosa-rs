use crate::error::{Result, param};
use crate::signal::savgol_interp;
use ndarray::{Array2, ArrayView2, Axis};

/// Local derivative estimates (deltas) of a feature matrix along `axis`
/// (`librosa.feature.delta` with `mode="interp"`), via a Savitzky-Golay filter of odd
/// `width` and polynomial order `order`. librosa defaults: `width=9, order=1, axis=1`
/// (the time axis of a `(features, frames)` matrix).
pub fn delta(
    data: ArrayView2<f64>,
    width: usize,
    order: usize,
    axis: usize,
) -> Result<Array2<f64>> {
    if axis > 1 {
        return param("axis must be 0 or 1");
    }
    let n = data.len_of(Axis(axis));
    if width > n {
        return param(format!(
            "when mode='interp', width={width} cannot exceed data.shape[axis]={n}"
        ));
    }
    if width < 3 || width % 2 != 1 {
        return param("width must be an odd integer >= 3");
    }
    if order == 0 {
        return param("order must be a positive integer");
    }
    let mut out = Array2::zeros(data.dim());
    for (src, mut dst) in data
        .lanes(Axis(axis))
        .into_iter()
        .zip(out.lanes_mut(Axis(axis)))
    {
        let v = savgol_interp(&src.to_vec(), width, order, order);
        for (d, x) in dst.iter_mut().zip(v) {
            *d = x;
        }
    }
    Ok(out)
}

/// Stack time-delayed copies of a feature matrix (`librosa.feature.stack_memory`), with
/// zero padding: output shape `(n_features * n_steps, n_frames)`.
pub fn stack_memory(data: ArrayView2<f64>, n_steps: usize, delay: usize) -> Result<Array2<f64>> {
    if n_steps < 1 {
        return param("n_steps must be a positive integer");
    }
    if delay == 0 {
        return param("delay must be a non-zero integer");
    }
    let (d, t) = data.dim();
    let mut out = Array2::zeros((d * n_steps, t));
    for step in 0..n_steps {
        let shift = step * delay;
        for f in 0..d {
            for j in shift..t {
                out[[step * d + f, j]] = data[[f, j - shift]];
            }
        }
    }
    Ok(out)
}
