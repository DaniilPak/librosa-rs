//! Sequence models (`librosa.sequence`): Viterbi decoding and transition matrices.

use crate::error::{Result, param};
use crate::util::{TINY, pad_center};
use crate::window::{Window, get_window};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Viterbi decoding of a first-order HMM (`librosa.sequence.viterbi`).
///
/// `prob` has shape `(n_states, n_steps)` with observation likelihoods in `[0, 1]`,
/// `transition[i, j]` is `P(j | i)`. `p_init` defaults to uniform. Transitions below
/// `transition_min_prob` are skipped during the search (librosa's speed-up).
/// Returns the most likely state sequence.
pub fn viterbi(
    prob: ArrayView2<f64>,
    transition: ArrayView2<f64>,
    p_init: Option<ArrayView1<f64>>,
    transition_min_prob: Option<f64>,
) -> Result<Vec<usize>> {
    let (n_states, n_steps) = prob.dim();
    if transition.dim() != (n_states, n_states) {
        return param(format!(
            "transition.shape={:?}, must be (n_states, n_states)={:?}",
            transition.dim(),
            (n_states, n_states)
        ));
    }
    if transition.iter().any(|&v| v < 0.0)
        || transition
            .rows()
            .into_iter()
            .any(|r| (r.sum() - 1.0).abs() > 1e-5)
    {
        return param("Invalid transition matrix: must be non-negative and sum to 1 on each row.");
    }
    if prob.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return param("Invalid probability values: must be between 0 and 1.");
    }
    if n_steps == 0 {
        return Ok(Vec::new());
    }
    let eps = TINY;
    let log_trans = transition.mapv(|v| (v + eps).ln());
    let log_p_init: Array1<f64> = match p_init {
        Some(p) => p.mapv(|v| (v + eps).ln()),
        None => Array1::from_elem(n_states, (1.0 / n_states as f64 + eps).ln()),
    };
    let threshold = match transition_min_prob {
        Some(t) if t > 0.0 => (t + eps).ln(),
        Some(t) if t < 0.0 => return param(format!("Invalid transition_min_prob={t}")),
        _ => f64::NEG_INFINITY,
    };
    let preds: Vec<Vec<usize>> = (0..n_states)
        .map(|j| {
            (0..n_states)
                .filter(|&k| log_trans[[k, j]] >= threshold)
                .collect()
        })
        .collect();
    if preds.iter().any(|p| p.is_empty()) {
        return param(
            "Empty transition matrix detected in Viterbi. Try reducing your minimum transition probability threshold.",
        );
    }
    let mut value = Array2::<f64>::zeros((n_steps, n_states));
    let mut ptr = Array2::<usize>::zeros((n_steps, n_states));
    for j in 0..n_states {
        value[[0, j]] = (prob[[j, 0]] + eps).ln() + log_p_init[j];
    }
    for t in 1..n_steps {
        for j in 0..n_states {
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0;
            for &k in &preds[j] {
                let cost = value[[t - 1, k]] + log_trans[[k, j]];
                if cost > best {
                    best = cost;
                    arg = k;
                }
            }
            ptr[[t, j]] = arg;
            value[[t, j]] = (prob[[j, t]] + eps).ln() + best;
        }
    }
    let mut state = vec![0usize; n_steps];
    let last = value.row(n_steps - 1);
    let mut best = f64::NEG_INFINITY;
    for (j, &v) in last.iter().enumerate() {
        if v > best {
            best = v;
            state[n_steps - 1] = j;
        }
    }
    for t in (0..n_steps - 1).rev() {
        state[t] = ptr[[t + 1, state[t + 1]]];
    }
    Ok(state)
}

/// Self-loop transition matrix: stay with probability `prob`, else move uniformly.
pub fn transition_loop(n_states: usize, prob: f64) -> Result<Array2<f64>> {
    if n_states < 2 {
        return param(format!(
            "n_states={n_states} must be a positive integer > 1"
        ));
    }
    if !(0.0..=1.0).contains(&prob) {
        return param(format!("prob={prob} must have values in the range [0, 1]"));
    }
    let off = (1.0 - prob) / (n_states - 1) as f64;
    Ok(Array2::from_shape_fn((n_states, n_states), |(i, j)| {
        if i == j { prob } else { off }
    }))
}

/// Banded (local) transition matrix with a window of `width` states around the diagonal
/// (`librosa.sequence.transition_local`).
pub fn transition_local(
    n_states: usize,
    width: usize,
    window: &Window,
    wrap: bool,
) -> Result<Array2<f64>> {
    if n_states < 2 {
        return param(format!(
            "n_states={n_states} must be a positive integer > 1"
        ));
    }
    if width < 1 {
        return param(format!("width={width} must be at least 1"));
    }
    let mut t = Array2::<f64>::zeros((n_states, n_states));
    for i in 0..n_states {
        let row = pad_center(get_window(window, width, false)?.view(), n_states)?;
        let shift = n_states / 2 + i + 1;
        let mut rolled = vec![0.0; n_states];
        for (k, v) in row.iter().enumerate() {
            rolled[(k + shift) % n_states] = *v;
        }
        if !wrap {
            let hi = n_states.min(i + width / 2 + 1);
            rolled[hi..].iter_mut().for_each(|v| *v = 0.0);
            let lo = i.saturating_sub(width / 2);
            rolled[..lo].iter_mut().for_each(|v| *v = 0.0);
        }
        for (k, v) in rolled.iter().enumerate() {
            t[[i, k]] = *v;
        }
    }
    for mut r in t.rows_mut() {
        let s = r.sum();
        r.mapv_inplace(|v| v / s);
    }
    Ok(t)
}
