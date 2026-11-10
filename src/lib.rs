#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
// Index loops mirror the reference (NumPy) formulas and read more clearly in DSP code.
#![allow(clippy::needless_range_loop)]

pub mod audio;
pub mod beat;
pub mod constantq;
pub mod convert;
pub mod decompose;
pub mod effects;
pub mod error;
pub mod feature;
mod fft;
pub mod filters;
pub mod onset;
pub mod pitch;
pub mod sequence;
mod signal;
pub mod spectrum;
pub mod util;
pub mod window;

pub use audio::*;
pub use constantq::{CqtParams, cqt, vqt};
pub use convert::*;
pub use error::{Error, Result};
pub use pitch::{
    PyinParams, PyinResult, TuningParams, YinParams, estimate_tuning, piptrack, pitch_tuning, pyin,
    yin,
};
pub use spectrum::*;
pub use util::PadMode;
pub use window::{Window, get_window};
