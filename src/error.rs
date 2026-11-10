//! Error type shared by every function in the crate.

use std::fmt;

/// Errors returned by librosa-rs.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An argument was out of range or inconsistent (librosa's `ParameterError`).
    Parameter(String),
    /// Reading an audio file failed.
    Io(std::io::Error),
    /// The audio file could not be decoded.
    Decode(String),
    /// The resampler could not be built or failed.
    Resample(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parameter(m) => write!(f, "parameter error: {m}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Decode(m) => write!(f, "decode error: {m}"),
            Error::Resample(m) => write!(f, "resample error: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn param<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Parameter(msg.into()))
}
