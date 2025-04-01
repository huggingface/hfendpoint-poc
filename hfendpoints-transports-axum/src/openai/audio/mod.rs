use crate::openai::python::AutomaticSpeechRecognitionEndpoint;
use hfendpoints_binding_python::ImportablePyModuleBuilder;
use pyo3::prelude::PyModule;
use pyo3::{Bound, PyResult, Python};

pub(crate) mod transcription;

pub const AUDIO_TAG: &str = "Audio";
pub const AUDIO_DESC: &str = "Learn how to turn audio into text or text into audio.";

/// Bind hfendpoints.openai.audio submodule into the exported Python wheel
pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    let module = ImportablePyModuleBuilder::new(py, name)?
        .defaults()?
        .add_class::<AutomaticSpeechRecognitionEndpoint>()?
        .finish();

    Ok(module)
}
