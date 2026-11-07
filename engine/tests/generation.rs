mod common;

use common::reference_load;
use engine::{Backend, Model, bf16_to_f32};

fn check_reference(backend: Backend) {
    let model = Model::load("Qwen3-1.7B", backend).unwrap();
    let (tokens, expected) = reference_load("reference_qwen3-1.7b_0.safetensors", "hidden_1");

    let first_layer = model.hidden(&tokens, 1);
    assert_eq!(expected.shape[0], 1);
    assert_eq!(first_layer.shape, expected.shape[1..]);
    let mut error = 0.0_f64;
    let mut scale = 0.0_f64;
    for (&actual, bits) in first_layer.data.iter().zip(expected.data) {
        let reference = f64::from(bf16_to_f32(bits));
        error += (f64::from(actual) - reference).powi(2);
        scale += reference.powi(2);
    }
    let relative_error = (error / scale).sqrt();
    assert!(
        relative_error < 0.01,
        "first layer relative error: {relative_error}"
    );

    let generated = model.generate(&tokens[..20], tokens.len() - 20);
    assert_eq!(generated, tokens);
}

#[test]
#[ignore = "requires Qwen weights and the generated Python reference fixture"]
fn cpu_matches_python() {
    check_reference(Backend::Cpu);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Qwen weights, the Python reference fixture and a Metal device"]
fn metal_matches_python() {
    check_reference(Backend::Metal);
}
