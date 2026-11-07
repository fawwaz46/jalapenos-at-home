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

#[test]
#[ignore = "requires both Qwen models and the generated Python reference fixture"]
fn speculative_matches_greedy_after_rejections() {
    use std::time::Instant;

    use engine::generate_speculative;

    let target = Model::load("Qwen3-1.7B", Backend::Cpu).unwrap();
    let draft = Model::load("Qwen3-0.6B", Backend::Cpu).unwrap();
    let (expected, _) = reference_load("reference_qwen3-1.7b_0.safetensors", "hidden_1");
    let prompt = &expected[..20];
    let count = expected.len() - prompt.len();

    let started = Instant::now();
    let greedy = target.generate(prompt, count);
    let greedy_time = started.elapsed();
    let started = Instant::now();
    let result = generate_speculative(&target, &draft, prompt, count, 5);
    let speculative_time = started.elapsed();
    assert_eq!(greedy, expected);
    assert_eq!(result.tokens, greedy);
    assert!(result.accepted > 0 && result.accepted < result.drafted);
    assert!(result.rounds < count);
    eprintln!(
        "greedy: {greedy_time:.2?}, speculative: {speculative_time:.2?}, accepted: {}/{}",
        result.accepted, result.drafted
    );

    let all_matched = generate_speculative(&target, &target, prompt, 3, 2);
    assert_eq!(all_matched.tokens, expected[..23]);
    assert_eq!(all_matched.accepted, 2);
    assert_eq!(all_matched.rounds, 1);
}
