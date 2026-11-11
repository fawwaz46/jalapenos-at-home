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

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Qwen weights, the Python reference fixture and a Metal device"]
fn gpu_matches_python() {
    check_reference(Backend::Gpu);
}

// Speculative decoding on the GPU-resident backend must reproduce plain greedy
// output, for bf16 and MXFP4, through rejected drafts and KV cache rollback.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires both Qwen models, the Python reference fixture and a Metal device"]
fn gpu_speculative_matches_greedy() {
    use engine::{DraftSchedule, generate_scheduled};

    let (expected, _) = reference_load("reference_qwen3-1.7b_0.safetensors", "hidden_1");
    let prompt = &expected[..20];
    for suffix in ["", "-MXFP4"] {
        let target = Model::load(&format!("Qwen3-1.7B{suffix}"), Backend::Gpu).unwrap();
        let draft = Model::load(&format!("Qwen3-0.6B{suffix}"), Backend::Gpu).unwrap();
        let greedy = target.generate(prompt, 40);
        if suffix.is_empty() {
            assert_eq!(greedy[..expected.len()], expected[..]);
        }
        for schedule in [DraftSchedule::fixed(5), DraftSchedule::adaptive(5, 8).unwrap()] {
            let result = generate_scheduled(&target, &draft, prompt, 40, schedule);
            assert_eq!(result.tokens, greedy, "{suffix}");
            assert!(result.accepted < result.drafted, "expected some rejections");
        }
    }
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

#[test]
#[ignore = "requires the Qwen models and Python reference fixture"]
fn adaptive_scheduler_matches_greedy_and_handles_short_budgets() {
    use engine::{DraftSchedule, generate_scheduled};

    let (expected, _) = reference_load("reference_qwen3-1.7b_0.safetensors", "hidden_1");
    let prompt = &expected[..20];
    for suffix in ["", "-MXFP4"] {
        let target =
            Model::load_with_prefill(&format!("Qwen3-1.7B{suffix}"), Backend::Cpu, Backend::Cpu)
                .unwrap();
        let draft =
            Model::load_with_prefill(&format!("Qwen3-0.6B{suffix}"), Backend::Cpu, Backend::Cpu)
                .unwrap();
        let greedy = target.generate(prompt, 15);
        for count in [0, 1, 15] {
            let result = generate_scheduled(
                &target,
                &draft,
                prompt,
                count,
                DraftSchedule::adaptive(5, 8).unwrap(),
            );
            assert_eq!(result.tokens, greedy[..prompt.len() + count]);
            assert_eq!(result.rounds, result.verify.passes);
            assert_eq!(result.rounds, result.round_stats.len());
            assert_eq!(
                result.drafted,
                result
                    .round_stats
                    .iter()
                    .map(|round| round.drafted)
                    .sum::<usize>()
            );
            assert_eq!(
                result.accepted,
                result
                    .round_stats
                    .iter()
                    .map(|round| round.accepted)
                    .sum::<usize>()
            );
            assert!(
                result
                    .round_stats
                    .iter()
                    .all(|round| (1..=8).contains(&round.gamma))
            );
            if count > 0 {
                assert_eq!(result.prefill.passes, 2);
                assert!(result.draft.passes < result.drafted + result.rounds);
            }
            if count == 15 {
                assert!(
                    result
                        .round_stats
                        .iter()
                        .any(|round| round.next_gamma != round.gamma)
                );
            }
        }
        let result = generate_scheduled(
            &target,
            &target,
            prompt,
            6,
            DraftSchedule::adaptive(2, 4).unwrap(),
        );
        assert_eq!(result.tokens, greedy[..prompt.len() + 6]);
        assert_eq!(result.round_stats[0].next_gamma, 3);
        // The last accepted proposal and the bonus share one draft pass.
        // The two rounds use one and two other drafting passes respectively.
        assert_eq!(result.draft.passes, 4);
    }
}
