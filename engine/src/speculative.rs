use std::time::Instant;

use serde::Serialize;

use crate::{DraftSchedule, Model};

#[derive(Default, Serialize)]
pub struct PhaseStats {
    pub passes: usize,
    pub elapsed_ms: f64,
}

#[derive(Serialize)]
pub struct RoundStats {
    pub gamma: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub next_gamma: usize,
}

pub struct SpeculativeResult {
    pub tokens: Vec<usize>,
    pub drafted: usize,
    pub accepted: usize,
    pub rounds: usize,
    pub prefill: PhaseStats,
    pub draft: PhaseStats,
    pub verify: PhaseStats,
    pub round_stats: Vec<RoundStats>,
}

pub fn generate_speculative(
    target: &Model,
    draft: &Model,
    prompt: &[usize],
    count: usize,
    gamma: usize,
) -> SpeculativeResult {
    generate_scheduled(target, draft, prompt, count, DraftSchedule::fixed(gamma))
}

// Both models must share token IDs. The target always has the final say.
pub fn generate_scheduled(
    target: &Model,
    draft: &Model,
    prompt: &[usize],
    count: usize,
    mut schedule: DraftSchedule,
) -> SpeculativeResult {
    let mut result = SpeculativeResult {
        tokens: prompt.to_vec(),
        drafted: 0,
        accepted: 0,
        rounds: 0,
        prefill: PhaseStats::default(),
        draft: PhaseStats::default(),
        verify: PhaseStats::default(),
        round_stats: Vec::new(),
    };
    let limit = prompt
        .len()
        .saturating_add(count)
        .min(target.context_limit())
        .min(draft.context_limit());
    if result.tokens.len() >= limit {
        return result;
    }

    let mut target_cache = target.cache();
    let mut draft_cache = draft.cache();
    let started = Instant::now();
    let mut first_prediction = Some(target.predict(prompt, &mut target_cache, false)[0]);
    let mut draft_next = draft.predict(prompt, &mut draft_cache, false)[0];
    result.prefill.passes = 2;
    result.prefill.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut pending = None;

    while result.tokens.len() < limit {
        let start = result.tokens.len();
        let gamma = schedule.gamma();
        let budget = gamma.min(limit - start);
        let mut proposed = Vec::with_capacity(budget);
        let started = Instant::now();
        for i in 0..budget {
            proposed.push(draft_next);
            // Don't run a forward pass just to predict beyond the draft budget.
            if i + 1 == budget || draft_next == target.eos() {
                break;
            }
            draft_next = draft.predict(&[draft_next], &mut draft_cache, false)[0];
            result.draft.passes += 1;
        }
        result.draft.elapsed_ms += started.elapsed().as_secs_f64() * 1000.0;

        // Verify the previous correction and the new proposals in one pass.
        let mut verification: Vec<_> = pending.take().into_iter().collect();
        verification.extend_from_slice(&proposed);
        let started = Instant::now();
        let checked = target.predict(&verification, &mut target_cache, true);
        result.verify.passes += 1;
        result.verify.elapsed_ms += started.elapsed().as_secs_f64() * 1000.0;
        let mut predictions = first_prediction.take().into_iter().chain(checked);
        let mut accepted = 0;
        let mut next = predictions.next().unwrap();
        while proposed.get(accepted) == Some(&next) {
            accepted += 1;
            next = predictions.next().unwrap();
        }

        schedule.observe(accepted, proposed.len());
        result.round_stats.push(RoundStats {
            gamma,
            drafted: proposed.len(),
            accepted,
            next_gamma: schedule.gamma(),
        });
        result.rounds += 1;
        result.drafted += proposed.len();
        result.accepted += accepted;
        result.tokens.extend_from_slice(&proposed[..accepted]);
        if result.tokens.last() == Some(&target.eos()) && accepted > 0
            || result.tokens.len() == limit
        {
            break;
        }

        target_cache.truncate(start + accepted);
        draft_cache.truncate((start + accepted).min(draft_cache.len));
        result.tokens.push(next);
        if next == target.eos() || result.tokens.len() == limit {
            break;
        }
        pending = Some(next);
        // If every proposal matched, the last accepted proposal hasn't entered
        // the draft cache yet. Batch it with the target's bonus token.
        let draft_tail = &result.tokens[draft_cache.len..];
        let started = Instant::now();
        draft_next = draft.predict(draft_tail, &mut draft_cache, false)[0];
        result.draft.passes += 1;
        result.draft.elapsed_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::generate_scheduled;
    use crate::{DraftSchedule, Model};

    #[test]
    fn accepted_and_corrected_eos_stop_generation() {
        let target = Model::constant_for_test(3);
        for draft_token in [1, 3] {
            let draft = Model::constant_for_test(draft_token);
            let result = generate_scheduled(
                &target,
                &draft,
                &[0],
                5,
                DraftSchedule::adaptive(3, 5).unwrap(),
            );
            assert_eq!(result.tokens, [0, 3]);
            assert_eq!(result.rounds, 1);
        }
    }

    #[test]
    fn rejected_draft_eos_and_gamma_one_preserve_target_output() {
        let target = Model::constant_for_test(2);
        let draft = Model::constant_for_test(3);
        for gamma in [1, 5] {
            let result = generate_scheduled(&target, &draft, &[0], 20, DraftSchedule::fixed(gamma));
            assert_eq!(result.tokens, target.generate(&[0], 20));
            assert_eq!(result.tokens.len(), 6);
            assert_eq!(result.accepted, 0);
        }
    }

    #[test]
    fn zero_budget_and_context_boundary_do_not_overrun() {
        let target = Model::constant_for_test(2);
        let empty = generate_scheduled(&target, &target, &[0], 0, DraftSchedule::fixed(5));
        assert_eq!(empty.tokens, [0]);
        assert_eq!(empty.prefill.passes, 0);
        let prompt = [0; 5];
        let result = generate_scheduled(&target, &target, &prompt, 20, DraftSchedule::fixed(5));
        assert_eq!(result.tokens, [0, 0, 0, 0, 0, 2]);
        assert_eq!(result.draft.passes, 0);
    }
}
