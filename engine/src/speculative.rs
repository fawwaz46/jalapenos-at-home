use crate::Model;

pub struct SpeculativeResult {
    pub tokens: Vec<usize>,
    pub drafted: usize,
    pub accepted: usize,
    pub rounds: usize,
}

// Both models must share token IDs. The target always has the final say.
pub fn generate_speculative(
    target: &Model,
    draft: &Model,
    prompt: &[usize],
    count: usize,
    gamma: usize,
) -> SpeculativeResult {
    assert!(gamma > 0);
    let mut result = SpeculativeResult {
        tokens: prompt.to_vec(),
        drafted: 0,
        accepted: 0,
        rounds: 0,
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
    let mut first_prediction = Some(target.predict(prompt, &mut target_cache, false)[0]);
    let mut pending = None;
    let mut draft_next = draft.predict(prompt, &mut draft_cache, false)[0];

    while result.tokens.len() < limit {
        let start = result.tokens.len();
        let mut proposed = Vec::with_capacity(gamma.min(limit - start));
        for _ in 0..gamma.min(limit - start) {
            proposed.push(draft_next);
            draft_next = draft.predict(&[draft_next], &mut draft_cache, false)[0];
            if proposed.last() == Some(&target.eos()) {
                break;
            }
        }

        // After the first round, verify the previous correction and the new
        // proposals together so the target only needs one pass per round.
        let mut verification: Vec<_> = pending.take().into_iter().collect();
        verification.extend_from_slice(&proposed);
        let checked = target.predict(&verification, &mut target_cache, true);
        let mut predictions = first_prediction.take().into_iter().chain(checked);
        let mut accepted = 0;
        let mut next = predictions.next().unwrap();
        while proposed.get(accepted) == Some(&next) {
            accepted += 1;
            next = predictions.next().unwrap();
        }

        result.rounds += 1;
        result.drafted += proposed.len();
        result.accepted += accepted;
        result.tokens.extend_from_slice(&proposed[..accepted]);
        if result.tokens.last() == Some(&target.eos()) && accepted > 0
            || result.tokens.len() == limit
        {
            break;
        }

        // On a mismatch drop the rejected token and everything conditioned on it.
        // If all proposals matched, this is a bonus token from the target.
        target_cache.truncate(start + accepted);
        draft_cache.truncate(start + accepted);
        result.tokens.push(next);
        if next == target.eos() || result.tokens.len() == limit {
            break;
        }
        pending = Some(next);
        draft_next = draft.predict(&[next], &mut draft_cache, false)[0];
    }
    result
}
