mod common;

use common::reference_load;
use engine::{embed, load};

// The embedding is a pure row copy, so it must equal Python's hidden_0 exactly.
#[test]
fn embedding_matches_hidden_0() {
    let weights = load("Qwen3-1.7B");
    let (token_ids, hidden_0) = reference_load("reference_qwen3-1.7b_0.safetensors", "hidden_0");

    let output = embed(&weights, &token_ids);

    assert_eq!(output.shape, vec![35, 2048]);
    assert!(
        output.data == hidden_0.data,
        "embedding output differs from hidden_0"
    );
}
