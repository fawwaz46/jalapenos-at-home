use std::collections::HashMap;
use std::fs;

use engine::{bf16_to_f32, load};
use serde::Deserialize;

// One item of the Python answer key (python/weight_sums.py).
#[derive(Deserialize)]
struct Expected {
    shape: Vec<usize>,
    sum: f64,
}

// Every tensor's count, shape and sum must match what Python read from the same files.
#[test]
fn weights_match_python() {
    let weights = load("Qwen3-1.7B");

    let key_text = fs::read_to_string("../data/weight_sums_qwen3-1.7b.json").expect("could not read answer key");
    let expected: HashMap<String, Expected> = serde_json::from_str(&key_text).expect("answer key is not valid JSON");

    assert_eq!(weights.tensors.len(), expected.len(), "tensor count");

    for (name, exp) in &expected {
        let tensor = &weights.tensors[name];
        assert_eq!(tensor.shape, exp.shape, "shape of {name}");

        let mut sum = 0.0f64;
        for &bits in &tensor.data {
            sum += bf16_to_f32(bits) as f64;
        }

        let diff = (sum - exp.sum).abs() / exp.sum.abs().max(1.0);
        assert!(diff < 1e-6, "sum of {name}: got {sum}, expected {}", exp.sum);
    }
}
