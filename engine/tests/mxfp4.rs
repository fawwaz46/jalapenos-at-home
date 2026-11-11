use engine::{Tensor, TensorF32, matmul};

fn packed_weight() -> Tensor {
    // One block, all 16 E2M1 codes repeated twice, scaled by 2^-1.
    let row = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
    Tensor::quantized(vec![1, 32], row.repeat(2), vec![126]).unwrap()
}

#[test]
fn matrix_multiply_decodes_nibbles_and_scales() {
    let weights = packed_weight();
    let mut input = vec![0.0; 32 * 32];
    for i in 0..32 {
        input[i * 32 + i] = 1.0;
    }
    let output = matmul(
        &TensorF32 {
            shape: vec![32, 32],
            data: input,
        },
        &weights,
    );
    let levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
    let expected: Vec<_> = (0..32)
        .map(|i| {
            let value = levels[i % 8];
            if i % 16 >= 8 { -value } else { value }
        })
        .collect();
    assert_eq!(output.shape, [32, 1]);
    assert_eq!(output.data, expected);
    assert!(weights.data.is_empty());
}

#[test]
fn invalid_quantized_storage_is_rejected() {
    assert!(Tensor::quantized(vec![1, 31], vec![0; 16], vec![127]).is_err());
    assert!(Tensor::quantized(vec![1, 32], vec![0; 15], vec![127]).is_err());
    assert!(Tensor::quantized(vec![1, 32], vec![0; 16], vec![255]).is_err());
}

#[test]
#[ignore = "run python/check_mxfp4.py first to create the quantized reference fixture"]
fn quantized_model_matches_python_hidden_state() {
    use engine::{Backend, Model};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/reference_qwen3-1.7b_mxfp4.json");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let prompt: Vec<usize> = serde_json::from_value(reference["prompt"].clone()).unwrap();
    let expected: Vec<Vec<f32>> = serde_json::from_value(reference["hidden_1"].clone()).unwrap();
    let name = reference["model"].as_str().unwrap();
    {
        let weights = engine::load(name);
        assert_eq!(weights.tensors.len(), 311);
        for weight in weights.tensors.values() {
            if weight.shape.len() == 2 {
                assert!(weight.data.is_empty());
                assert!(weight.mxfp4.is_some());
            } else {
                assert!(weight.mxfp4.is_none());
                assert_eq!(weight.data.len(), weight.shape[0]);
            }
        }
    }
    let expected: Vec<_> = expected.into_iter().flatten().collect();
    let mut backends = vec![Backend::Cpu];
    if cfg!(target_os = "macos") {
        backends.push(Backend::Gpu);
    }
    for backend in backends {
        let model = Model::load(name, backend).unwrap();
        let actual = model.hidden(&prompt, 1);
        assert_eq!(actual.data.len(), expected.len());
        let error: f64 = actual
            .data
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum();
        let scale: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
        let relative = (error / scale).sqrt();
        assert!(relative < 1e-5, "{backend:?}: relative error {relative}");
    }
}

#[test]
fn matrix_rows_and_blocks_use_different_scales() {
    let weights = Tensor::quantized(vec![2, 64], vec![0x77; 64], vec![127, 126, 125, 124]).unwrap();
    let mut input = vec![0.0; 3 * 64];
    input[0] = 1.0;
    input[64 + 32] = 1.0;
    input[128] = 1.0;
    input[128 + 32] = 1.0;
    let result = matmul(
        &TensorF32 {
            shape: vec![3, 64],
            data: input,
        },
        &weights,
    );
    assert_eq!(result.shape, [3, 2]);
    assert_eq!(result.data, [6.0, 1.5, 3.0, 0.75, 9.0, 2.25]);
}
