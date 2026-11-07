use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    Tensor, TensorF32, Weights, attention, bf16_to_f32, embed, load, matmul, mlp, rms_norm,
};

#[derive(Clone, Copy)]
pub enum Backend {
    Cpu,
    Metal,
}

#[derive(Deserialize)]
struct Config {
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    eos_token_id: usize,
}

pub struct Model {
    #[cfg(target_os = "macos")]
    metal: Option<crate::metal::Metal>,
    weights: Weights,
    config: Config,
}

impl Model {
    pub fn load(name: &str, backend: Backend) -> Result<Self, Box<dyn std::error::Error>> {
        let config = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../weights")
            .join(name)
            .join("config.json");
        let config = serde_json::from_slice(&fs::read(config)?)?;
        #[cfg(target_os = "macos")]
        let metal = match backend {
            Backend::Cpu => None,
            Backend::Metal => Some(crate::metal::Metal::new()?),
        };
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, Backend::Metal) {
            return Err("Metal is only available on macOS".into());
        }
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal,
            weights: load(name),
            config,
        })
    }

    fn project(&self, input: &TensorF32, weights: &Tensor) -> TensorF32 {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return metal.matmul(input, weights);
        }
        matmul(input, weights)
    }

    pub fn hidden(&self, tokens: &[usize], layers: usize) -> TensorF32 {
        assert!(!tokens.is_empty() && tokens.len() <= self.config.max_position_embeddings);
        assert!(layers <= self.config.num_hidden_layers);

        let embedded = embed(&self.weights, tokens);
        let mut x = TensorF32 {
            shape: embedded.shape,
            data: embedded.data.into_iter().map(bf16_to_f32).collect(),
        };
        let project = |input: &TensorF32, weights: &Tensor| self.project(input, weights);
        for layer in 0..layers {
            let prefix = format!("model.layers.{layer}");
            let normalized = rms_norm(
                &x,
                &self.weights.tensors[&format!("{prefix}.input_layernorm.weight")],
            );
            let attn = attention(&normalized, &self.weights, layer, &project);
            for (x, attn) in x.data.iter_mut().zip(attn.data) {
                *x += attn;
            }

            let normalized = rms_norm(
                &x,
                &self.weights.tensors[&format!("{prefix}.post_attention_layernorm.weight")],
            );
            let ff = mlp(&normalized, &self.weights, layer, &project);
            for (x, ff) in x.data.iter_mut().zip(ff.data) {
                *x += ff;
            }
        }
        x
    }

    pub fn next_token(&self, tokens: &[usize]) -> usize {
        let hidden = self.hidden(tokens, self.config.num_hidden_layers);
        let width = hidden.shape[1];
        let last = TensorF32 {
            shape: vec![1, width],
            data: hidden.data[hidden.data.len() - width..].to_vec(),
        };
        let last = rms_norm(&last, &self.weights.tensors["model.norm.weight"]);
        let weights = self
            .weights
            .tensors
            .get("lm_head.weight")
            .unwrap_or(&self.weights.tensors["model.embed_tokens.weight"]);
        let logits = self.project(&last, weights);
        argmax(&logits.data)
    }

    pub fn generate(&self, prompt: &[usize], count: usize) -> Vec<usize> {
        let mut tokens = prompt.to_vec();
        for _ in 0..count {
            let token = self.next_token(&tokens);
            tokens.push(token);
            if token == self.config.eos_token_id {
                break;
            }
        }
        tokens
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
