use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    LayerCache, Tensor, TensorF32, Weights, attention, bf16_to_f32, embed, matmul, mlp, rms_norm,
    try_load,
};

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "lowercase")]
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
    backend: Backend,
    prefill_backend: Backend,
    weights: Weights,
    config: Config,
}

pub(crate) struct KvCache {
    layers: Vec<LayerCache>,
    pub(crate) len: usize,
}

impl KvCache {
    pub(crate) fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        for layer in &mut self.layers {
            layer.truncate(len);
        }
        self.len = len;
    }
}

impl Model {
    pub fn load(name: &str, backend: Backend) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_with_prefill(name, backend, backend)
    }

    pub fn load_with_prefill(
        name: &str,
        backend: Backend,
        prefill_backend: Backend,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../weights")
            .join(name)
            .join("config.json");
        let config = serde_json::from_slice(&fs::read(config)?)?;
        #[cfg(target_os = "macos")]
        let metal =
            if matches!(backend, Backend::Metal) || matches!(prefill_backend, Backend::Metal) {
                Some(crate::metal::Metal::new()?)
            } else {
                None
            };
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, Backend::Metal) || matches!(prefill_backend, Backend::Metal) {
            return Err("Metal is only available on macOS".into());
        }
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal,
            backend,
            prefill_backend,
            weights: try_load(name)?,
            config,
        })
    }

    fn project(&self, input: &TensorF32, weights: &Tensor, backend: Backend) -> TensorF32 {
        #[cfg(target_os = "macos")]
        if matches!(backend, Backend::Metal) {
            return self
                .metal
                .as_ref()
                .expect("Metal was initialized at load time")
                .matmul(input, weights);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = backend;
        matmul(input, weights)
    }

    pub(crate) fn cache(&self) -> KvCache {
        KvCache {
            layers: (0..self.config.num_hidden_layers)
                .map(|_| LayerCache::default())
                .collect(),
            len: 0,
        }
    }

    pub(crate) fn context_limit(&self) -> usize {
        self.config.max_position_embeddings
    }

    pub(crate) fn eos(&self) -> usize {
        self.config.eos_token_id
    }

    pub fn hidden(&self, tokens: &[usize], layers: usize) -> TensorF32 {
        self.append_hidden(tokens, layers, &mut self.cache(), self.prefill_backend)
    }

    fn append_hidden(
        &self,
        tokens: &[usize],
        layers: usize,
        cache: &mut KvCache,
        backend: Backend,
    ) -> TensorF32 {
        assert!(!tokens.is_empty() && cache.len + tokens.len() <= self.context_limit());
        assert!(layers <= self.config.num_hidden_layers);

        let embedded = embed(&self.weights, tokens);
        let mut x = TensorF32 {
            shape: embedded.shape,
            data: embedded.data.into_iter().map(bf16_to_f32).collect(),
        };
        let project = |input: &TensorF32, weights: &Tensor| self.project(input, weights, backend);
        for layer in 0..layers {
            let prefix = format!("model.layers.{layer}");
            let normalized = rms_norm(
                &x,
                &self.weights.tensors[&format!("{prefix}.input_layernorm.weight")],
            );
            let attn = attention(
                &normalized,
                &self.weights,
                layer,
                cache.len,
                &mut cache.layers[layer],
                &project,
            );
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
        cache.len += tokens.len();
        x
    }

    pub fn next_token(&self, tokens: &[usize]) -> usize {
        self.predict(tokens, &mut self.cache(), false)[0]
    }

    // Each returned token is the prediction after the corresponding input row.
    // Ordinary generation only needs the last row, while verification needs all of them.
    pub(crate) fn predict(&self, tokens: &[usize], cache: &mut KvCache, all: bool) -> Vec<usize> {
        let backend = if cache.len == 0 {
            self.prefill_backend
        } else {
            self.backend
        };
        let mut hidden = self.append_hidden(tokens, self.config.num_hidden_layers, cache, backend);
        let width = hidden.shape[1];
        if !all {
            let last_row = hidden.data.len() - width;
            hidden.data.copy_within(last_row.., 0);
            hidden.data.truncate(width);
            hidden.shape[0] = 1;
        }
        let normalized = rms_norm(&hidden, &self.weights.tensors["model.norm.weight"]);
        let weights = self
            .weights
            .tensors
            .get("lm_head.weight")
            .unwrap_or(&self.weights.tensors["model.embed_tokens.weight"]);
        let logits = self.project(&normalized, weights, backend);
        logits
            .data
            .chunks_exact(logits.shape[1])
            .map(argmax)
            .collect()
    }

    pub fn generate(&self, prompt: &[usize], count: usize) -> Vec<usize> {
        let mut tokens = prompt.to_vec();
        let count = count.min(self.context_limit().saturating_sub(prompt.len()));
        if count == 0 {
            return tokens;
        }

        let mut cache = self.cache();
        let mut next = self.predict(prompt, &mut cache, false)[0];
        for i in 0..count {
            tokens.push(next);
            if next == self.eos() || i + 1 == count {
                break;
            }
            next = self.predict(&[next], &mut cache, false)[0];
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

impl std::str::FromStr for Backend {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            _ => Err("backend must be cpu or metal"),
        }
    }
}

#[cfg(test)]
impl Model {
    pub(crate) fn constant_for_test(token: usize) -> Self {
        use std::collections::HashMap;

        let one = (1.0_f32.to_bits() >> 16) as u16;
        let mut head = vec![0; 8];
        head[token * 2] = one;
        Self {
            #[cfg(target_os = "macos")]
            metal: None,
            backend: Backend::Cpu,
            prefill_backend: Backend::Cpu,
            config: Config {
                num_hidden_layers: 0,
                max_position_embeddings: 6,
                eos_token_id: 3,
            },
            weights: Weights {
                model_name: "constant".into(),
                tensors: HashMap::from([
                    (
                        "model.embed_tokens.weight".into(),
                        Tensor::bf16(vec![4, 2], [one, 0].repeat(4)),
                    ),
                    (
                        "model.norm.weight".into(),
                        Tensor::bf16(vec![2], vec![one, one]),
                    ),
                    ("lm_head.weight".into(), Tensor::bf16(vec![4, 2], head)),
                ]),
            },
        }
    }
}
