use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;

use serde::Deserialize;

#[cfg(target_os = "macos")]
mod cpu;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod speculative;
pub use model::{Backend, Model};
pub use speculative::{SpeculativeResult, generate_speculative};

// Fixed for every Qwen3 model we use (0.6B, 1.7B, 8B).
const HEAD_DIM: usize = 128; // numbers per head
const ROPE_THETA: f32 = 1_000_000.0; // base for RoPE's rotation speeds

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<u16>,
}
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Weights {
    pub model_name: String,
    pub tensors: HashMap<String, Tensor>,
}

// bf16 is the top half of an f32, so shifting it up gives the exact value.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[derive(Deserialize, Debug)]
struct HeaderEntry {
    shape: Vec<usize>,
    data_offsets: Vec<usize>,
}

// Loads every tensor from every .safetensors file in ../weights/<name>.
pub fn load(name: &str) -> Weights {
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../weights")
        .join(name);

    // every .safetensors file in the model folder = the shards
    let mut paths = Vec::new();
    for item in fs::read_dir(&folder).expect("could not open model folder") {
        let path = item.expect("could not read folder item").path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            paths.push(path);
        }
    }
    paths.sort();

    // made once, outside the loop, so both shards add to the same map
    let mut tensor_map: HashMap<String, Tensor> = HashMap::new();

    for path in paths {
        let mut file = File::open(&path).expect("could not load da file");
        let mut first8 = [0u8; 8];
        file.read_exact(&mut first8)
            .expect("could not read 8 bytes");

        let header_len = u64::from_le_bytes(first8);

        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)
            .expect("could not read da header");
        let header_text = String::from_utf8(header_bytes).expect("dat shi is NOT a string gang");

        let mut header: HashMap<String, serde_json::Value> =
            serde_json::from_str(&header_text).expect("header is not valid JSON");
        header.remove("__metadata__");

        let mut contents: HashMap<String, HeaderEntry> = HashMap::new();
        for (name, value) in header {
            let entry: HeaderEntry = serde_json::from_value(value).expect("bad header entry");
            contents.insert(name, entry);
        }

        let mut sorted: Vec<(String, HeaderEntry)> = contents.into_iter().collect();
        sorted.sort_by_key(|pair| pair.1.data_offsets[0]);

        for (name, entry) in sorted {
            let sz: usize = entry.data_offsets[1] - entry.data_offsets[0];
            let mut buff = vec![0u8; sz];

            file.read_exact(&mut buff).expect("couldn't read bytes");
            let mut temp = Vec::new();

            for chunk in buff.chunks_exact(2) {
                let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                temp.push(value)
            }
            let tensor = Tensor {
                shape: entry.shape,
                data: temp,
            };
            tensor_map.insert(name, tensor);
        }
    }

    Weights {
        model_name: name.to_string(),
        tensors: tensor_map,
    }
}

/*
1. Write a function that takes the loaded Weights and the token ID list, and returns a Tensor.
2. Get model.embed_tokens.weight from the weights' tensors map.
3. Get the row length from that tensor's shape. It's the second number, 2048.
4. Make an empty list for the output numbers.
5. For each token ID: its row starts at ID × row length and ends at start + row length. Take that section of the embedding table's data and add all of it to the output list.
6. Return a Tensor with shape [number of IDs, row length] and the output list as its data.
*/
// Copies each token ID's row out of the embedding table, stacked into one Tensor.
pub fn embed(weights: &Weights, token_ids: &[usize]) -> Tensor {
    let table = &weights.tensors["model.embed_tokens.weight"];
    let row_len = table.shape[1];

    let mut output = Vec::new();

    for &id in token_ids {
        let start = id * row_len;
        let end = start + row_len;
        output.extend_from_slice(&table.data[start..end]);
    }

    Tensor {
        shape: vec![token_ids.len(), row_len],
        data: output,
    }
}

#[cfg(target_os = "macos")]
pub use cpu::matmul;

#[cfg(not(target_os = "macos"))]
pub fn matmul(x: &TensorF32, w: &Tensor) -> TensorF32 {
    let tok_rows = x.shape[0];
    let tok_len = x.shape[1];
    let weight_rows = w.shape[0];

    let mut output = Vec::new();

    for t in 0..tok_rows {
        for r in 0..weight_rows {
            let mut sum = 0.0;
            for i in 0..tok_len {
                sum += x.data[t * tok_len + i] * bf16_to_f32(w.data[r * tok_len + i]);
            }
            output.push(sum);
        }
    }

    TensorF32 {
        shape: vec![tok_rows, weight_rows],
        data: output,
    }
}

pub fn rms_norm(x: &TensorF32, w: &Tensor) -> TensorF32 {
    let tok_rows = x.shape[0];
    let row_len = x.shape[1];

    let mut output = Vec::new();
    for t in 0..tok_rows {
        let mut sum = 0.0;
        for i in 0..row_len {
            sum += x.data[t * row_len + i] * x.data[t * row_len + i];
        }
        let average = sum / (row_len as f32);
        let size = (average + 1e-6).sqrt();

        for i in 0..row_len {
            output.push(x.data[t * row_len + i] / size * bf16_to_f32(w.data[i]));
        }
    }

    TensorF32 {
        shape: vec![tok_rows, row_len],
        data: output,
    }
}
// Rotates each (i, i+64) pair in every 128-number head by an angle that depends on the token's position.
// Works for q (16 heads) and k (8 heads); the head count is row length / 128.
fn rope(x: &TensorF32, offset: usize) -> TensorF32 {
    let tok_rows = x.shape[0];
    let row_len = x.shape[1];
    let head_dim = HEAD_DIM;
    let half = head_dim / 2;
    let heads = row_len / head_dim;

    let mut out = x.data.clone();

    for t in 0..tok_rows {
        let position = (offset + t) as f32;
        for h in 0..heads {
            let head_start = t * row_len + h * head_dim;
            for i in 0..half {
                let freq = 1.0 / ROPE_THETA.powf(i as f32 / half as f32);
                let angle = position * freq;
                let a = x.data[head_start + i];
                let b = x.data[head_start + i + half];
                out[head_start + i] = a * angle.cos() - b * angle.sin();
                out[head_start + i + half] = b * angle.cos() + a * angle.sin();
            }
        }
    }

    TensorF32 {
        shape: vec![tok_rows, row_len],
        data: out,
    }
}

#[derive(Default)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    width: usize,
}

impl LayerCache {
    fn truncate(&mut self, tokens: usize) {
        self.keys.truncate(tokens * self.width);
        self.values.truncate(tokens * self.width);
    }
}

fn attention(
    x: &TensorF32,
    weights: &Weights,
    layer: usize,
    offset: usize,
    cache: &mut LayerCache,
    project: &impl Fn(&TensorF32, &Tensor) -> TensorF32,
) -> TensorF32 {
    //grab weights from tensor
    let q_proj = &weights.tensors[&format!("model.layers.{layer}.self_attn.q_proj.weight")];
    let k_proj = &weights.tensors[&format!("model.layers.{layer}.self_attn.k_proj.weight")];
    let v_proj = &weights.tensors[&format!("model.layers.{layer}.self_attn.v_proj.weight")];
    let q_norm = &weights.tensors[&format!("model.layers.{layer}.self_attn.q_norm.weight")];
    let k_norm = &weights.tensors[&format!("model.layers.{layer}.self_attn.k_norm.weight")];

    //compute q k and v for the tokens
    let q = project(x, q_proj);
    let k = project(x, k_proj);
    let v = project(x, v_proj);

    // split into heads
    let tok_rows = x.shape[0];
    let q_heads = q.shape[1] / HEAD_DIM;
    let k_heads = k.shape[1] / HEAD_DIM;

    //store the heads
    let q_split = TensorF32 {
        shape: vec![tok_rows * q_heads, HEAD_DIM],
        data: q.data,
    };
    let k_split = TensorF32 {
        shape: vec![tok_rows * k_heads, HEAD_DIM],
        data: k.data,
    };

    //normalize the heads
    let q_normed = rms_norm(&q_split, q_norm);
    let k_normed = rms_norm(&k_split, k_norm);

    //put back to one row per token, so rope's row number = token position
    let q_back = TensorF32 {
        shape: vec![tok_rows, q_heads * HEAD_DIM],
        data: q_normed.data,
    };
    let k_back = TensorF32 {
        shape: vec![tok_rows, k_heads * HEAD_DIM],
        data: k_normed.data,
    };

    //run rope on them
    let q_rope = rope(&q_back, offset);
    let k_rope = rope(&k_back, offset);

    //attention setup
    let q_len = q_heads * HEAD_DIM; // 2048: one token's row in q_rope and output
    let kv_len = k_heads * HEAD_DIM; // 1024: one token's row in k_rope and v
    debug_assert_eq!(cache.keys.len(), offset * kv_len);
    cache.width = kv_len;
    cache.keys.extend(k_rope.data);
    cache.values.extend(v.data);
    let group = q_heads / k_heads; // 2: q heads per kv head
    let scale = (HEAD_DIM as f32).sqrt(); // 11.31: scores get divided by this
    let mut output = vec![0.0f32; tok_rows * q_len]; // [35, 2048] of zeros, filled by the loop

    for t in 0..tok_rows {
        // every token
        for h in 0..q_heads {
            // every q head
            let kv_h = h / group; // which k/v head this q head reads
            let q_start = t * q_len + h * HEAD_DIM; // where this head starts in q_rope and output

            //scores: one per visible token, how well this q head matches that token's k head
            let mut scores = Vec::new();
            for s in 0..=offset + t {
                let k_start = s * kv_len + kv_h * HEAD_DIM; // skip s tokens, then kv_h heads
                let mut dot = 0.0;
                for i in 0..HEAD_DIM {
                    dot += q_rope.data[q_start + i] * cache.keys[k_start + i];
                }
                scores.push(dot / scale);
            }

            //softmax: make every score positive (exp), so they can become shares that add to 1
            let mut max = f32::NEG_INFINITY;
            for &sc in &scores {
                if sc > max {
                    max = sc;
                } // biggest score
            }
            let mut total = 0.0;
            for sc in scores.iter_mut() {
                *sc = (*sc - max).exp(); // minus max = overflow safety, same shares
                total += *sc;
            }

            //mix v: add each token's v head, times its share, into this head's spot in output
            for (s, score) in scores.into_iter().enumerate() {
                let share = score / total;
                let v_start = s * kv_len + kv_h * HEAD_DIM; // same jump as k
                for i in 0..HEAD_DIM {
                    output[q_start + i] += share * cache.values[v_start + i];
                }
            }
        }
    }

    //o_proj: blend the 16 heads' results together, [35, 2048] -> [35, 2048]
    let o_proj = &weights.tensors[&format!("model.layers.{layer}.self_attn.o_proj.weight")];
    let out = TensorF32 {
        shape: vec![tok_rows, q_len],
        data: output,
    };
    project(&out, o_proj)
}

fn mlp(
    x: &TensorF32,
    weights: &Weights,
    layer: usize,
    project: &impl Fn(&TensorF32, &Tensor) -> TensorF32,
) -> TensorF32 {
    let prefix = format!("model.layers.{layer}.mlp");
    let mut gate = project(x, &weights.tensors[&format!("{prefix}.gate_proj.weight")]);
    let up = project(x, &weights.tensors[&format!("{prefix}.up_proj.weight")]);
    for (gate, up) in gate.data.iter_mut().zip(up.data) {
        *gate = (*gate / (1.0 + (-*gate).exp())) * up;
    }
    project(
        &gate,
        &weights.tensors[&format!("{prefix}.down_proj.weight")],
    )
}
