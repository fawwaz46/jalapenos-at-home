use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;

use serde::Deserialize;

#[cfg(target_os = "macos")]
mod cpu;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod mxfp4;
pub use mxfp4::Mxfp4;
mod speculative;
pub use model::{Backend, Model};
pub use speculative::{SpeculativeResult, generate_speculative};

// Fixed for every Qwen3 model we use (0.6B, 1.7B, 8B).
const HEAD_DIM: usize = 128; // numbers per head
const ROPE_THETA: f32 = 1_000_000.0; // base for RoPE's rotation speeds

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<u16>,
    pub mxfp4: Option<Mxfp4>,
}

impl Tensor {
    pub fn bf16(shape: Vec<usize>, data: Vec<u16>) -> Self {
        Self {
            shape,
            data,
            mxfp4: None,
        }
    }

    pub fn quantized(
        shape: Vec<usize>,
        packed: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<Self, &'static str> {
        let mxfp4 = Mxfp4 { packed, scales };
        mxfp4.validate(&shape)?;
        Ok(Self {
            shape,
            data: Vec::new(),
            mxfp4: Some(mxfp4),
        })
    }

    pub(crate) fn values(&self, range: Range<usize>) -> Vec<f32> {
        match &self.mxfp4 {
            Some(mxfp4) => mxfp4.values(range),
            None => self.data[range].iter().copied().map(bf16_to_f32).collect(),
        }
    }
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

#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub fn load(name: &str) -> Weights {
    try_load(name).expect("could not load model weights")
}

fn try_load(name: &str) -> Result<Weights, Box<dyn std::error::Error>> {
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../weights")
        .join(name);
    let mut paths = fs::read_dir(&folder)?
        .map(|item| item.map(|item| item.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
    paths.sort();
    if paths.is_empty() {
        return Err("model directory has no safetensors files".into());
    }
    let mut tensors = HashMap::new();
    let mut packed = HashMap::new();
    let mut scales = HashMap::new();
    for path in paths {
        let mut file = File::open(&path)?;
        let mut first8 = [0; 8];
        file.read_exact(&mut first8)?;
        let header_len = u64::from_le_bytes(first8);
        if header_len > 100_000_000 || header_len + 8 > file.metadata()?.len() {
            return Err("invalid safetensors header length".into());
        }
        let mut header_bytes = vec![0; header_len as usize];
        file.read_exact(&mut header_bytes)?;
        let mut header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)?;
        let metadata = header.remove("__metadata__");
        let quantized = metadata
            .as_ref()
            .and_then(|m| m.get("format"))
            .and_then(|v| v.as_str())
            == Some("jalapenos-mxfp4-v1");
        let mut entries = header
            .into_iter()
            .map(|(name, value)| {
                serde_json::from_value::<HeaderEntry>(value).map(|entry| (name, entry))
            })
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|(_, entry)| entry.data_offsets[0]);
        for (name, entry) in entries {
            let [start, end] = entry.data_offsets;
            let base = 8 + header_len;
            if start > end || end > file.metadata()?.len() - base {
                return Err(format!("invalid tensor offsets: {name}").into());
            }
            let mut bytes = vec![0; (end - start).try_into()?];
            file.seek(SeekFrom::Start(base + start))?;
            file.read_exact(&mut bytes)?;
            let elements = entry
                .shape
                .iter()
                .try_fold(1usize, |size, &dim| size.checked_mul(dim))
                .ok_or("tensor shape overflow")?;
            if quantized && entry.dtype == "U8" {
                if elements != bytes.len() {
                    return Err(format!("invalid packed tensor size: {name}").into());
                }
                if let Some(name) = name.strip_suffix(".mxfp4_packed") {
                    if packed
                        .insert(name.to_owned(), (entry.shape, bytes))
                        .is_some()
                    {
                        return Err("duplicate packed tensor".into());
                    }
                } else if let Some(name) = name.strip_suffix(".mxfp4_scales") {
                    if scales
                        .insert(name.to_owned(), (entry.shape, bytes))
                        .is_some()
                    {
                        return Err("duplicate scale tensor".into());
                    }
                } else {
                    return Err(format!("unexpected U8 tensor: {name}").into());
                }
            } else {
                if !(entry.dtype == "BF16" || quantized && entry.dtype == "U16")
                    || elements.checked_mul(2) != Some(bytes.len())
                {
                    return Err(format!("expected BF16 tensor: {name}").into());
                }
                let data = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                if tensors
                    .insert(name, Tensor::bf16(entry.shape, data))
                    .is_some()
                {
                    return Err("duplicate tensor".into());
                }
            }
        }
    }
    for (name, (shape, packed)) in packed {
        let (scale_shape, scales) = scales.remove(&name).ok_or("missing MXFP4 scales")?;
        if shape.len() != 2 || shape[1] % 16 != 0 || scale_shape != [shape[0], shape[1] / 16] {
            return Err(format!("invalid MXFP4 tensor shape: {name}").into());
        }
        let shape = vec![shape[0], shape[1].checked_mul(2).ok_or("shape overflow")?];
        let tensor = Tensor::quantized(shape, packed, scales)?;
        if tensors.insert(name, tensor).is_some() {
            return Err("duplicate tensor".into());
        }
    }
    if !scales.is_empty() {
        return Err("MXFP4 scales have no matching packed weights".into());
    }
    Ok(Weights {
        model_name: name.to_owned(),
        tensors,
    })
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
        if table.mxfp4.is_some() {
            output.extend(
                table
                    .values(start..end)
                    .into_iter()
                    .map(|value| (value.to_bits() >> 16) as u16),
            );
        } else {
            output.extend_from_slice(&table.data[start..end]);
        }
    }

    Tensor::bf16(vec![token_ids.len(), row_len], output)
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
                sum += x.data[t * tok_len + i]
                    * match &w.mxfp4 {
                        Some(q) => q.value(r * tok_len + i),
                        None => bf16_to_f32(w.data[r * tok_len + i]),
                    };
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
