use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;

use serde::Deserialize;

pub struct Tensor{
    pub shape: Vec<usize>,
    pub data: Vec<u16>,
}

pub struct Weights{
   pub model_name: String,
   pub tensors: HashMap<String, Tensor>

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
    let folder = format!("../weights/{}", name);

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
        file.read_exact(&mut first8).expect("could not read 8 bytes");

        let header_len = u64::from_le_bytes(first8);

        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes).expect("could not read da header");
        let header_text = String::from_utf8(header_bytes).expect("dat shi is NOT a string gang");

        let mut header: HashMap<String, serde_json::Value> = serde_json::from_str(&header_text).expect("header is not valid JSON");
        header.remove("__metadata__");

        let mut contents: HashMap<String, HeaderEntry> = HashMap::new();
        for (name, value) in header {
            let entry: HeaderEntry = serde_json::from_value(value).expect("bad header entry");
            contents.insert(name, entry);
        }

        let mut sorted: Vec<(String, HeaderEntry)> = contents.into_iter().collect();
        sorted.sort_by_key(|pair| pair.1.data_offsets[0]);

        for(name, entry) in sorted{
            let sz: usize = entry.data_offsets[1] - entry.data_offsets[0];
            let mut buff = vec![0u8; sz];

            file.read_exact(&mut buff).expect("couldn't read bytes");
            let mut temp = Vec::new();

            for chunk in buff.chunks_exact(2){
                let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                temp.push(value)
            }
            let tensor = Tensor { shape: entry.shape, data: temp };
            tensor_map.insert(name, tensor);
        }
    }

    Weights { model_name: name.to_string(), tensors: tensor_map }
}
