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

//write a function that takes in a file and returns the list of tokenIDs and a Tensor for hidden_0
pub fn embedding_load(file_name: &str) -> (Vec<usize>, Tensor){
    let path: String = format!("../data/{}", file_name);

    let mut file = File::open(&path).expect("could not open file");
    let mut first8 = [0u8; 8];
    file.read_exact(&mut first8).expect("could not read 8 bytes");

    let header_len = u64::from_le_bytes(first8);
    let mut header_bytes = vec![0u8; header_len as usize];

    file.read_exact(&mut header_bytes).expect("could not read header bytes into buffer");

    let header_text = String::from_utf8(header_bytes).expect("could not turn header bytes into string");
    let mut header: HashMap<String, serde_json::Value> = serde_json::from_str(&header_text).expect("could not store values in map, maybe a json error?");

    header.remove("__metadata__");

    let mut contents: HashMap<String, HeaderEntry> = HashMap::new();

    for (name, value) in header{
        let entry: HeaderEntry = serde_json::from_value(value).expect("could not destructure into header value");
        contents.insert(name, entry);
    }

    let mut sorted: Vec<(String, HeaderEntry)> = contents.into_iter().collect();
    sorted.sort_by_key(|pair| pair.1.data_offsets[0]);

    let mut tok_ids = Vec::new();
    let mut hidden_0 = Tensor { shape: Vec::new(), data: Vec::new() };

    for (name, entry) in sorted{
        
        let sz: usize = entry.data_offsets[1] - entry.data_offsets[0];
        let mut buff = vec![0u8; sz];

        
        file.read_exact(&mut buff).expect("could not read tensor bytes");
        
        let mut temp = Vec::new();
        
        if name == "token_ids"{
            
            for chunk in buff.chunks_exact(8){
                let value = i64::from_le_bytes(chunk.try_into().unwrap()) as usize;
                tok_ids.push(value)
            }
        } else if name == "hidden_0"{
            for chunk in buff.chunks_exact(2){
                let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                temp.push(value)
            }
            hidden_0 = Tensor { shape: entry.shape, data: temp };
        }
    }

    (tok_ids, hidden_0)
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
pub fn embed(weights: &Weights, token_ids: &Vec<usize>) -> Tensor {
    let table = &weights.tensors["model.embed_tokens.weight"];
    let row_len = table.shape[1];

    let mut output = Vec::new();

    for &id in token_ids {
        let start = id * row_len;
        let end = start + row_len;
        output.extend_from_slice(&table.data[start..end]);
    }

    Tensor { shape: vec![token_ids.len(), row_len], data: output }
}
