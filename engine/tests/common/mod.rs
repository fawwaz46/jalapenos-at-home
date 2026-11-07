use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use engine::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct HeaderEntry {
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub fn reference_load(file_name: &str, hidden_name: &str) -> (Vec<usize>, Tensor) {
    let mut file = File::open(format!("../data/{file_name}")).expect("open reference fixture");
    let mut first8 = [0; 8];
    file.read_exact(&mut first8).unwrap();
    let header_len = u64::from_le_bytes(first8);
    let mut header_bytes = vec![0; header_len as usize];
    file.read_exact(&mut header_bytes).unwrap();

    let mut header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes).unwrap();
    let mut read_tensor = |name: &str| {
        let entry: HeaderEntry = serde_json::from_value(header.remove(name).unwrap()).unwrap();
        let [start, end] = entry.data_offsets;
        file.seek(SeekFrom::Start(8 + header_len + start)).unwrap();
        let mut bytes = vec![0; (end - start) as usize];
        file.read_exact(&mut bytes).unwrap();
        (entry.shape, bytes)
    };

    let (_, token_bytes) = read_tensor("token_ids");
    let token_ids = token_bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize)
        .collect();
    let (shape, hidden_bytes) = read_tensor(hidden_name);
    let data = hidden_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (token_ids, Tensor { shape, data })
}
