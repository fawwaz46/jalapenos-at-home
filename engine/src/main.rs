use std::time::Instant;

use engine::{embed, embedding_load, load};

fn main(){
    let start = Instant::now();
    let weights = load("Qwen3-1.7B");
    println!("{}: {} tensors loaded in {:.2?}", weights.model_name, weights.tensors.len(), start.elapsed());

    let (token_ids, hidden_0) = embedding_load("reference_qwen3-1.7b_0.safetensors");
    println!("token ids: {} (first 3: {:?})", token_ids.len(), &token_ids[..3]);

    let output = embed(&weights, &token_ids);
    println!("embedding shape: {:?}", output.shape);            // expect [35, 2048]
    println!("embedding first 3: {:?}", &output.data[..3]);     // expect [48404, 48224, 15377]
    println!("matches hidden_0: {}", output.data == hidden_0.data);
}
