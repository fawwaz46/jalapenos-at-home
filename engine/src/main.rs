use std::time::Instant;

use engine::load;

fn main(){
    let start = Instant::now();
    let weights = load("Qwen3-1.7B");
    println!("{}: {} tensors loaded in {:.2?}", weights.model_name, weights.tensors.len(), start.elapsed());
}
