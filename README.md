# jalapenos at home

i wanted to see what it takes to run qwen from scratch on my mac. the python version gives me something to check against. the rust version loads the same weights and generates text. the metal path does the big matrix multiplies on the gpu, and there's a cpu path too.

from the repo, build with `cargo build --release --manifest-path engine/Cargo.toml`. with qwen3-1.7b in `weights/Qwen3-1.7B` and the python deps installed, run `python python/run_rust.py "To make a simple loaf of bread" --backend cpu`. use `--backend metal` to try it on the gpu. the weights stay out of git.

add `--draft` to let qwen3-0.6b guess a few tokens at a time and have the bigger model check them. put the small model in `weights/Qwen3-0.6B` first. it prints how many guesses got through and drops the rejected ones from the kv cache.

still want to add mxfp4, schedule prefill/draft/verify across the cpu and gpu, and show the whole thing running live with benchmarks.
