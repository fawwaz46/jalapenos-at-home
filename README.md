# jalapenos at home

i wanted to see what it takes to run qwen from scratch on my mac. the python version gives me something to check against. the rust version loads the same weights and generates text. the metal path does the big matrix multiplies on the gpu, and there's a cpu path too.

from the repo, build with `cargo build --release --manifest-path engine/Cargo.toml`. with qwen3-1.7b in `weights/Qwen3-1.7B` and the python deps installed, run `python python/run_rust.py "To make a simple loaf of bread" --backend cpu`. use `--backend metal` to try it on the gpu. the weights stay out of git.

add `--draft` to let qwen3-0.6b guess a few tokens at a time and have the bigger model check them. put the small model in `weights/Qwen3-0.6B` first. it prints how many guesses got through and drops the rejected ones from the kv cache.

to make the 4-bit versions, run `python python/quantize_mxfp4.py Qwen3-0.6B Qwen3-1.7B` in the same python environment. this writes separate `-MXFP4` model folders. add `--mxfp4` when running the engine, with or without `--draft`. weights use e2m1 with one e8m0 scale per 32 values. normalization weights stay bf16 and activations stay float32. the metal kernel unpacks weights while multiplying, and the cpu path can check the same files.

the packing follows the [ocp mx format](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), rounding to nearest even and saturating at the largest fp4 value. scales and the retained bf16 vectors make the weight data about 3.76x smaller. output can change after quantization. the cpu path is checked against transformers; gpu validation is still pending.

`python python/check_mxfp4.py` checks the quantized rust output against transformers using unpacked float32 weights. add `--backend metal` to check the gpu path too. run `cargo test --release --manifest-path engine/Cargo.toml --lib -- --ignored` for the metal kernel checks. these need access to a metal device.

add `--adaptive` with `--draft` to adjust how many tokens get guessed each round. it starts at `--gamma` (5 by default), grows by one when all guesses match, and halves when fewer than half match. `--max-gamma` caps it at 8 by default. fixed gamma still works without the adaptive flag.

`--backend` picks where verification runs. `--draft-backend cpu` puts drafting on the cpu, and `--prefill-backend metal` puts prompt processing on the gpu for both models. leave them out to use each model's normal backend. the scheduler batches the previous target correction into verification, and combines deferred draft work with the next target token to avoid redundant model passes. it doesn't run the cpu and gpu phases concurrently.

for example, `python python/run_rust.py "To make a simple loaf of bread" --backend metal --draft --draft-backend cpu --prefill-backend metal --adaptive --mxfp4 --stats`. stats go to stderr and show phase timings, forward-pass counts, and each round's draft budget and acceptance. timings measure model execution after loading weights. gpu checks and any speedup claims still need metal validation.

still want to show the whole thing running live with benchmarks.
