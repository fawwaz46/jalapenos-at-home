# jalapenos at home

qwen3 running from scratch on my macbook, in rust. it loads the raw weights, runs the whole forward pass on the gpu with metal, and borrows ideas from openai's jalapeño chip: 4-bit mxfp4 weights, a small model drafting tokens for a big one, and no round trips to the cpu between steps.

## what came of it

m5 macbook air (32 gb), 64 tokens, median of 5 runs. output matches pytorch token for token.

| setup | qwen3-1.7b tok/s | peak gb | qwen3-8b tok/s | peak gb |
|---|---|---|---|---|
| pytorch (fp16) | 36.6 | 5.9 | 8.2 | 21.6 |
| bf16 | 35.1 | 5.6 | 9.0 | 22.5 |
| mxfp4 | 96.2 | 1.8 | 27.0 | 6.6 |
| bf16 + drafting | 40.7 | 6.8 | 16.1 | 24.1 |
| mxfp4 + drafting | 61.6 | 2.3 | 21.0 | 7.2 |

4-bit weights are faster and smaller but less accurate: perplexity goes up 24% on 1.7b and 11% on 8b.

## how it was made

python baseline and rust transformer written by hand. gpu backend, quantization, speed work, tests, and website were written with claude with me directing and testing
