// numbers are from runs on the M5 MacBook Air (32 GB), 2026-11-11

export const links = {
	// Not shown on the site yet; add it back to the list in sections/Contact.tsx.
	github: "https://github.com/fawwaz46",
	x: "https://x.com/46waz",
	linkedin: "https://www.linkedin.com/in/fawwaz-abayomi/",
};

export type BuildStep = {
	title: string;
	summary: string;
	body: string[];
	facts: [string, string][];
};

export const buildSteps: BuildStep[] = [
	{
		title: "The answer key",
		summary: "Python + Transformers, saving every layer's output",
		body: [
			"Before writing any engine code, a Python reference runs Qwen3-1.7B on a fixed prompt and saves the hidden state after every layer, plus the final logits. Every Rust component is checked against these files.",
			"A second run in float32 sets the noise floor: how much bf16 rounding alone moves each layer. That turns \"does it match?\" into a number instead of a guess.",
		],
		facts: [
			["Snapshots", "29 layers × 35 tokens"],
			["bf16 noise, last layer", "1.8% relative error"],
			["Cosine similarity", "≥ 0.9996"],
		],
	},
	{
		title: "Loading weights",
		summary: "Parsing safetensors by hand",
		body: [
			"The model's weights ship as safetensors files. Each one opens with a header listing every tensor: its name, its shape, and where its bytes sit in the file. Everything after the header is the raw numbers.",
			"The loader reads that header itself, then copies each tensor straight into memory, keeping it in the original 16-bit format. No Python or ML library involved.",
		],
		facts: [
			["Tensors (1.7B)", "311"],
			["Checksum error vs Python", "2 × 10⁻¹⁶"],
			["Load time", "≈ 1.4 s"],
		],
	},
	{
		title: "The transformer, in Rust",
		summary: "Embedding → 28 layers → logits",
		body: [
			"Each layer is built from small functions: RMSNorm, attention with rotary position embeddings, and a SwiGLU MLP. Attention is grouped-query: 16 query heads share 8 key/value heads, and a KV cache keeps past keys and values so each new token only computes itself.",
			"The last layer projects onto all 151,936 vocabulary entries and picks the most likely next token.",
		],
		facts: [
			["Hidden size", "2,048"],
			["RoPE base", "1,000,000"],
			["Output vs Transformers", "Token-for-token match"],
		],
	},
	{
		title: "Metal kernels",
		summary: "The whole forward pass runs on the GPU",
		body: [
			"The first version sent each matrix multiply to the GPU separately and waited for it: 197 round trips per token, with norms, attention and activations on the CPU in between.",
			"The current version keeps everything on the GPU. Weights, the KV cache and activations live in GPU memory, every step of a pass is recorded into one submission, and only the chosen token ID comes back. Separate kernels handle one token, a handful of tokens, and whole prompts.",
		],
		facts: [
			["GPU submissions per token", "1 (was 197)"],
			["Speed, first → current (1.7B bf16)", "14.2 → 35.1 tok/s"],
			["Output vs Transformers", "Token-for-token match"],
		],
	},
	{
		title: "Speculative decoding",
		summary: "A small model guesses, the big model checks",
		body: [
			"Qwen3-0.6B drafts a few tokens. The big model (Qwen3-1.7B or 8B) checks all of them in one forward pass, keeps every guess up to the first wrong one, and adds its own token. Rejected guesses are removed from both models' KV caches.",
			"The big model always has the final say, so the output is identical to running it alone. That's tested, including rounds where guesses get rejected.",
		],
		facts: [
			["Draft / target", "0.6B / 1.7B or 8B"],
			["Acceptance (recorded runs)", "43–60%"],
			["Output vs plain greedy", "Identical"],
		],
	},
	{
		title: "MXFP4 quantization",
		summary: "4-bit weights, unpacked inside the kernel",
		body: [
			"A converter packs the weights into the OCP microscaling format: 4-bit e2m1 values with one shared e8m0 scale per 32 weights, rounding to nearest even. Norm weights stay bf16 and activations stay float32.",
			"The Metal kernel unpacks the 4-bit values while it multiplies, so the full-size weights never exist in memory. Both CPU and GPU paths are checked against Transformers running the same unpacked weights.",
		],
		facts: [
			["Qwen3-1.7B", "3.8 GB → 1.0 GB"],
			["Qwen3-8B", "15 GB → 4.1 GB"],
			["Format", "e2m1 + e8m0 / 32"],
		],
	},
	{
		title: "The scheduler",
		summary: "How far to guess, and what rides along",
		body: [
			"The previous round's correction is batched into the next verification pass, and leftover draft work rides along with the next token, so no pass is spent on one stray token.",
			"The draft length adapts: it grows by one when every guess lands and halves when fewer than half do, capped at 8.",
		],
		facts: [
			["Start / max guesses", "5 / 8"],
			["Checking 8 guesses vs 1 token (bf16)", "About the same time"],
			["Runs phases concurrently", "Not yet"],
		],
	},
];

export type Bench = {
	label: string;
	tokS: number;
	peakGB: number;
	/** Draft guesses the big model kept, out of those proposed. */
	acceptance?: string;
	/** An outside reference rather than this engine. */
	reference?: boolean;
};

// 64 tokens, greedy, median of 5 runs. speed = 64 / (t(65 tokens) - t(1 token)),
// so load and prompt time cancel out. PyTorch = Transformers on mps, fp16.
export const benches: Record<"1.7B" | "8B", Bench[]> = {
	"1.7B": [
		{ label: "PyTorch, fp16", reference: true, tokS: 36.6, peakGB: 5.9 },
		{ label: "bf16", tokS: 35.1, peakGB: 5.6 },
		{ label: "MXFP4", tokS: 96.2, peakGB: 1.8 },
		{ label: "bf16 + drafting", tokS: 40.7, peakGB: 6.8, acceptance: "33 of 65" },
		{ label: "MXFP4 + drafting", tokS: 61.6, peakGB: 2.3, acceptance: "40 of 70" },
	],
	"8B": [
		{ label: "PyTorch, fp16", reference: true, tokS: 8.2, peakGB: 21.6 },
		{ label: "bf16", tokS: 9.0, peakGB: 22.5 },
		{ label: "MXFP4", tokS: 27.0, peakGB: 6.6 },
		{ label: "bf16 + drafting", tokS: 16.1, peakGB: 24.1, acceptance: "37 of 67" },
		{ label: "MXFP4 + drafting", tokS: 21.0, peakGB: 7.2, acceptance: "41 of 69" },
	],
};

// Qwen3-1.7B tokens/s for each version of the engine
export const versions: { name: string; detail: string; speeds: [number, number, number, number] }[] = [
	{ name: "First version", detail: "One GPU round trip per matrix multiply", speeds: [14.2, 14.9, 8.2, 8.4] },
	{ name: "Whole pass on the GPU", detail: "One submission per token", speeds: [28.6, 33.2, 16.6, 17.6] },
	{ name: "Current", detail: "Kernels for small batches, 4-bit blocks and 32-bit loads", speeds: [35.1, 96.2, 40.7, 61.6] },
];

// perplexity on ~7k tokens of text, bf16 weights vs the MXFP4 files
export const quality: { model: string; bf16: number; mxfp4: number }[] = [
	{ model: "Qwen3-0.6B", bf16: 26.6, mxfp4: 34.0 },
	{ model: "Qwen3-1.7B", bf16: 19.4, mxfp4: 24.0 },
	{ model: "Qwen3-8B", bf16: 11.8, mxfp4: 13.1 },
];

export const jalapenoMap: { idea: string; chip: string; here: string; status: "match" | "partial" | "none" }[] = [
	{
		idea: "MXFP4 math",
		chip: "Native format of the matrix engine — 13.4 PFLOPs of MXFP4 per die.",
		here: "Weights are stored in MXFP4. Apple's GPU has no 4-bit math, so the kernel decodes them to floats as it multiplies.",
		status: "partial",
	},
	{
		idea: "One shared pool",
		chip: "Prefill and decode share one homogeneous pool instead of separate chips.",
		here: "Every phase runs on the one GPU.",
		status: "match",
	},
	{
		idea: "Draft + main together",
		chip: "Draft and main models share the same chips and fabric for locality.",
		here: "Both models live in the same unified memory on the same GPU.",
		status: "match",
	},
	{
		idea: "Data stays put",
		chip: "Weights and KV cache are placed so they don't move.",
		here: "Weights, KV cache and activations stay in GPU memory. Only token IDs come back.",
		status: "match",
	},
	{
		idea: "No fixed overheads",
		chip: "A persistent \"gigakernel\" loops on-device; launch and barrier latency removed.",
		here: "One GPU submission per token instead of 197. Inside it, several hundred kernels still launch one after another.",
		status: "partial",
	},
];

export const sources = [
	{ name: "SemiAnalysis", url: "https://newsletter.semianalysis.com/p/openai-jalapeno-better-than-nvidia" },
	{ name: "Tom's Hardware", url: "https://www.tomshardware.com/tech-industry/artificial-intelligence/hot-chips-2026-openais-jalapeno-ai-asic-unpacked-accelerator-developed-using-ai-achieves-efficiency-and-throughput-gains-against-power-hungry-blackwell" },
	{ name: "IEEE Spectrum", url: "https://spectrum.ieee.org/llms-for-chip-design" },
	{ name: "ServeTheHome", url: "https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/" },
];
