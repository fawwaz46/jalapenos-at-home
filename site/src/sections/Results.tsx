import { motion } from "motion/react";
import type { ReactNode } from "react";
import { Ledger } from "../components/Ledger";
import { Eyebrow, Pill } from "../components/Pill";
import { Visualizer } from "../components/Visualizer";
import { benches, quality, versions } from "../data/content";

const ease = [0.16, 1, 0.3, 1] as const;

function Reveal({ children, className = "", delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
	return (
		<motion.div
			initial={{ opacity: 0, y: 40 }}
			whileInView={{ opacity: 1, y: 0 }}
			viewport={{ once: true, margin: "-8%" }}
			transition={{ duration: 0.9, delay, ease }}
			className={className}
		>
			{children}
		</motion.div>
	);
}

const headline = [
	{ value: "27", label: "tokens / sec", note: "Qwen3-8B with 4-bit weights, in 6.6 GB" },
	{ value: "3.4×", label: "less memory", note: "22.5 → 6.6 GB peak on 8B with MXFP4" },
	{ value: "1:1", label: "with Transformers", note: "Same tokens as PyTorch on 1.7B and 8B" },
	{ value: "+11%", label: "perplexity", note: "What 4-bit weights cost in quality on 8B" },
];

const limits = [
	["Test size", "One prompt, one Mac"],
	["Decoding", "Greedy only"],
	["Baseline", "PyTorch at 16-bit, not 4-bit"],
];

const setups = ["bf16", "MXFP4", "bf16 + drafting", "MXFP4 + drafting"];

export function Results({ onNext }: { onNext: () => void }) {
	return (
		<section className="scroll-area h-full w-full">
			<div className="px-6 pt-24 pb-24 md:px-10 lg:pt-28">
				<Eyebrow index="04">Results</Eyebrow>
				<h2 className="mt-8 text-[17vw] font-medium leading-[0.8] tracking-[-0.06em] md:text-[10.5vw]">
					What came
					<br />
					of it
				</h2>

				<div className="mt-16 grid grid-cols-2 border-t border-line lg:grid-cols-4">
					{headline.map((h, i) => (
						<Reveal key={h.label} delay={i * 0.08} className="border-b border-line py-6 pr-6 lg:border-b-0">
							<div className="text-[18vw] font-medium leading-none tracking-[-0.06em] md:text-[7vw]">{h.value}</div>
							<div className="mt-3 text-lg font-medium tracking-[-0.02em]">{h.label}</div>
							<div className="mt-1 text-[15px] text-paper/70">{h.note}</div>
						</Reveal>
					))}
				</div>

				<Block index="A" title="Watching it think" lede="A small model guesses ahead. The big model checks every guess in one pass and keeps the ones it agrees with.">
					<Visualizer />
				</Block>

				<Block
					index="B"
					title="The benchmark"
					lede="Two model sizes, 16-bit and 4-bit weights, with and without drafting, next to PyTorch on the same Mac."
				>
					<div className="mb-8 flex flex-wrap gap-x-8 gap-y-2 text-[15px]">
						<span className="flex items-center gap-2.5">
							<span className="h-[10px] w-8 rounded-r-[4px] bg-paper" aria-hidden />
							This engine
						</span>
						<span className="flex items-center gap-2.5">
							<span className="h-[10px] w-8 rounded-r-[4px] bg-paper/30" aria-hidden />
							Baseline: PyTorch (Transformers) on the same Mac
						</span>
					</div>
					<div className="grid gap-x-16 gap-y-14 lg:grid-cols-2">
						<Ledger model="Qwen3-1.7B" rows={benches["1.7B"]} />
						<Ledger model="Qwen3-8B" rows={benches["8B"]} />
					</div>
					<p className="mt-6 max-w-[80ch] text-[14px] leading-snug text-paper/70">
						64 tokens per run, median of 5 runs. Every setup produced the same output.
					</p>

					<Reveal className="mt-20">
						<h4 className="text-3xl font-medium tracking-[-0.04em] md:text-4xl">Three versions of the engine</h4>
						<p className="mt-2 max-w-[60ch] text-[16px] leading-snug text-paper/75">
							Qwen3-1.7B, tokens per second. The output stayed identical through every change.
						</p>
						<div className="mt-5 overflow-x-auto">
							<table className="w-full min-w-[640px] border-collapse border-t border-line text-left">
								<thead>
									<tr className="border-b border-line text-[14px] text-paper/75">
										<th className="py-3 pr-4 font-normal">Version</th>
										{setups.map((name) => (
											<th key={name} className="py-3 pr-4 text-right font-normal">
												{name}
											</th>
										))}
									</tr>
								</thead>
								<tbody>
									{versions.map((v, row) => (
										<tr
											key={v.name}
											className={`border-b border-line ${row === versions.length - 1 ? "" : "text-paper/70"}`}
										>
											<td className="py-4 pr-4">
												<div className="text-[17px] tracking-[-0.01em]">{v.name}</div>
												<div className="text-[14px] opacity-85">{v.detail}</div>
											</td>
											{v.speeds.map((speed, i) => (
												<td
													key={setups[i]}
													className="py-4 pr-4 text-right text-[26px] font-medium tabular-nums tracking-[-0.03em]"
												>
													{speed.toFixed(1)}
												</td>
											))}
										</tr>
									))}
								</tbody>
							</table>
						</div>
					</Reveal>
				</Block>

				<Block
					index="C"
					title="What 4 bits cost"
					lede="4-bit weights are faster and smaller, but less accurate."
				>
					<Quality />
				</Block>

				<div className="mt-24 flex justify-start">
					<Pill onClick={onNext}>Why I built it</Pill>
				</div>
			</div>
		</section>
	);
}

function Block({ index, title, lede, children }: { index: string; title: string; lede: string; children: ReactNode }) {
	return (
		<section className="mt-28">
			<Reveal className="mb-8 grid gap-4 md:grid-cols-12 md:items-end">
				<div className="md:col-span-6">
					<div className="font-mono text-[12px] uppercase tracking-[0.08em] text-gray">{index}</div>
					<h3 className="mt-3 text-4xl font-medium tracking-[-0.04em] md:text-5xl">{title}</h3>
				</div>
				{lede && <p className="max-w-[44ch] text-lg leading-snug text-paper/60 md:col-span-6">{lede}</p>}
			</Reveal>
			{children}
		</section>
	);
}

function Quality() {
	return (
		<div className="grid gap-x-16 gap-y-10 lg:grid-cols-12">
			<Reveal className="lg:col-span-7">
				<div className="overflow-x-auto">
					<table className="w-full min-w-[520px] border-collapse border-t border-line text-left">
						<thead>
							<tr className="border-b border-line text-[14px] text-paper/75">
								<th className="py-3 pr-4 font-normal">Model</th>
								<th className="py-3 pr-4 text-right font-normal">Perplexity, bf16</th>
								<th className="py-3 pr-4 text-right font-normal">Perplexity, MXFP4</th>
								<th className="py-3 text-right font-normal">Change</th>
							</tr>
						</thead>
						<tbody>
							{quality.map((q) => (
								<tr key={q.model} className="border-b border-line">
									<td className="py-4 pr-4 text-[17px] tracking-[-0.01em]">{q.model}</td>
									<td className="py-4 pr-4 text-right text-[26px] font-medium tabular-nums tracking-[-0.03em]">{q.bf16.toFixed(1)}</td>
									<td className="py-4 pr-4 text-right text-[26px] font-medium tabular-nums tracking-[-0.03em]">{q.mxfp4.toFixed(1)}</td>
									<td className="py-4 text-right text-[26px] font-medium tabular-nums tracking-[-0.03em]">
										+{Math.round((q.mxfp4 / q.bf16 - 1) * 100)}%
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
				<p className="mt-4 text-[14px] leading-snug text-paper/70">
					Lower perplexity is better.
				</p>
			</Reveal>
			<Reveal delay={0.1} className="lg:col-span-5">
				<dl className="border-t border-line">
					<div className="border-b border-line py-3 text-[14px] text-paper/75">What this doesn't cover</div>
					{limits.map(([label, value]) => (
						<div key={label} className="flex min-h-[72px] items-center justify-between gap-6 border-b border-line">
							<dt className="text-[17px] tracking-[-0.01em]">{label}</dt>
							<dd className="text-right text-[17px] tracking-[-0.01em] text-paper/70">{value}</dd>
						</div>
					))}
				</dl>
			</Reveal>
		</div>
	);
}
