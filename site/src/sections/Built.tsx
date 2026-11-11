import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { Eyebrow, Pill } from "../components/Pill";
import { buildSteps } from "../data/content";

const ease = [0.16, 1, 0.3, 1] as const;

export function Built({ onNext }: { onNext: () => void }) {
	const [open, setOpen] = useState<number | null>(0);

	return (
		<section className="scroll-area h-full w-full">
			<div className="grid min-h-full grid-cols-1 gap-12 px-6 pt-24 pb-16 md:px-10 lg:grid-cols-12 lg:gap-6 lg:pt-28">
				<div className="lg:col-span-5 lg:sticky lg:top-28 lg:self-start">
					<Eyebrow index="02">The engine</Eyebrow>
					<h2 className="mt-8 font-medium tracking-[-0.06em]">
						<span className="block text-[15.2vw] leading-[0.8] lg:text-[8.7vw]">How it's</span>
						<span className="block text-[27vw] leading-[0.78] lg:text-[15.4vw]">built</span>
					</h2>
					<p className="mt-8 max-w-[38ch] text-lg leading-snug tracking-[-0.02em] text-paper/60">
						From a Python answer key, to a transformer written in Rust, to Metal kernels on the GPU. Each layer
						was checked against the one before it.
					</p>
					<div className="mt-10 hidden lg:block">
						<Pill onClick={onNext}>Where it meets Jalapeño</Pill>
					</div>
				</div>

				<div className="lg:col-span-6 lg:col-start-7">
					<ol className="border-t border-line">
						{buildSteps.map((step, i) => {
							const isOpen = open === i;
							return (
								<li key={step.title} className="border-b border-line">
									<button
										type="button"
										onClick={() => setOpen(isOpen ? null : i)}
										aria-expanded={isOpen}
										className="group flex w-full items-baseline gap-5 py-5 text-left md:py-6"
									>
										<span className="w-8 shrink-0 font-mono text-[12px] text-gray">{String(i + 1).padStart(2, "0")}</span>
										<span className="flex-1">
											<span className="block text-2xl font-medium tracking-[-0.03em] transition-colors group-hover:text-chili-hot md:text-[2rem] md:leading-tight">
												{step.title}
											</span>
											<span className="mt-1 block text-[15px] text-paper/50">{step.summary}</span>
										</span>
										<motion.span
											animate={{ rotate: isOpen ? 45 : 0 }}
											transition={{ duration: 0.4, ease }}
											className="relative block h-4 w-4 shrink-0 self-center"
											aria-hidden
										>
											<span className="absolute top-1/2 left-0 h-px w-4 bg-paper" />
											<span className="absolute top-0 left-1/2 h-4 w-px bg-paper" />
										</motion.span>
									</button>
									<AnimatePresence initial={false}>
										{isOpen && (
											<motion.div
												key="body"
												initial={{ height: 0, opacity: 0 }}
												animate={{ height: "auto", opacity: 1 }}
												exit={{ height: 0, opacity: 0 }}
												transition={{ duration: 0.55, ease }}
												className="overflow-hidden"
											>
												<div className="grid gap-8 pb-8 pl-13 md:grid-cols-[1fr_auto]">
													<div className="space-y-4 text-[17px] leading-relaxed text-paper/75">
														{step.body.map((p) => (
															<p key={p.slice(0, 24)}>{p}</p>
														))}
													</div>
													<dl className="space-y-4 md:w-56">
														{step.facts.map(([k, v]) => (
															<div key={k}>
																<dt className="font-mono text-[11px] uppercase tracking-[0.08em] text-gray">{k}</dt>
																<dd className="mt-1 text-lg font-medium tracking-[-0.02em]">{v}</dd>
															</div>
														))}
													</dl>
												</div>
											</motion.div>
										)}
									</AnimatePresence>
								</li>
							);
						})}
					</ol>
					<div className="mt-12 lg:hidden">
						<Pill onClick={onNext}>Where it meets Jalapeño</Pill>
					</div>
				</div>
			</div>
		</section>
	);
}
