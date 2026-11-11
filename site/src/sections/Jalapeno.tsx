import { motion } from "motion/react";
import type { ReactNode } from "react";
import { Eyebrow, Pill } from "../components/Pill";
import { jalapenoMap } from "../data/content";

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

export function Jalapeno({ onNext }: { onNext: () => void }) {
	return (
		<section className="scroll-area h-full w-full">
			<div className="px-6 pt-24 pb-24 md:px-10 lg:pt-28">
				<Eyebrow index="03">The chip</Eyebrow>
				<div className="mt-8 grid gap-8 md:grid-cols-12 md:items-end">
					<h2 className="text-[15vw] font-medium leading-[0.8] tracking-[-0.06em] md:col-span-8 md:text-[9.5vw]">
						Where it meets
						<br />
						Jalapeño
					</h2>
					<p className="max-w-[36ch] text-lg leading-snug tracking-[-0.02em] text-paper/70 md:col-span-4 md:pb-2">
						Jalapeño is OpenAI's inference chip, shown at Hot Chips 2026. These are its ideas, and how they carry
						over to a laptop.
					</p>
				</div>

				<div className="mt-16">
				<div className="border-t border-line">
					{jalapenoMap.map((row, i) => (
						<Reveal
							key={row.idea}
							delay={i * 0.05}
							className="grid gap-3 border-b border-line py-6 md:grid-cols-12 md:gap-6"
						>
							<div className="text-2xl font-medium tracking-[-0.03em] md:col-span-3">{row.idea}</div>
							<div className="md:col-span-4">
								<div className="font-mono text-[11px] uppercase tracking-[0.08em] text-gray">Jalapeño</div>
								<p className="mt-1 text-[16px] leading-snug text-paper/75">{row.chip}</p>
							</div>
							<div className="md:col-span-3">
								<div className="font-mono text-[11px] uppercase tracking-[0.08em] text-gray">On the MacBook</div>
								<p className="mt-1 text-[16px] leading-snug text-paper/75">{row.here}</p>
							</div>
							<div className="md:col-span-2 md:text-right">
								<span
									className={`inline-block whitespace-nowrap rounded-full px-3 py-1 text-[13px] font-medium ${
										row.status === "match"
											? "bg-paper text-dark"
											: row.status === "partial"
												? "border border-black/40 text-paper"
												: "border border-dashed border-black/40 text-paper/60"
									}`}
								>
									{row.status === "match" ? "Carried over" : row.status === "partial" ? "Partly" : "Not yet"}
								</span>
							</div>
						</Reveal>
					))}
				</div>
				</div>

				<div className="mt-16 flex justify-start">
					<Pill onClick={onNext}>What came of it</Pill>
				</div>
			</div>
		</section>
	);
}
