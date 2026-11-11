import { motion, useInView } from "motion/react";
import { useRef } from "react";
import type { Bench } from "../data/content";

/** One model: a thin bar and number for speed, a plain number for memory. */
export function Ledger({ model, rows }: { model: string; rows: Bench[] }) {
	const ref = useRef<HTMLTableElement>(null);
	const inView = useInView(ref, { once: true, margin: "-10%" });
	const max = Math.max(...rows.map((r) => r.tokS));

	return (
		<table ref={ref} className="w-full border-collapse text-left">
			<caption className="pb-5 text-left text-3xl font-medium tracking-[-0.04em] md:text-4xl">{model}</caption>
			<thead>
				<tr className="border-y border-line text-[14px] text-paper/75">
					<th scope="col" className="py-3 pr-4 font-normal">
						Setup
					</th>
					<th scope="col" className="w-[38%] py-3 pr-4 font-normal">
						<span className="sr-only">Speed, drawn to scale</span>
					</th>
					<th scope="col" className="py-3 pr-4 text-right font-normal">
						Tokens / sec
					</th>
					<th scope="col" className="py-3 text-right font-normal">
						Peak GB
					</th>
				</tr>
			</thead>
			<tbody>
				{rows.map((r, i) => (
					<tr key={r.label} className="border-b border-line">
						<th scope="row" className="py-4 pr-4 text-[17px] font-normal leading-tight tracking-[-0.01em]">
							{r.label}
							{r.reference && <span className="block text-[14px] text-paper/70">Baseline</span>}
							{r.acceptance && (
								<span className="block text-[14px] text-paper/70">{r.acceptance} guesses kept</span>
							)}
						</th>
						<td className="py-4 pr-4 align-middle" aria-hidden>
							<motion.div
								initial={{ width: 0 }}
								animate={{ width: inView ? `${(r.tokS / max) * 100}%` : 0 }}
								transition={{ duration: 1, delay: 0.1 + i * 0.07, ease: [0.16, 1, 0.3, 1] }}
								className={`h-[10px] rounded-r-[4px] ${r.reference ? "bg-paper/30" : "bg-paper"}`}
							/>
						</td>
						<td className="py-4 pr-4 text-right text-[26px] font-medium tabular-nums tracking-[-0.03em]">
							{r.tokS.toFixed(1)}
						</td>
						<td className="py-4 text-right text-[26px] font-medium tabular-nums tracking-[-0.03em] text-paper/70">
							{r.peakGB.toFixed(1)}
						</td>
					</tr>
				))}
			</tbody>
		</table>
	);
}
