import { motion } from "motion/react";
import { links } from "../data/content";

const items = [
	{ name: "Project Repo", href: links.repo },
	{ name: "X / Twitter", href: links.x },
	{ name: "LinkedIn", href: links.linkedin },
];

export function Contact({ onBack }: { onBack: () => void }) {
	return (
		<section className="relative flex h-full w-full flex-col justify-between bg-chili px-6 pt-24 pb-8 text-paper md:px-10">
			<motion.p
				initial={{ opacity: 0, y: 30 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.9, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
				className="font-mono text-[12px] uppercase tracking-[0.08em] text-paper/70"
			>
				05 — Say hi
			</motion.p>

			<ul>
				{items.map((item, i) => (
					<motion.li
						key={item.name}
						initial={{ opacity: 0, y: 80 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 1, delay: 0.15 + i * 0.1, ease: [0.16, 1, 0.3, 1] }}
						className="border-t border-paper/25 last:border-b"
					>
						<a
							href={item.href}
							target="_blank"
							rel="noreferrer"
							className="group flex items-center justify-between py-3 md:py-4"
						>
							<span className="text-[14vw] font-medium leading-[0.9] tracking-[-0.06em] transition-transform duration-500 group-hover:translate-x-4 md:text-[8vw]">
								{item.name}
							</span>
							<svg
								viewBox="0 0 24 24"
								className="h-10 w-10 transition-transform duration-500 group-hover:-translate-y-1 group-hover:translate-x-1 md:h-16 md:w-16"
								aria-hidden
							>
								<path d="M6 18L18 6M8 6h10v10" stroke="currentColor" strokeWidth="1.5" fill="none" />
							</svg>
						</a>
					</motion.li>
				))}
			</ul>

			<div className="flex items-end justify-between font-mono text-[11px] uppercase tracking-[0.08em]">
				<button type="button" onClick={onBack} className="underline underline-offset-4 hover:opacity-70">
					Back
				</button>
				<span>We got Jalapenos at home</span>
			</div>
		</section>
	);
}
