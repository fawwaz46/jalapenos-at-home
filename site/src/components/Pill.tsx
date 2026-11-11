import { motion } from "motion/react";
import type { ReactNode } from "react";

type PillProps = {
	children: ReactNode;
	onClick?: () => void;
	href?: string;
	variant?: "light" | "ghost" | "dark";
	arrow?: boolean;
	className?: string;
};

const styles = {
	light: "bg-paper text-dark",
	ghost: "bg-black/5 text-paper backdrop-blur-md border border-black/10",
	dark: "bg-dark text-paper",
};

/** Behold-style pill, fully rounded. */
export function Pill({ children, onClick, href, variant = "light", arrow = true, className = "" }: PillProps) {
	const cls = `group inline-flex h-12 items-center gap-3 rounded-full px-6 text-[16px] font-medium tracking-[-0.01em] transition-opacity hover:opacity-85 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-paper ${styles[variant]} ${className}`;
	const inner = (
		<>
			<span>{children}</span>
			{arrow && (
				<span className="relative block h-3 w-4 overflow-hidden" aria-hidden>
					<svg viewBox="0 0 16 12" className="absolute inset-0 h-3 w-4 transition-transform duration-300 group-hover:translate-x-4">
						<path d="M0 6h14M9 1l5 5-5 5" stroke="currentColor" strokeWidth="1.5" fill="none" />
					</svg>
					<svg viewBox="0 0 16 12" className="absolute inset-0 h-3 w-4 -translate-x-4 transition-transform duration-300 group-hover:translate-x-0">
						<path d="M0 6h14M9 1l5 5-5 5" stroke="currentColor" strokeWidth="1.5" fill="none" />
					</svg>
				</span>
			)}
		</>
	);
	if (href)
		return (
			<motion.a whileTap={{ scale: 0.96 }} href={href} target="_blank" rel="noreferrer" className={cls}>
				{inner}
			</motion.a>
		);
	return (
		<motion.button whileTap={{ scale: 0.96 }} type="button" onClick={onClick} className={cls}>
			{inner}
		</motion.button>
	);
}

/** Small mono label with a hairline, used to open every section. */
export function Eyebrow({ index, children }: { index: string; children: ReactNode }) {
	return (
		<div className="flex items-center gap-3 font-mono text-[12px] uppercase tracking-[0.08em] text-gray">
			<span className="text-paper">{index}</span>
			<span className="h-px w-8 bg-line" />
			<span>{children}</span>
		</div>
	);
}
