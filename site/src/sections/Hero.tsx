import { motion } from "motion/react";
import { Liquid } from "../components/Liquid";
import { Pill } from "../components/Pill";

const rise = (delay: number) => ({
	initial: { y: 40, opacity: 0, filter: "blur(8px)" },
	animate: { y: 0, opacity: 1, filter: "blur(0px)" },
	transition: { duration: 1.1, delay, ease: [0.16, 1, 0.3, 1] as const },
});

export function Hero({ onNext, leaving }: { onNext: () => void; leaving: boolean }) {
	return (
		<section className="relative flex h-full w-full items-center justify-center overflow-hidden">
			<div className="absolute inset-0">
				<Liquid surge={leaving ? 0.35 : 0} />
			</div>

			<div className="relative z-10 -mt-[18vh] flex flex-col items-center px-6 text-center">
				<motion.h1
					{...rise(0.15)}
					className="text-[13vw] font-medium leading-[0.86] tracking-[-0.055em] md:text-[8.6vw]"
				>
					We Got Jalapenos
					<br />
					at Home
				</motion.h1>
				<motion.p
					{...rise(0.35)}
					className="mt-6 max-w-[34ch] text-lg font-medium leading-tight tracking-[-0.02em] text-paper/60 md:text-2xl"
				>
					Turned my MacBook into OpenAI's Jalapeno chip.
				</motion.p>
				<motion.div {...rise(0.55)} className="mt-10">
					<Pill onClick={onNext}>How it's built</Pill>
				</motion.div>
			</div>

			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: 1, duration: 1 }}
				className="absolute inset-x-0 bottom-6 z-10 flex justify-between px-6 font-mono text-[11px] uppercase tracking-[0.08em] text-paper/70 md:px-10"
			>
				<span>Rust · Metal · Qwen3</span>
				<span className="hidden md:block">MXFP4 · Speculative decoding · Scheduler</span>
				<span>M5 MacBook Air</span>
			</motion.div>
		</section>
	);
}
