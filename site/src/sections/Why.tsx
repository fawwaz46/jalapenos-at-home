import { motion } from "motion/react";
import { Pill } from "../components/Pill";

const rise = (delay: number) => ({
	initial: { y: 50, opacity: 0, filter: "blur(10px)" },
	animate: { y: 0, opacity: 1, filter: "blur(0px)" },
	transition: { duration: 1.1, delay, ease: [0.16, 1, 0.3, 1] as const },
});

export function Why({ onNext }: { onNext: () => void }) {
	return (
		<section className="relative flex h-full w-full items-center justify-center overflow-hidden px-6">
			<div className="flex max-w-5xl flex-col items-center text-center">
				<motion.h2 {...rise(0.6)} className="text-[16vw] font-medium leading-[0.84] tracking-[-0.06em] md:text-[9vw]">
					Why I built it
				</motion.h2>
				<motion.p
					{...rise(0.8)}
					className="mt-8 max-w-[40ch] text-lg leading-snug tracking-[-0.02em] text-paper/65 md:text-2xl"
				>
					I wanted to understand inference from the inside: what actually happens between a prompt and the next
					token, and where the time really goes.
				</motion.p>
				<motion.p
					{...rise(0.95)}
					className="mt-5 max-w-[40ch] text-lg leading-snug tracking-[-0.02em] text-paper/65 md:text-2xl"
				>
					I was told the way to learn something deeply is to build something difficult. So I took the ideas behind a
					data-center chip and rebuilt them on a laptop.
				</motion.p>
				<motion.div {...rise(1.15)} className="mt-12">
					<Pill onClick={onNext}>Find me</Pill>
				</motion.div>
			</div>
		</section>
	);
}
