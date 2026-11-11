import { motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { type Transition, WaveTransition } from "./components/WaveTransition";
import { Built } from "./sections/Built";
import { Contact } from "./sections/Contact";
import { Hero } from "./sections/Hero";
import { Jalapeno } from "./sections/Jalapeno";
import { Results } from "./sections/Results";
import { Why } from "./sections/Why";

const PAGES = ["Home", "How it's built", "Where it meets Jalapeño", "What came of it", "Why I built it", "Find me"] as const;
const LAST = PAGES.length - 1;

export function App() {
	const [page, setPage] = useState(0);
	const [transition, setTransition] = useState<Transition | null>(null);
	const [leaving, setLeaving] = useState(false);
	const [entered, setEntered] = useState<Transition | null>(null);
	const target = useRef(0);
	const ids = useRef(0);

	const go = useCallback(
		(to: number) => {
			if (transition || to === page || to < 0 || to > LAST) return;
			target.current = to;
			const kind = to === LAST ? "flood" : page === LAST ? "drain" : "wave";
			setLeaving(true);
			setTransition({ id: ++ids.current, kind, dir: to > page ? 1 : -1 });
		},
		[page, transition],
	);

	const onSwap = useCallback(() => {
		setEntered(transition);
		setLeaving(false);
		setPage(target.current);
	}, [transition]);

	useEffect(() => {
		const key = (e: KeyboardEvent) => {
			if (e.key === "ArrowRight") go(page + 1);
			if (e.key === "ArrowLeft") go(page - 1);
		};
		addEventListener("keydown", key);
		return () => removeEventListener("keydown", key);
	}, [go, page]);

	const next = () => go(page + 1);
	const sliding = (transition ?? entered)?.kind === "wave";
	const dir = (transition ?? entered)?.dir ?? 1;
	// pages move with the liquid: up when going forward, down when going back
	const exit = sliding
		? { y: `${-20 * dir}vh`, opacity: 0.3, transition: { duration: 0.85, ease: [0.7, 0, 0.84, 0] as const } }
		: { y: "-4vh", scale: 0.97, opacity: 0.4, transition: { duration: 0.9, ease: [0.7, 0, 0.84, 0] as const } };
	const enterFrom = entered ? (sliding ? { y: `${30 * dir}vh`, opacity: 0 } : { y: "6vh", opacity: 0 }) : false;

	return (
		<div className={`grain relative h-full w-full overflow-hidden ${page > 0 && page < LAST ? "theme-red" : ""}`}>
			<header
				className={`pointer-events-none fixed inset-x-0 top-0 z-40 flex items-center justify-between px-6 py-5 font-mono text-[11px] uppercase tracking-[0.08em] md:px-10 ${
					page === LAST ? "text-paper" : "bg-gradient-to-b from-ink via-ink/80 to-transparent pb-10 text-paper"
				}`}
			>
				<button type="button" onClick={() => go(0)} className="pointer-events-auto hover:opacity-70">
					Jalapenos at home
				</button>
				<nav className="pointer-events-auto flex items-center gap-4" aria-label="Sections">
					<span className="hidden opacity-60 md:inline">{PAGES[page]}</span>
					<span className="tabular-nums">
						{String(page + 1).padStart(2, "0")} / {String(PAGES.length).padStart(2, "0")}
					</span>
					<button type="button" onClick={() => go(page - 1)} disabled={page === 0} aria-label="Previous section" className="disabled:opacity-25">
						←
					</button>
					<button type="button" onClick={() => go(page + 1)} disabled={page === LAST} aria-label="Next section" className="disabled:opacity-25">
						→
					</button>
				</nav>
			</header>

			<motion.main
				key={page}
				initial={enterFrom}
				animate={leaving ? exit : { x: 0, y: 0, scale: 1, opacity: 1 }}
				transition={{ duration: 1.05, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
				className="h-full w-full"
			>
				{page === 0 && <Hero onNext={next} leaving={leaving} />}
				{page === 1 && <Built onNext={next} />}
				{page === 2 && <Jalapeno onNext={next} />}
				{page === 3 && <Results onNext={next} />}
				{page === 4 && <Why onNext={next} />}
				{page === 5 && <Contact onBack={() => go(page - 1)} />}
			</motion.main>

			<WaveTransition transition={transition} onSwap={onSwap} onDone={() => setTransition(null)} />
		</div>
	);
}
