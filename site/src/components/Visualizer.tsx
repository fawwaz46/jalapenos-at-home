import { AnimatePresence, motion, useInView } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import data from "../data/runs.json";

type Round = (typeof data.runs)[number]["rounds"][number];
type Phase = "draft" | "verify" | "commit";
type Piece = { text: string; from: "draft" | "target" };

const MAX_GAMMA = 8;

const show = (piece: string) => piece.replace(/\n/g, "↵ ");

function Chip({ text, state }: { text: string; state: "pending" | "accept" | "reject" }) {
	const lead = text.startsWith(" ");
	const styles = {
		pending: "border-dashed border-black/35 text-paper/70",
		accept: "border-accept bg-accept/20 text-paper",
		reject: "border-reject text-paper/60 line-through decoration-reject",
	};
	return (
		<motion.span
			layout
			initial={{ y: -14, opacity: 0, scale: 0.9 }}
			animate={{ y: 0, opacity: 1, scale: 1 }}
			exit={{ y: 18, opacity: 0, scale: 0.9, transition: { duration: 0.35 } }}
			transition={{ type: "spring", stiffness: 420, damping: 30 }}
			className={`mx-[1px] inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 align-baseline text-[0.8em] ${styles[state]} ${lead ? "ml-2" : ""}`}
		>
			{show(text.trimStart()) || "␣"}
			{state !== "pending" && (
				<span className="font-mono text-[0.7em]" aria-label={state === "accept" ? "accepted" : "rejected"}>
					{state === "accept" ? "✓" : "✗"}
				</span>
			)}
		</motion.span>
	);
}

export function Visualizer() {
	const [runIndex, setRunIndex] = useState(0);
	const [playing, setPlaying] = useState(true);
	const box = useRef<HTMLDivElement>(null);
	const visible = useInView(box, { margin: "-20%" });
	const [realtime, setRealtime] = useState(true);
	const [round, setRound] = useState(0);
	const [shown, setShown] = useState(0);
	const [phase, setPhase] = useState<Phase>("draft");
	const run = data.runs[runIndex];
	const current: Round | undefined = run.rounds[round];
	const done = round >= run.rounds.length;

	// everything committed before the current round
	const committed = useMemo(() => {
		const out: Piece[] = [];
		for (const r of run.rounds.slice(0, round)) {
			for (const t of r.proposed.slice(0, r.accepted)) out.push({ text: t, from: "draft" });
			if (r.target !== null) out.push({ text: r.target, from: "target" });
		}
		return out;
	}, [run, round]);

	const totals = useMemo(() => {
		const past = run.rounds.slice(0, round);
		return {
			accepted: past.reduce((n, r) => n + r.accepted, 0),
			drafted: past.reduce((n, r) => n + r.proposed.length, 0),
		};
	}, [run, round]);

	const restart = (index = runIndex) => {
		setRunIndex(index);
		setRound(0);
		setShown(0);
		setPhase("draft");
		setPlaying(true);
	};

	// the replay clock
	useEffect(() => {
		if (!playing || !visible || done || !current) return;
		const realRound = run.elapsed_ms / run.rounds.length;
		const n = current.proposed.length;
		const speed = realtime
			? { guess: (realRound * 0.55) / n, verify: realRound * 0.3, commit: realRound * 0.15 }
			: { guess: 170, verify: 650, commit: 900 };
		let delay: number;
		let next: () => void;
		if (phase === "draft" && shown < n) {
			delay = speed.guess;
			next = () => setShown((s) => s + 1);
		} else if (phase === "draft") {
			delay = speed.verify;
			next = () => setPhase("verify");
		} else if (phase === "verify") {
			delay = speed.commit;
			next = () => setPhase("commit");
		} else {
			delay = realtime ? 1 : 260;
			next = () => {
				setRound((r) => r + 1);
				setShown(0);
				setPhase("draft");
			};
		}
		const timer = setTimeout(next, delay);
		return () => clearTimeout(timer);
	}, [playing, visible, done, current, phase, shown, realtime, run]);

	const gamma = current?.gamma ?? run.rounds[run.rounds.length - 1].next_gamma;
	const tokens = committed.length;

	return (
		<div ref={box} className="rounded-2xl border border-line bg-panel">
			<div className="flex flex-wrap items-center justify-between gap-3 border-b border-line p-4 md:px-6">
				<div className="flex flex-wrap gap-2">
					{data.runs.map((r, i) => (
						<button
							key={r.prompt}
							type="button"
							onClick={() => restart(i)}
							className={`rounded-full px-4 py-2 text-[14px] font-medium tracking-[-0.01em] transition-colors ${
								i === runIndex ? "bg-paper text-dark" : "bg-black/5 text-paper/70 hover:bg-black/10"
							}`}
						>
							{r.prompt.split(" ").slice(0, 4).join(" ")}…
						</button>
					))}
				</div>
				<div className="flex items-center gap-2 text-[14px] font-medium tracking-[-0.01em]">
					<button
						type="button"
						onClick={() => setRealtime((v) => !v)}
						className="rounded-full border border-line px-4 py-2 text-paper/80 hover:border-black/40"
					>
						{realtime ? "Slow down" : "Speed up"}
					</button>
					<button
						type="button"
						onClick={() => (done ? restart() : setPlaying((p) => !p))}
						className="rounded-full border border-line px-4 py-2 text-paper/80 hover:border-black/40"
					>
						{done ? "Replay" : playing ? "Pause" : "Play"}
					</button>
				</div>
			</div>

			<div className="grid lg:grid-cols-[1fr_17rem]">
				<div className="min-h-[22rem] p-5 text-[1.35rem] leading-[1.7] tracking-[-0.01em] whitespace-pre-wrap md:p-8 md:text-[1.6rem]">
					<span className="text-paper/40">{run.prompt}</span>
					{committed.map((p, i) => (
						<span
							// biome-ignore lint: tokens are positional
							key={i}
							className={p.from === "draft" ? "underline decoration-accept decoration-2 underline-offset-[6px]" : ""}
						>
							{show(p.text)}
						</span>
					))}
					{/* keyed by round: when a round commits, its chips are replaced by text at once */}
					<AnimatePresence key={round} mode="popLayout">
						{current &&
							current.proposed.slice(0, shown).map((t, i) => {
								const state =
									phase === "draft" ? "pending" : i < current.accepted ? "accept" : "reject";
								if (phase === "commit" && state === "reject") return null;
								return <Chip key={`${round}-${i}`} text={t} state={state} />;
							})}
						{current && phase === "commit" && current.target !== null && (
							<motion.span
								key={`${round}-target`}
								initial={{ opacity: 0, y: -10 }}
								animate={{ opacity: 1, y: 0 }}
								className="mx-[1px] rounded-md border border-paper px-1.5 py-0.5 text-[0.8em]"
							>
								{show(current.target.trimStart()) || "␣"}
							</motion.span>
						)}
					</AnimatePresence>
					{!done && <span className="ml-1 inline-block h-[1em] w-[2px] translate-y-[3px] animate-pulse bg-chili" />}
				</div>

				<aside className="space-y-6 border-t border-line p-5 md:p-6 lg:border-t-0 lg:border-l">
					<Stat label="Round" value={`${Math.min(round + 1, run.rounds.length)} / ${run.rounds.length}`} />
					<div>
						<div className="font-mono text-[11px] uppercase tracking-[0.08em] text-gray">Guess budget</div>
						<div className="mt-2 flex gap-1" role="img" aria-label={`${gamma} of ${MAX_GAMMA} guesses`}>
							{Array.from({ length: MAX_GAMMA }, (_, i) => (
								<motion.span
									// biome-ignore lint: fixed cells
									key={i}
									animate={{ backgroundColor: i < gamma ? "#0a0a0a" : "rgba(0,0,0,0.08)" }}
									className="h-6 flex-1 rounded-[3px]"
								/>
							))}
						</div>
						<div className="mt-1 text-sm text-paper/50">
							{gamma} {gamma === 1 ? "guess" : "guesses"} this round. Grows when all land, halves when most miss.
						</div>
					</div>
					<Stat label="Guesses accepted" value={`${totals.accepted} / ${totals.drafted}`} />
					<Stat
						label="Tokens per big-model pass"
						value={round ? (tokens / round).toFixed(2) : "–"}
						note={`${tokens} tokens from ${round} passes. Without drafting: 1.00.`}
					/>
					<div className="space-y-2 border-t border-line pt-5 text-sm text-paper/60">
						<Legend className="border-dashed border-black/35">Small model's guess</Legend>
						<Legend className="border-accept bg-accept/20" mark="✓">Big model agrees</Legend>
						<Legend className="border-reject line-through decoration-reject" mark="✗">Rejected, rolled back</Legend>
						<Legend className="border-paper">Big model's own token</Legend>
					</div>
				</aside>
			</div>

			<p className="border-t border-line px-5 py-4 text-[14px] leading-snug text-paper/70 md:px-6">
				Replay of a recorded run: Qwen3-0.6B drafting for Qwen3-1.7B.
			</p>
		</div>
	);
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
	return (
		<div>
			<div className="font-mono text-[11px] uppercase tracking-[0.08em] text-gray">{label}</div>
			<div className="mt-1 text-3xl font-medium tracking-[-0.03em] tabular-nums">{value}</div>
			{note && <div className="mt-1 text-sm text-paper/50">{note}</div>}
		</div>
	);
}

function Legend({ className, mark, children }: { className: string; mark?: string; children: string }) {
	return (
		<div className="flex items-center gap-3">
			<span className={`inline-flex h-5 w-9 items-center justify-center rounded-[4px] border font-mono text-[10px] ${className}`}>
				{mark}
			</span>
			{children}
		</div>
	);
}
