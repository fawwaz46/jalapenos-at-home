import { useEffect, useRef } from "react";
import { createNoise2D } from "simplex-noise";

export type Transition = {
	id: number;
	/** wave: a band of liquid sweeps the screen. flood: it rises. drain: it falls away. */
	kind: "wave" | "flood" | "drain";
	/** 1 = forward (upward), -1 = back */
	dir: 1 | -1;
};

export const CHILI = "#e8341c";
const HOT = "#ff6a4d";
const DEEP = "#7d1609";

const DURATION = { wave: 1600, flood: 1400, drain: 1300 } as const;

const easeInOut = (p: number) => (p < 0.5 ? 4 * p * p * p : 1 - (-2 * p + 2) ** 3 / 2);

/**
 * Full-screen liquid that covers the page while it changes underneath.
 * Calls onSwap once the screen is fully covered and onDone when it has gone.
 */
export function WaveTransition({
	transition,
	onSwap,
	onDone,
}: {
	transition: Transition | null;
	onSwap: () => void;
	onDone: () => void;
}) {
	const canvas = useRef<HTMLCanvasElement>(null);
	const callbacks = useRef({ onSwap, onDone });
	callbacks.current = { onSwap, onDone };

	useEffect(() => {
		if (!transition) return;
		const el = canvas.current!;
		const ctx = el.getContext("2d")!;
		const noise = createNoise2D(() => 0.7);
		const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
		const dpr = Math.min(devicePixelRatio, 2);
		const W = innerWidth;
		const H = innerHeight;
		el.width = W * dpr;
		el.height = H * dpr;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

		const { kind, dir } = transition;
		const duration = reduced ? 350 : DURATION[kind];
		const start = performance.now();
		let swapped = false;
		let raf = 0;

		const swap = () => {
			if (!swapped) {
				swapped = true;
				callbacks.current.onSwap();
			}
		};

		/** y position of a horizontal liquid surface at x */
		const surface = (base: number, x: number, t: number, seed: number) => {
			const u = x / W;
			return base + H * (noise(u * 1.8 + seed, t * 0.8) * 0.045 + noise(u * 6 + seed, t * 1.7) * 0.012);
		};

		/** a band of liquid between two wavy surfaces */
		const fillBand = (lead: number, trail: number, t: number, seed: number, color: string | CanvasGradient) => {
			const step = 10;
			ctx.beginPath();
			for (let x = -step; x <= W + step; x += step) ctx.lineTo(x, surface(lead, x, t, seed));
			for (let x = W + step; x >= -step; x -= step) ctx.lineTo(x, surface(trail, x, t, seed + 5));
			ctx.closePath();
			ctx.fillStyle = color;
			ctx.fill();
		};

		const fillSurface = (level: number, t: number, seed: number, color: string) => {
			const step = 10;
			ctx.beginPath();
			ctx.moveTo(0, H);
			for (let x = 0; x <= W + step; x += step) {
				const u = x / W;
				const y = level + H * (noise(u * 2 + seed, t * 0.8) * 0.04 + noise(u * 6 + seed, t * 1.6) * 0.012);
				ctx.lineTo(x, y);
			}
			ctx.lineTo(W, H);
			ctx.closePath();
			ctx.fillStyle = color;
			ctx.fill();
		};

		const frame = (now: number) => {
			const p = Math.min(1, (now - start) / duration);
			const s = easeInOut(p);
			const t = now / 1000;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, W, H);

			if (reduced) {
				// a plain red fade for people who asked for less motion
				ctx.globalAlpha = Math.sin(p * Math.PI);
				ctx.fillStyle = CHILI;
				ctx.fillRect(0, 0, W, H);
				ctx.globalAlpha = 1;
				if (p >= 0.5) swap();
			} else if (kind === "wave") {
				// rises from the bottom (forward) or falls from the top (back)
				if (dir === -1) ctx.setTransform(dpr, 0, 0, -dpr, 0, H * dpr);
				const lead = H * (1.05 - 2.95 * s);
				const trail = lead + 1.8 * H;
				fillBand(lead + 0.35 * H, trail, t, 3, DEEP);
				fillBand(lead - 0.045 * H, trail - 0.12 * H, t, 1, HOT);
				const g = ctx.createLinearGradient(0, lead, 0, trail);
				g.addColorStop(0, CHILI);
				g.addColorStop(0.6, "#c92811");
				g.addColorStop(1, "#a11f0c");
				fillBand(lead, trail - 0.07 * H, t, 0, g);
				// fully covered while the top surface is above the screen and the bottom one below it
				if (s >= 0.5) swap();
			} else {
				const rising = kind === "flood";
				const level = rising ? H * (1.1 - 1.4 * s) : H * (-0.3 + 1.4 * s);
				if (!rising) swap();
				fillSurface(level - H * 0.05, t, 2, rising ? HOT : DEEP);
				fillSurface(level, t, 0, CHILI);
				if (rising && s >= 0.97) swap();
			}

			if (p < 1) raf = requestAnimationFrame(frame);
			else {
				swap();
				ctx.clearRect(0, 0, W * 2, H * 2);
				callbacks.current.onDone();
			}
		};
		raf = requestAnimationFrame(frame);
		return () => cancelAnimationFrame(raf);
	}, [transition]);

	return (
		<canvas
			ref={canvas}
			aria-hidden
			className="pointer-events-none fixed inset-0 z-50 h-full w-full"
			style={{ visibility: transition ? "visible" : "hidden" }}
		/>
	);
}
