import { useEffect, useRef } from "react";
import { createNoise2D } from "simplex-noise";

type Layer = { level: number; amp: number; freq: number; speed: number; fill: [string, string] };

// level = surface height as a fraction of the canvas, measured from the top
const LAYERS: Layer[] = [
	{ level: 0.56, amp: 0.07, freq: 0.9, speed: 0.1, fill: ["#ffb3a3", "#f0614a"] },
	{ level: 0.63, amp: 0.06, freq: 1.3, speed: 0.14, fill: ["#ff6a4d", "#c92811"] },
	{ level: 0.71, amp: 0.05, freq: 1.8, speed: 0.19, fill: ["#e8341c", "#9e1c0c"] },
];

/** A slow, layered red liquid that swells toward the cursor. */
export function Liquid({ surge = 0 }: { surge?: number }) {
	const canvas = useRef<HTMLCanvasElement>(null);
	const surgeRef = useRef(surge);
	surgeRef.current = surge;

	useEffect(() => {
		const el = canvas.current!;
		const ctx = el.getContext("2d")!;
		const noise = createNoise2D(() => 0.42);
		const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
		const pointer = { x: 0.5, target: 0.5, strength: 0, targetStrength: 0 };
		let raf = 0;
		let w = 0;
		let h = 0;
		let lift = 0;

		const resize = () => {
			const dpr = Math.min(devicePixelRatio, 2);
			w = el.clientWidth;
			h = el.clientHeight;
			el.width = w * dpr;
			el.height = h * dpr;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		};
		const move = (e: PointerEvent) => {
			const r = el.getBoundingClientRect();
			pointer.target = (e.clientX - r.left) / r.width;
			const y = (e.clientY - r.top) / r.height;
			pointer.targetStrength = Math.max(0, 1 - Math.abs(y - 0.6) * 2.2);
		};
		const leave = () => (pointer.targetStrength = 0);

		const draw = (now: number) => {
			const t = reduced ? 0 : now / 1000;
			pointer.x += (pointer.target - pointer.x) * 0.06;
			pointer.strength += (pointer.targetStrength - pointer.strength) * 0.05;
			lift += (surgeRef.current - lift) * 0.08;
			ctx.clearRect(0, 0, w, h);

			LAYERS.forEach((layer, i) => {
				const step = 6;
				ctx.beginPath();
				ctx.moveTo(0, h);
				for (let x = 0; x <= w + step; x += step) {
					const u = x / w;
					const n =
						noise(u * layer.freq * 2, t * layer.speed + i * 10) * 0.7 +
						noise(u * layer.freq * 5, t * layer.speed * 1.7 + i * 20) * 0.3;
					// the cursor pulls the surface up in a soft bump
					const d = u - pointer.x;
					const bump = Math.exp(-(d * d) / 0.012) * pointer.strength * 0.08 * (1 + i * 0.4);
					const y = h * (layer.level - lift * (0.25 + i * 0.05) + n * layer.amp - bump);
					ctx.lineTo(x, y);
				}
				ctx.lineTo(w, h);
				ctx.closePath();
				const top = h * (layer.level - layer.amp - lift * 0.3);
				const g = ctx.createLinearGradient(0, top, 0, h);
				g.addColorStop(0, layer.fill[0]);
				g.addColorStop(1, layer.fill[1]);
				ctx.fillStyle = g;
				ctx.fill();
				if (i === LAYERS.length - 1) {
					ctx.strokeStyle = "rgba(255, 170, 140, 0.35)";
					ctx.lineWidth = 1.25;
					ctx.stroke();
				}
			});
			raf = requestAnimationFrame(draw);
		};

		resize();
		const ro = new ResizeObserver(resize);
		ro.observe(el);
		addEventListener("pointermove", move);
		document.addEventListener("pointerleave", leave);
		raf = requestAnimationFrame(draw);
		return () => {
			cancelAnimationFrame(raf);
			ro.disconnect();
			removeEventListener("pointermove", move);
			document.removeEventListener("pointerleave", leave);
		};
	}, []);

	return <canvas ref={canvas} className="absolute inset-0 h-full w-full" aria-hidden />;
}
