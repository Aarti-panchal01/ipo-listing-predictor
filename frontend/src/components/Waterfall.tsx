"use client";

import { useState } from "react";
import type { Explanation } from "@/lib/api";

const fmt = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`;

function fmtValue(v: number | string | null): string {
  if (v === null || v === undefined) return "missing";
  if (typeof v === "string") return v;
  if (Math.abs(v) >= 1000) return v.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  return v.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

/** SHAP waterfall: starts at the model's average prediction, each feature pushes
 *  the running total up (blue) or down (red), ending at this IPO's prediction.
 *  Features outside the top list are folded into one "all other features" row. */
export default function Waterfall({ data }: { data: Explanation }) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = data.contributions;
  const rest = data.predicted_gain_pct - data.base_value - shown.reduce((s, c) => s + c.shap, 0);
  const rows = [
    ...shown.map((c) => ({ label: c.label, value: fmtValue(c.value), shap: c.shap })),
    ...(Math.abs(rest) >= 0.05 ? [{ label: "All other features", value: "", shap: rest }] : []),
  ];

  let run = data.base_value;
  const steps = rows.map((r) => {
    const start = run;
    run += r.shap;
    return { ...r, start, end: run };
  });
  const xs = [data.base_value, data.predicted_gain_pct, ...steps.flatMap((s) => [s.start, s.end]), 0];
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pad = (hi - lo) * 0.08 || 1;
  const min = lo - pad;
  const max = hi + pad;
  const pct = (v: number) => ((v - min) / (max - min)) * 100;

  return (
    <figure className="w-full">
      <figcaption className="sr-only">
        SHAP waterfall from the average prediction {data.base_value.toFixed(1)}% to this IPO&apos;s prediction{" "}
        {data.predicted_gain_pct.toFixed(1)}%.
      </figcaption>
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-up" aria-hidden /> pushes the prediction up
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-down" aria-hidden /> pushes it down
        </span>
        <span>units: percentage points of listing gain</span>
      </div>

      <div className="grid grid-cols-[minmax(0,9.5rem)_1fr_3.5rem] items-center gap-x-3 text-xs sm:grid-cols-[minmax(0,13rem)_1fr_4rem] sm:text-sm">
        <Row label="Average IPO (model base)" value="" bar={null} num={`${data.base_value.toFixed(1)}%`} marker={pct(data.base_value)} zero={pct(0)} />
        {steps.map((s, i) => {
          const left = pct(Math.min(s.start, s.end));
          const width = Math.max(Math.abs(pct(s.end) - pct(s.start)), 0.6);
          return (
            <Row
              key={s.label}
              label={s.label}
              value={s.value}
              num={fmt(s.shap)}
              zero={pct(0)}
              highlighted={hover === i}
              onHover={(h) => setHover(h ? i : null)}
              bar={{ left, width, up: s.shap >= 0, start: s.start, end: s.end }}
            />
          );
        })}
        <Row label="This IPO (prediction)" value="" bar={null} num={`${data.predicted_gain_pct.toFixed(1)}%`} marker={pct(data.predicted_gain_pct)} zero={pct(0)} strong />
      </div>
    </figure>
  );
}

function Row({
  label, value, num, bar, marker, zero, strong, highlighted, onHover,
}: {
  label: string;
  value: string;
  num: string;
  zero: number;
  bar: { left: number; width: number; up: boolean; start: number; end: number } | null;
  marker?: number;
  strong?: boolean;
  highlighted?: boolean;
  onHover?: (h: boolean) => void;
}) {
  return (
    <>
      <div className={`truncate py-1.5 ${strong ? "font-semibold" : ""}`} title={value ? `${label} = ${value}` : label}>
        {label}
        {value && <span className="ml-1 text-muted">= {value}</span>}
      </div>
      <div
        className={`relative h-7 rounded ${highlighted ? "bg-surface-2" : ""}`}
        onMouseEnter={() => onHover?.(true)}
        onMouseLeave={() => onHover?.(false)}
      >
        <div className="absolute inset-y-0 w-px bg-[var(--axis)]" style={{ left: `${zero}%` }} aria-hidden />
        {bar && (
          <div
            className={`absolute top-1.5 h-4 rounded-[3px] ${bar.up ? "bg-up" : "bg-down"}`}
            style={{ left: `${bar.left}%`, width: `${bar.width}%` }}
          />
        )}
        {marker !== undefined && (
          <div className="absolute top-1 h-5 w-[3px] -translate-x-1/2 rounded bg-ink" style={{ left: `${marker}%` }} aria-hidden />
        )}
        {highlighted && bar && (
          <div
            role="tooltip"
            className="pointer-events-none absolute -top-9 z-10 whitespace-nowrap rounded-md border border-line bg-surface px-2 py-1 text-xs shadow-sm"
            style={{ left: `${Math.min(bar.left, 70)}%` }}
          >
            {label}: {num} pts ({bar.start.toFixed(1)}% → {bar.end.toFixed(1)}%)
          </div>
        )}
      </div>
      <div className={`tabular text-right font-mono ${strong ? "font-semibold" : "text-ink-2"}`}>{num}</div>
    </>
  );
}
