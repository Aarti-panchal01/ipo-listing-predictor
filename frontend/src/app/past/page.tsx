"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { ErrorBox, Loading } from "@/components/Status";
import { getPast, type PastIPO } from "@/lib/api";

type SortKey = "listing_date" | "company" | "listing_gain_pct" | "predicted_gain_pct" | "error";

export default function PastPage() {
  const [rows, setRows] = useState<PastIPO[] | null>(null);
  const [cut, setCut] = useState(0.5);
  const [err, setErr] = useState<string | null>(null);
  const [wake, setWake] = useState<{ attempt: number; max: number } | null>(null);
  const [year, setYear] = useState("all");
  const [sector, setSector] = useState("all");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "listing_date", dir: -1 });

  const load = useCallback(() => {
    setErr(null);
    setRows(null);
    getPast((attempt, max) => setWake({ attempt, max }))
      .then((d) => { setRows(d.items); setCut(d.decision_threshold); })
      .catch((e: Error) => setErr(e.message))
      .finally(() => setWake(null));
  }, []);
  useEffect(load, [load]);

  const years = useMemo(() => [...new Set((rows ?? []).map((r) => r.listing_date.slice(0, 4)))].sort(), [rows]);
  const sectors = useMemo(() => [...new Set((rows ?? []).map((r) => r.sector))].sort(), [rows]);

  const filtered = useMemo(() => {
    const f = (rows ?? []).filter(
      (r) => (year === "all" || r.listing_date.startsWith(year)) && (sector === "all" || r.sector === sector),
    );
    const val = (r: PastIPO) => (sort.key === "error" ? Math.abs(r.listing_gain_pct - r.predicted_gain_pct) : r[sort.key]);
    return [...f].sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort.dir);
  }, [rows, year, sector, sort]);

  const stats = useMemo(() => {
    if (!filtered.length) return null;
    const mae = filtered.reduce((s, r) => s + Math.abs(r.listing_gain_pct - r.predicted_gain_pct), 0) / filtered.length;
    const hits = filtered.filter((r) => (r.prob_positive >= cut) === (r.listing_gain_pct > 0)).length;
    const pos = filtered.filter((r) => r.listing_gain_pct > 0).length;
    return { n: filtered.length, mae, hit: hits / filtered.length, pos: pos / filtered.length };
  }, [filtered, cut]);

  const domain = useMemo(() => {
    const v = filtered.flatMap((r) => [r.listing_gain_pct, r.predicted_gain_pct]);
    if (!v.length) return [-50, 100] as [number, number];
    return [Math.floor(Math.min(...v, 0) / 10) * 10, Math.ceil(Math.max(...v) / 10) * 10] as [number, number];
  }, [filtered]);

  const header = (key: SortKey, label: string, right = false) => (
    <th scope="col" className={`px-3 py-2 font-medium ${right ? "text-right" : "text-left"}`}
      aria-sort={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      <button type="button" className="hover:text-ink" onClick={() => setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : -1 }))}>
        {label}{sort.key === key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
      </button>
    </th>
  );

  return (
    <div className="space-y-6">
      <section className="max-w-3xl">
        <p className="font-mono text-xs uppercase tracking-widest text-muted">Past IPOs</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">How the model did on IPOs it never saw</h1>
        <p className="mt-3 text-ink-2">
          Every mainboard IPO listed from January 2024 onward is in the held-out test set. The model was trained only on
          2012 to 2023, so these are honest out-of-sample predictions, including its misses.
        </p>
      </section>

      {!rows && !err && <Loading label="Loading test-set IPOs…" attempt={wake?.attempt} max={wake?.max} />}
      {err && <ErrorBox message={err} onRetry={load} />}

      {rows && (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm">
              <span className="text-ink-2">Year</span>
              <select value={year} onChange={(e) => setYear(e.target.value)} className="mt-1 block rounded-md border border-line bg-surface px-3 py-2">
                <option value="all">All years</option>
                {years.map((y) => <option key={y}>{y}</option>)}
              </select>
            </label>
            <label className="text-sm">
              <span className="text-ink-2">Sector</span>
              <select value={sector} onChange={(e) => setSector(e.target.value)} className="mt-1 block rounded-md border border-line bg-surface px-3 py-2">
                <option value="all">All sectors</option>
                {sectors.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
          </div>

          {stats && (
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["IPOs shown", stats.n.toString()],
                ["Mean absolute error", `${stats.mae.toFixed(1)} pts`],
                ["Direction correct", `${(stats.hit * 100).toFixed(0)}%`],
                ["Actually listed positive", `${(stats.pos * 100).toFixed(0)}%`],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg border border-line bg-surface p-3">
                  <dt className="text-xs text-muted">{k}</dt>
                  <dd className="mt-1 font-mono text-2xl font-semibold">{v}</dd>
                </div>
              ))}
            </dl>
          )}

          <figure className="rounded-lg border border-line bg-surface p-4">
            <figcaption className="mb-2 text-sm font-semibold">Predicted vs actual listing gain (%)</figcaption>
            <p className="mb-3 text-xs text-muted">Each dot is one IPO. On the diagonal = perfect prediction; above it = listed better than predicted.</p>
            <div className="h-[340px] sm:h-[420px]">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 8, right: 12, bottom: 28, left: 4 }}>
                  <CartesianGrid stroke="var(--grid)" />
                  <XAxis type="number" dataKey="predicted_gain_pct" domain={domain} name="Predicted" unit="%"
                    tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)"
                    label={{ value: "Predicted", position: "insideBottom", offset: -16, fill: "var(--ink-2)", fontSize: 12 }} />
                  <YAxis type="number" dataKey="listing_gain_pct" domain={domain} name="Actual" unit="%" width={52}
                    tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--axis)" />
                  <ReferenceLine segment={[{ x: domain[0], y: domain[0] }, { x: domain[1], y: domain[1] }]} stroke="var(--axis)" />
                  <ReferenceLine y={0} stroke="var(--axis)" />
                  <ReferenceLine x={0} stroke="var(--axis)" />
                  <Tooltip cursor={false} content={<ScatterTip />} />
                  <Scatter data={filtered} fill="var(--series-1)" fillOpacity={0.75} stroke="var(--surface)" strokeWidth={1.5} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </figure>

          <div className="overflow-x-auto rounded-lg border border-line bg-surface">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="border-b border-line text-ink-2">
                <tr>
                  {header("listing_date", "Listed")}
                  {header("company", "Company")}
                  <th scope="col" className="px-3 py-2 text-left font-medium">Sector</th>
                  {header("predicted_gain_pct", "Predicted", true)}
                  {header("listing_gain_pct", "Actual", true)}
                  {header("error", "Miss", true)}
                  <th scope="col" className="px-3 py-2 text-right font-medium">P(up)</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {filtered.map((r) => {
                  const miss = r.listing_gain_pct - r.predicted_gain_pct;
                  const wrong = (r.prob_positive >= cut) !== (r.listing_gain_pct > 0);
                  return (
                    <tr key={r.ipo_id} className="border-b border-line last:border-0 hover:bg-surface-2">
                      <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-ink-2">{r.listing_date}</td>
                      <td className="px-3 py-2">{r.company}</td>
                      <td className="px-3 py-2 text-ink-2">{r.sector}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.predicted_gain_pct.toFixed(1)}%</td>
                      <td className={`px-3 py-2 text-right font-mono ${r.listing_gain_pct > 0 ? "text-good" : "text-bad"}`}>
                        {r.listing_gain_pct > 0 ? "+" : ""}{r.listing_gain_pct.toFixed(1)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-ink-2">{miss > 0 ? "+" : ""}{miss.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {Math.round(r.prob_positive * 100)}%{wrong && <span className="ml-1 text-bad" title="direction wrong">✕</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {!filtered.length && <p className="p-4 text-sm text-muted">No IPOs match these filters.</p>}
          </div>
        </>
      )}
    </div>
  );
}

function ScatterTip({ active, payload }: { active?: boolean; payload?: { payload: PastIPO }[] }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <p className="font-semibold">{r.company}</p>
      <p className="text-muted">{r.listing_date} · {r.sector}</p>
      <p className="mt-1 tabular">Predicted <strong>{r.predicted_gain_pct.toFixed(1)}%</strong> · Actual <strong>{r.listing_gain_pct.toFixed(1)}%</strong></p>
      <p className="tabular">Total subscription {r.total_x?.toFixed(1) ?? "n/a"}x</p>
    </div>
  );
}
