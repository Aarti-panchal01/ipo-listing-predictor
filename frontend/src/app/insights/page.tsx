"use client";

import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { ErrorBox, Loading } from "@/components/Status";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Json = any;

const AXIS = { tick: { fill: "var(--muted)", fontSize: 12 }, stroke: "var(--axis)" };
const FEATURE_LABELS: Record<string, string> = {
  log_qib_x: "QIB subscription", log_nii_x: "NII subscription", log_bnii_x: "bNII subscription", log_snii_x: "sNII subscription",
  log_retail_x: "Retail subscription", log_employee_x: "Employee subscription", log_total_x: "Total subscription",
  log_issue_size_cr: "Issue size", log_anchor_portion_cr: "Anchor book", log_price_band_upper: "Price band",
  log_lot_size: "Lot size", log_min_investment: "Min. investment", log_revenue_cr: "Revenue", log_mcap_post_cr: "Market cap",
  slog_pat_cr: "Profit after tax", fresh_issue_pct: "Fresh issue %", anchor_pct_of_issue: "Anchor % of issue",
  retail_reserved_pct: "Retail quota %", listing_month: "Listing month", issue_days_open: "Days open", nifty_ret_7d: "Nifty 7d return", nifty_ret_30d: "Nifty 30d return",
  vix_close: "India VIX", ipo_heat_30d: "IPOs closed, last 30d", prev_gain_mean_90d: "Recent IPOs' avg gain",
  prev_count_90d: "IPOs listed, last 90d", revenue_growth_pct: "Revenue growth", pat_margin_pct: "PAT margin",
  roe_pct: "ROE", debt_equity: "Debt/Equity", pe_post: "Issue P/E", promoter_post_pct: "Promoter holding",
  promoter_dilution_pct: "Promoter dilution", exchange: "Exchange", sector: "Sector",
};
const label = (f: string) => FEATURE_LABELS[f] ?? f;
const n = (v: number | null | undefined, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(d));

function Card({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <figure className="rounded-lg border border-line bg-surface p-4 sm:p-5">
      <figcaption>
        <h3 className="font-semibold">{title}</h3>
        {sub && <p className="mt-1 text-sm text-ink-2">{sub}</p>}
      </figcaption>
      <div className="mt-4">{children}</div>
    </figure>
  );
}

function Tip({ active, payload, fmt }: { active?: boolean; payload?: any[]; fmt: (p: any) => React.ReactNode }) {
  if (!active || !payload?.length) return null;
  return <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-sm">{fmt(payload[0].payload)}</div>;
}

function MetricTable({ rows, cols }: { rows: Json[]; cols: { key: string; label: string; d?: number; best?: "min" | "max" }[] }) {
  const best: Record<string, number> = {};
  for (const c of cols) if (c.best) {
    const vals = rows.map((r) => r[c.key]).filter((v) => typeof v === "number" && !Number.isNaN(v));
    best[c.key] = c.best === "min" ? Math.min(...vals) : Math.max(...vals);
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <thead className="border-b border-line text-left text-ink-2">
          <tr>{cols.map((c, i) => <th key={c.key} scope="col" className={`px-2 py-2 font-medium ${i ? "text-right" : ""}`}>{c.label}</th>)}</tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r, j) => (
            <tr key={j} className="border-b border-line last:border-0">
              {cols.map((c, i) => {
                const v = r[c.key];
                const isBest = c.best && v === best[c.key];
                return (
                  <td key={c.key} className={`px-2 py-2 ${i ? "text-right font-mono" : ""} ${isBest ? "font-semibold text-accent" : ""}`}>
                    {typeof v === "number" ? n(v, c.d ?? 2) : String(v ?? "")}{isBest ? " ●" : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted">● best in column.</p>
    </div>
  );
}

export default function InsightsPage() {
  const [d, setD] = useState<{ eda: Json; classical: Json; deep: Json; explain: Json } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all(["eda", "classical", "deep", "explain"].map((k) => fetch(`/insights/${k}.json`).then((r) => {
      if (!r.ok) throw new Error(`Could not load ${k}.json`);
      // Python's json writes bare NaN for "not applicable" metrics (e.g. ROC-AUC of the median baseline)
      return r.text().then((t) => JSON.parse(t.replace(/\bNaN\b/g, "null")));
    })))
      .then(([eda, classical, deep, explain]) => setD({ eda, classical, deep, explain }))
      .catch((e: Error) => setErr(e.message));
  }, []);

  if (err) return <ErrorBox message={err} />;
  if (!d) return <Loading label="Loading results…" />;
  const { eda, classical, deep, explain } = d;
  const shapTop = explain.shap_global.slice(0, 12).map((r: Json) => ({ ...r, label: label(r.feature) }));
  const blendRows = explain.blend.map((r: Json) => ({ ...r }));

  return (
    <div className="space-y-10">
      <section className="max-w-3xl">
        <p className="font-mono text-xs uppercase tracking-widest text-muted">Insights</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">What the data and the models say</h1>
        <p className="mt-3 text-ink-2">
          {eda.n_ipos} mainboard IPOs listed between {eda.period[0]} and {eda.period[1]}. Train: listed up to 2022 ({eda.splits.train}),
          validation: 2023 ({eda.splits.val}), test: 2024 onward ({eda.splits.test}). Every chart below is generated from the
          notebooks&apos; exported results, not screenshots.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">1 · The market changes every year</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Median listing-day gain by year (%)" sub="The 2021 boom and the 2024-25 wave look very different from 2015-2019.">
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={eda.by_year} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="year" {...AXIS} />
                  <YAxis {...AXIS} width={40} unit="%" />
                  <ReferenceLine y={0} stroke="var(--axis)" />
                  <Tooltip content={<Tip fmt={(p) => <><b>{p.year}</b>: median {n(p.median_gain)}%, {p.ipos} IPOs, {n(p.pct_positive, 0)}% positive</>} />} />
                  <Line dataKey="median_gain" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Mainboard IPOs listed per year">
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={eda.by_year} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="year" {...AXIS} />
                  <YAxis {...AXIS} width={40} />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(p) => <><b>{p.year}</b>: {p.ipos} IPOs</>} />} />
                  <Bar dataKey="ipos" fill="var(--series-1)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">2 · Subscription is the strongest single signal, but noisy</h2>
        <Card title="Total subscription (log scale) vs listing-day gain" sub="Each dot is an IPO. Heavier demand usually means a better listing, with wide scatter.">
          <div className="h-80">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 20 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="log_total_x" name="log(1+total_x)" {...AXIS}
                  tickFormatter={(v) => `${Math.round(Math.expm1(v))}x`}
                  label={{ value: "Total subscription (log axis)", position: "insideBottom", offset: -12, fill: "var(--ink-2)", fontSize: 12 }} />
                <YAxis type="number" dataKey="gain" unit="%" {...AXIS} width={48} />
                <ReferenceLine y={0} stroke="var(--axis)" />
                <Tooltip cursor={false} content={<Tip fmt={(p) => <><b>{p.company}</b><br />{n(Math.expm1(p.log_total_x))}x subscribed, listed {n(p.gain)}%</>} />} />
                <Scatter data={eda.scatter_total_x} fill="var(--series-1)" fillOpacity={0.6} stroke="var(--surface)" strokeWidth={1} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">3 · Model comparison on the 2024+ test set</h2>
        <Card title="Baseline vs XGBoost vs deep learning vs blend"
          sub={`Blend = ${n(explain.blend_weight_xgb, 2)} × XGBoost + ${n(1 - explain.blend_weight_xgb, 2)} × ${explain.best_dl}, weight tuned on 2023. Production model: ${explain.production_choice}.`}>
          <MetricTable rows={blendRows} cols={[
            { key: "model", label: "Model" }, { key: "MAE", label: "MAE (pts)", best: "min" }, { key: "RMSE", label: "RMSE", best: "min" },
            { key: "R2", label: "R²", best: "max" }, { key: "ROC_AUC", label: "ROC-AUC", best: "max" }, { key: "Balanced_Accuracy", label: "Bal. acc.", best: "max" },
          ]} />
        </Card>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Cross-validated MAE on 2012-2022 (TimeSeriesSplit, 5 folds)" sub="Lower is better. Bars show the mean; ± is the spread across folds.">
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={deep.cv} layout="vertical" margin={{ top: 0, right: 48, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" {...AXIS} />
                  <YAxis type="category" dataKey="model" {...AXIS} width={110} />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(p) => <><b>{p.model}</b>: {n(p.mean, 2)} ± {n(p.std, 2)}</>} />} />
                  <Bar dataKey="mean" fill="var(--series-1)" radius={[0, 4, 4, 0]} barSize={18}>
                    <LabelList dataKey="mean" position="right" formatter={(v: number) => v.toFixed(1)} fill="var(--ink-2)" fontSize={12} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Hyperparameter search: same budget, same CV" sub="Validation MAE is one year ahead (2023).">
            <MetricTable rows={classical.tuning} cols={[
              { key: "method", label: "Method" }, { key: "fits", label: "Fits", d: 0 }, { key: "cv_MAE", label: "CV MAE", best: "min" },
              { key: "val_MAE", label: "2023 MAE", best: "min" }, { key: "seconds", label: "Seconds", d: 1 },
            ]} />
          </Card>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">4 · What drives the predictions (SHAP)</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Global importance: mean |SHAP| on test IPOs" sub="Average size of each feature's push on the predicted gain, in percentage points.">
            <div className="h-[400px]">
              <ResponsiveContainer>
                <BarChart data={shapTop} layout="vertical" margin={{ top: 0, right: 44, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" {...AXIS} />
                  <YAxis type="category" dataKey="label" {...AXIS} width={150} interval={0} />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(p) => <><b>{p.label}</b>: {n(p.mean_abs_shap, 2)} pts</>} />} />
                  <Bar dataKey="mean_abs_shap" fill="var(--series-1)" radius={[0, 4, 4, 0]} barSize={16}>
                    <LabelList dataKey="mean_abs_shap" position="right" formatter={(v: number) => v.toFixed(1)} fill="var(--ink-2)" fontSize={12} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Built-in importance vs SHAP: rank by method" sub="XGBoost 'weight' counts splits, 'gain' averages loss reduction, SHAP measures actual effect on predictions. They disagree.">
            <MetricTable rows={explain.importance.slice(0, 10).map((r: Json) => ({ ...r, feature: label(r.feature) }))} cols={[
              { key: "feature", label: "Feature" }, { key: "mean_abs_SHAP_rank", label: "SHAP", d: 0 }, { key: "gain_rank", label: "Gain", d: 0 },
              { key: "weight_rank", label: "Weight", d: 0 }, { key: "cover_rank", label: "Cover", d: 0 }, { key: "permutation_MAE_increase_rank", label: "Permutation", d: 0 },
            ]} />
          </Card>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {explain.local.map((l: Json) => (
            <Card key={l.label} title={`${l.label[0].toUpperCase()}${l.label.slice(1)}: ${l.company}`}
              sub={`Listed ${l.listing_date}. Actual ${n(l.actual)}%, predicted ${n(l.predicted)}%.`}>
              <ul className="space-y-1.5 text-sm">
                {l.top.slice(0, 5).map((t: Json) => (
                  <li key={t.feature} className="flex items-center justify-between gap-2">
                    <span className="truncate">{label(t.feature)}</span>
                    <span className={`tabular font-mono ${t.shap >= 0 ? "text-up" : "text-down"}`}>{t.shap >= 0 ? "+" : ""}{n(t.shap)}</span>
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">5 · Grey-market premium (GMP) experiment</h2>
        <Card title={`Does subscription add value beyond GMP? (fit on ${explain.gmp_n.fit} IPOs from 2020-2023, test on ${explain.gmp_n.test})`}
          sub="GMP = last public reading dated before listing day. Unofficial data, kept out of the production model.">
          <MetricTable rows={explain.gmp} cols={[
            { key: "inputs", label: "Inputs" }, { key: "MAE", label: "MAE", best: "min" }, { key: "R2", label: "R²", best: "max" },
            { key: "ROC_AUC", label: "ROC-AUC", best: "max" },
          ]} />
        </Card>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">6 · Data quality</h2>
        <Card title="Missing values per feature (%)" sub="Missingness is structured: bNII/sNII split only exists from 2022, financial ratios are thinner for older issues.">
          <div className="h-[520px]">
            <ResponsiveContainer>
              <BarChart data={eda.missing_pct.filter((m: Json) => m.pct > 0).map((m: Json) => ({ ...m }))} layout="vertical" margin={{ top: 0, right: 44, left: 8, bottom: 0 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" unit="%" {...AXIS} domain={[0, 100]} />
                <YAxis type="category" dataKey="feature" {...AXIS} width={150} interval={0} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<Tip fmt={(p) => <><b>{p.feature}</b>: {n(p.pct)}% missing</>} />} />
                <Bar dataKey="pct" fill="var(--series-2)" radius={[0, 4, 4, 0]} barSize={12}>
                  {eda.missing_pct.filter((m: Json) => m.pct > 0).map((m: Json) => <Cell key={m.feature} />)}
                  <LabelList dataKey="pct" position="right" formatter={(v: number) => `${v.toFixed(0)}%`} fill="var(--ink-2)" fontSize={11} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>
    </div>
  );
}
