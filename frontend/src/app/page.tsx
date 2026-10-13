"use client";

import { useCallback, useEffect, useState } from "react";
import { ErrorBox, Loading } from "@/components/Status";
import Waterfall from "@/components/Waterfall";
import {
  ApiError, explain, getMeta, predict,
  type Explanation, type InputSpec, type IPOInput, type Meta, type Prediction,
} from "@/lib/api";

type Field = { key: string; label: string; unit?: string; step?: string; required?: boolean };
type Group = { title: string; hint: string; fields: Field[]; optional?: boolean };

const GROUPS: Group[] = [
  {
    title: "Issue terms",
    hint: "From the red herring prospectus (RHP).",
    fields: [
      { key: "issue_size_cr", label: "Issue size", unit: "₹ cr", required: true },
      { key: "price_band_upper", label: "Price band, upper", unit: "₹", required: true },
      { key: "fresh_issue_pct", label: "Fresh issue share", unit: "% of issue", required: true },
      { key: "lot_size", label: "Lot size", unit: "shares" },
      { key: "min_investment", label: "Retail minimum", unit: "₹" },
      { key: "anchor_portion_cr", label: "Anchor book", unit: "₹ cr" },
      { key: "retail_reserved_pct", label: "Retail quota", unit: "%" },
    ],
  },
  {
    title: "Final-day subscription",
    hint: "Times subscribed when bidding closes (ex-anchor).",
    fields: [
      { key: "total_x", label: "Total", unit: "x", required: true },
      { key: "qib_x", label: "QIB", unit: "x" },
      { key: "nii_x", label: "NII / HNI", unit: "x" },
      { key: "bnii_x", label: "bNII (>₹10L)", unit: "x" },
      { key: "snii_x", label: "sNII (<₹10L)", unit: "x" },
      { key: "retail_x", label: "Retail", unit: "x" },
      { key: "employee_x", label: "Employee", unit: "x" },
    ],
  },
  {
    title: "Timing and market",
    hint: "Measured at the issue close date.",
    fields: [
      { key: "listing_month", label: "Listing month", unit: "1-12", step: "1", required: true },
      { key: "issue_days_open", label: "Days open", unit: "days", step: "1" },
      { key: "nifty_ret_7d", label: "Nifty 50, 7-day return", unit: "%" },
      { key: "nifty_ret_30d", label: "Nifty 50, 30-day return", unit: "%" },
      { key: "vix_close", label: "India VIX", unit: "" },
      { key: "ipo_heat_30d", label: "IPOs closed in last 30 days", unit: "count", step: "1" },
      { key: "prev_gain_mean_90d", label: "Avg listing gain, last 90 days", unit: "%" },
      { key: "prev_count_90d", label: "IPOs listed, last 90 days", unit: "count", step: "1" },
    ],
  },
  {
    title: "Company financials",
    hint: "Latest full financial year in the RHP. Leave blank if unknown.",
    optional: true,
    fields: [
      { key: "revenue_cr", label: "Revenue", unit: "₹ cr" },
      { key: "pat_cr", label: "Profit after tax", unit: "₹ cr" },
      { key: "revenue_growth_pct", label: "Revenue growth", unit: "%" },
      { key: "pat_margin_pct", label: "PAT margin", unit: "%" },
      { key: "roe_pct", label: "ROE", unit: "%" },
      { key: "debt_equity", label: "Debt / equity", unit: "x" },
      { key: "pe_post", label: "Issue P/E (post)", unit: "x" },
      { key: "mcap_post_cr", label: "Market cap at issue price", unit: "₹ cr" },
      { key: "promoter_post_pct", label: "Promoter holding after", unit: "%" },
      { key: "promoter_dilution_pct", label: "Promoter dilution", unit: "pts" },
    ],
  },
];

function bounds(spec?: InputSpec): { min?: number; max?: number } {
  if (!spec) return {};
  const inner = spec.anyOf?.find((a) => (a as InputSpec).minimum !== undefined || (a as InputSpec).maximum !== undefined) as InputSpec | undefined;
  const s = inner ?? spec;
  return { min: s.minimum ?? s.exclusiveMinimum, max: s.maximum };
}

const toForm = (inputs: IPOInput) =>
  Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v === null || v === undefined ? "" : String(typeof v === "number" ? +v.toFixed(2) : v)]));

export default function PredictorPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [wake, setWake] = useState<{ attempt: number; max: number } | null>(null);
  const [form, setForm] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<{ p: Prediction; e: Explanation } | null>(null);
  const [showFin, setShowFin] = useState(true);

  const loadMeta = useCallback(() => {
    setMetaErr(null);
    setWake({ attempt: 0, max: 6 });
    getMeta((attempt, max) => setWake({ attempt, max }))
      .then((m) => {
        setMeta(m);
        setForm(toForm(m.example.inputs));
      })
      .catch((e: Error) => setMetaErr(e.message))
      .finally(() => setWake(null));
  }, []);

  useEffect(loadMeta, [loadMeta]);

  const payload = (): IPOInput => {
    const out: IPOInput = { exchange: form.exchange || "BSE,NSE", sector: form.sector || "Unknown" };
    for (const g of GROUPS) for (const f of g.fields) {
      const v = form[f.key];
      if (v !== undefined && v !== "") out[f.key] = Number(v);
    }
    return out;
  };

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setBusy(true);
    setErr(null);
    setResult(null);
    const body = payload();
    try {
      const onRetry = (attempt: number, max: number) => setWake({ attempt, max });
      const [p, e] = await Promise.all([predict(body, onRetry), explain(body, onRetry)]);
      setResult({ p, e });
      requestAnimationFrame(() => document.getElementById("result")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : "Unexpected error");
    } finally {
      setBusy(false);
      setWake(null);
    }
  };

  const set = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className="space-y-8">
      <section className="max-w-3xl">
        <p className="font-mono text-xs uppercase tracking-widest text-muted">Predictor</p>
        <h1 className="mt-2 text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
          Will this IPO list above its issue price?
        </h1>
        <p className="mt-3 text-ink-2">
          Enter what is known before listing day: issue terms, final-day subscription and market conditions. A gradient-boosted
          model trained on Indian mainboard IPOs from 2012 to 2023 estimates the listing-day open versus the issue price,
          and SHAP shows which inputs drove that estimate.
        </p>
      </section>

      {!meta && !metaErr && <Loading label="Connecting to the prediction server…" attempt={wake?.attempt} max={wake?.max} />}
      {metaErr && <ErrorBox message={metaErr} onRetry={loadMeta} />}

      {meta && (
        <form onSubmit={submit} className="grid gap-6 lg:grid-cols-[1fr_minmax(0,22rem)]" noValidate={false}>
          <div className="space-y-5">
            <div className="rounded-lg border border-line bg-surface p-4 text-sm text-ink-2">
              Prefilled with <strong className="text-ink">{meta.example.company}</strong>, listed{" "}
              {new Date(meta.example.listing_date).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}
              . It actually opened at <strong className="text-ink">{meta.example.actual_listing_gain_pct.toFixed(1)}%</strong> vs its
              issue price, so you can check the model against a real outcome.{" "}
              <button type="button" className="underline underline-offset-2 hover:text-ink" onClick={() => setForm(toForm(meta.example.inputs))}>
                Reset to example
              </button>
            </div>

            <fieldset className="grid gap-4 rounded-lg border border-line bg-surface p-4 sm:grid-cols-2">
              <legend className="px-1 text-sm font-semibold">Company</legend>
              <label className="text-sm">
                <span className="text-ink-2">Sector</span>
                <select value={form.sector ?? "Unknown"} onChange={(e) => set("sector", e.target.value)}
                  className="mt-1 w-full rounded-md border border-line bg-page px-3 py-2">
                  {meta.sectors.map((s) => <option key={s}>{s}</option>)}
                </select>
              </label>
              <label className="text-sm">
                <span className="text-ink-2">Listing exchange</span>
                <select value={form.exchange ?? "BSE,NSE"} onChange={(e) => set("exchange", e.target.value)}
                  className="mt-1 w-full rounded-md border border-line bg-page px-3 py-2">
                  {meta.exchanges.map((s) => <option key={s} value={s}>{s.replace(",", " + ")}</option>)}
                </select>
              </label>
            </fieldset>

            {GROUPS.map((g) => (
              <fieldset key={g.title} className="rounded-lg border border-line bg-surface p-4">
                <legend className="px-1 text-sm font-semibold">{g.title}</legend>
                <div className="flex items-start justify-between gap-3">
                  <p className="text-xs text-muted">{g.hint}</p>
                  {g.optional && (
                    <button type="button" onClick={() => setShowFin((s) => !s)} className="shrink-0 text-xs underline underline-offset-2" aria-expanded={showFin}>
                      {showFin ? "Hide" : "Show"}
                    </button>
                  )}
                </div>
                {(!g.optional || showFin) && (
                  <div className="mt-3 grid gap-x-4 gap-y-3 sm:grid-cols-2 xl:grid-cols-3">
                    {g.fields.map((f) => {
                      const b = bounds(meta.inputs[f.key]);
                      return (
                        <label key={f.key} className="text-sm">
                          <span className="flex justify-between gap-2 text-ink-2">
                            <span>{f.label}{f.required && <span className="text-bad"> *</span>}</span>
                            <span className="text-xs text-muted">{f.unit}</span>
                          </span>
                          <input
                            type="number" inputMode="decimal" step={f.step ?? "any"}
                            min={b.min} max={b.max} required={f.required}
                            value={form[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)}
                            placeholder={f.required ? "required" : "optional"}
                            className="tabular mt-1 w-full rounded-md border border-line bg-page px-3 py-2 font-mono invalid:border-down"
                          />
                        </label>
                      );
                    })}
                  </div>
                )}
              </fieldset>
            ))}
          </div>

          <aside className="lg:sticky lg:top-4 lg:self-start">
            <div className="rounded-lg border border-line bg-surface p-4">
              <button type="submit" disabled={busy}
                className="w-full rounded-md bg-ink px-4 py-3 font-medium text-page transition-opacity hover:opacity-90 disabled:opacity-60">
                {busy ? "Predicting…" : "Predict listing gain"}
              </button>
              <p className="mt-3 text-xs text-muted">
                Model {meta.model_version}, trained on {meta.train_period}. Test MAE on {meta.test_period}:{" "}
                <span className="tabular">{meta.test_metrics.MAE.toFixed(1)}</span> pts.
              </p>
            </div>
            <div className="mt-4 space-y-3">
              {busy && <Loading label="Running the model…" attempt={wake?.attempt} max={wake?.max} />}
              {err && <ErrorBox message={err} />}
            </div>
          </aside>
        </form>
      )}

      {result && <Result p={result.p} e={result.e} />}
    </div>
  );
}

function Result({ p, e }: { p: Prediction; e: Explanation }) {
  const up = p.predicted_gain_pct > 0;
  return (
    <section id="result" aria-live="polite" className="scroll-mt-4 overflow-hidden rounded-xl border border-line bg-surface">
      <div className="grid gap-6 p-5 sm:grid-cols-3 sm:p-6">
        <div>
          <p className="text-xs uppercase tracking-wider text-muted">Predicted listing gain</p>
          <p className={`mt-1 font-mono text-5xl font-semibold tracking-tight ${up ? "text-good" : "text-bad"}`}>
            {up ? "+" : ""}{p.predicted_gain_pct.toFixed(1)}%
          </p>
          <p className="mt-1 text-xs text-muted">listing-day open vs issue price</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-muted">Chance of listing above issue price</p>
          <p className="mt-1 font-mono text-5xl font-semibold tracking-tight">{Math.round(p.prob_positive * 100)}%</p>
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-surface-2" role="img" aria-label={`${Math.round(p.prob_positive * 100)} percent`}>
            <div className="h-full rounded-full bg-accent" style={{ width: `${p.prob_positive * 100}%` }} />
          </div>
        </div>
        <div className="text-sm text-ink-2">
          <p className="text-xs uppercase tracking-wider text-muted">How much to trust this</p>
          <p className="mt-1">{p.confidence_note}</p>
        </div>
      </div>
      <div className="perforated" aria-hidden />
      <div className="p-5 sm:p-6">
        <h2 className="text-lg font-semibold">Why: the inputs that moved this prediction</h2>
        <p className="mb-4 mt-1 text-sm text-ink-2">
          SHAP values from the gain model. Each bar is how much one input moved the estimate away from the average IPO.
        </p>
        <Waterfall data={e} />
        <p className="mt-4 text-xs text-muted">{e.note} Not investment advice.</p>
      </div>
    </section>
  );
}
