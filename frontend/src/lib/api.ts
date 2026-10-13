// Client for the FastAPI backend on Hugging Face Spaces.
// Free Spaces sleep when idle; the first request can take 30-60 s while the
// container wakes up, so every call retries with backoff and reports progress.

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

export type IPOInput = Record<string, number | string | null>;

export interface Prediction {
  predicted_gain_pct: number;
  prob_positive: number;
  predicted_direction: "positive" | "flat_or_negative";
  confidence_note: string;
  model_version: string;
}

export interface Contribution {
  feature: string;
  label: string;
  value: number | string | null;
  shap: number;
}

export interface Explanation {
  base_value: number;
  predicted_gain_pct: number;
  contributions: Contribution[];
  note: string;
}

export interface PastIPO {
  ipo_id: number;
  company: string;
  listing_date: string;
  sector: string;
  issue_price: number;
  issue_size_cr: number;
  total_x: number | null;
  listing_gain_pct: number;
  listed_positive: number;
  predicted_gain_pct: number;
  prob_positive: number;
}

export interface InputSpec {
  type?: string;
  anyOf?: { type: string }[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  description?: string;
  enum?: string[];
  default?: unknown;
}

export interface Meta {
  model_version: string;
  trained_at: string;
  model: string;
  train_period: string;
  test_period: string;
  features: string[];
  inputs: Record<string, InputSpec>;
  sectors: string[];
  exchanges: string[];
  test_metrics: Record<string, number>;
  baseline_test_metrics: Record<string, number | string>;
  example: { company: string; listing_date: string; actual_listing_gain_pct: number; inputs: IPOInput };
  disclaimer: string;
}

export class ApiError extends Error {
  constructor(message: string, public status?: number, public detail?: unknown) {
    super(message);
  }
}

type Progress = (attempt: number, maxAttempts: number) => void;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function api<T>(
  path: string,
  init: RequestInit = {},
  { attempts = 6, timeoutMs = 25000, onRetry }: { attempts?: number; timeoutMs?: number; onRetry?: Progress } = {},
): Promise<T> {
  if (!API_BASE) throw new ApiError("NEXT_PUBLIC_API_URL is not set");
  let lastErr: unknown;
  for (let i = 1; i <= attempts; i++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        ...init,
        signal: ctrl.signal,
        headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
      });
      clearTimeout(timer);
      if (res.status === 422 || res.status === 400) {
        const body = await res.json().catch(() => ({}));
        throw new ApiError(formatValidation(body), res.status, body);
      }
      // 502/503/504 = Space still waking up; retry
      if (res.status >= 500) throw new Error(`HTTP ${res.status}`);
      if (!res.ok) throw new ApiError(`Request failed (HTTP ${res.status})`, res.status);
      return (await res.json()) as T;
    } catch (e) {
      clearTimeout(timer);
      if (e instanceof ApiError) throw e;
      lastErr = e;
      if (i < attempts) {
        onRetry?.(i, attempts);
        await sleep(Math.min(2000 * 2 ** (i - 1), 15000));
      }
    }
  }
  throw new ApiError(
    "The prediction server did not respond. It may be waking up from sleep; please try again in a minute.",
    undefined,
    lastErr,
  );
}

function formatValidation(body: { detail?: unknown }): string {
  const d = body?.detail;
  if (Array.isArray(d)) {
    return d
      .map((e: { loc?: (string | number)[]; msg?: string }) => {
        const field = (e.loc ?? []).filter((x) => x !== "body").join(".");
        return field ? `${field}: ${e.msg}` : e.msg;
      })
      .join(" · ");
  }
  return typeof d === "string" ? d : "Some inputs are invalid.";
}

export const getMeta = (onRetry?: Progress) => api<Meta>("/meta", {}, { onRetry });
export const predict = (body: IPOInput, onRetry?: Progress) =>
  api<Prediction>("/predict", { method: "POST", body: JSON.stringify(body) }, { onRetry });
export const explain = (body: IPOInput, onRetry?: Progress) =>
  api<Explanation>("/explain?top=12", { method: "POST", body: JSON.stringify(body) }, { onRetry });
export const getPast = (onRetry?: Progress) =>
  api<{ count: number; items: PastIPO[] }>("/past-ipos", {}, { onRetry });
