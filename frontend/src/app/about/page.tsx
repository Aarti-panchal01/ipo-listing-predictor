import Link from "next/link";

export const metadata = { title: "About · Listing Day" };

const STEPS = [
  ["Collect", "Public Chittorgarh IPO pages and report data (issue terms, final-day subscription, listing-day prices), yfinance for Nifty 50 and India VIX, InvestorGain for the optional GMP experiment. Raw responses are cached so nothing is fetched twice; requests are spaced 2-3 seconds apart and respect robots.txt. Login-gated data is never used."],
  ["Clean and engineer", "Log-transform the heavy-tailed subscription multiples and sizes, compute market context at the issue close date, and build 'IPO heat' features from the trailing 30/90 days only."],
  ["Split by time", "Train on IPOs listed 2012-2022, validate on 2023, test once on 2024 onward. No random split: the model never sees the future."],
  ["Model", "Baselines, decision tree, extra trees, gradient boosting, XGBoost (tuned with Optuna, early stopping on 2023), TabNet and a fastai MLP, plus a validation-weighted blend."],
  ["Explain", "Exact TreeSHAP values from XGBoost for every prediction, compared against XGBoost's built-in gain/weight/cover importance and permutation importance."],
  ["Serve", "FastAPI on Hugging Face Spaces (Docker) and this Next.js app on Vercel."],
];

export default function AboutPage() {
  return (
    <article className="max-w-3xl space-y-10">
      <header>
        <p className="font-mono text-xs uppercase tracking-widest text-muted">About</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Method, data, and what this can&apos;t do</h1>
        <p className="mt-3 text-ink-2">
          A course project for <em>Machine Learning with Business Data</em> (UE24AM342AA4, PES University). The question: using
          only information available before an Indian mainboard IPO lists, how well can we predict its listing-day gain, and
          whether it lists at a profit?
        </p>
      </header>

      <section>
        <h2 className="text-xl font-semibold">How it works</h2>
        <ol className="mt-4 space-y-4">
          {STEPS.map(([t, body], i) => (
            <li key={t} className="grid grid-cols-[2.25rem_1fr] gap-3">
              <span className="font-mono text-sm text-muted">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <h3 className="font-medium">{t}</h3>
                <p className="mt-1 text-sm text-ink-2">{body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Targets</h2>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-ink-2">
          <li><strong className="text-ink">Listing gain %</strong> = (listing-day open − issue price) / issue price × 100, NSE open where available, else BSE.</li>
          <li><strong className="text-ink">Listed positive</strong> = listing gain &gt; 0 (an issue that opens flat counts as not positive).</li>
          <li>Listing-day close gain is also in the dataset as a secondary target.</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Leakage handling</h2>
        <p className="mt-3 text-sm text-ink-2">
          No input may use information from listing day or later. The data pipeline enforces this in code: every feature must be
          registered with its source and the moment it becomes known, columns that look like listing-day data are refused,
          market data must be dated on or before the issue close date, the close date must be before the listing date, and any
          feature that correlates suspiciously well with the target fails the build. Two places where the obvious feature
          would have leaked, and what we did instead:
        </p>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-ink-2">
          <li>&ldquo;IPOs listing that month&rdquo; counts issues that list <em>after</em> this one. We use a trailing 30-day count of issues that already closed.</li>
          <li>The grey-market tracker&apos;s GMP column is updated around 9:30 am <em>on listing day</em>. We use the last reading dated the day before listing, and only in a separate experiment.</li>
        </ul>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Limitations</h2>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-ink-2">
          <li>Small data: a few hundred mainboard IPOs. Errors of 10-15 percentage points are typical, and big surprises are common.</li>
          <li>Regime shifts: the 2021 and 2024 IPO waves behave differently from earlier years. The model is trained on the past.</li>
          <li>Sector comes from Yahoo Finance by NSE symbol; companies that delisted or changed symbol are &ldquo;Unknown&rdquo;.</li>
          <li>Category-wise subscription is the final-day figure; intraday bidding patterns are not used.</li>
          <li>Grey-market premium is unofficial and unregulated; it is excluded from the production model.</li>
          <li>SME IPOs behave very differently (different rules, smaller sizes) and are not modelled.</li>
        </ul>
      </section>

      <section className="rounded-lg border border-line bg-warn-bg p-4 text-sm text-warn-ink">
        <h2 className="font-semibold">Disclaimer</h2>
        <p className="mt-1">
          This is an educational project. It is not investment advice, not a recommendation to apply for or sell any IPO, and not
          SEBI-registered investment advice. Do your own research or consult a SEBI-registered adviser.
        </p>
      </section>

      <p className="text-sm text-ink-2">
        Code, notebooks, data dictionary and report:{" "}
        <a className="underline underline-offset-2" href="https://github.com/Aarti-panchal01/ipo-listing-predictor">GitHub repository</a>. Model
        results by method are on the <Link className="underline underline-offset-2" href="/insights">Insights</Link> page.
      </p>
    </article>
  );
}
