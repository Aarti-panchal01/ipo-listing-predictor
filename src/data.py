"""Build the processed mainboard (and optional SME) IPO datasets from cached raw data.

Run:  python -m src.data          (from the repo root)

Every feature is something an investor could know BEFORE the listing day:
issue terms from the RHP, final-day subscription, and market data up to the
issue close date. Listing-day prices are only used to build the targets.
`leakage_check()` enforces this and is called on every build.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .parse import parse_detail, parse_gmp_history

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

TARGETS = ["listing_gain_pct", "listing_close_gain_pct", "listed_positive"]

# ---------------------------------------------------------------- raw loaders


def _report_rows(report: int, segment: str) -> list[dict]:
    rows = []
    for f in sorted((RAW / "api").glob(f"r{report}_{segment}_*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")).get("reportTableData", []))
    return rows


def _id_slug(html_link: str) -> tuple[int | None, str | None]:
    m = re.search(r"/ipo/([^/]+)/(\d+)/", html_link or "")
    return (int(m.group(2)), m.group(1)) if m else (None, None)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _f(x) -> float:
    if x is None or x == "":
        return np.nan
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def load_list(segment: str = "mainboard") -> pd.DataFrame:
    out = []
    for r in _report_rows(82, segment):
        ipo_id, slug = _id_slug(r.get("Company", ""))
        if ipo_id is None:
            continue
        out.append(
            {
                "ipo_id": ipo_id,
                "slug": slug,
                "company": _strip_tags(r.get("Company")),
                "pricing_method": r.get("Pricing Method"),
                "open_date": pd.to_datetime(r.get("Opening Date"), format="%d-%b-%Y", errors="coerce"),
                "close_date": pd.to_datetime(r.get("Closing Date"), format="%d-%b-%Y", errors="coerce"),
                "listing_date": pd.to_datetime(r.get("Listing Date"), format="%d-%b-%Y", errors="coerce"),
                "issue_price": _f(r.get("Issue Price (Rs.)")),
                "issue_size_cr": _f(r.get("Total Issue Amount (Incl.Firm reservations) (Rs.cr.)")),
                "fresh_issue_cr": _f(r.get("Fresh Capital (Rs.cr.)")),
                "ofs_cr": _f(r.get("Offer for sale (Rs.cr.)")),
                "fresh_issue_pct": _f(r.get("Fresh Capital as % of Issue Amount")),
                "ofs_pct": _f(r.get("Offer for sale as % of Issue Amount")),
                "exchange": (r.get("Listing at") or "").replace(" ", ""),
                "nse_symbol": r.get("~nse_symbol") or None,
                "bse_code": r.get("~bse_script_code") or None,
                "isin": r.get("~isin") or None,
            }
        )
    df = pd.DataFrame(out).drop_duplicates("ipo_id")
    return df


def load_subscription(segment: str = "mainboard") -> pd.DataFrame:
    report = 21 if segment == "mainboard" else 22
    out = []
    for r in _report_rows(report, segment):
        ipo_id = r.get("~id") or _id_slug(r.get("Company", ""))[0]
        if ipo_id is None:
            continue
        out.append(
            {
                "ipo_id": int(ipo_id),
                "qib_x": _f(r.get("QIB (x)")),
                "nii_x": _f(r.get("NII (x)")),
                "bnii_x": _f(r.get("bNII (x)")),
                "snii_x": _f(r.get("sNII (x)")),
                "retail_x": _f(r.get("Retail (x)")),
                "employee_x": _f(r.get("Employee (x)")),
                "shareholder_x": _f(r.get("Shareholder (x)")),
                "total_x": _f(r.get("Total (x)")),
                "applications": _f(r.get("Applications")),
                "subscription_as_on": r.get("Subscription as on"),
            }
        )
    return pd.DataFrame(out).drop_duplicates("ipo_id", keep="last")


def load_details(ids: list[int]) -> pd.DataFrame:
    files = {int(p.name.split("_", 1)[0]): p for p in (RAW / "html").glob("*.html")}
    out = []
    for i in ids:
        if i in files:
            d = parse_detail(files[i].read_text(encoding="utf-8"))
            d["ipo_id"] = i
            out.append(d)
    return pd.DataFrame(out)


def load_gmp(df: pd.DataFrame) -> pd.DataFrame:
    """Last public pre-listing GMP per IPO (InvestorGain, 2020+), matched on
    NSE symbol or BSE code. Returns ipo_id, gmp_rs, gmp_pct, gmp_date."""
    ig = RAW / "investorgain"
    rows = []
    for f in sorted(ig.glob("r377_*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")).get("reportTableData", []))
    by_key = {}
    for r in rows:
        if r.get("~IPO_Category") != "IPO" or not r.get("~URLRewrite_Folder_Name"):
            continue
        for part in str(r.get("Symbol", "")).split(","):
            part = part.strip()
            if part:
                by_key[part] = r["~URLRewrite_Folder_Name"]
    out = []
    for _, row in df.iterrows():
        path = by_key.get(str(row.get("nse_symbol"))) or by_key.get(str(row.get("bse_code")))
        if not path:
            continue
        f = ig / "html" / (path.strip("/").replace("/", "_") + ".html")
        if not f.exists():
            continue
        g = parse_gmp_history(f.read_text(encoding="utf-8"), row["listing_date"])
        out.append({"ipo_id": row["ipo_id"], **g})
    return pd.DataFrame(out, columns=["ipo_id", "gmp_rs", "gmp_pct", "gmp_date"])


# ---------------------------------------------------------------- market data


def load_market(refresh: bool = False) -> pd.DataFrame:
    """Daily Nifty 50 and India VIX closes from yfinance, cached to data/external."""
    EXT.mkdir(parents=True, exist_ok=True)
    f = EXT / "market_daily.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, parse_dates=["date"]).set_index("date")
    import yfinance as yf

    frames = {}
    for tick, name in (("^NSEI", "nifty"), ("^INDIAVIX", "vix")):
        h = yf.download(tick, start="2011-01-01", progress=False, auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"yfinance returned no data for {tick}")
        close = h["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[name] = close
    m = pd.DataFrame(frames)
    m.index = pd.to_datetime(m.index).tz_localize(None)
    m.index.name = "date"
    m.to_csv(f)
    return m


def load_sectors(symbols: list[str], refresh: bool = False) -> dict[str, dict]:
    """Sector/industry by NSE symbol from yfinance, cached. Sector is a static
    company attribute (known before listing), but yfinance only knows symbols
    that still trade, so delisted/renamed companies stay missing."""
    f = EXT / "sectors.json"
    cache = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    todo = [s for s in symbols if s and (refresh or s not in cache)]
    if todo:
        import time

        import yfinance as yf

        for i, s in enumerate(todo):
            try:
                info = yf.Ticker(f"{s}.NS").info or {}
                cache[s] = {"sector": info.get("sector"), "industry": info.get("industry")}
            except Exception as e:  # noqa: BLE001 - record and move on
                cache[s] = {"sector": None, "industry": None, "error": type(e).__name__}
            if i % 20 == 0:
                f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            time.sleep(0.5)
        f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


# ---------------------------------------------------------------- features


def _market_features(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Nifty returns and VIX measured at the issue close date (inclusive),
    using only trading days on or before it."""
    nifty = m["nifty"].dropna()
    vix = m["vix"].dropna()

    def asof(series, d):
        s = series.loc[:d]
        return (s.iloc[-1], s.index[-1]) if len(s) else (np.nan, pd.NaT)

    r7, r30, v, v_date = [], [], [], []
    for d in df["close_date"]:
        if pd.isna(d):
            r7.append(np.nan); r30.append(np.nan); v.append(np.nan); v_date.append(pd.NaT)
            continue
        n0, _ = asof(nifty, d)
        n7, _ = asof(nifty, d - pd.Timedelta(days=7))
        n30, _ = asof(nifty, d - pd.Timedelta(days=30))
        r7.append((n0 / n7 - 1) * 100)
        r30.append((n0 / n30 - 1) * 100)
        vv, vd = asof(vix, d)
        v.append(vv)
        v_date.append(vd)
    df["nifty_ret_7d"] = r7
    df["nifty_ret_30d"] = r30
    df["vix_close"] = v
    df["_vix_asof_date"] = v_date
    return df


def _heat_features(df: pd.DataFrame) -> pd.DataFrame:
    """IPO market 'heat', computed strictly from the past.

    ipo_heat_30d: mainboard issues that CLOSED in the 30 days up to this
        issue's close date (excluding itself). The brief suggested "IPOs listing
        that month", but that counts issues listing AFTER this one, which is
        future information, so we use a trailing window instead.
    prev_gain_mean_90d / prev_count_90d: mean listing gain of issues that had
        already LISTED before this issue closed (90-day window).
    """
    heat, pg, pc = [], [], []
    for i, row in df.iterrows():
        c = row["close_date"]
        win = df[(df["close_date"] > c - pd.Timedelta(days=30)) & (df["close_date"] <= c) & (df.index != i)]
        heat.append(len(win))
        past = df[(df["listing_date"] < c) & (df["listing_date"] >= c - pd.Timedelta(days=90))]
        pc.append(len(past))
        pg.append(past["listing_gain_pct"].mean() if len(past) else np.nan)
    df["ipo_heat_30d"] = heat
    df["prev_gain_mean_90d"] = pg
    df["prev_count_90d"] = pc
    return df


def _targets(df: pd.DataFrame) -> pd.DataFrame:
    # a 0.0 price on the page is a placeholder for "no trade data", not a real price
    for c in [c for c in df.columns if c.startswith("ld_")]:
        df.loc[df[c] <= 0, c] = np.nan
    open_px = df["ld_nse_open"].where(df["ld_nse_open"].notna(), df.get("ld_bse_open"))
    close_px = df["ld_nse_last_trade"].where(df["ld_nse_last_trade"].notna(), df.get("ld_bse_last_trade"))
    df["listing_open_price"] = open_px
    df["listing_close_price"] = close_px
    df["listing_gain_pct"] = (open_px - df["issue_price"]) / df["issue_price"] * 100
    df["listing_close_gain_pct"] = (close_px - df["issue_price"]) / df["issue_price"] * 100
    df["listed_positive"] = (df["listing_gain_pct"] > 0).astype("Int64").where(df["listing_gain_pct"].notna())
    return df


# Feature registry: name -> (source, when it becomes known). Anything not in
# here can not be used as a model feature (enforced by leakage_check).
FEATURES: dict[str, tuple[str, str]] = {
    "issue_size_cr": ("Chittorgarh report 82 (RHP)", "before issue opens"),
    "fresh_issue_pct": ("Chittorgarh report 82 (RHP)", "before issue opens"),
    "ofs_pct": ("Chittorgarh report 82 (RHP)", "before issue opens"),
    "price_band_upper": ("Chittorgarh detail page (RHP)", "before issue opens"),
    "issue_price": ("Chittorgarh report 82", "price band announcement / basis of allotment, before listing"),
    "lot_size": ("Chittorgarh detail page (RHP)", "before issue opens"),
    "min_investment": ("Chittorgarh detail page (RHP)", "before issue opens"),
    "exchange": ("Chittorgarh report 82", "before issue opens"),
    "sector": ("yfinance Ticker.info by NSE symbol", "static company attribute"),
    "anchor_portion_cr": ("Chittorgarh detail page", "anchor bid day (day before open)"),
    "anchor_pct_of_issue": ("derived: anchor_portion_cr / issue_size_cr", "anchor bid day"),
    "retail_reserved_pct": ("Chittorgarh detail page (RHP)", "before issue opens"),
    "qib_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "nii_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "bnii_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "snii_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "retail_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "employee_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "total_x": ("Chittorgarh report 21 (final day)", "issue close date"),
    "listing_year": ("derived from listing date (scheduled in RHP timetable)", "before issue opens"),
    "listing_month": ("derived from listing date (scheduled in RHP timetable)", "before issue opens"),
    "issue_days_open": ("derived: close_date - open_date", "before issue opens"),
    "days_close_to_listing": ("derived: listing_date - close_date (scheduled)", "before issue opens"),
    "nifty_ret_7d": ("yfinance ^NSEI, closes up to issue close date", "issue close date"),
    "nifty_ret_30d": ("yfinance ^NSEI, closes up to issue close date", "issue close date"),
    "vix_close": ("yfinance ^INDIAVIX, close on/before issue close date", "issue close date"),
    "ipo_heat_30d": ("derived: issues closed in trailing 30 days", "issue close date"),
    "prev_gain_mean_90d": ("derived: mean listing gain of issues already listed, trailing 90 days", "issue close date"),
    "prev_count_90d": ("derived: count of issues already listed, trailing 90 days", "issue close date"),
    "revenue_cr": ("Chittorgarh detail page (RHP financials, latest FY)", "before issue opens"),
    "pat_cr": ("Chittorgarh detail page (RHP financials, latest FY)", "before issue opens"),
    "revenue_growth_pct": ("derived from RHP financials", "before issue opens"),
    "pat_margin_pct": ("Chittorgarh detail page (RHP KPI)", "before issue opens"),
    "roe_pct": ("Chittorgarh detail page (RHP KPI)", "before issue opens"),
    "debt_equity": ("Chittorgarh detail page (RHP KPI)", "before issue opens"),
    "pe_post": ("Chittorgarh detail page (issue P/E, post-issue)", "price band announcement"),
    "mcap_post_cr": ("Chittorgarh detail page (market cap at offer price)", "price band announcement"),
    "promoter_post_pct": ("Chittorgarh detail page (RHP shareholding)", "before issue opens"),
    "promoter_dilution_pct": ("derived: promoter_pre_pct - promoter_post_pct", "before issue opens"),
}

# GMP lives outside FEATURES on purpose: it is unofficial, only exists from
# 2020, and is used in a separate "with vs without GMP" experiment.
GMP_FEATURES: dict[str, tuple[str, str]] = {
    "gmp_pct": ("InvestorGain GMP page, latest public reading dated before listing day", "evening before listing"),
}

FORBIDDEN_PATTERNS = (r"^ld_", r"listing_(open|close)_price", r"listing_gain", r"close_gain", r"listed_positive", r"cmp", r"ltp")


def leakage_check(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Fail loudly if any feature could carry listing-day (or later) information."""
    problems = []
    for c in feature_cols:
        if c not in FEATURES:
            problems.append(f"{c}: not in the FEATURES registry (unknown availability)")
        if any(re.search(p, c) for p in FORBIDDEN_PATTERNS):
            problems.append(f"{c}: matches a listing-day/target pattern")
    # market data must be observed on or before the issue close date
    if "_vix_asof_date" in df:
        bad = df["_vix_asof_date"] > df["close_date"]
        if bad.any():
            problems.append(f"vix_close observed after close date for {int(bad.sum())} rows")
    # close date must be strictly before listing (so close-date data is pre-listing)
    bad = df["close_date"] >= df["listing_date"]
    if bad.any():
        problems.append(f"{int(bad.sum())} rows have close_date >= listing_date")
    # no feature should be a near copy of the target
    y = df["listing_gain_pct"]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            r = df[[c]].assign(y=y).dropna()
            if len(r) > 30 and abs(r[c].corr(r["y"], method="spearman")) > 0.95:
                problems.append(f"{c}: |spearman| with target > 0.95, suspicious")
    if "gmp_date" in df:
        bad = df["gmp_date"].notna() & (pd.to_datetime(df["gmp_date"]) >= df["listing_date"].dt.normalize())
        if bad.any():
            problems.append(f"gmp_date on/after listing date for {int(bad.sum())} rows")
    if problems:
        raise AssertionError("LEAKAGE CHECK FAILED:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- build


def build_mainboard(start_year: int = 2012) -> pd.DataFrame:
    lst = load_list("mainboard")
    sub = load_subscription("mainboard")
    det = load_details(lst["ipo_id"].tolist())
    df = lst.merge(sub, on="ipo_id", how="left").merge(det, on="ipo_id", how="left")

    df = df[df["listing_date"].notna() & (df["listing_date"] <= pd.Timestamp.today().normalize())]
    df = df[df["listing_date"].dt.year >= start_year]
    df = _targets(df)
    n_before = len(df)
    df = df[df["listing_gain_pct"].notna()].copy()
    print(f"dropped {n_before - len(df)} listed issues with no listing-day price")

    df["price_band_upper"] = df["price_band_upper"].fillna(df["issue_price"])
    # older pages have no "Anchor Portion (Rs Cr)" table; derive it from anchor shares x issue price
    if "anchor_shares" in df:
        df["anchor_portion_cr"] = df["anchor_portion_cr"].fillna(df["anchor_shares"] * df["issue_price"] / 1e7)
    df["anchor_pct_of_issue"] = df["anchor_portion_cr"] / df["issue_size_cr"] * 100
    df["listing_year"] = df["listing_date"].dt.year
    df["listing_month"] = df["listing_date"].dt.month
    df["issue_days_open"] = (df["close_date"] - df["open_date"]).dt.days
    df["days_close_to_listing"] = (df["listing_date"] - df["close_date"]).dt.days
    df["revenue_growth_pct"] = (df["revenue_cr"] / df["revenue_cr_prev"] - 1) * 100
    df.loc[~np.isfinite(df["revenue_growth_pct"]), "revenue_growth_pct"] = np.nan
    df["promoter_dilution_pct"] = df["promoter_pre_pct"] - df["promoter_post_pct"]

    df = df.sort_values("close_date").reset_index(drop=True)
    df = _market_features(df, load_market())
    df = _heat_features(df)

    sectors = load_sectors(df["nse_symbol"].dropna().unique().tolist())
    df["sector"] = df["nse_symbol"].map(lambda s: (sectors.get(s) or {}).get("sector") if s else None)
    df["industry"] = df["nse_symbol"].map(lambda s: (sectors.get(s) or {}).get("industry") if s else None)

    gmp = load_gmp(df)
    df = df.merge(gmp, on="ipo_id", how="left")

    df["split"] = np.select(
        [df["listing_year"] <= 2022, df["listing_year"] == 2023], ["train", "val"], default="test"
    )
    leakage_check(df, list(FEATURES))
    return df


def build_sme() -> pd.DataFrame:
    """Optional SME extension dataset (2020+), from the cached InvestorGain GMP
    tracker rows: issue price, size, total subscription, listing open/close.
    SME issues follow different rules (market makers, 90%+ price caps on BSE SME,
    lot sizes worth 1-2 lakh) so they are never mixed into the mainboard model.
    The tracker's GMP column is the listing-morning value, so it is dropped."""
    rows = []
    for f in sorted((RAW / "investorgain").glob("r377_*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")).get("reportTableData", []))

    def money(x):
        t = _strip_tags(str(x or "")).replace("&#8377;", "").replace("₹", "")
        m = re.search(r"-?[\d,]*\.?\d+", t)
        return float(m.group(0).replace(",", "")) if m else np.nan

    out = []
    for r in rows:
        if r.get("~IPO_Category") != "SME":
            continue
        out.append({
            "company": _strip_tags(r.get("IPO", "")).replace(" SME", "").strip(),
            "listing_date": pd.to_datetime(r.get("Listing Dt"), format="%d-%b-%y", errors="coerce"),
            "issue_size_cr": money(r.get("Size")),
            "total_x": money(r.get("Sub")),
            "issue_price": money(r.get("Price")),
            "listing_open_price": money(r.get("Listing Price")),
            "listing_close_price": money(r.get("Listing Day Close")),
        })
    df = pd.DataFrame(out).dropna(subset=["listing_date", "issue_price", "listing_open_price"])
    df = df[df["listing_date"] <= pd.Timestamp.today().normalize()].drop_duplicates(["company", "listing_date"])
    df["listing_gain_pct"] = (df["listing_open_price"] - df["issue_price"]) / df["issue_price"] * 100
    df["listing_close_gain_pct"] = (df["listing_close_price"] - df["issue_price"]) / df["issue_price"] * 100
    df["listed_positive"] = (df["listing_gain_pct"] > 0).astype(int)
    return df.sort_values("listing_date").reset_index(drop=True)


def write_dictionary(df: pd.DataFrame, path: Path) -> None:
    other = {
        "ipo_id": "Chittorgarh IPO id", "slug": "Chittorgarh URL slug", "company": "Company name",
        "pricing_method": "Bookbuilding or Fixed Price", "open_date": "Issue open date",
        "close_date": "Issue close date", "listing_date": "Listing date", "fresh_issue_cr": "Fresh issue (Rs cr)",
        "ofs_cr": "Offer for sale (Rs cr)", "nse_symbol": "NSE symbol", "bse_code": "BSE scrip code", "isin": "ISIN",
        "listing_open_price": "TARGET SOURCE: listing-day open (NSE, else BSE)",
        "listing_close_price": "TARGET SOURCE: listing-day last trade (NSE, else BSE)",
        "listing_gain_pct": "TARGET (primary): (listing open - issue price) / issue price * 100",
        "listing_close_gain_pct": "TARGET (secondary): (listing close - issue price) / issue price * 100",
        "listed_positive": "TARGET (classification): listing_gain_pct > 0",
        "split": "train (listed <= 2022) / val (2023) / test (2024+)",
        "gmp_rs": "GMP experiment: GMP in Rs (same reading as gmp_pct)",
        "gmp_date": "Date of the GMP reading used (always before listing date)",
        "industry": "yfinance industry (finer than sector; not used as a feature)",
    }
    lines = [
        "# Data dictionary: `data/processed/ipo_mainboard.csv`",
        "",
        f"Rows: **{len(df)}** mainboard IPOs listed {df['listing_date'].min():%d %b %Y} to {df['listing_date'].max():%d %b %Y}. "
        f"Split sizes: {df['split'].value_counts().to_dict()}.",
        "",
        "Columns marked **feature** are in the `FEATURES` registry in `src/data.py`; `leakage_check()` refuses to build "
        "if any feature is outside the registry or looks like listing-day information.",
        "",
        "| Column | Role | Source | Known at | Missing % |",
        "|---|---|---|---|---|",
    ]
    for c in df.columns:
        if c.startswith("_"):
            continue
        miss = f"{df[c].isna().mean() * 100:.1f}"
        if c in FEATURES or c in GMP_FEATURES:
            src, when = FEATURES.get(c) or GMP_FEATURES[c]
            role = "feature" if c in FEATURES else "GMP experiment only"
            lines.append(f"| `{c}` | {role} | {src} | {when} | {miss} |")
        elif c in other:
            role = "target" if c in TARGETS else ("target source" if "TARGET SOURCE" in other[c] else "id / info")
            lines.append(f"| `{c}` | {role} | {other[c]} | | {miss} |")
        elif c.startswith("ld_"):
            lines.append(f"| `{c}` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | {miss} |")
        else:
            lines.append(f"| `{c}` | raw / unused | Chittorgarh | | {miss} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    df = build_mainboard()
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(PROC / "ipo_mainboard.csv", index=False)
    write_dictionary(df, DOCS / "data_dictionary.md")
    print(f"mainboard: {len(df)} rows; splits {df['split'].value_counts().to_dict()}")
    sme = build_sme()
    sme.to_csv(PROC / "ipo_sme.csv", index=False)
    print(f"sme (extension, 2020+): {len(sme)} rows")


if __name__ == "__main__":
    main()
