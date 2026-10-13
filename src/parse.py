"""Parse cached Chittorgarh IPO detail pages into flat dictionaries.

Only tables that describe the issue as filed (RHP data) and the listing-day
price table are read. The listing-day table is used for TARGETS only; the
leakage check in data.py makes sure none of those columns become features.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

NUM = re.compile(r"-?[\d,]*\.?\d+")


def num(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.replace("₹", "").replace("Rs.", "")
    m = NUM.search(t.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _tables(html: str) -> list[list[list[str]]]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for t in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in t.find_all("tr")]
        out.append([r for r in rows if r])
    return out


def _kv(rows: list[list[str]]) -> dict[str, str]:
    return {r[0]: r[1] for r in rows if len(r) >= 2}


def parse_detail(html: str) -> dict:
    d: dict = {}
    for rows in _tables(html):
        if not rows:
            continue
        first = rows[0][0] if rows[0] else ""
        kv = _kv(rows)

        if "Price Band" in kv or "Lot Size" in kv or "Issue Price" in kv:
            band = kv.get("Price Band", "")
            nums = [num(x) for x in re.split(r"\bto\b", band)] if band else []
            nums = [n for n in nums if n is not None]
            if nums:
                d["price_band_lower"] = nums[0]
                d["price_band_upper"] = nums[-1]
            d["issue_price_detail"] = num(kv.get("Issue Price"))
            d["face_value"] = num(kv.get("Face Value"))
            d["lot_size"] = num(kv.get("Lot Size"))
            d["sale_type"] = kv.get("Sale Type")
            d["issue_type"] = kv.get("Issue Type")
            d["listing_at_detail"] = kv.get("Listing At")

        elif "Total Issue Size" in kv:
            d["shares_pre_issue"] = num(kv.get("Share Holding Pre Issue"))
            d["shares_post_issue"] = num(kv.get("Share Holding Post Issue"))

        elif first == "Investor Category" and len(rows[0]) >= 3:
            # some pages have both "% of Net Issue" and "% of Total Issue" columns
            hdr = rows[0]
            pct_col = hdr.index("% of Total Issue") if "% of Total Issue" in hdr else 2
            for r in rows[1:]:
                if len(r) <= pct_col:
                    continue
                label, pct = r[0], num(r[pct_col])
                if "Ex. Anchor" in label or "Ex Anchor" in label:
                    continue
                if label.startswith("QIB"):
                    d["qib_reserved_pct"] = pct
                elif "Anchor" in label:
                    d["anchor_reserved_pct"] = pct
                    d["anchor_shares"] = num(r[1])
                elif label.startswith("NII"):
                    d["nii_reserved_pct"] = pct
                elif label.startswith("Retail"):
                    d["retail_reserved_pct"] = pct
                elif label.startswith("Employee"):
                    d["employee_reserved_pct"] = pct
                elif label.startswith("Shareholder"):
                    d["shareholder_reserved_pct"] = pct

        elif first == "Application" and len(rows) > 1:
            for r in rows[1:]:
                if r[0] == "Retail (Min)" and len(r) >= 4:
                    d["min_investment"] = num(r[3])

        elif "Anchor Portion (₹ Cr.)" in kv:
            d["anchor_portion_cr"] = num(kv.get("Anchor Portion (₹ Cr.)"))

        elif first == "Period Ended" and len(rows[0]) >= 2:
            periods = rows[0][1:]
            table = {r[0]: r[1:] for r in rows[1:] if len(r) > 1}
            # latest full financial year (31 Mar) column, and the one before it
            fy_idx = [i for i, p in enumerate(periods) if p.startswith("31 Mar")]
            if fy_idx:
                i0 = fy_idx[0]
                i1 = fy_idx[1] if len(fy_idx) > 1 else None
                d["fin_period"] = periods[i0]

                def pick(key, i):
                    vals = table.get(key)
                    return num(vals[i]) if vals and i is not None and i < len(vals) else None

                for key, col in (
                    ("Total Income", "revenue_cr"),
                    ("Revenue", "revenue_cr"),
                    ("Profit After Tax", "pat_cr"),
                    ("NET Worth", "net_worth_cr"),
                    ("Net Worth", "net_worth_cr"),
                    ("Total Borrowing", "borrowing_cr"),
                    ("Assets", "assets_cr"),
                ):
                    if key in table and d.get(col) is None:
                        d[col] = pick(key, i0)
                        if col in ("revenue_cr", "pat_cr"):
                            d[col + "_prev"] = pick(key, i1)

        elif first == "KPI" or ("ROE" in kv and "RoNW" in kv) or "ROCE" in kv:
            d["roe_pct"] = num(kv.get("ROE"))
            d["roce_pct"] = num(kv.get("ROCE"))
            d["debt_equity"] = num(kv.get("Debt/Equity"))
            d["pat_margin_pct"] = num(kv.get("PAT Margin"))

        elif first == "Valuation Metric":
            for r in rows[1:]:
                if r[0].startswith("P/E"):
                    d["pe_pre"] = num(r[1]) if len(r) > 1 else None
                    d["pe_post"] = num(r[2]) if len(r) > 2 else None
                elif r[0].startswith("Market Cap"):
                    d["mcap_pre_cr"] = num(r[1]) if len(r) > 1 else None
                    d["mcap_post_cr"] = num(r[2]) if len(r) > 2 else None

        elif first == "Category" and len(rows[0]) >= 3 and rows[0][1] == "Pre IPO":
            for r in rows[1:]:
                if r[0].startswith("Promoter"):
                    d["promoter_pre_pct"] = num(r[1]) if len(r) > 1 else None
                    d["promoter_post_pct"] = num(r[2]) if len(r) > 2 else None

        elif first == "Category" and len(rows[0]) >= 2 and rows[0][1].startswith("Subscription"):
            for r in rows[1:]:
                if r[0] == "Total" and len(r) > 1:
                    d["total_x_detail"] = num(r[1])

        elif first == "Price Details":
            # listing-day trading table: TARGET SOURCE ONLY (never a feature)
            hdr = rows[0]
            for r in rows[1:]:
                for j, exch in enumerate(hdr[1:], start=1):
                    if j < len(r):
                        key = r[0].lower().replace(" ", "_")
                        d[f"ld_{exch.lower()}_{key}"] = num(r[j])
    return d


def parse_gmp_history(html: str, listing_date) -> dict:
    """Latest public GMP reading dated strictly before the listing date.

    Rows look like ['28-Dec 17:27', '₹32 (7.14%) ▼', ...] and carry no year,
    so the year is inferred from the listing date (a December reading for a
    January listing belongs to the previous year).
    """
    import pandas as pd

    out = {"gmp_rs": None, "gmp_pct": None, "gmp_date": None}
    if listing_date is None or pd.isna(listing_date):
        return out
    best = None
    for rows in _tables(html):
        if not rows or rows[0][0] != "GMP Date":
            continue
        for r in rows[1:]:
            if len(r) < 2:
                continue
            m = re.match(r"(\d{1,2})-([A-Za-z]{3})", r[0])
            if not m:
                continue
            d = pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{listing_date.year}", format="%d-%b-%Y", errors="coerce")
            if pd.isna(d):
                continue
            if d > listing_date + pd.Timedelta(days=31):
                d = d - pd.DateOffset(years=1)
            if d >= listing_date.normalize():
                continue  # listing-day reading: leakage, skip
            g = re.search(r"(-?[\d,.]+)\s*\((-?[\d.]+)%\)", r[1].replace("\u20b9", ""))
            if not g:
                continue
            if best is None or d > best[0]:
                best = (d, float(g.group(1).replace(",", "")), float(g.group(2)))
    if best:
        out = {"gmp_rs": best[1], "gmp_pct": best[2], "gmp_date": best[0]}
    return out
