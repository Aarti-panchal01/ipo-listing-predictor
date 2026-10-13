"""Polite, cached scraper for public Chittorgarh IPO data.

Sources (both allow general crawling in robots.txt as of 13 Oct 2026):
  * webnodejs.chittorgarh.com JSON report API (the same endpoint the public
    report pages call from the browser)
      - report 82: IPO list (dates, issue price, fresh/OFS split, exchange)
      - report 21: mainboard subscription status by investor category
      - report 22: SME subscription status
  * www.chittorgarh.com/ipo/<slug>/<id>/ detail pages (lot size, price band,
    anchor portion, financials, P/E, promoter holding, listing-day prices)

Every response is cached under data/raw/ so nothing is fetched twice, and
there is a 2-3 second randomised delay between live requests. No login-gated
content is fetched: the category-wise subscription table on the detail page is
paywalled, so we use the public report 21 instead.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API_CACHE = RAW / "api"
HTML_CACHE = RAW / "html"

API = "https://webnodejs.chittorgarh.com/cloud/report/data-read"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 "
    "(student research project; ipo-listing-predictor)"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Referer": "https://www.chittorgarh.com/"})


def _sleep() -> None:
    time.sleep(random.uniform(2.0, 3.0))


def _get(url: str, retries: int = 3) -> requests.Response:
    for attempt in range(retries + 2):
        try:
            r = SESSION.get(url, timeout=45)
        except requests.RequestException as e:
            print(f"  network error ({type(e).__name__}), retry {attempt + 1}", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429):
            raise RuntimeError(f"Blocked ({r.status_code}) at {url}; stopping instead of working around it")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Giving up on {url} after repeated errors")


def fy(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def report_page(report: int, segment: str, year: int, page: int) -> list[dict]:
    """One page (5 rows) of a report for a calendar year. Cached."""
    API_CACHE.mkdir(parents=True, exist_ok=True)
    f = API_CACHE / f"r{report}_{segment}_{year}_p{page:03d}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")).get("reportTableData", [])
    url = f"{API}/{report}/{page}/1/{year}/{fy(year)}/0/{segment}/0?search=&v=1"
    r = _get(url)
    data = r.json()
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _sleep()
    if data.get("msg") in (-1, "-1"):
        return []
    return data.get("reportTableData", [])


def report_year(report: int, segment: str, year: int, max_pages: int = 200) -> list[dict]:
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        chunk = report_page(report, segment, year, page)
        key = tuple(sorted(str(x.get("~id") or x.get("~URLRewrite_Folder_Name")) for x in chunk))
        if not chunk or key in seen:
            break
        seen.add(key)
        rows.extend(chunk)
        if len(chunk) < 5:
            break
    return rows


def detail_page(slug: str, ipo_id: int) -> str:
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    f = HTML_CACHE / f"{ipo_id}_{slug}.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    r = _get(f"https://www.chittorgarh.com/ipo/{slug}/{ipo_id}/")
    f.write_text(r.text, encoding="utf-8")
    _sleep()
    return r.text


def main(argv: list[str]) -> None:
    what = argv[1] if len(argv) > 1 else "lists"
    years = range(2012, 2027)
    if what in ("lists", "sme"):
        segs = (("mainboard", 21),) if what == "lists" else (("sme", 22),)
        for seg, sub_report in segs:
            for y in years:
                n_list = len(report_year(82, seg, y))
                n_sub = len(report_year(sub_report, seg, y))
                print(f"{seg} {y}: list={n_list} subscription={n_sub}", flush=True)
    elif what == "details":
        rows = []
        for y in years:
            rows.extend(report_year(82, "mainboard", y))
        import re

        todo = {}
        for r in rows:
            m = re.search(r"/ipo/([^/]+)/(\d+)/", r["Company"])
            if m:
                todo[int(m.group(2))] = m.group(1)
        print(f"{len(todo)} mainboard detail pages", flush=True)
        for i, (ipo_id, slug) in enumerate(sorted(todo.items())):
            detail_page(slug, ipo_id)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        print("details done", flush=True)
    elif what == "gmp":
        scrape_gmp()



# ---------------------------------------------------------------- GMP (optional)
# InvestorGain (Chittorgarh's sister site) publishes a GMP performance tracker
# (report 377, from 2020) and a per-IPO GMP page whose public preview shows the
# 3 most recent GMP readings. We keep the latest reading dated strictly BEFORE
# the listing date; the tracker's own "GMP" column is the listing-morning value
# (last updated ~09:30 on listing day), which would be leakage, so it is not used.
IG_API = "https://webnodejs.investorgain.com/cloud/v2/report/data-read"
IG_CACHE = RAW / "investorgain"


def ig_tracker_year(year: int) -> list[dict]:
    IG_CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in range(1, 20):
        f = IG_CACHE / f"r377_{year}_p{page:02d}.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
        else:
            r = SESSION.get(
                f"{IG_API}/377/{page}/1/{year}/{fy(year)}/0/all?search=&v=1",
                headers={"Referer": "https://www.investorgain.com/"},
                timeout=45,
            )
            r.raise_for_status()
            data = r.json()
            f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            _sleep()
        chunk = data.get("reportTableData", [])
        rows.extend(chunk)
        if page >= int(data.get("totalPages") or 1) or not chunk:
            break
    return rows


def ig_gmp_page(path: str) -> str:
    f = IG_CACHE / "html" / (path.strip("/").replace("/", "_") + ".html")
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return f.read_text(encoding="utf-8")
    r = _get("https://www.investorgain.com" + path)
    f.write_text(r.text, encoding="utf-8")
    _sleep()
    return r.text


def scrape_gmp() -> None:
    rows = []
    for y in range(2020, 2027):
        got = ig_tracker_year(y)
        print(f"gmp tracker {y}: {len(got)}", flush=True)
        rows.extend(got)
    main = [r for r in rows if r.get("~IPO_Category") == "IPO" and r.get("~URLRewrite_Folder_Name")]
    print(f"{len(main)} mainboard GMP pages", flush=True)
    for i, r in enumerate(main):
        ig_gmp_page(r["~URLRewrite_Folder_Name"])
        if i % 25 == 0:
            print(f"  {i}/{len(main)}", flush=True)
    print("gmp done", flush=True)


if __name__ == "__main__":
    main(sys.argv)
