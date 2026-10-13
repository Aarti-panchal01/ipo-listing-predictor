# Data dictionary: `data/processed/ipo_mainboard.csv`

Rows: **604** mainboard IPOs listed 09 Mar 2012 to 08 Oct 2026. Split sizes: {'test': 290, 'train': 256, 'val': 58}.

Columns marked **feature** are in the `FEATURES` registry in `src/data.py`; `leakage_check()` refuses to build if any feature is outside the registry or looks like listing-day information.

| Column | Role | Source | Known at | Missing % |
|---|---|---|---|---|
| `ipo_id` | id / info | Chittorgarh IPO id | | 0.0 |
| `slug` | id / info | Chittorgarh URL slug | | 0.0 |
| `company` | id / info | Company name | | 0.0 |
| `pricing_method` | id / info | Bookbuilding or Fixed Price | | 0.0 |
| `open_date` | id / info | Issue open date | | 0.0 |
| `close_date` | id / info | Issue close date | | 0.0 |
| `listing_date` | id / info | Listing date | | 0.0 |
| `issue_price` | feature | Chittorgarh report 82 | price band announcement / basis of allotment, before listing | 0.0 |
| `issue_size_cr` | feature | Chittorgarh report 82 (RHP) | before issue opens | 0.0 |
| `fresh_issue_cr` | id / info | Fresh issue (Rs cr) | | 21.2 |
| `ofs_cr` | id / info | Offer for sale (Rs cr) | | 19.4 |
| `fresh_issue_pct` | feature | Chittorgarh report 82 (RHP) | before issue opens | 0.0 |
| `ofs_pct` | feature | Chittorgarh report 82 (RHP) | before issue opens | 0.0 |
| `exchange` | feature | Chittorgarh report 82 | before issue opens | 0.0 |
| `nse_symbol` | id / info | NSE symbol | | 0.5 |
| `bse_code` | id / info | BSE scrip code | | 0.3 |
| `isin` | id / info | ISIN | | 0.0 |
| `qib_x` | feature | Chittorgarh report 21 (final day) | issue close date | 0.2 |
| `nii_x` | feature | Chittorgarh report 21 (final day) | issue close date | 0.0 |
| `bnii_x` | feature | Chittorgarh report 21 (final day) | issue close date | 37.4 |
| `snii_x` | feature | Chittorgarh report 21 (final day) | issue close date | 37.4 |
| `retail_x` | feature | Chittorgarh report 21 (final day) | issue close date | 0.0 |
| `employee_x` | feature | Chittorgarh report 21 (final day) | issue close date | 58.8 |
| `shareholder_x` | raw / unused | Chittorgarh | | 95.9 |
| `total_x` | feature | Chittorgarh report 21 (final day) | issue close date | 0.0 |
| `applications` | raw / unused | Chittorgarh | | 42.9 |
| `subscription_as_on` | raw / unused | Chittorgarh | | 0.0 |
| `price_band_lower` | raw / unused | Chittorgarh | | 0.2 |
| `price_band_upper` | feature | Chittorgarh detail page (RHP) | before issue opens | 0.0 |
| `issue_price_detail` | raw / unused | Chittorgarh | | 0.2 |
| `face_value` | raw / unused | Chittorgarh | | 0.2 |
| `lot_size` | feature | Chittorgarh detail page (RHP) | before issue opens | 0.2 |
| `sale_type` | raw / unused | Chittorgarh | | 0.2 |
| `issue_type` | raw / unused | Chittorgarh | | 0.2 |
| `listing_at_detail` | raw / unused | Chittorgarh | | 0.2 |
| `shares_pre_issue` | raw / unused | Chittorgarh | | 0.0 |
| `shares_post_issue` | raw / unused | Chittorgarh | | 0.0 |
| `qib_reserved_pct` | raw / unused | Chittorgarh | | 0.2 |
| `anchor_reserved_pct` | raw / unused | Chittorgarh | | 7.0 |
| `anchor_shares` | raw / unused | Chittorgarh | | 7.0 |
| `nii_reserved_pct` | raw / unused | Chittorgarh | | 0.0 |
| `retail_reserved_pct` | feature | Chittorgarh detail page (RHP) | before issue opens | 0.0 |
| `min_investment` | feature | Chittorgarh detail page (RHP) | before issue opens | 0.0 |
| `fin_period` | raw / unused | Chittorgarh | | 0.5 |
| `revenue_cr` | feature | Chittorgarh detail page (RHP financials, latest FY) | before issue opens | 1.2 |
| `revenue_cr_prev` | raw / unused | Chittorgarh | | 2.3 |
| `pat_cr` | feature | Chittorgarh detail page (RHP financials, latest FY) | before issue opens | 0.5 |
| `pat_cr_prev` | raw / unused | Chittorgarh | | 1.7 |
| `net_worth_cr` | raw / unused | Chittorgarh | | 2.2 |
| `borrowing_cr` | raw / unused | Chittorgarh | | 13.1 |
| `assets_cr` | raw / unused | Chittorgarh | | 0.7 |
| `roe_pct` | feature | Chittorgarh detail page (RHP KPI) | before issue opens | 48.8 |
| `roce_pct` | raw / unused | Chittorgarh | | 48.2 |
| `debt_equity` | feature | Chittorgarh detail page (RHP KPI) | before issue opens | 51.3 |
| `pat_margin_pct` | feature | Chittorgarh detail page (RHP KPI) | before issue opens | 47.2 |
| `mcap_pre_cr` | raw / unused | Chittorgarh | | 30.8 |
| `mcap_post_cr` | feature | Chittorgarh detail page (market cap at offer price) | price band announcement | 29.1 |
| `promoter_pre_pct` | raw / unused | Chittorgarh | | 4.0 |
| `promoter_post_pct` | feature | Chittorgarh detail page (RHP shareholding) | before issue opens | 4.6 |
| `total_x_detail` | raw / unused | Chittorgarh | | 0.0 |
| `ld_bse_final_issue_price` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 0.7 |
| `ld_nse_final_issue_price` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 1.8 |
| `ld_bse_open` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 0.7 |
| `ld_nse_open` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 1.8 |
| `ld_bse_low` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 0.7 |
| `ld_nse_low` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 1.8 |
| `ld_bse_high` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 0.7 |
| `ld_nse_high` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 1.8 |
| `ld_bse_last_trade` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 0.7 |
| `ld_nse_last_trade` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 1.8 |
| `employee_reserved_pct` | raw / unused | Chittorgarh | | 58.8 |
| `shareholder_reserved_pct` | raw / unused | Chittorgarh | | 95.9 |
| `pe_pre` | raw / unused | Chittorgarh | | 37.4 |
| `pe_post` | feature | Chittorgarh detail page (issue P/E, post-issue) | price band announcement | 48.3 |
| `ld_nse sme_final_issue_price` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_nse sme_open` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_nse sme_low` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_nse sme_high` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_nse sme_last_trade` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `anchor_portion_cr` | feature | Chittorgarh detail page | anchor bid day (day before open) | 7.0 |
| `ld_bse sme_final_issue_price` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_bse sme_open` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_bse sme_low` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_bse sme_high` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `ld_bse sme_last_trade` | listing-day raw (never a feature) | Chittorgarh detail page | listing day | 100.0 |
| `listing_open_price` | target source | TARGET SOURCE: listing-day open (NSE, else BSE) | | 0.0 |
| `listing_close_price` | target source | TARGET SOURCE: listing-day last trade (NSE, else BSE) | | 0.0 |
| `listing_gain_pct` | target | TARGET (primary): (listing open - issue price) / issue price * 100 | | 0.0 |
| `listing_close_gain_pct` | target | TARGET (secondary): (listing close - issue price) / issue price * 100 | | 0.0 |
| `listed_positive` | target | TARGET (classification): listing_gain_pct > 0 | | 0.0 |
| `anchor_pct_of_issue` | feature | derived: anchor_portion_cr / issue_size_cr | anchor bid day | 7.0 |
| `listing_year` | feature | derived from listing date (scheduled in RHP timetable) | before issue opens | 0.0 |
| `listing_month` | feature | derived from listing date (scheduled in RHP timetable) | before issue opens | 0.0 |
| `issue_days_open` | feature | derived: close_date - open_date | before issue opens | 0.0 |
| `days_close_to_listing` | feature | derived: listing_date - close_date (scheduled) | before issue opens | 0.0 |
| `revenue_growth_pct` | feature | derived from RHP financials | before issue opens | 2.5 |
| `promoter_dilution_pct` | feature | derived: promoter_pre_pct - promoter_post_pct | before issue opens | 4.6 |
| `nifty_ret_7d` | feature | yfinance ^NSEI, closes up to issue close date | issue close date | 0.0 |
| `nifty_ret_30d` | feature | yfinance ^NSEI, closes up to issue close date | issue close date | 0.0 |
| `vix_close` | feature | yfinance ^INDIAVIX, close on/before issue close date | issue close date | 0.0 |
| `ipo_heat_30d` | feature | derived: issues closed in trailing 30 days | issue close date | 0.0 |
| `prev_gain_mean_90d` | feature | derived: mean listing gain of issues already listed, trailing 90 days | issue close date | 1.0 |
| `prev_count_90d` | feature | derived: count of issues already listed, trailing 90 days | issue close date | 0.0 |
| `sector` | feature | yfinance Ticker.info by NSE symbol | static company attribute | 3.3 |
| `industry` | id / info | yfinance industry (finer than sector; not used as a feature) | | 3.3 |
| `gmp_rs` | id / info | GMP experiment: GMP in Rs (same reading as gmp_pct) | | 25.7 |
| `gmp_pct` | GMP experiment only | InvestorGain GMP page, latest public reading dated before listing day | evening before listing | 25.7 |
| `gmp_date` | id / info | Date of the GMP reading used (always before listing date) | | 25.7 |
| `split` | id / info | train (listed <= 2022) / val (2023) / test (2024+) | | 0.0 |
