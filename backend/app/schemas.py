"""Request/response models. Ranges are deliberately a little wider than the
training data so realistic new IPOs are accepted, but obvious typos (a 5000x
retail subscription, a 300% Nifty move) are rejected with a clear message."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Exchange = Literal["BSE,NSE", "NSE", "BSE"]
Sector = Literal[
    "Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
    "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate",
    "Technology", "Utilities", "Unknown",
]


def _f(lo, hi, desc, **kw):
    return Field(default=None, ge=lo, le=hi, description=desc, **kw)


class IPOInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # issue terms (RHP)
    issue_size_cr: float = Field(..., gt=0, le=60000, description="Total issue size, Rs crore")
    price_band_upper: float = Field(..., gt=0, le=20000, description="Upper end of price band, Rs")
    fresh_issue_pct: float = Field(..., ge=0, le=100, description="Fresh issue as % of issue size (rest is OFS)")
    lot_size: Optional[float] = _f(1, 100000, "Shares per lot")
    min_investment: Optional[float] = _f(1000, 500000, "Retail minimum investment, Rs")
    anchor_portion_cr: Optional[float] = _f(0, 30000, "Anchor book size, Rs crore")
    retail_reserved_pct: Optional[float] = _f(0, 100, "Retail quota, % of issue")
    exchange: Exchange = "BSE,NSE"
    sector: Sector = "Unknown"

    # final-day subscription (times)
    qib_x: Optional[float] = _f(0, 2000, "QIB subscription (x)")
    nii_x: Optional[float] = _f(0, 2000, "NII/HNI subscription (x)")
    bnii_x: Optional[float] = _f(0, 2000, "bNII subscription (x), post-2022 issues")
    snii_x: Optional[float] = _f(0, 2000, "sNII subscription (x), post-2022 issues")
    retail_x: Optional[float] = _f(0, 1000, "Retail subscription (x)")
    employee_x: Optional[float] = _f(0, 1000, "Employee quota subscription (x)")
    total_x: float = Field(..., ge=0, le=2000, description="Overall subscription (x), ex-anchor")

    # timing
    listing_month: int = Field(..., ge=1, le=12)
    issue_days_open: Optional[float] = _f(1, 15, "Days between open and close")

    # market context at issue close
    nifty_ret_7d: Optional[float] = _f(-30, 30, "Nifty 50 % return over 7 days to issue close")
    nifty_ret_30d: Optional[float] = _f(-50, 50, "Nifty 50 % return over 30 days to issue close")
    vix_close: Optional[float] = _f(5, 90, "India VIX on issue close date")
    ipo_heat_30d: Optional[float] = _f(0, 80, "Mainboard issues that closed in the previous 30 days")
    prev_gain_mean_90d: Optional[float] = _f(-100, 400, "Mean listing gain of issues listed in previous 90 days, %")
    prev_count_90d: Optional[float] = _f(0, 150, "Issues listed in previous 90 days")

    # company (RHP)
    revenue_cr: Optional[float] = _f(0, 1_000_000, "Latest FY total income, Rs crore")
    pat_cr: Optional[float] = _f(-100_000, 200_000, "Latest FY profit after tax, Rs crore")
    revenue_growth_pct: Optional[float] = _f(-100, 10000, "Revenue growth vs previous FY, %")
    pat_margin_pct: Optional[float] = _f(-1000, 100, "PAT margin, %")
    roe_pct: Optional[float] = _f(-500, 500, "Return on equity, %")
    debt_equity: Optional[float] = _f(0, 100, "Debt to equity")
    pe_post: Optional[float] = _f(0, 5000, "Post-issue P/E at upper band")
    mcap_post_cr: Optional[float] = _f(0, 2_000_000, "Market cap at issue price, Rs crore")
    promoter_post_pct: Optional[float] = _f(0, 100, "Promoter holding after issue, %")
    promoter_dilution_pct: Optional[float] = _f(-10, 100, "Promoter pre minus post holding, % points")

    @model_validator(mode="after")
    def _consistency(self):
        if self.anchor_portion_cr is not None and self.anchor_portion_cr > self.issue_size_cr:
            raise ValueError("anchor_portion_cr cannot exceed issue_size_cr")
        return self


class Prediction(BaseModel):
    predicted_gain_pct: float
    prob_positive: float
    decision_threshold: float
    predicted_direction: Literal["positive", "flat_or_negative"]
    confidence_note: str
    model_version: str


class Contribution(BaseModel):
    feature: str
    label: str
    value: Optional[float | str]
    shap: float


class Explanation(BaseModel):
    base_value: float
    predicted_gain_pct: float
    contributions: list[Contribution]
    note: str
