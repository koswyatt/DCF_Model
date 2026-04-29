"""
Streamlit 10-K DCF Valuation App with Optional AI Extraction
-------------------------------------------------------------
Run locally:
    streamlit run streamlit_10k_dcf_app_ai_upload.py

Recommended requirements.txt:
    streamlit
    pandas
    numpy
    pypdf
    openai

Deployment note:
    For AI extraction, add OPENAI_API_KEY to Streamlit Cloud secrets.
    The app still works manually without an API key.
"""

import json
import re
from io import BytesIO
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports: app still runs if these are missing
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

st.set_page_config(page_title="AI-Assisted 10-K DCF App", page_icon="📊", layout="wide")
st.title("📊 AI-Assisted 10-K DCF Valuation App")
st.caption("Upload a 10-K, let AI pre-fill values, then review and adjust before valuing the company.")

# -----------------------------
# Defaults
# -----------------------------
DEFAULTS = {
    "company_name": "Example Company",
    "fiscal_year": 2025,
    "revenue_t": 10000.0,
    "ebit_t": 1500.0,
    "net_income_t": 1000.0,
    "tax_expense_t": 350.0,
    "depreciation_t": 500.0,
    "capex_t": 600.0,
    "interest_expense_t": 200.0,
    "cash_t": 800.0,
    "total_debt_t": 2500.0,
    "diluted_shares_t": 500.0,
    "current_assets_t": 4000.0,
    "current_liabilities_t": 3000.0,
    "revenue_t1": 9500.0,
    "ebit_t1": 1350.0,
    "current_assets_t1": 3700.0,
    "current_liabilities_t1": 2850.0,
    "revenue_t2": 9000.0,
    "ebit_t2": 1200.0,
    "market_cap": 0.0,
}

DEFAULT_RISK_FREE_RATE = 0.0435
DEFAULT_EQUITY_RISK_PREMIUM = 0.0475
DEFAULT_PRETAX_COST_OF_DEBT = 0.0650
DEFAULT_EFFECTIVE_TAX_RATE = 0.2550
DEFAULT_TERMINAL_GROWTH = 0.0250
DEFAULT_PROJECTION_YEARS = 5

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("units", "Millions")

# -----------------------------
# Helpers
# -----------------------------
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator in [0, None] or pd.isna(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default


def cagr(beginning_value: float, ending_value: float, periods: int) -> float:
    if beginning_value <= 0 or ending_value <= 0 or periods <= 0:
        return 0.0
    return (ending_value / beginning_value) ** (1 / periods) - 1


def format_dollars(x: float) -> str:
    return "N/A" if pd.isna(x) else f"${x:,.2f}"


def format_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:.2f}%"


def estimate_beta_from_industry(industry_risk: str) -> float:
    return {
        "Defensive / Utilities / Staples": 0.75,
        "Average market risk": 1.00,
        "Cyclical / Consumer discretionary / Industrials": 1.15,
        "High-growth / Technology / Biotech": 1.30,
        "Highly leveraged or distressed": 1.50,
    }.get(industry_risk, 1.0)


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("$", "").replace(",", "")
    if s in {"", "N/A", "NA", "null", "None"}:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    try:
        num = float(re.findall(r"-?\d+(?:\.\d+)?", s)[0])
        return -num if neg else num
    except Exception:
        return None


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    uploaded_file.seek(0)
    if name.endswith(".pdf"):
        if PdfReader is None:
            st.error("PDF extraction requires pypdf. Add `pypdf` to requirements.txt.")
            return ""
        reader = PdfReader(BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(f"\n--- PAGE {i + 1} ---\n" + (page.extract_text() or ""))
            except Exception:
                continue
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


def reduce_10k_context(text: str, max_chars: int = 65000) -> str:
    """Keep sections most likely to contain DCF inputs to reduce API cost."""
    terms = [
        "consolidated statements of operations", "consolidated statement of operations",
        "consolidated income", "consolidated balance sheets", "consolidated balance sheet",
        "consolidated statements of cash flows", "cash flows", "net sales", "revenue",
        "operating income", "income tax", "depreciation", "amortization", "capital expenditures",
        "property and equipment", "cash and cash equivalents", "current assets", "current liabilities",
        "long-term debt", "interest expense", "shares", "earnings per share"
    ]
    low = text.lower()
    windows = []
    for term in terms:
        start = 0
        while True:
            idx = low.find(term, start)
            if idx == -1:
                break
            a, b = max(0, idx - 3500), min(len(text), idx + 5500)
            windows.append(text[a:b])
            start = idx + len(term)
    if not windows:
        return text[:max_chars]
    combined = "\n\n".join(windows)
    # Remove extreme repeated whitespace
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined[:max_chars]


def regex_guess(text: str) -> Dict[str, Any]:
    """Very rough fallback. AI is better; this only catches common labels."""
    t = re.sub(r"\s+", " ", text)
    patterns = {
        "revenue_t": r"(?:net sales|total revenues?|revenue)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "ebit_t": r"(?:operating income|income from operations)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "net_income_t": r"(?:net income|net earnings)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "tax_expense_t": r"(?:income tax expense|provision for income taxes)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "depreciation_t": r"(?:depreciation and amortization|depreciation)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "capex_t": r"(?:capital expenditures|purchases of property and equipment|additions to property and equipment)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "cash_t": r"(?:cash and cash equivalents)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
        "total_debt_t": r"(?:total debt|long-term debt|short-term borrowings)\D{0,80}([\(\)-]?[\$]?[0-9][0-9,\.]+)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, t, flags=re.I)
        if m:
            val = parse_number(m.group(1))
            if val is not None:
                out[key] = val
    return out


def ai_extract_10k(text: str, api_key: str, model: str) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("OpenAI package not installed. Add `openai` to requirements.txt.")
    client = OpenAI(api_key=api_key)
    context = reduce_10k_context(text)
    schema_keys = list(DEFAULTS.keys())
    prompt = f"""
You are extracting raw financial statement values from a company's Form 10-K for a DCF app.
Return ONLY valid JSON. Do not include explanations.

Rules:
- Use numbers exactly as reported in the filing table, before scaling. If the filing says dollars in millions, return millions. If thousands, return thousands.
- Do not calculate values unless explicitly stated below.
- For total_debt_t, if a single total debt line is not available, you may sum short-term borrowings/current debt + long-term debt and set confidence lower.
- For capex_t, use capital expenditures, purchases of property and equipment, or additions to property and equipment from cash flows. Return as a positive number.
- For ebit_t, use operating income/income from operations.
- For shares, prefer diluted weighted-average shares; if unavailable, use shares outstanding.
- Include confidence from 0 to 1 and source_note with the line/section name where you found each value.
- Use null for missing values.

Return this JSON shape:
{{
  "detected_units": "Millions|Thousands|Actual dollars|Unknown",
  "fields": {{
    "company_name": {{"value": null, "confidence": 0, "source_note": ""}},
    "fiscal_year": {{"value": null, "confidence": 0, "source_note": ""}},
    "revenue_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "revenue_t1": {{"value": null, "confidence": 0, "source_note": ""}},
    "revenue_t2": {{"value": null, "confidence": 0, "source_note": ""}},
    "ebit_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "ebit_t1": {{"value": null, "confidence": 0, "source_note": ""}},
    "ebit_t2": {{"value": null, "confidence": 0, "source_note": ""}},
    "net_income_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "tax_expense_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "depreciation_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "capex_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "interest_expense_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "cash_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "total_debt_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "diluted_shares_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "current_assets_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "current_liabilities_t": {{"value": null, "confidence": 0, "source_note": ""}},
    "current_assets_t1": {{"value": null, "confidence": 0, "source_note": ""}},
    "current_liabilities_t1": {{"value": null, "confidence": 0, "source_note": ""}}
  }}
}}

10-K text excerpts:
{context}
"""
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    raw = getattr(response, "output_text", "")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def apply_extraction(extraction: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    units = extraction.get("detected_units")
    if units in ["Millions", "Thousands", "Actual dollars"]:
        st.session_state["units"] = units
    fields = extraction.get("fields", {}) if "fields" in extraction else extraction
    for key, payload in fields.items():
        if key not in DEFAULTS:
            continue
        if isinstance(payload, dict):
            val = payload.get("value")
            conf = payload.get("confidence", "")
            note = payload.get("source_note", "")
        else:
            val, conf, note = payload, "", ""
        if key == "company_name" and val:
            st.session_state[key] = str(val)
        elif key == "fiscal_year":
            num = parse_number(val)
            if num:
                st.session_state[key] = int(num)
        else:
            num = parse_number(val)
            if num is not None:
                st.session_state[key] = float(num)
        rows.append({"Field": key, "Extracted Value": val, "Confidence": conf, "Source note": note})
    return pd.DataFrame(rows)

# -----------------------------
# Upload + AI extraction
# -----------------------------
st.header("0) Optional: Upload a 10-K to Pre-Fill Inputs")
with st.expander("Upload 10-K and extract values", expanded=True):
    uploaded_10k = st.file_uploader("Upload a 10-K PDF, HTML, TXT, or copied SEC filing text", type=["pdf", "html", "htm", "txt"])
    col_ai1, col_ai2, col_ai3 = st.columns([1.1, 1, 1])
    with col_ai1:
        extraction_mode = st.radio("Extraction method", ["AI extraction", "Simple regex fallback"], horizontal=True)
    with col_ai2:
        model_name = st.text_input("OpenAI model", value="gpt-5.5-mini")
    with col_ai3:
        secret_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key_input = st.text_input("OpenAI API key", value=secret_key, type="password", help="Best practice: store this in Streamlit secrets as OPENAI_API_KEY.")

    if uploaded_10k and st.button("Extract and pre-fill model inputs"):
        with st.spinner("Reading 10-K and extracting financial statement values..."):
            text = extract_text_from_upload(uploaded_10k)
            if text:
                try:
                    if extraction_mode == "AI extraction":
                        if not api_key_input:
                            st.error("Add an OpenAI API key or switch to Simple regex fallback.")
                        else:
                            extraction = ai_extract_10k(text, api_key_input, model_name)
                            df_extract = apply_extraction(extraction)
                            st.success("Extraction complete. Review every field below before relying on the valuation.")
                            st.dataframe(df_extract, use_container_width=True, hide_index=True)
                            st.rerun()
                    else:
                        guessed = regex_guess(text)
                        df_extract = apply_extraction(guessed)
                        st.warning("Regex extraction is rough. Review and correct all fields.")
                        st.dataframe(df_extract, use_container_width=True, hide_index=True)
                        st.rerun()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    st.info("Try the regex fallback, copy/paste the filing into a .txt file, or manually enter the missing fields.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model Controls")
units = st.sidebar.selectbox("Input units", ["Millions", "Thousands", "Actual dollars"], key="units")
unit_multiplier = {"Millions": 1_000_000, "Thousands": 1_000, "Actual dollars": 1}[units]
projection_years = st.sidebar.slider("Projection years", 3, 10, DEFAULT_PROJECTION_YEARS)
valuation_method = st.sidebar.selectbox("Terminal value method", ["Perpetuity growth", "Exit EBITDA multiple"], index=0)

st.sidebar.subheader("Broad Assumption Defaults")
risk_free_rate = st.sidebar.number_input("Risk-free rate (%)", value=DEFAULT_RISK_FREE_RATE * 100, step=0.05) / 100
equity_risk_premium = st.sidebar.number_input("Equity risk premium (%)", value=DEFAULT_EQUITY_RISK_PREMIUM * 100, step=0.05) / 100
pretax_cost_of_debt = st.sidebar.number_input("Pre-tax cost of debt (%)", value=DEFAULT_PRETAX_COST_OF_DEBT * 100, step=0.10) / 100
tax_rate_default = st.sidebar.number_input("Default tax rate (%)", value=DEFAULT_EFFECTIVE_TAX_RATE * 100, step=0.10) / 100
terminal_growth = st.sidebar.number_input("Terminal growth rate (%)", value=DEFAULT_TERMINAL_GROWTH * 100, step=0.10) / 100
exit_ebitda_multiple = st.sidebar.number_input("Exit EBITDA multiple", value=8.0, step=0.5)

# -----------------------------
# Inputs
# -----------------------------
st.header("1) Company Information")
col_a, col_b, col_c = st.columns(3)
with col_a:
    company_name = st.text_input("Company name", key="company_name")
with col_b:
    fiscal_year = st.number_input("Most recent fiscal year", min_value=1990, max_value=2100, step=1, key="fiscal_year")
with col_c:
    industry_risk = st.selectbox("Industry risk profile", ["Defensive / Utilities / Staples", "Average market risk", "Cyclical / Consumer discretionary / Industrials", "High-growth / Technology / Biotech", "Highly leveraged or distressed"], index=1)

st.header("2) Review / Adjust 10-K Inputs")
st.write("AI pre-fill is a starting point only. Verify every field against the 10-K before using the valuation.")

st.subheader("Most Recent Year")
col1, col2, col3 = st.columns(3)
with col1:
    revenue_t = st.number_input("Revenue / net sales", min_value=0.0, step=100.0, key="revenue_t")
    ebit_t = st.number_input("Operating income / EBIT", step=100.0, key="ebit_t")
    net_income_t = st.number_input("Net income", step=100.0, key="net_income_t")
    tax_expense_t = st.number_input("Income tax expense", min_value=0.0, step=10.0, key="tax_expense_t")
with col2:
    depreciation_t = st.number_input("Depreciation & amortization", min_value=0.0, step=10.0, key="depreciation_t")
    capex_t = st.number_input("Capital expenditures / purchases of PP&E", min_value=0.0, step=10.0, key="capex_t")
    interest_expense_t = st.number_input("Interest expense", min_value=0.0, step=10.0, key="interest_expense_t")
    cash_t = st.number_input("Cash and cash equivalents", min_value=0.0, step=50.0, key="cash_t")
with col3:
    total_debt_t = st.number_input("Total debt", min_value=0.0, step=100.0, key="total_debt_t")
    diluted_shares_t = st.number_input("Diluted weighted-average shares or shares outstanding", min_value=0.0001, step=1.0, key="diluted_shares_t")
    current_assets_t = st.number_input("Current assets", min_value=0.0, step=100.0, key="current_assets_t")
    current_liabilities_t = st.number_input("Current liabilities", min_value=0.0, step=100.0, key="current_liabilities_t")

st.subheader("Prior Years")
col4, col5, col6 = st.columns(3)
with col4:
    revenue_t1 = st.number_input("Prior-year revenue / net sales", min_value=0.0, step=100.0, key="revenue_t1")
    ebit_t1 = st.number_input("Prior-year operating income / EBIT", step=100.0, key="ebit_t1")
with col5:
    current_assets_t1 = st.number_input("Prior-year current assets", min_value=0.0, step=100.0, key="current_assets_t1")
    current_liabilities_t1 = st.number_input("Prior-year current liabilities", min_value=0.0, step=100.0, key="current_liabilities_t1")
with col6:
    revenue_t2 = st.number_input("Two-years-ago revenue / net sales", min_value=0.0, step=100.0, key="revenue_t2")
    ebit_t2 = st.number_input("Two-years-ago operating income / EBIT", step=100.0, key="ebit_t2")

st.header("3) Optional Market Inputs")
col7, col8, col9 = st.columns(3)
with col7:
    use_manual_beta = st.checkbox("Input company beta manually", value=False)
    manual_beta = st.number_input("Company beta", value=1.00, step=0.05, disabled=not use_manual_beta)
with col8:
    use_actual_tax = st.checkbox("Use actual effective tax rate from latest year", value=True)
with col9:
    market_cap = st.number_input("Market capitalization, if known", min_value=0.0, step=100.0, key="market_cap")

# -----------------------------
# Convert units
# -----------------------------
revenue = revenue_t * unit_multiplier
ebit = ebit_t * unit_multiplier
net_income = net_income_t * unit_multiplier
tax_expense = tax_expense_t * unit_multiplier
depreciation = depreciation_t * unit_multiplier
capex = capex_t * unit_multiplier
interest_expense = interest_expense_t * unit_multiplier
cash = cash_t * unit_multiplier
total_debt = total_debt_t * unit_multiplier
current_assets = current_assets_t * unit_multiplier
current_liabilities = current_liabilities_t * unit_multiplier
revenue_prior = revenue_t1 * unit_multiplier
ebit_prior = ebit_t1 * unit_multiplier
current_assets_prior = current_assets_t1 * unit_multiplier
current_liabilities_prior = current_liabilities_t1 * unit_multiplier
revenue_two_years_ago = revenue_t2 * unit_multiplier
ebit_two_years_ago = ebit_t2 * unit_multiplier
market_cap_actual = market_cap * unit_multiplier
shares = diluted_shares_t

# -----------------------------
# Historical calculations
# -----------------------------
working_capital = current_assets - current_liabilities
working_capital_prior = current_assets_prior - current_liabilities_prior
change_nwc = working_capital - working_capital_prior
historical_growth_1yr = safe_div(revenue - revenue_prior, revenue_prior)
historical_growth_2yr_cagr = cagr(revenue_two_years_ago, revenue, 2)
base_revenue_growth = np.nanmean([historical_growth_1yr, historical_growth_2yr_cagr])
base_revenue_growth = max(min(base_revenue_growth, 0.20), -0.10)
ebit_margin = safe_div(ebit, revenue)
da_percent_revenue = safe_div(depreciation, revenue)
capex_percent_revenue = safe_div(capex, revenue)
nwc_percent_revenue = safe_div(working_capital, revenue)
change_nwc_percent_revenue = safe_div(change_nwc, revenue)
pre_tax_income_estimate = ebit - interest_expense
actual_effective_tax_rate = safe_div(tax_expense, max(pre_tax_income_estimate, 0.0001), default=tax_rate_default)
actual_effective_tax_rate = max(0.0, min(actual_effective_tax_rate, 0.40))
tax_rate = actual_effective_tax_rate if use_actual_tax else tax_rate_default
beta = manual_beta if use_manual_beta else estimate_beta_from_industry(industry_risk)
cost_of_equity = risk_free_rate + beta * equity_risk_premium
after_tax_cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)
if market_cap_actual > 0:
    equity_weight_base = safe_div(market_cap_actual, market_cap_actual + total_debt, default=0.80)
else:
    equity_proxy = max(net_income / max(cost_of_equity, 0.0001), revenue)
    equity_weight_base = safe_div(equity_proxy, equity_proxy + total_debt, default=0.80)
debt_weight_base = 1 - equity_weight_base
wacc = equity_weight_base * cost_of_equity + debt_weight_base * after_tax_cost_of_debt

# -----------------------------
# Forecast assumptions
# -----------------------------
st.header("4) Forecast Assumptions")
col10, col11, col12, col13 = st.columns(4)
with col10:
    forecast_growth = st.number_input("Forecast revenue growth (%)", value=base_revenue_growth * 100, step=0.25) / 100
with col11:
    forecast_ebit_margin = st.number_input("Forecast EBIT margin (%)", value=ebit_margin * 100, step=0.25) / 100
with col12:
    forecast_da_percent = st.number_input("D&A as % of revenue", value=da_percent_revenue * 100, step=0.10) / 100
with col13:
    forecast_capex_percent = st.number_input("CapEx as % of revenue", value=capex_percent_revenue * 100, step=0.10) / 100
col14, col15, col16, col17 = st.columns(4)
with col14:
    forecast_nwc_percent = st.number_input("Net working capital as % of revenue", value=nwc_percent_revenue * 100, step=0.25) / 100
with col15:
    forecast_tax_rate = st.number_input("Forecast tax rate (%)", value=tax_rate * 100, step=0.25) / 100
with col16:
    st.metric("Calculated beta", f"{beta:.2f}")
with col17:
    st.metric("Calculated WACC", format_pct(wacc))

# -----------------------------
# Projection
# -----------------------------
projection = []
previous_revenue = revenue
previous_nwc = working_capital
for year in range(1, projection_years + 1):
    projected_revenue = previous_revenue * (1 + forecast_growth)
    projected_ebit = projected_revenue * forecast_ebit_margin
    nopat = projected_ebit * (1 - forecast_tax_rate)
    projected_da = projected_revenue * forecast_da_percent
    projected_capex = projected_revenue * forecast_capex_percent
    projected_nwc = projected_revenue * forecast_nwc_percent
    projected_change_nwc = projected_nwc - previous_nwc
    fcf = nopat + projected_da - projected_capex - projected_change_nwc
    discount_factor = 1 / ((1 + wacc) ** year)
    pv_fcf = fcf * discount_factor
    ebitda = projected_ebit + projected_da
    projection.append({"Year": int(fiscal_year + year), "Revenue": projected_revenue, "EBIT": projected_ebit, "EBITDA": ebitda, "NOPAT": nopat, "D&A": projected_da, "CapEx": projected_capex, "Change in NWC": projected_change_nwc, "FCF": fcf, "Discount Factor": discount_factor, "PV of FCF": pv_fcf})
    previous_revenue = projected_revenue
    previous_nwc = projected_nwc
projection_df = pd.DataFrame(projection)
last_fcf = projection_df["FCF"].iloc[-1]
last_ebitda = projection_df["EBITDA"].iloc[-1]
if valuation_method == "Perpetuity growth":
    if wacc <= terminal_growth:
        st.error("WACC must be greater than terminal growth for the perpetuity growth method.")
        terminal_value = np.nan
        pv_terminal_value = np.nan
    else:
        terminal_value = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
else:
    terminal_value = last_ebitda * exit_ebitda_multiple
    pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
pv_fcfs = projection_df["PV of FCF"].sum()
enterprise_value = pv_fcfs + pv_terminal_value
equity_value = enterprise_value - total_debt + cash
value_per_share = safe_div(equity_value, shares)

# -----------------------------
# Results
# -----------------------------
st.header("5) Valuation Results")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Enterprise Value", format_dollars(enterprise_value))
r2.metric("Equity Value", format_dollars(equity_value))
r3.metric("Implied Value per Share", format_dollars(value_per_share))
r4.metric("PV of Terminal Value", format_dollars(pv_terminal_value))

with st.expander("Key calculated historical ratios", expanded=True):
    ratios = pd.DataFrame({
        "Metric": ["1-year revenue growth", "2-year revenue CAGR", "EBIT margin", "D&A / Revenue", "CapEx / Revenue", "NWC / Revenue", "Change in NWC / Revenue", "Effective tax rate used", "Cost of equity", "After-tax cost of debt", "Equity weight", "Debt weight", "WACC"],
        "Value": [format_pct(historical_growth_1yr), format_pct(historical_growth_2yr_cagr), format_pct(ebit_margin), format_pct(da_percent_revenue), format_pct(capex_percent_revenue), format_pct(nwc_percent_revenue), format_pct(change_nwc_percent_revenue), format_pct(forecast_tax_rate), format_pct(cost_of_equity), format_pct(after_tax_cost_of_debt), format_pct(equity_weight_base), format_pct(debt_weight_base), format_pct(wacc)]
    })
    st.dataframe(ratios, use_container_width=True, hide_index=True)

st.subheader("DCF Projection Table")
display_df = projection_df.copy()
for col in ["Revenue", "EBIT", "EBITDA", "NOPAT", "D&A", "CapEx", "Change in NWC", "FCF", "PV of FCF"]:
    display_df[col] = display_df[col].map(format_dollars)
display_df["Discount Factor"] = display_df["Discount Factor"].map(lambda x: f"{x:.4f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)
st.subheader("Free Cash Flow Projection")
st.line_chart(projection_df[["Year", "FCF"]].set_index("Year"), use_container_width=True)

# -----------------------------
# Sensitivity
# -----------------------------
st.header("6) Sensitivity Analysis")
wacc_range = np.arange(max(0.01, wacc - 0.02), wacc + 0.0201, 0.005)
tg_range = np.arange(max(0.0, terminal_growth - 0.01), terminal_growth + 0.0101, 0.005)
sensitivity = pd.DataFrame(index=[f"{x * 100:.1f}%" for x in tg_range])
for w in wacc_range:
    values = []
    for tg in tg_range:
        if w <= tg:
            values.append(np.nan)
        else:
            tv = last_fcf * (1 + tg) / (w - tg)
            pv_tv = tv / ((1 + w) ** projection_years)
            pv_fcf_alt = sum(projection_df.loc[i, "FCF"] / ((1 + w) ** (i + 1)) for i in range(len(projection_df)))
            eq = pv_fcf_alt + pv_tv - total_debt + cash
            values.append(safe_div(eq, shares))
    sensitivity[f"WACC {w * 100:.1f}%"] = values
st.dataframe(sensitivity.style.format("${:,.2f}"), use_container_width=True)

# -----------------------------
# Methodology
# -----------------------------
st.header("7) Methodology / AI Use Disclosure")
st.markdown("""
**Free cash flow:** FCF = EBIT × (1 − Tax Rate) + D&A − CapEx − Change in Net Working Capital  
**WACC:** Equity Weight × Cost of Equity + Debt Weight × After-Tax Cost of Debt  
**Cost of equity:** Risk-Free Rate + Beta × Equity Risk Premium  

**AI extraction limitation:** The AI module is used only to pre-fill fields from the uploaded 10-K. It can misread tables, signs, or units, so the user should verify every extracted value against the filing before relying on the valuation.
""")
