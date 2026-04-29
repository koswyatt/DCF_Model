"""
Fast-Launch Streamlit 10-K DCF App
----------------------------------
Launches immediately as a manual DCF model. The 10-K upload/analyzer is delayed
until the user clicks "Analyze uploaded 10-K".

Run:
    streamlit run streamlit_10k_dcf_app_fast_launch.py

requirements.txt:
    streamlit
    pandas
    openai
    pypdf

Streamlit Cloud secret for AI extraction:
    OPENAI_API_KEY = "your_api_key_here"
"""

import json
import re
from io import BytesIO
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fast 10-K DCF App", page_icon="📊", layout="wide")

# -----------------------------------------------------------------------------
# Basic helpers - intentionally lightweight
# -----------------------------------------------------------------------------

def safe_div(n: float, d: float, default: float = 0.0) -> float:
    try:
        return default if d in (0, None) else n / d
    except Exception:
        return default


def cagr(begin: float, end: float, years: int) -> float:
    if begin <= 0 or end <= 0 or years <= 0:
        return 0.0
    return (end / begin) ** (1 / years) - 1


def pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def money(x: float) -> str:
    return f"${x:,.2f}"


def clean_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"na", "n/a", "none", "null"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[$,%\s,]", "", s).replace("(", "").replace(")", "")
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None


def set_if_present(data: Dict[str, Any], key: str):
    val = clean_number(data.get(key))
    if val is not None:
        st.session_state[key] = val


def text_value(data: Dict[str, Any], key: str):
    val = data.get(key)
    if val not in (None, "", "N/A"):
        st.session_state[key] = str(val)

# -----------------------------------------------------------------------------
# Session defaults
# -----------------------------------------------------------------------------

DEFAULTS = {
    "company_name": "Example Company",
    "fiscal_year": "2025",
    "units": "Millions",
    "revenue_t": 10000.0,
    "revenue_t1": 9500.0,
    "revenue_t2": 9000.0,
    "ebit_t": 1500.0,
    "ebit_t1": 1350.0,
    "tax_expense_t": 350.0,
    "pretax_income_t": 1350.0,
    "depreciation_t": 500.0,
    "capex_t": 600.0,
    "cash_t": 800.0,
    "total_debt_t": 2500.0,
    "diluted_shares_t": 500.0,
    "current_assets_t": 4000.0,
    "current_liabilities_t": 3000.0,
    "current_assets_t1": 3700.0,
    "current_liabilities_t1": 2850.0,
    "risk_free_rate": 4.35,
    "equity_risk_premium": 4.75,
    "beta": 1.00,
    "pretax_cost_of_debt": 6.50,
    "terminal_growth": 2.50,
    "projection_years": 5,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# -----------------------------------------------------------------------------
# Delayed 10-K analyzer functions
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def extract_pdf_text_cached(file_bytes: bytes, max_pages: int = 45) -> str:
    """Only runs after the user clicks analyze. Cached by file bytes."""
    from pypdf import PdfReader  # delayed import speeds initial launch

    reader = PdfReader(BytesIO(file_bytes))
    page_count = min(len(reader.pages), max_pages)
    chunks = []
    for i in range(page_count):
        try:
            txt = reader.pages[i].extract_text() or ""
            if txt.strip():
                chunks.append(f"\n--- Page {i + 1} ---\n{txt}")
        except Exception:
            continue
    return "\n".join(chunks)


def extract_relevant_sections(text: str, max_chars: int = 35000) -> str:
    """Keep the prompt smaller and faster by selecting likely financial pages."""
    keywords = [
        "consolidated statements of operations", "consolidated statements of income",
        "consolidated balance sheets", "consolidated statements of cash flows",
        "revenue", "net sales", "operating income", "income taxes",
        "depreciation", "capital expenditures", "cash and cash equivalents",
        "long-term debt", "diluted shares", "current assets", "current liabilities",
    ]
    pages = re.split(r"\n--- Page \d+ ---\n", text)
    scored = []
    for p in pages:
        low = p.lower()
        score = sum(1 for k in keywords if k in low)
        if score:
            scored.append((score, p))
    selected = "\n\n".join(p for _, p in sorted(scored, reverse=True)[:10])
    return (selected or text)[:max_chars]


def regex_fallback(text: str) -> Dict[str, Any]:
    """Very rough fallback. User should review all values."""
    out: Dict[str, Any] = {}
    patterns = {
        "revenue_t": r"(?:total\s+)?(?:net\s+sales|revenues?|sales)\s+[$]?\s*([\(\)\d,\.]+)",
        "cash_t": r"cash and cash equivalents\s+[$]?\s*([\(\)\d,\.]+)",
        "total_debt_t": r"(?:total debt|long-term debt[^\n]*?)\s+[$]?\s*([\(\)\d,\.]+)",
        "capex_t": r"(?:capital expenditures|purchases of property[^\n]*?)\s+[$]?\s*([\(\)\d,\.]+)",
        "depreciation_t": r"depreciation(?: and amortization)?\s+[$]?\s*([\(\)\d,\.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            out[key] = clean_number(m.group(1))
    return out


def ai_extract_10k(text: str) -> Dict[str, Any]:
    from openai import OpenAI  # delayed import speeds initial launch

    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))
    schema_keys = list(DEFAULTS.keys())
    prompt = f"""
Extract DCF input fields from this 10-K text. Return ONLY valid JSON.
Use numbers exactly as reported in the filing. Do not calculate values unless the field is directly stated.
For missing fields, use null. Assume values are in the filing's stated units.

Fields: {schema_keys}

10-K text:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract financial statement line items into strict JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("📊 Fast-Launch 10-K DCF Valuation App")
st.caption("Manual DCF loads first. Upload/analyze a 10-K only when you need automatic pre-fill.")

with st.expander("Optional delayed 10-K analyzer", expanded=False):
    st.write("Upload a PDF 10-K, then click the button. Nothing is read or sent to AI until you click analyze.")
    upload = st.file_uploader("Upload 10-K PDF", type=["pdf"])
    use_ai = st.checkbox("Use OpenAI to pre-fill inputs", value=True)
    max_pages = st.slider("Maximum pages to scan", min_value=15, max_value=100, value=45, step=5)

    if st.button("Analyze uploaded 10-K", disabled=upload is None):
        if upload is None:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Extracting text from selected pages..."):
                raw_text = extract_pdf_text_cached(upload.getvalue(), max_pages=max_pages)
                relevant_text = extract_relevant_sections(raw_text)

            extracted: Dict[str, Any] = {}
            if use_ai:
                if "OPENAI_API_KEY" not in st.secrets:
                    st.warning("No OPENAI_API_KEY secret found. Using rough regex fallback instead.")
                    extracted = regex_fallback(relevant_text)
                else:
                    try:
                        with st.spinner("Asking AI to extract filing values..."):
                            extracted = ai_extract_10k(relevant_text)
                    except Exception as e:
                        st.warning(f"AI extraction failed, using rough regex fallback. Error: {e}")
                        extracted = regex_fallback(relevant_text)
            else:
                extracted = regex_fallback(relevant_text)

            for key in ["company_name", "fiscal_year", "units"]:
                text_value(extracted, key)
            for key in DEFAULTS:
                if key not in ["company_name", "fiscal_year", "units"]:
                    set_if_present(extracted, key)

            st.success("Inputs were pre-filled where values could be found. Review everything below before relying on the valuation.")
            with st.expander("Show extracted JSON"):
                st.json(extracted)

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

left, right = st.columns([1, 1])
with left:
    st.subheader("10-K line items")
    st.text_input("Company name", key="company_name")
    st.text_input("Fiscal year", key="fiscal_year")
    st.selectbox("Statement units", ["Actual dollars", "Thousands", "Millions", "Billions"], key="units")

    st.number_input("Revenue / net sales - current year", min_value=0.0, step=100.0, key="revenue_t")
    st.number_input("Revenue / net sales - prior year", min_value=0.0, step=100.0, key="revenue_t1")
    st.number_input("Revenue / net sales - two years ago", min_value=0.0, step=100.0, key="revenue_t2")
    st.number_input("Operating income / EBIT - current year", step=100.0, key="ebit_t")
    st.number_input("Operating income / EBIT - prior year", step=100.0, key="ebit_t1")
    st.number_input("Income tax expense", min_value=0.0, step=10.0, key="tax_expense_t")
    st.number_input("Pretax income", step=10.0, key="pretax_income_t")
    st.number_input("Depreciation & amortization", min_value=0.0, step=10.0, key="depreciation_t")
    st.number_input("Capital expenditures / purchases of PP&E", min_value=0.0, step=10.0, key="capex_t")

with right:
    st.subheader("Balance sheet and market inputs")
    st.number_input("Cash and cash equivalents", min_value=0.0, step=10.0, key="cash_t")
    st.number_input("Total debt", min_value=0.0, step=10.0, key="total_debt_t")
    st.number_input("Diluted shares outstanding", min_value=0.0001, step=1.0, key="diluted_shares_t")
    st.number_input("Current assets - current year", min_value=0.0, step=10.0, key="current_assets_t")
    st.number_input("Current liabilities - current year", min_value=0.0, step=10.0, key="current_liabilities_t")
    st.number_input("Current assets - prior year", min_value=0.0, step=10.0, key="current_assets_t1")
    st.number_input("Current liabilities - prior year", min_value=0.0, step=10.0, key="current_liabilities_t1")

    st.subheader("Editable valuation assumptions")
    st.slider("Projection years", 3, 10, key="projection_years")
    st.number_input("Risk-free rate (%)", step=0.05, key="risk_free_rate")
    st.number_input("Equity risk premium (%)", step=0.05, key="equity_risk_premium")
    st.number_input("Beta", min_value=0.1, max_value=3.0, step=0.05, key="beta")
    st.number_input("Pre-tax cost of debt (%)", min_value=0.0, step=0.05, key="pretax_cost_of_debt")
    st.number_input("Terminal growth (%)", min_value=0.0, max_value=6.0, step=0.05, key="terminal_growth")

# -----------------------------------------------------------------------------
# Calculations
# -----------------------------------------------------------------------------

revenue_t = st.session_state.revenue_t
revenue_t1 = st.session_state.revenue_t1
revenue_t2 = st.session_state.revenue_t2
ebit_t = st.session_state.ebit_t
ebit_t1 = st.session_state.ebit_t1
pretax = st.session_state.pretax_income_t
tax_expense = st.session_state.tax_expense_t
capex = st.session_state.capex_t
dep = st.session_state.depreciation_t
cash = st.session_state.cash_t
debt = st.session_state.total_debt_t
shares = st.session_state.diluted_shares_t
nwc_t = st.session_state.current_assets_t - st.session_state.current_liabilities_t
nwc_t1 = st.session_state.current_assets_t1 - st.session_state.current_liabilities_t1
change_nwc = nwc_t - nwc_t1

growth_hist = cagr(revenue_t2, revenue_t, 2) if revenue_t2 else safe_div(revenue_t - revenue_t1, revenue_t1)
ebit_margin = safe_div(ebit_t, revenue_t)
tax_rate = min(max(safe_div(tax_expense, pretax, 0.255), 0.0), 0.40)
dep_pct_sales = safe_div(dep, revenue_t)
capex_pct_sales = safe_div(capex, revenue_t)
nwc_pct_sales = safe_div(nwc_t, revenue_t)

cost_equity = (st.session_state.risk_free_rate / 100) + st.session_state.beta * (st.session_state.equity_risk_premium / 100)
after_tax_cost_debt = (st.session_state.pretax_cost_of_debt / 100) * (1 - tax_rate)
# Simple default capital structure when market cap is not entered: 80/20 equity/debt
wacc = 0.80 * cost_equity + 0.20 * after_tax_cost_debt
terminal_growth = st.session_state.terminal_growth / 100
years = int(st.session_state.projection_years)

# Keep assumptions reasonable for auto projection
forecast_growth = min(max(growth_hist, -0.05), 0.12)

rows = []
prev_revenue = revenue_t
for year in range(1, years + 1):
    rev = prev_revenue * (1 + forecast_growth)
    ebit = rev * ebit_margin
    nopat = ebit * (1 - tax_rate)
    depreciation = rev * dep_pct_sales
    capex_forecast = rev * capex_pct_sales
    nwc_needed = rev * nwc_pct_sales
    prev_nwc_needed = prev_revenue * nwc_pct_sales
    delta_nwc = nwc_needed - prev_nwc_needed
    fcf = nopat + depreciation - capex_forecast - delta_nwc
    pv_fcf = fcf / ((1 + wacc) ** year)
    rows.append({
        "Year": year,
        "Revenue": rev,
        "EBIT": ebit,
        "NOPAT": nopat,
        "D&A": depreciation,
        "CapEx": capex_forecast,
        "ΔNWC": delta_nwc,
        "FCF": fcf,
        "PV of FCF": pv_fcf,
    })
    prev_revenue = rev

df = pd.DataFrame(rows)
last_fcf = float(df["FCF"].iloc[-1])
if wacc <= terminal_growth:
    terminal_value = 0.0
    pv_terminal = 0.0
    warning_tv = True
else:
    terminal_value = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    warning_tv = False
enterprise_value = float(df["PV of FCF"].sum()) + pv_terminal
equity_value = enterprise_value - debt + cash
value_per_share = equity_value / shares

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Valuation results")
if warning_tv:
    st.error("Terminal growth must be lower than WACC. Lower terminal growth or raise WACC assumptions.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated WACC", pct(wacc))
m2.metric("Enterprise value", money(enterprise_value))
m3.metric("Equity value", money(equity_value))
m4.metric("Value per share", money(value_per_share))

st.subheader("Auto-derived assumptions")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Forecast revenue growth", pct(forecast_growth))
a2.metric("EBIT margin", pct(ebit_margin))
a3.metric("Effective tax rate", pct(tax_rate))
a4.metric("NWC / sales", pct(nwc_pct_sales))

st.subheader("Projection table")
st.dataframe(df.style.format({
    "Revenue": "{:,.2f}", "EBIT": "{:,.2f}", "NOPAT": "{:,.2f}",
    "D&A": "{:,.2f}", "CapEx": "{:,.2f}", "ΔNWC": "{:,.2f}",
    "FCF": "{:,.2f}", "PV of FCF": "{:,.2f}",
}), use_container_width=True)

st.subheader("FCF projection")
st.line_chart(df.set_index("Year")["FCF"])

with st.expander("Notes and limitations"):
    st.write(
        "This is an educational DCF model. AI and regex extraction can misread filings, especially tables, "
        "so every pre-filled value should be checked against the 10-K. The app intentionally avoids heavy "
        "imports and PDF processing at startup to keep Streamlit Cloud launches fast."
    )
