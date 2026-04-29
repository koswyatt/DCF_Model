"""
Fast-Launch Streamlit 10-K DCF App - Enhanced 10-K Analyzer
------------------------------------------------------------
Manual DCF loads immediately. The 10-K analyzer runs only after upload + button click.
The analyzer searches for statement pages using common US GAAP and industry title variants,
then uses OpenAI extraction if configured, with a regex fallback.

Run:
    streamlit run streamlit_10k_dcf_app_fast_launch_enhanced.py

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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fast 10-K DCF App", page_icon="📊", layout="wide")

# -----------------------------------------------------------------------------
# Helpers
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
    if not s or s.lower() in {"na", "n/a", "none", "null", "--", "—"}:
        return None
    neg = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = re.sub(r"[$,%\s,]", "", s).replace("(", "").replace(")", "").replace("−", "-")
    s = s.lstrip("-")
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None


def set_if_present(data: Dict[str, Any], key: str) -> bool:
    val = clean_number(data.get(key))
    if val is not None:
        st.session_state[key] = val
        return True
    return False


def text_value(data: Dict[str, Any], key: str) -> bool:
    val = data.get(key)
    if val not in (None, "", "N/A", "null"):
        st.session_state[key] = str(val)
        return True
    return False


def first_number_in_line(line: str, prefer: int = 0) -> Optional[float]:
    """Return a number from a filing table row. prefer=0 usually selects current year."""
    nums = re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", line)
    cleaned = [clean_number(n) for n in nums]
    cleaned = [n for n in cleaned if n is not None]
    if not cleaned:
        return None
    if prefer < len(cleaned):
        return cleaned[prefer]
    return cleaned[0]

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

REQUIRED_NUMERIC_FIELDS = [
    "revenue_t", "revenue_t1", "revenue_t2", "ebit_t", "ebit_t1", "tax_expense_t",
    "pretax_income_t", "depreciation_t", "capex_t", "cash_t", "total_debt_t",
    "diluted_shares_t", "current_assets_t", "current_liabilities_t",
    "current_assets_t1", "current_liabilities_t1",
]

# -----------------------------------------------------------------------------
# Statement title and line-item aliases
# -----------------------------------------------------------------------------

STATEMENT_TITLE_ALIASES = {
    "income_statement": [
        "consolidated statements of income",
        "consolidated statement of income",
        "consolidated statements of operations",
        "consolidated statement of operations",
        "consolidated statements of earnings",
        "consolidated statement of earnings",
        "consolidated statement of operations and comprehensive income",
        "consolidated statements of operations and comprehensive income",
        "consolidated statement of operations and comprehensive loss",
        "consolidated statements of operations and comprehensive loss",
        "consolidated statement of operations and comprehensive gain",
        "consolidated statements of operations and comprehensive gain",
        "consolidated statements of comprehensive income",
        "consolidated statement of comprehensive income",
        "consolidated statements of comprehensive loss",
        "consolidated income statement",
        "statement of income",
        "statement of operations",
        "statement of earnings",
    ],
    "balance_sheet": [
        "consolidated balance sheets",
        "consolidated balance sheet",
        "consolidated statements of financial position",
        "consolidated statement of financial position",
        "balance sheets",
        "balance sheet",
        "statement of financial position",
    ],
    "cash_flow_statement": [
        "consolidated statements of cash flows",
        "consolidated statement of cash flows",
        "consolidated statements of cash flow",
        "consolidated statement of cash flow",
        "statements of cash flows",
        "statement of cash flows",
        "cash flow statements",
    ],
    "equity_eps_notes": [
        "earnings per share",
        "net income per share",
        "loss per share",
        "weighted average shares",
        "weighted-average shares",
        "shares used in computing",
        "basic and diluted",
    ],
}

LINE_ALIASES = {
    "revenue": [
        "revenue", "revenues", "total revenues", "net revenues", "net revenue",
        "net sales", "sales", "sales revenue", "total net sales", "service revenue",
        "product revenue", "operating revenues", "total operating revenues",
        "customer revenue", "contract revenue", "subscription revenue",
    ],
    "ebit": [
        "operating income", "operating loss", "income from operations", "loss from operations",
        "operating profit", "profit from operations", "earnings from operations",
        "income before interest and income taxes", "ebit",
    ],
    "pretax_income": [
        "income before income taxes", "loss before income taxes",
        "income before provision for income taxes", "loss before provision for income taxes",
        "income before taxes", "earnings before income taxes", "pretax income",
    ],
    "tax_expense": [
        "income tax expense", "provision for income taxes", "benefit from income taxes",
        "income tax benefit", "provision for taxes", "tax expense", "benefit for income taxes",
    ],
    "depreciation": [
        "depreciation and amortization", "depreciation & amortization", "depreciation",
        "amortization", "depreciation, depletion and amortization", "dd&a",
    ],
    "capex": [
        "capital expenditures", "capital expenditure", "purchases of property and equipment",
        "purchase of property and equipment", "additions to property and equipment",
        "payments for property and equipment", "property and equipment additions",
        "purchases of property, plant and equipment", "additions to property, plant and equipment",
        "capitalized software", "payments for acquisition of property and equipment",
    ],
    "cash": [
        "cash and cash equivalents", "cash, cash equivalents and restricted cash",
        "cash and short-term investments", "cash equivalents", "cash",
    ],
    "total_debt": [
        "total debt", "total borrowings", "short-term debt", "current portion of long-term debt",
        "long-term debt", "long term debt", "debt, current", "debt, non-current",
        "notes payable", "senior notes", "finance lease liabilities", "lease obligations",
    ],
    "diluted_shares": [
        "weighted average diluted shares", "weighted-average diluted shares",
        "diluted weighted average shares", "weighted average shares diluted",
        "shares used in computing diluted", "diluted shares", "weighted-average shares outstanding diluted",
    ],
    "current_assets": ["total current assets", "current assets"],
    "current_liabilities": ["total current liabilities", "current liabilities"],
}

# -----------------------------------------------------------------------------
# Delayed PDF extraction and analyzer
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def extract_pdf_pages_cached(file_bytes: bytes, max_pages: int = 160) -> List[Tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        try:
            from PyPDF2 import PdfReader
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PDF support is not installed. Add `pypdf` to requirements.txt, commit/push it, and reboot the app."
            ) from exc

    reader = PdfReader(BytesIO(file_bytes))
    limit = min(len(reader.pages), max_pages)
    pages: List[Tuple[int, str]] = []
    for i in range(limit):
        try:
            txt = reader.pages[i].extract_text() or ""
            if txt.strip():
                pages.append((i + 1, txt))
        except Exception:
            continue
    return pages


def score_statement_page(page_text: str) -> Tuple[int, List[str]]:
    low = re.sub(r"\s+", " ", page_text.lower())
    hits: List[str] = []
    score = 0
    for stmt_type, titles in STATEMENT_TITLE_ALIASES.items():
        for title in titles:
            if title in low:
                score += 12
                hits.append(f"{stmt_type}: {title}")
                break
    for group, aliases in LINE_ALIASES.items():
        for alias in aliases:
            if alias in low:
                score += 1
                break
    # Penalize table of contents / index pages that mention lots of statements but have little data.
    if "table of contents" in low or "index to financial statements" in low:
        score -= 6
    if len(re.findall(r"\(?\d[\d,]+\)?", page_text)) > 20:
        score += 4
    return score, hits


def select_relevant_statement_pages(pages: List[Tuple[int, str]], max_selected_pages: int = 18) -> Tuple[str, pd.DataFrame]:
    scored_rows = []
    for page_no, txt in pages:
        score, hits = score_statement_page(txt)
        if score > 0:
            scored_rows.append({"page": page_no, "score": score, "hits": "; ".join(hits), "text": txt})
    scored_rows.sort(key=lambda r: r["score"], reverse=True)

    # Include top scored pages and neighbor pages, since headers often appear on one page and table continues.
    selected_nums = set()
    for row in scored_rows[:max_selected_pages]:
        for p in (row["page"] - 1, row["page"], row["page"] + 1):
            if p > 0:
                selected_nums.add(p)
    selected_nums = set(sorted(selected_nums)[:max_selected_pages])

    page_dict = dict(pages)
    selected_text = []
    for p in sorted(selected_nums):
        if p in page_dict:
            selected_text.append(f"\n--- Page {p} ---\n{page_dict[p]}")

    diagnostics = pd.DataFrame([{k: v for k, v in row.items() if k != "text"} for row in scored_rows[:25]])
    return "\n".join(selected_text) if selected_text else "\n".join(f"\n--- Page {p} ---\n{t}" for p, t in pages[:max_selected_pages]), diagnostics


def regex_extract_from_statement_pages(text: str) -> Dict[str, Any]:
    """Fallback extraction using aliases. It is intentionally conservative."""
    out: Dict[str, Any] = {}
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

    def find_alias_value(alias_group: str, prefer: int = 0, current_only: bool = False) -> Optional[float]:
        aliases = LINE_ALIASES[alias_group]
        for line in lines:
            low = line.lower()
            if any(re.search(rf"\b{re.escape(a)}\b", low) for a in aliases):
                # Avoid selecting subtotal lines when a specific total exists later only if applicable.
                val = first_number_in_line(line, prefer=prefer)
                if val is not None:
                    return val
        return None

    # Income statement rows commonly have current/prior/two-years-ago in left-to-right order.
    revs = []
    for line in lines:
        low = line.lower()
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in LINE_ALIASES["revenue"]):
            vals = [clean_number(n) for n in re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", line)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 1:
                revs = vals[:3]
                break
    if len(revs) > 0: out["revenue_t"] = revs[0]
    if len(revs) > 1: out["revenue_t1"] = revs[1]
    if len(revs) > 2: out["revenue_t2"] = revs[2]

    ebits = []
    for line in lines:
        low = line.lower()
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in LINE_ALIASES["ebit"]):
            vals = [clean_number(n) for n in re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", line)]
            vals = [v for v in vals if v is not None]
            if vals:
                ebits = vals[:2]
                break
    if len(ebits) > 0: out["ebit_t"] = ebits[0]
    if len(ebits) > 1: out["ebit_t1"] = ebits[1]

    mappings = {
        "pretax_income_t": "pretax_income",
        "tax_expense_t": "tax_expense",
        "depreciation_t": "depreciation",
        "capex_t": "capex",
        "cash_t": "cash",
        "diluted_shares_t": "diluted_shares",
    }
    for field, group in mappings.items():
        val = find_alias_value(group)
        if val is not None:
            out[field] = abs(val) if field in {"tax_expense_t", "depreciation_t", "capex_t"} else val

    # Current assets and liabilities often have current/prior columns.
    ca_vals = []
    cl_vals = []
    for line in lines:
        low = line.lower()
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in LINE_ALIASES["current_assets"]):
            vals = [clean_number(n) for n in re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", line)]
            ca_vals = [v for v in vals if v is not None][:2]
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in LINE_ALIASES["current_liabilities"]):
            vals = [clean_number(n) for n in re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", line)]
            cl_vals = [v for v in vals if v is not None][:2]
    if len(ca_vals) > 0: out["current_assets_t"] = ca_vals[0]
    if len(ca_vals) > 1: out["current_assets_t1"] = ca_vals[1]
    if len(cl_vals) > 0: out["current_liabilities_t"] = cl_vals[0]
    if len(cl_vals) > 1: out["current_liabilities_t1"] = cl_vals[1]

    # Debt may be split. Sum current debt + long-term debt if total debt is not directly found.
    debt_direct = find_alias_value("total_debt")
    if debt_direct is not None:
        out["total_debt_t"] = abs(debt_direct)

    # Filing unit clues.
    low_text = text.lower()
    if "in millions" in low_text or "amounts in millions" in low_text or "dollars in millions" in low_text:
        out["units"] = "Millions"
    elif "in thousands" in low_text or "amounts in thousands" in low_text or "dollars in thousands" in low_text:
        out["units"] = "Thousands"
    elif "in billions" in low_text:
        out["units"] = "Billions"

    return out


def ai_extract_10k(statement_text: str, diagnostics_text: str = "") -> Dict[str, Any]:
    from openai import OpenAI  # delayed import

    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))
    keys = list(DEFAULTS.keys())
    prompt = f"""
You are extracting directly observable 10-K line items for a DCF app.
Return ONLY valid JSON using these exact keys: {keys}

Rules:
- Use only values directly stated in the filing text. Do not calculate margins, WACC, FCF, or valuation.
- It is acceptable to map US GAAP / industry-standard alternate labels to the app fields.
- Prefer consolidated financial statements over parent-only or segment tables.
- Prefer the most recent fiscal year as *_t, prior year as *_t1, and two-years-ago as *_t2.
- Preserve the filing's reporting units: Actual dollars, Thousands, Millions, or Billions.
- If a value is not clearly found, use null.
- For capex_t, use cash paid for purchases/additions of PP&E or capital expenditures from investing cash flows.
- For depreciation_t, use depreciation and amortization from operating cash flows or notes.
- For total_debt_t, use total debt if given. If separate current and long-term debt are both stated near each other, you may add them and include the sum.
- For diluted_shares_t, use weighted-average diluted shares outstanding, not EPS.
- For ebit_t, use operating income/loss or income/loss from operations.
- For tax_expense_t, use income tax expense/provision. Use positive amount for expense; negative amount only if clearly a benefit.

Accepted statement title variants include but are not limited to:
{json.dumps(STATEMENT_TITLE_ALIASES, indent=2)}

Accepted line-item label variants include but are not limited to:
{json.dumps(LINE_ALIASES, indent=2)}

Statement page diagnostics:
{diagnostics_text}

10-K statement text:
{statement_text[:60000]}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract financial statement line items into strict JSON for a valuation app."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------------------------------------------------------
# UI: delayed analyzer
# -----------------------------------------------------------------------------

st.title("📊 Fast-Launch 10-K DCF Valuation App")
st.caption("Manual DCF loads first. The 10-K analyzer runs only after upload + button click.")

with st.expander("Optional delayed 10-K analyzer", expanded=False):
    st.write(
        "Upload a PDF 10-K, then click analyze. The app searches for financial statement pages using common title variants "
        "before extracting values. Review all pre-filled values before relying on the valuation."
    )
    upload = st.file_uploader("Upload 10-K PDF", type=["pdf"])
    use_ai = st.checkbox("Use OpenAI to pre-fill inputs", value=True)
    max_pages = st.slider("Maximum pages to scan", min_value=40, max_value=250, value=160, step=10)
    selected_page_cap = st.slider("Statement/context pages to send to extractor", min_value=8, max_value=30, value=18, step=2)

    if st.button("Analyze uploaded 10-K", disabled=upload is None):
        if upload is None:
            st.warning("Upload a PDF first.")
        else:
            try:
                with st.spinner("Reading PDF pages and locating financial statements..."):
                    pages = extract_pdf_pages_cached(upload.getvalue(), max_pages=max_pages)
                    statement_text, diagnostics = select_relevant_statement_pages(pages, max_selected_pages=selected_page_cap)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            extracted: Dict[str, Any] = {}
            ai_error = None
            if use_ai and "OPENAI_API_KEY" in st.secrets:
                try:
                    with st.spinner("Extracting 10-K line items from statement pages..."):
                        extracted = ai_extract_10k(statement_text, diagnostics.to_string(index=False) if not diagnostics.empty else "")
                except Exception as e:
                    ai_error = str(e)
                    extracted = {}
            elif use_ai:
                st.warning("No OPENAI_API_KEY secret found. Using regex fallback only.")

            fallback = regex_extract_from_statement_pages(statement_text)
            # Fill missing AI fields with fallback values, but do not override AI values.
            merged = dict(fallback)
            merged.update({k: v for k, v in extracted.items() if v not in (None, "", "null")})
            extracted = merged

            changed = []
            for key in ["company_name", "fiscal_year", "units"]:
                if text_value(extracted, key):
                    changed.append(key)
            for key in REQUIRED_NUMERIC_FIELDS:
                if set_if_present(extracted, key):
                    changed.append(key)

            missing = [k for k in REQUIRED_NUMERIC_FIELDS if clean_number(extracted.get(k)) is None]
            if changed:
                st.success(f"Pre-filled {len(changed)} fields. Review and adjust the inputs below.")
            else:
                st.warning("No DCF inputs were confidently extracted. Try increasing page scan limits or uploading the SEC filing PDF directly.")
            if ai_error:
                st.warning(f"AI extraction failed and regex fallback was used. Error: {ai_error}")
            if missing:
                st.info("Fields not found clearly: " + ", ".join(missing))

            with st.expander("Statement page matches"):
                if diagnostics.empty:
                    st.write("No strong statement-page matches found. The app used the first available PDF pages as fallback context.")
                else:
                    st.dataframe(diagnostics, use_container_width=True)

            with st.expander("Show extracted JSON"):
                st.json(extracted)

            with st.expander("Show selected statement/context text"):
                st.text(statement_text[:30000])

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
    st.number_input("Income tax expense", step=10.0, key="tax_expense_t")
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

growth_hist = cagr(revenue_t2, revenue_t, 2) if revenue_t2 else safe_div(revenue_t - revenue_t1, revenue_t1)
ebit_margin = safe_div(ebit_t, revenue_t)
tax_rate = min(max(safe_div(tax_expense, pretax, 0.255), 0.0), 0.40)
dep_pct_sales = safe_div(dep, revenue_t)
capex_pct_sales = safe_div(capex, revenue_t)
nwc_pct_sales = safe_div(nwc_t, revenue_t)

cost_equity = (st.session_state.risk_free_rate / 100) + st.session_state.beta * (st.session_state.equity_risk_premium / 100)
after_tax_cost_debt = (st.session_state.pretax_cost_of_debt / 100) * (1 - tax_rate)
wacc = 0.80 * cost_equity + 0.20 * after_tax_cost_debt
terminal_growth = st.session_state.terminal_growth / 100
years = int(st.session_state.projection_years)
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
        "This is an educational DCF model. The analyzer searches for consolidated statement title variants such as "
        "statements of income, operations, comprehensive income/loss, balance sheets, statements of financial position, "
        "and cash flow statements. AI/PDF extraction can still misread tables, so verify each pre-filled value against the 10-K."
    )
