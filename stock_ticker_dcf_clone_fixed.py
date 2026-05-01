import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DCF Model by stock ticker", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top left, #172238 0, #0b1220 35%, #070b12 100%); color: #e8eefc;}
h1, h2, h3 {letter-spacing: -0.03em;}
div[data-testid="stMetric"] {background: linear-gradient(135deg, rgba(96,165,250,.13), rgba(110,231,183,.08)); border: 1px solid rgba(255,255,255,.10); padding: 18px; border-radius: 18px; box-shadow: 0 12px 30px rgba(0,0,0,.18);}
.section-card {background: rgba(18,26,43,.78); border: 1px solid rgba(255,255,255,.08); border-radius: 22px; padding: 22px; margin: 12px 0 18px 0; box-shadow: 0 18px 45px rgba(0,0,0,.22);}
.small-muted {color: #9fb0d0; font-size: 0.92rem;}
.pill {display: inline-block; padding: 5px 10px; border-radius: 999px; margin-right: 8px; border: 1px solid rgba(110,231,183,.35); color: #dffcf0; background: rgba(110,231,183,.08); font-size: .85rem;}
.warning-card {border: 1px solid rgba(251,191,36,.28); background: rgba(251,191,36,.08); padding: 14px 16px; border-radius: 16px;}
</style>
""", unsafe_allow_html=True)

def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def fmt_money(x, digits=1):
    x = safe_float(x, 0); neg = x < 0; x = abs(x)
    if x >= 1_000_000_000_000: s = f"${x/1_000_000_000_000:,.{digits}f}T"
    elif x >= 1_000_000_000: s = f"${x/1_000_000_000:,.{digits}f}B"
    elif x >= 1_000_000: s = f"${x/1_000_000:,.{digits}f}M"
    elif x >= 1_000: s = f"${x/1_000:,.{digits}f}K"
    else: s = f"${x:,.0f}"
    return f"({s})" if neg else s

def fmt_pct(x, digits=2):
    try: return f"{x*100:.{digits}f}%"
    except Exception: return "N/A"

def first_existing(df, labels):
    if df is None or df.empty: return None
    idx = [str(i).lower() for i in df.index]
    for label in labels:
        lab = label.lower()
        for pos, name in enumerate(idx):
            if lab in name:
                return df.iloc[pos]
    return None

def latest_value(series, default=0.0):
    if series is None: return default
    if isinstance(series, pd.Series):
        for val in series:
            if pd.notna(val): return safe_float(val, default)
    return safe_float(series, default)

def row_latest(df, labels, default=0.0):
    return latest_value(first_existing(df, labels), default)

def get_history_row(df, labels):
    row = first_existing(df, labels)
    return pd.Series(dtype=float) if row is None else pd.to_numeric(row, errors="coerce")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker_symbol: str):
    import yfinance as yf
    symbol = ticker_symbol.strip().upper()
    t = yf.Ticker(symbol)
    return {
        "symbol": symbol,
        "income": t.income_stmt,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
        "quarterly_income": t.quarterly_income_stmt,
        "quarterly_balance": t.quarterly_balance_sheet,
        "quarterly_cashflow": t.quarterly_cashflow,
        "info": t.info or {},
        "history": t.history(period="1y"),
    }

def estimate_revenue_growth(income):
    rev_row = get_history_row(income, ["Total Revenue", "Operating Revenue", "Revenue", "Net Sales"])
    vals = list(rev_row.dropna())
    if len(vals) >= 2 and vals[1]:
        return min(max((vals[0] / vals[1]) - 1, -0.25), 0.35)
    return 0.05

def infer_financials(data):
    income, balance, cashflow, info = data["income"], data["balance"], data["cashflow"], data["info"]
    revenue = row_latest(income, ["Total Revenue", "Operating Revenue", "Revenue", "Net Sales"])
    ebit = row_latest(income, ["EBIT", "Operating Income", "Income From Operations"])
    ebt = row_latest(income, ["Pretax Income", "Income Before Tax", "Income Before Income Taxes"])
    tax_expense = abs(row_latest(income, ["Tax Provision", "Income Tax Expense", "Provision For Income Taxes"], 0))
    net_income = row_latest(income, ["Net Income", "Net Income Common Stockholders"])
    depreciation = row_latest(cashflow, ["Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion"])
    capex = abs(row_latest(cashflow, ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE", "Investments In Property Plant And Equipment"]))
    cfo = row_latest(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Net Cash Provided By Operating Activities"])
    fcf_yahoo = row_latest(cashflow, ["Free Cash Flow"], cfo - capex if cfo or capex else 0)
    cash = row_latest(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"])
    total_debt = row_latest(balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation"], 0)
    if total_debt == 0:
        total_debt = row_latest(balance, ["Long Term Debt"], 0) + row_latest(balance, ["Current Debt", "Short Long Term Debt"], 0)
    current_assets = row_latest(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = row_latest(balance, ["Current Liabilities", "Total Current Liabilities"])
    nwc = current_assets - current_liabilities if current_assets or current_liabilities else 0
    shares = safe_float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or row_latest(income, ["Diluted Average Shares", "Basic Average Shares"], 0))
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    beta = safe_float(info.get("beta"), 1.0)
    market_cap = safe_float(info.get("marketCap"), price * shares if price and shares else 0)
    trailing_eps = safe_float(info.get("trailingEps"), net_income / shares if shares else 0)
    forward_eps = safe_float(info.get("forwardEps"), 0)
    ev = safe_float(info.get("enterpriseValue"), market_cap + total_debt - cash)
    return {
        "name": info.get("longName") or info.get("shortName") or data["symbol"], "sector": info.get("sector", "N/A"), "industry": info.get("industry", "N/A"),
        "revenue": revenue, "ebit": ebit, "ebt": ebt, "tax_expense": tax_expense, "net_income": net_income, "depreciation": depreciation,
        "capex": capex, "cfo": cfo, "fcf_yahoo": fcf_yahoo, "cash": cash, "total_debt": total_debt, "current_assets": current_assets,
        "current_liabilities": current_liabilities, "nwc": nwc, "shares": shares, "price": price, "beta": beta, "market_cap": market_cap,
        "ev": ev, "trailing_eps": trailing_eps, "forward_eps": forward_eps, "trailing_pe": safe_float(info.get("trailingPE"), price / trailing_eps if trailing_eps else 0),
        "forward_pe": safe_float(info.get("forwardPE"), price / forward_eps if forward_eps else 0), "ev_ebitda": safe_float(info.get("enterpriseToEbitda"), 0),
        "dividend_yield": safe_float(info.get("dividendYield"), 0), "tax_rate": min(max(tax_expense / ebt if ebt > 0 and tax_expense >= 0 else 0.25, 0.0), 0.40),
        "ebit_margin": ebit / revenue if revenue else 0.15, "fcf_margin": fcf_yahoo / revenue if revenue else 0.05, "revenue_growth": estimate_revenue_growth(income),
        "capex_pct_revenue": capex / revenue if revenue else 0.04, "da_pct_revenue": depreciation / revenue if revenue else 0.03, "nwc_pct_revenue": nwc / revenue if revenue else 0.08,
    }

def calculate_wacc(beta, risk_free_rate, erp, pre_tax_cost_debt, tax_rate, market_cap, total_debt):
    cost_equity = risk_free_rate + beta * erp
    ev = max(market_cap + total_debt, 1)
    eq_w = market_cap / ev if market_cap > 0 else 0.85
    debt_w = total_debt / ev if total_debt > 0 else 0.15
    if market_cap <= 0 and total_debt <= 0: eq_w, debt_w = 0.85, 0.15
    after_tax_debt = pre_tax_cost_debt * (1 - tax_rate)
    return eq_w * cost_equity + debt_w * after_tax_debt, cost_equity, after_tax_debt, eq_w, debt_w

def build_dcf(base_revenue, years, revenue_growth, terminal_growth, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, wacc, debt, cash, shares):
    rows, pv_fcfs = [], []
    revenue = base_revenue; prior_nwc = base_revenue * nwc_pct
    for year in range(1, years + 1):
        revenue *= (1 + revenue_growth)
        ebit = revenue * ebit_margin; nopat = ebit * (1 - tax_rate)
        da = revenue * da_pct; capex = revenue * capex_pct
        nwc = revenue * nwc_pct; change_nwc = nwc - prior_nwc
        fcf = nopat + da - capex - change_nwc
        df = 1 / ((1 + wacc) ** year); pv_fcf = fcf * df
        pv_fcfs.append(pv_fcf)
        rows.append({"Year": year, "Revenue": revenue, "Revenue Growth": revenue_growth, "EBIT": ebit, "EBIT Margin": ebit_margin, "NOPAT": nopat, "D&A": da, "CapEx": capex, "Change in NWC": change_nwc, "Free Cash Flow": fcf, "Discount Factor": df, "PV of FCF": pv_fcf})
        prior_nwc = nwc
    final_fcf = rows[-1]["Free Cash Flow"] if rows else 0
    tv = np.nan if wacc <= terminal_growth else final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = np.nan if np.isnan(tv) else tv / ((1 + wacc) ** years)
    enterprise_value = sum(pv_fcfs) + safe_float(pv_tv, 0)
    equity_value = enterprise_value - debt + cash
    intrinsic_price = equity_value / shares if shares else np.nan
    return pd.DataFrame(rows), {"pv_fcfs": sum(pv_fcfs), "terminal_value": tv, "pv_terminal": pv_tv, "enterprise_value": enterprise_value, "equity_value": equity_value, "intrinsic_price": intrinsic_price}

def sensitivity(base_revenue, years, revenue_growth, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, debt, cash, shares, wacc_values, tg_values):
    table = pd.DataFrame(index=[fmt_pct(x, 1) for x in wacc_values], columns=[fmt_pct(x, 1) for x in tg_values])
    for w in wacc_values:
        for g in tg_values:
            _, s = build_dcf(base_revenue, years, revenue_growth, g, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, w, debt, cash, shares)
            table.loc[fmt_pct(w,1), fmt_pct(g,1)] = None if np.isnan(s["intrinsic_price"]) else round(s["intrinsic_price"], 2)
    return table

def display_statement(df, title, max_rows=18):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No data returned from Yahoo Finance for this statement."); return
    d = df.copy(); d.columns = [str(c.date()) if hasattr(c, "date") else str(c) for c in d.columns]
    st.dataframe(d.head(max_rows).style.format("${:,.0f}", na_rep="—"), use_container_width=True)

st.title("DCF Model by stock ticker")
st.markdown('<span class="pill">Yahoo Finance data pull</span><span class="pill">Auto-built DCF</span><span class="pill">Editable assumptions</span><span class="pill">Sensitivity table</span>', unsafe_allow_html=True)
st.markdown('<p class="small-muted">Search a ticker, review pulled financials, adjust assumptions, and compare intrinsic value to the market price.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Search")
    ticker_input = st.text_input("Stock ticker", value="AAPL")
    search_clicked = st.button("Pull Yahoo Finance data", type="primary", use_container_width=True)
    st.divider(); st.header("Model defaults")
    default_rf = st.number_input("Risk-free rate", value=4.35, min_value=0.0, max_value=10.0, step=0.05, format="%.2f") / 100
    default_erp = st.number_input("Equity risk premium", value=5.00, min_value=0.0, max_value=12.0, step=0.10, format="%.2f") / 100
    default_cod = st.number_input("Pre-tax cost of debt", value=5.50, min_value=0.0, max_value=20.0, step=0.10, format="%.2f") / 100
    default_tg = st.number_input("Terminal growth", value=2.50, min_value=-2.0, max_value=6.0, step=0.10, format="%.2f") / 100
    default_years = st.slider("Projection years", 3, 10, 5)

if "ticker_data" not in st.session_state: st.session_state.ticker_data = None
if search_clicked:
    try:
        with st.spinner(f"Pulling {ticker_input.upper()} from Yahoo Finance..."):
            st.session_state.ticker_data = fetch_ticker_data(ticker_input)
    except Exception as e:
        st.error(f"Could not load ticker data. Check the ticker and try again. Error: {e}")

data = st.session_state.ticker_data
if data is None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How to use this app")
    st.write("Enter a ticker in the sidebar and click **Pull Yahoo Finance data**. The app pulls financial statements, market data, multiples, and DCF inputs from Yahoo Finance through yfinance.")
    st.write("The assumptions remain editable because Yahoo Finance data can be incomplete or company-specific.")
    st.markdown('</div>', unsafe_allow_html=True); st.stop()

f = infer_financials(data)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
left, right = st.columns([2, 1])
with left:
    st.subheader(f"{f['name']} ({data['symbol']})")
    st.write(f"**Sector:** {f['sector']}  \n**Industry:** {f['industry']}")
with right:
    st.metric("Current Price", fmt_money(f["price"], 2)); st.metric("Market Cap", fmt_money(f["market_cap"]))
st.markdown('</div>', unsafe_allow_html=True)

for cols in [[("Revenue", f["revenue"]), ("EBIT Margin", f["ebit_margin"], "pct"), ("Free Cash Flow", f["fcf_yahoo"]), ("Net Debt", f["total_debt"] - f["cash"])], [("Trailing P/E", f["trailing_pe"], "x"), ("Forward P/E", f["forward_pe"], "x"), ("EV / EBITDA", f["ev_ebitda"], "x"), ("Beta", f["beta"], "num")]]:
    c = st.columns(4)
    for i, item in enumerate(cols):
        label, val = item[0], item[1]; typ = item[2] if len(item)>2 else "money"
        if typ=="pct": text = fmt_pct(val)
        elif typ=="x": text = f"{val:.2f}x" if val else "N/A"
        elif typ=="num": text = f"{val:.2f}"
        else: text = fmt_money(val)
        c[i].metric(label, text)

tabs = st.tabs(["Step 1 — Pulled Financials", "Step 2 — Forecast Assumptions", "Step 3 — Free Cash Flow", "Step 4 — Cost of Capital / WACC", "Step 5 — Terminal Value", "Step 6 — Valuation Output", "Sensitivity", "Raw Yahoo Data"])

with tabs[0]:
    st.header("Step 1 — Financial information pulled from Yahoo Finance")
    summary = pd.DataFrame({"Metric": ["Revenue", "EBIT", "Net Income", "Operating Cash Flow", "Capital Expenditures", "Free Cash Flow", "Cash", "Total Debt", "Shares Outstanding", "Market Cap", "Enterprise Value"], "Value": [f["revenue"], f["ebit"], f["net_income"], f["cfo"], f["capex"], f["fcf_yahoo"], f["cash"], f["total_debt"], f["shares"], f["market_cap"], f["ev"]]})
    st.dataframe(summary.style.format({"Value": "${:,.0f}"}), use_container_width=True)

with tabs[1]:
    st.header("Step 2 — Forecast assumptions")
    a,b,c = st.columns(3)
    revenue_growth = a.number_input("Annual revenue growth", value=float(round(f["revenue_growth"]*100,2)), min_value=-50.0, max_value=75.0, step=0.25)/100
    ebit_margin = a.number_input("EBIT margin", value=float(round(f["ebit_margin"]*100,2)), min_value=-50.0, max_value=75.0, step=0.25)/100
    tax_rate = b.number_input("Effective tax rate", value=float(round(f["tax_rate"]*100,2)), min_value=0.0, max_value=50.0, step=0.25)/100
    da_pct = b.number_input("D&A as % of revenue", value=float(round(f["da_pct_revenue"]*100,2)), min_value=0.0, max_value=50.0, step=0.25)/100
    capex_pct = c.number_input("CapEx as % of revenue", value=float(round(f["capex_pct_revenue"]*100,2)), min_value=0.0, max_value=75.0, step=0.25)/100
    nwc_pct = c.number_input("Net working capital as % of revenue", value=float(round(f["nwc_pct_revenue"]*100,2)), min_value=-50.0, max_value=75.0, step=0.25)/100
    st.info("Yahoo Finance gives the starting point. Adjust these assumptions using the 10-K, MD&A, industry trends, and your judgment.")

with tabs[3]:
    st.header("Step 4 — Cost of Capital / WACC")
    a,b,c = st.columns(3)
    beta = a.number_input("Beta", value=float(round(f["beta"], 2)), min_value=-1.0, max_value=5.0, step=0.05)
    risk_free_rate = a.number_input("Risk-free rate", value=float(round(default_rf*100,2)), min_value=0.0, max_value=10.0, step=0.05)/100
    erp = b.number_input("Equity risk premium", value=float(round(default_erp*100,2)), min_value=0.0, max_value=15.0, step=0.10)/100
    cod = b.number_input("Pre-tax cost of debt", value=float(round(default_cod*100,2)), min_value=0.0, max_value=25.0, step=0.10)/100
    market_cap_adj = c.number_input("Market value of equity", value=float(max(f["market_cap"],0)), min_value=0.0, step=1_000_000.0)
    debt_adj = c.number_input("Market/book value of debt", value=float(max(f["total_debt"],0)), min_value=0.0, step=1_000_000.0)
    wacc, cost_equity, after_tax_debt, eq_w, debt_w = calculate_wacc(beta, risk_free_rate, erp, cod, tax_rate, market_cap_adj, debt_adj)
    x = st.columns(4); x[0].metric("Cost of Equity", fmt_pct(cost_equity)); x[1].metric("After-tax Cost of Debt", fmt_pct(after_tax_debt)); x[2].metric("Equity Weight", fmt_pct(eq_w)); x[3].metric("WACC", fmt_pct(wacc))
    st.latex(r"WACC = \frac{E}{D+E}R_e + \frac{D}{D+E}R_d(1-T)")

if 'revenue_growth' not in locals():
    revenue_growth, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct = f["revenue_growth"], f["ebit_margin"], f["tax_rate"], f["da_pct_revenue"], f["capex_pct_revenue"], f["nwc_pct_revenue"]
if 'wacc' not in locals():
    wacc, cost_equity, after_tax_debt, eq_w, debt_w = calculate_wacc(f["beta"], default_rf, default_erp, default_cod, tax_rate, f["market_cap"], f["total_debt"])

with tabs[4]:
    st.header("Step 5 — Terminal value")
    years = st.slider("Projection period", 3, 10, default_years, key="terminal_years")
    terminal_growth = st.number_input("Long-term terminal growth rate", value=float(round(default_tg*100,2)), min_value=-2.0, max_value=6.0, step=0.10)/100
    st.latex(r"Terminal\ Value = \frac{FCF_n(1+g)}{WACC-g}")
    if terminal_growth >= wacc: st.error("Terminal growth must be lower than WACC.")
if 'years' not in locals(): years, terminal_growth = default_years, default_tg

dcf_df, dcf_summary = build_dcf(f["revenue"], years, revenue_growth, terminal_growth, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, wacc, f["total_debt"], f["cash"], f["shares"])

with tabs[2]:
    st.header("Step 3 — Free cash flow forecast")
    st.latex(r"FCF = EBIT(1-T) + D\&A - CapEx - \Delta NWC")
    money_cols = ["Revenue", "EBIT", "NOPAT", "D&A", "CapEx", "Change in NWC", "Free Cash Flow", "PV of FCF"]
    st.dataframe(dcf_df.style.format({**{col: "${:,.0f}" for col in money_cols}, "Revenue Growth": "{:.2%}", "EBIT Margin": "{:.2%}", "Discount Factor": "{:.3f}"}), use_container_width=True)
    st.line_chart(dcf_df.set_index("Year")[["Revenue", "Free Cash Flow"]])

with tabs[5]:
    st.header("Step 6 — Valuation output")
    o = st.columns(4); o[0].metric("PV of FCFs", fmt_money(dcf_summary["pv_fcfs"])); o[1].metric("PV of Terminal Value", fmt_money(dcf_summary["pv_terminal"])); o[2].metric("Enterprise Value", fmt_money(dcf_summary["enterprise_value"])); o[3].metric("Equity Value", fmt_money(dcf_summary["equity_value"]))
    intrinsic = dcf_summary["intrinsic_price"]; upside = (intrinsic / f["price"] - 1) if f["price"] and not np.isnan(intrinsic) else np.nan
    v = st.columns(3); v[0].metric("Intrinsic Value / Share", fmt_money(intrinsic, 2)); v[1].metric("Current Market Price", fmt_money(f["price"], 2)); v[2].metric("Implied Upside / Downside", fmt_pct(upside) if not np.isnan(upside) else "N/A")
    if not np.isnan(upside):
        if upside > .15: st.success("Based on these assumptions, the model indicates potential undervaluation.")
        elif upside < -.15: st.error("Based on these assumptions, the model indicates potential overvaluation.")
        else: st.warning("Based on these assumptions, the stock appears relatively close to estimated fair value.")

with tabs[6]:
    st.header("Sensitivity analysis")
    wacc_values = np.arange(max(wacc-.02, .01), wacc+.021, .01)
    tg_values = np.arange(max(terminal_growth-.01, -.01), terminal_growth+.011, .005)
    st.dataframe(sensitivity(f["revenue"], years, revenue_growth, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, f["total_debt"], f["cash"], f["shares"], wacc_values, tg_values), use_container_width=True)
    st.caption("Rows are WACC. Columns are terminal growth. Values are intrinsic value per share.")

with tabs[7]:
    st.header("Raw Yahoo Finance financial statements")
    a,b,c = st.columns(3)
    with a: display_statement(data["income"], "Income Statement")
    with b: display_statement(data["balance"], "Balance Sheet")
    with c: display_statement(data["cashflow"], "Cash Flow Statement")
    with st.expander("Raw company info dictionary"):
        st.json(data["info"])

st.markdown('<div class="warning-card"><b>Model note:</b> Yahoo Finance data can be incomplete, delayed, or company-specific. This app is for education and screening, not investment advice. Always verify against the original 10-K / 10-Q.</div>', unsafe_allow_html=True)
