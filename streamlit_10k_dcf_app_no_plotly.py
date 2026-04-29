"""
Streamlit 10-K DCF Valuation App
--------------------------------
Purpose:
    User enters values directly available in a company's Form 10-K.
    The app calculates historical ratios, projects free cash flow,
    estimates WACC, terminal value, enterprise value, equity value,
    and implied value per share.

Run locally:
    streamlit run streamlit_10k_dcf_app.py

Deploy:
    Upload this file to GitHub with a requirements.txt containing:
        streamlit
        pandas
        numpy
        plotly
    Then connect the repo to Streamlit Community Cloud.
"""

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="10-K DCF Valuation App", page_icon="📊", layout="wide")
st.title("📊 10-K Driven DCF Valuation App")
st.caption(
    "Enter only data that can usually be found directly in a Form 10-K. "
    "The app calculates margins, working capital changes, free cash flow, WACC, terminal value, and implied share value."
)

# ------------------------------------------------------------
# Default assumptions - user can override
# ------------------------------------------------------------
DEFAULT_RISK_FREE_RATE = 0.0435       # Approx. current 10Y Treasury default
DEFAULT_EQUITY_RISK_PREMIUM = 0.0475  # Broad-market default; override for your class/source
DEFAULT_PRETAX_COST_OF_DEBT = 0.0650  # Broad investment-grade-ish assumption; override if company-specific debt note gives rate
DEFAULT_EFFECTIVE_TAX_RATE = 0.2550   # Approx. 21% federal + blended state impact assumption
DEFAULT_TERMINAL_GROWTH = 0.0250      # Common mature-company long-run nominal growth assumption
DEFAULT_MARKET_BETA = 1.00
DEFAULT_PROJECTION_YEARS = 5

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if denominator in [0, None] or np.isnan(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default


def cagr(beginning_value: float, ending_value: float, periods: int) -> float:
    """Compound annual growth rate, safely calculated."""
    if beginning_value <= 0 or ending_value <= 0 or periods <= 0:
        return 0.0
    return (ending_value / beginning_value) ** (1 / periods) - 1


def format_dollars(x: float) -> str:
    """Format dollars with negatives shown clearly."""
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"


def format_pct(x: float) -> str:
    """Format percentage."""
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def estimate_beta_from_industry(industry_risk: str) -> float:
    """
    Broad beta defaults when the 10-K does not provide a beta.
    These are intentionally rough and should be overridden when using a class-provided beta source.
    """
    beta_map = {
        "Defensive / Utilities / Staples": 0.75,
        "Average market risk": 1.00,
        "Cyclical / Consumer discretionary / Industrials": 1.15,
        "High-growth / Technology / Biotech": 1.30,
        "Highly leveraged or distressed": 1.50,
    }
    return beta_map.get(industry_risk, DEFAULT_MARKET_BETA)


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Model Controls")
units = st.sidebar.selectbox(
    "Input units",
    ["Millions", "Thousands", "Actual dollars"],
    index=0,
    help="Use the same unit convention as the 10-K tables. Most 10-K financial statements are in millions or thousands."
)
unit_multiplier = {"Millions": 1_000_000, "Thousands": 1_000, "Actual dollars": 1}[units]

projection_years = st.sidebar.slider("Projection years", 3, 10, DEFAULT_PROJECTION_YEARS)

valuation_method = st.sidebar.selectbox(
    "Terminal value method",
    ["Perpetuity growth", "Exit EBITDA multiple"],
    index=0
)

st.sidebar.subheader("Broad Assumption Defaults")
risk_free_rate = st.sidebar.number_input("Risk-free rate (%)", value=DEFAULT_RISK_FREE_RATE * 100, step=0.05) / 100
equity_risk_premium = st.sidebar.number_input("Equity risk premium (%)", value=DEFAULT_EQUITY_RISK_PREMIUM * 100, step=0.05) / 100
pretax_cost_of_debt = st.sidebar.number_input("Pre-tax cost of debt (%)", value=DEFAULT_PRETAX_COST_OF_DEBT * 100, step=0.10) / 100
tax_rate_default = st.sidebar.number_input("Default tax rate (%)", value=DEFAULT_EFFECTIVE_TAX_RATE * 100, step=0.10) / 100
terminal_growth = st.sidebar.number_input("Terminal growth rate (%)", value=DEFAULT_TERMINAL_GROWTH * 100, step=0.10) / 100
exit_ebitda_multiple = st.sidebar.number_input("Exit EBITDA multiple", value=8.0, step=0.5)

# ------------------------------------------------------------
# Company and market inputs
# ------------------------------------------------------------
st.header("1) Company Information")
col_a, col_b, col_c = st.columns(3)
with col_a:
    company_name = st.text_input("Company name", value="Example Company")
with col_b:
    fiscal_year = st.number_input("Most recent fiscal year", min_value=1990, max_value=2100, value=2025, step=1)
with col_c:
    industry_risk = st.selectbox(
        "Industry risk profile",
        [
            "Defensive / Utilities / Staples",
            "Average market risk",
            "Cyclical / Consumer discretionary / Industrials",
            "High-growth / Technology / Biotech",
            "Highly leveraged or distressed",
        ],
        index=1,
        help="Used only to estimate beta if you do not input a company beta."
    )

st.header("2) 10-K Historical Inputs")
st.write(
    "Enter the values exactly as shown in the 10-K financial statements. "
    "Use positive numbers for costs, CapEx, debt, and working capital accounts."
)

with st.expander("Where to find these in a 10-K", expanded=False):
    st.markdown(
        """
        - **Revenue / Net sales:** Consolidated Statement of Operations or Income Statement  
        - **Operating income / EBIT:** Consolidated Statement of Operations  
        - **Income tax expense:** Consolidated Statement of Operations or tax footnote  
        - **Net income:** Consolidated Statement of Operations  
        - **D&A:** Consolidated Statement of Cash Flows, usually under operating activities  
        - **Capital expenditures:** Consolidated Statement of Cash Flows, usually purchases of property and equipment  
        - **Current assets / current liabilities:** Consolidated Balance Sheet  
        - **Cash and cash equivalents:** Consolidated Balance Sheet  
        - **Total debt:** Balance Sheet and debt footnote; include short-term borrowings and long-term debt  
        - **Interest expense:** Income Statement or debt/interest footnote  
        - **Shares outstanding / diluted shares:** Cover page, equity footnote, or EPS note  
        """
    )

st.subheader("Most Recent Year")
col1, col2, col3 = st.columns(3)
with col1:
    revenue_t = st.number_input("Revenue / net sales", min_value=0.0, value=10_000.0, step=100.0)
    ebit_t = st.number_input("Operating income / EBIT", value=1_500.0, step=100.0)
    net_income_t = st.number_input("Net income", value=1_000.0, step=100.0)
    tax_expense_t = st.number_input("Income tax expense", min_value=0.0, value=350.0, step=10.0)
with col2:
    depreciation_t = st.number_input("Depreciation & amortization", min_value=0.0, value=500.0, step=10.0)
    capex_t = st.number_input("Capital expenditures / purchases of PP&E", min_value=0.0, value=600.0, step=10.0)
    interest_expense_t = st.number_input("Interest expense", min_value=0.0, value=200.0, step=10.0)
    cash_t = st.number_input("Cash and cash equivalents", min_value=0.0, value=800.0, step=50.0)
with col3:
    total_debt_t = st.number_input("Total debt", min_value=0.0, value=2_500.0, step=100.0)
    diluted_shares_t = st.number_input("Diluted weighted-average shares or shares outstanding", min_value=0.0001, value=500.0, step=1.0)
    current_assets_t = st.number_input("Current assets", min_value=0.0, value=4_000.0, step=100.0)
    current_liabilities_t = st.number_input("Current liabilities", min_value=0.0, value=3_000.0, step=100.0)

st.subheader("Prior Year")
col4, col5, col6 = st.columns(3)
with col4:
    revenue_t1 = st.number_input("Prior-year revenue / net sales", min_value=0.0, value=9_500.0, step=100.0)
    ebit_t1 = st.number_input("Prior-year operating income / EBIT", value=1_350.0, step=100.0)
with col5:
    current_assets_t1 = st.number_input("Prior-year current assets", min_value=0.0, value=3_700.0, step=100.0)
    current_liabilities_t1 = st.number_input("Prior-year current liabilities", min_value=0.0, value=2_850.0, step=100.0)
with col6:
    revenue_t2 = st.number_input("Two-years-ago revenue / net sales", min_value=0.0, value=9_000.0, step=100.0)
    ebit_t2 = st.number_input("Two-years-ago operating income / EBIT", value=1_200.0, step=100.0)

# ------------------------------------------------------------
# Optional market inputs
# ------------------------------------------------------------
st.header("3) Optional Inputs from Market Data or Company Notes")
col7, col8, col9 = st.columns(3)
with col7:
    use_manual_beta = st.checkbox("Input company beta manually", value=False)
    manual_beta = st.number_input("Company beta", value=1.00, step=0.05, disabled=not use_manual_beta)
with col8:
    use_manual_tax = st.checkbox("Use actual effective tax rate from latest year", value=True)
with col9:
    market_cap = st.number_input(
        "Market capitalization, if known",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help="Optional. If left at zero, the app uses book debt and estimated equity value weights for WACC."
    )

# ------------------------------------------------------------
# Convert all statement values to actual dollars for calculations
# ------------------------------------------------------------
raw_inputs = {
    "Revenue": revenue_t,
    "EBIT": ebit_t,
    "Net Income": net_income_t,
    "Tax Expense": tax_expense_t,
    "D&A": depreciation_t,
    "CapEx": capex_t,
    "Interest Expense": interest_expense_t,
    "Cash": cash_t,
    "Total Debt": total_debt_t,
    "Shares": diluted_shares_t,
    "Current Assets": current_assets_t,
    "Current Liabilities": current_liabilities_t,
    "Prior Revenue": revenue_t1,
    "Prior EBIT": ebit_t1,
    "Prior Current Assets": current_assets_t1,
    "Prior Current Liabilities": current_liabilities_t1,
    "Two-Year Revenue": revenue_t2,
    "Two-Year EBIT": ebit_t2,
    "Market Cap": market_cap,
}

# Multiply dollar line items, but not shares, beta, percentages, or multiples.
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
shares = diluted_shares_t  # Leave shares in whatever convention the user entered. Value/share is in the same convention.

# ------------------------------------------------------------
# Historical calculations
# ------------------------------------------------------------
working_capital = current_assets - current_liabilities
working_capital_prior = current_assets_prior - current_liabilities_prior
change_nwc = working_capital - working_capital_prior

historical_growth_1yr = safe_div(revenue - revenue_prior, revenue_prior)
historical_growth_2yr_cagr = cagr(revenue_two_years_ago, revenue, 2)
base_revenue_growth = np.nanmean([historical_growth_1yr, historical_growth_2yr_cagr])
base_revenue_growth = max(min(base_revenue_growth, 0.20), -0.10)  # Cap extreme auto-forecasting

ebit_margin = safe_div(ebit, revenue)
da_percent_revenue = safe_div(depreciation, revenue)
capex_percent_revenue = safe_div(capex, revenue)
nwc_percent_revenue = safe_div(working_capital, revenue)
change_nwc_percent_revenue = safe_div(change_nwc, revenue)

pre_tax_income_estimate = ebit - interest_expense
actual_effective_tax_rate = safe_div(tax_expense, max(pre_tax_income_estimate, 0.0001), default=tax_rate_default)
actual_effective_tax_rate = max(0.0, min(actual_effective_tax_rate, 0.40))
tax_rate = actual_effective_tax_rate if use_manual_tax else tax_rate_default

beta = manual_beta if use_manual_beta else estimate_beta_from_industry(industry_risk)
cost_of_equity = risk_free_rate + beta * equity_risk_premium
after_tax_cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)

# Initial WACC weights; if market cap is not entered, use book debt and an implied equity proxy from net income/cost of equity.
if market_cap_actual > 0:
    equity_weight_base = safe_div(market_cap_actual, market_cap_actual + total_debt, default=0.80)
else:
    equity_proxy = max(net_income / max(cost_of_equity, 0.0001), revenue)  # rough fallback only
    equity_weight_base = safe_div(equity_proxy, equity_proxy + total_debt, default=0.80)

debt_weight_base = 1 - equity_weight_base
wacc = equity_weight_base * cost_of_equity + debt_weight_base * after_tax_cost_of_debt

# ------------------------------------------------------------
# User projection adjustments
# ------------------------------------------------------------
st.header("4) Forecast Assumptions Calculated from 10-K Inputs")
st.write("The app calculates these from the 10-K data above. You can override them for scenario analysis.")

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

# ------------------------------------------------------------
# Build DCF projection
# ------------------------------------------------------------
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

    projection.append({
        "Year": int(fiscal_year + year),
        "Revenue": projected_revenue,
        "EBIT": projected_ebit,
        "EBITDA": ebitda,
        "NOPAT": nopat,
        "D&A": projected_da,
        "CapEx": projected_capex,
        "Change in NWC": projected_change_nwc,
        "FCF": fcf,
        "Discount Factor": discount_factor,
        "PV of FCF": pv_fcf,
    })

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

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
st.header("5) Valuation Results")
res1, res2, res3, res4 = st.columns(4)
res1.metric("Enterprise Value", format_dollars(enterprise_value))
res2.metric("Equity Value", format_dollars(equity_value))
res3.metric("Implied Value per Share", format_dollars(value_per_share))
res4.metric("PV of Terminal Value", format_dollars(pv_terminal_value))

with st.expander("Key calculated historical ratios", expanded=True):
    ratios = pd.DataFrame({
        "Metric": [
            "1-year revenue growth",
            "2-year revenue CAGR",
            "EBIT margin",
            "D&A / Revenue",
            "CapEx / Revenue",
            "NWC / Revenue",
            "Change in NWC / Revenue",
            "Effective tax rate used",
            "Cost of equity",
            "After-tax cost of debt",
            "Equity weight",
            "Debt weight",
            "WACC",
        ],
        "Value": [
            format_pct(historical_growth_1yr),
            format_pct(historical_growth_2yr_cagr),
            format_pct(ebit_margin),
            format_pct(da_percent_revenue),
            format_pct(capex_percent_revenue),
            format_pct(nwc_percent_revenue),
            format_pct(change_nwc_percent_revenue),
            format_pct(forecast_tax_rate),
            format_pct(cost_of_equity),
            format_pct(after_tax_cost_of_debt),
            format_pct(equity_weight_base),
            format_pct(debt_weight_base),
            format_pct(wacc),
        ]
    })
    st.dataframe(ratios, use_container_width=True, hide_index=True)

st.subheader("DCF Projection Table")
display_df = projection_df.copy()
for col in ["Revenue", "EBIT", "EBITDA", "NOPAT", "D&A", "CapEx", "Change in NWC", "FCF", "PV of FCF"]:
    display_df[col] = display_df[col].map(format_dollars)
display_df["Discount Factor"] = display_df["Discount Factor"].map(lambda x: f"{x:.4f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.subheader("Free Cash Flow Projection")
chart_df = projection_df[["Year", "FCF"]].set_index("Year")
st.line_chart(chart_df, use_container_width=True)

# ------------------------------------------------------------
# Sensitivity analysis
# ------------------------------------------------------------
st.header("6) Sensitivity Analysis")
st.write("This table shows implied value per share under different WACC and terminal growth assumptions.")

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
            pv_fcf_alt = sum(
                projection_df.loc[i, "FCF"] / ((1 + w) ** (i + 1))
                for i in range(len(projection_df))
            )
            ev = pv_fcf_alt + pv_tv
            eq = ev - total_debt + cash
            values.append(safe_div(eq, shares))
    sensitivity[f"WACC {w * 100:.1f}%"] = values

st.dataframe(sensitivity.style.format("${:,.2f}"), use_container_width=True)

# ------------------------------------------------------------
# Methodology
# ------------------------------------------------------------
st.header("7) Methodology")
st.markdown(
    f"""
    **Free cash flow formula:**  
    FCF = EBIT × (1 − Tax Rate) + D&A − CapEx − Change in Net Working Capital

    **WACC formula:**  
    WACC = Equity Weight × Cost of Equity + Debt Weight × After-Tax Cost of Debt

    **Cost of equity:**  
    Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium

    **Terminal value:**  
    - Perpetuity Growth: Final-Year FCF × (1 + g) / (WACC − g)  
    - Exit Multiple: Final-Year EBITDA × Exit EBITDA Multiple

    **Important limitation:** This is a structured academic/analytical model, not investment advice. 
    For professional valuation, review management guidance, segment trends, industry-specific beta, lease debt treatment, unusual items, acquisitions, stock-based compensation, and off-balance-sheet obligations.
    """
)
