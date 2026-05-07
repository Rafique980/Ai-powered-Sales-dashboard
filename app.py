import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import google.generativeai as genai

# ------------------ GEMINI SETUP ------------------
model_ai = None

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_ai = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ PAGE ------------------
st.set_page_config(
    page_title="AI Sales Dashboard",
    layout="wide",
    page_icon="📊"
)

# ------------------ THEME / CSS ------------------
PRIMARY = "#00E5FF"
ACCENT  = "#7C5CFF"
BG      = "#0B1220"
PANEL   = "#111A2E"
PANEL_2 = "#162038"
TEXT    = "#E6EEF8"
MUTED   = "#8AA0BF"

st.markdown(f"""
<style>
.stApp {{
    background: radial-gradient(1200px 600px at 10% -10%, #16224a 0%, transparent 60%),
                radial-gradient(900px 500px at 100% 0%, #0e2a3a 0%, transparent 55%),
                {BG};
    color: {TEXT};
}}

section.main > div {{
    padding-top: 1rem;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1326 0%, #0b1730 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}

.sidebar-brand {{
    display:flex;
    align-items:center;
    gap:.6rem;
    padding:14px 12px;
    margin-bottom:8px;
    border-radius:14px;
    background: linear-gradient(135deg, rgba(0,229,255,.15), rgba(124,92,255,.15));
    border: 1px solid rgba(255,255,255,0.08);
}}

.sidebar-brand .logo {{
    width:36px;
    height:36px;
    border-radius:10px;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
    color:#0b1220;
    font-weight:800;
}}

.sidebar-brand .title {{
    font-weight:700;
    font-size:1rem;
}}

.sidebar-brand .sub {{
    color:{MUTED};
    font-size:.75rem;
}}

.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    color: #0b1220;
    font-weight: 700;
    border: 0;
    border-radius: 12px;
    padding: .55rem 1.1rem;
}}

.hero-title {{
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}}

.hero-sub {{
    color:{MUTED};
}}

.metric-card {{
    background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px 20px;
}}

.metric-card .label {{
    color:{MUTED};
    font-size:.8rem;
}}

.metric-card .value {{
    color:{TEXT};
    font-size:1.7rem;
    font-weight:800;
}}

.ai-card {{
    background: linear-gradient(160deg, rgba(124,92,255,.10), rgba(0,229,255,.06));
    border: 1px solid rgba(124,92,255,.35);
    border-left: 4px solid {ACCENT};
    border-radius: 14px;
    padding: 16px 18px;
    color: {TEXT};
    line-height: 1.55;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ HELPERS ------------------
def section(title, icon="✨"):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.7rem;margin:1.4rem 0 .8rem 0;">
            <div style="width:6px;height:26px;border-radius:4px;background:linear-gradient(180deg,{PRIMARY},{ACCENT});"></div>
            <div style="font-size:1.25rem;">{icon}</div>
            <h3 style="margin:0;color:{TEXT};">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def metric_card(col, label, value):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="logo">📊</div>
    <div>
        <div class="title">Sales IQ</div>
        <div class="sub">Gemini AI Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown(
    '<div class="hero-title">📊 AI Sales Forecast Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-sub">Forecast sales trends and analyze business insights using Gemini AI.</div>',
    unsafe_allow_html=True
)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------ DATA PREVIEW ------------------
section("Data Preview", "🗂️")
st.dataframe(df.head(), use_container_width=True)

# ------------------ SETTINGS ------------------
st.sidebar.markdown("### ⚙️ Settings")

date_col = st.sidebar.selectbox("📅 Date Column", df.columns)
sales_col = st.sidebar.selectbox("💰 Sales Column", df.columns)
product_col = st.sidebar.selectbox("📦 Product Column", df.columns)

# ------------------ CLEAN DATA ------------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

df = df.dropna(subset=[date_col, sales_col])

if df.empty:
    st.error("No valid data after cleaning")
    st.stop()

# ------------------ SALES TREND ------------------
sales = df.groupby(date_col)[sales_col].sum().sort_index()

section("Sales Trend", "📈")

fig_sales = px.line(
    x=sales.index,
    y=sales.values,
    labels={"x": "Date", "y": "Sales"}
)

fig_sales.update_traces(line=dict(color=PRIMARY, width=3))

st.plotly_chart(style_fig(fig_sales), use_container_width=True)

# ------------------ MONTHLY SALES ------------------
sales_monthly = sales.groupby(pd.Grouper(freq="ME")).sum()

section("Monthly Sales Trend", "🗓️")

fig_month = px.area(
    x=sales_monthly.index,
    y=sales_monthly.values,
    labels={"x": "Month", "y": "Sales"}
)

fig_month.update_traces(
    line=dict(color=ACCENT, width=3),
    fillcolor="rgba(124,92,255,0.2)"
)

st.plotly_chart(style_fig(fig_month), use_container_width=True)

# ------------------ METRICS ------------------
section("Key Metrics", "🎯")

c1, c2, c3 = st.columns(3)

metric_card(c1, "Total Sales", f"{sales_monthly.sum():,.0f}")
metric_card(c2, "Average Monthly", f"{sales_monthly.mean():,.0f}")
metric_card(c3, "Maximum Monthly", f"{sales_monthly.max():,.0f}")

# ------------------ TOP PRODUCTS ------------------
section("Top Products", "🏆")

top_products = (
    df.groupby(product_col)[sales_col]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

fig_top = px.bar(
    x=top_products.values,
    y=top_products.index.astype(str),
    orientation="h",
    labels={"x": "Sales", "y": "Product"},
    color=top_products.values,
    color_continuous_scale=[[0, ACCENT], [1, PRIMARY]]
)

fig_top.update_layout(
    coloraxis_showscale=False,
    yaxis=dict(autorange="reversed")
)

st.plotly_chart(style_fig(fig_top), use_container_width=True)

# ------------------ FORECAST ------------------
section("Forecast", "🔮")

steps = st.slider("Months to Forecast", 1, 12, 6)

with st.spinner("Training ARIMA model..."):

    model = ARIMA(sales_monthly, order=(1,1,1))
    model_fit = model.fit()

    forecast_res = model_fit.get_forecast(steps=steps)

    forecast = forecast_res.predicted_mean
    conf = forecast_res.conf_int()

forecast_index = pd.date_range(
    start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
    periods=steps,
    freq=pd.offsets.MonthEnd()
)

forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast": forecast.values,
    "Lower": conf.iloc[:, 0].values,
    "Upper": conf.iloc[:, 1].values
})

c1, c2 = st.columns([1, 2])

with c1:
    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

with c2:

    fig_fc = go.Figure()

    fig_fc.add_trace(go.Scatter(
        x=sales_monthly.index,
        y=sales_monthly.values,
        name="Actual",
        line=dict(color=PRIMARY, width=3)
    ))

    fig_fc.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        name="Forecast",
        line=dict(color=ACCENT, width=3, dash="dash")
    ))

    st.plotly_chart(style_fig(fig_fc), use_container_width=True)

# ------------------ GEMINI AI INSIGHTS ------------------
section("Gemini AI Insights", "🤖")

if model_ai:

    if st.button("✨ Generate AI Analysis"):

        with st.spinner("Gemini AI analyzing data..."):

            prompt = f"""
You are a professional business analyst.

Analyze this sales dataset.

Total Sales:
{sales_monthly.sum()}

Average Monthly Sales:
{sales_monthly.mean()}

Maximum Monthly Sales:
{sales_monthly.max()}

Top Products:
{top_products.to_string()}

Forecast:
{forecast.values[:5].tolist()}

Provide:
1. Business insights
2. Risks
3. Recommendations
4. Growth opportunities
"""

            try:

                response = model_ai.generate_content(prompt)

                content = response.text.replace("\n", "<br>")

                st.markdown(
                    f"""
                    <div class="ai-card">
                        <h4>🤖 Gemini AI Analyst</h4>
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Gemini Error: {e}")

else:
    st.warning("Add GEMINI_API_KEY in Streamlit secrets")

# ------------------ GEMINI CHAT ------------------
section("Ask Gemini About Your Data", "💬")

query = st.text_input(
    "Ask a question about your data"
)

if query and model_ai:

    with st.spinner("Gemini is thinking..."):

        try:

            context = f"""
Dataset Summary:

Total Sales:
{sales_monthly.sum()}

Top Products:
{top_products.to_string()}

Forecast:
{forecast.values[:5].tolist()}
"""

            final_prompt = context + "\n\nUser Question:\n" + query

            response = model_ai.generate_content(final_prompt)

            content = response.text.replace("\n", "<br>")

            st.markdown(
                f"""
                <div class="ai-card">
                    <h4>💬 Gemini Response</h4>
                    {content}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Gemini Error: {e}")
