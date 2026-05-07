import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from openai import OpenAI

# ------------------ OPENAI SETUP ------------------
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------ PAGE ------------------
st.set_page_config(page_title="AI Sales Dashboard", layout="wide", page_icon="📊")

# ------------------ THEME / CSS ------------------
PRIMARY = "#00E5FF"      # electric teal
ACCENT  = "#7C5CFF"      # violet accent
BG      = "#0B1220"      # deep navy
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
section.main > div {{ padding-top: 1rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1326 0%, #0b1730 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
.sidebar-brand {{
    display:flex; align-items:center; gap:.6rem;
    padding: 14px 12px; margin-bottom: 8px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(0,229,255,.15), rgba(124,92,255,.15));
    border: 1px solid rgba(255,255,255,0.08);
}}
.sidebar-brand .logo {{
    width:36px;height:36px;border-radius:10px;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    display:flex;align-items:center;justify-content:center;
    font-size:20px;color:#0b1220;font-weight:800;
    box-shadow: 0 8px 24px rgba(0,229,255,.35);
}}
.sidebar-brand .title {{ font-weight:700;font-size:1rem;line-height:1; }}
.sidebar-brand .sub   {{ color:{MUTED};font-size:.75rem;margin-top:3px; }}

/* Inputs */
div[data-baseweb="select"] > div, .stTextInput input, .stFileUploader, .stSlider {{
    background: {PANEL} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {PRIMARY} !important; border-color:{PRIMARY} !important;
}}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    color: #0b1220; font-weight: 700; border: 0;
    border-radius: 12px; padding: .55rem 1.1rem;
    box-shadow: 0 8px 24px rgba(0,229,255,.25);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px); filter: brightness(1.05);
    box-shadow: 0 12px 30px rgba(124,92,255,.35);
}}

/* Section headers */
.section-header {{
    display:flex; align-items:center; gap:.7rem;
    margin: 1.4rem 0 .8rem 0;
}}
.section-header .bar {{
    width: 6px; height: 26px; border-radius: 4px;
    background: linear-gradient(180deg, {PRIMARY}, {ACCENT});
    box-shadow: 0 0 16px rgba(0,229,255,.5);
}}
.section-header .icon {{ font-size: 1.25rem; }}
.section-header h3 {{
    margin:0; color:{TEXT}; font-size:1.15rem; font-weight:700; letter-spacing:.2px;
}}

/* Metric cards */
.metric-card {{
    background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 12px 30px rgba(0,0,0,.35);
    position: relative; overflow: hidden;
}}
.metric-card::before {{
    content:""; position:absolute; inset:0;
    background: radial-gradient(400px 120px at 0% 0%, rgba(0,229,255,.18), transparent 60%);
    pointer-events:none;
}}
.metric-card .row   {{ display:flex; align-items:center; gap:12px; }}
.metric-card .icon  {{
    width:42px;height:42px;border-radius:12px;
    background: rgba(0,229,255,.12);
    display:flex;align-items:center;justify-content:center;
    font-size:20px; border:1px solid rgba(0,229,255,.25);
}}
.metric-card .label {{ color:{MUTED}; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }}
.metric-card .value {{ color:{TEXT}; font-size:1.7rem; font-weight:800; margin-top:4px; }}

/* AI callout */
.ai-card {{
    background: linear-gradient(160deg, rgba(124,92,255,.10), rgba(0,229,255,.06));
    border: 1px solid rgba(124,92,255,.35);
    border-left: 4px solid {ACCENT};
    border-radius: 14px; padding: 16px 18px;
    color: {TEXT}; line-height: 1.55;
    box-shadow: 0 10px 30px rgba(0,0,0,.3);
}}
.ai-card .head {{ display:flex; align-items:center; gap:.5rem; margin-bottom:.4rem; font-weight:700; }}

/* Dataframes */
.stDataFrame, [data-testid="stDataFrame"] {{
    border-radius: 12px; overflow:hidden;
    border:1px solid rgba(255,255,255,.06);
}}

/* Hero title */
.hero-title {{
    font-size: 2.1rem; font-weight: 800; letter-spacing:.2px;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin: 6px 0 0 0;
}}
.hero-sub {{ color:{MUTED}; margin-bottom: 8px; }}
hr.fancy {{ border:0; height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.15), transparent); margin: 6px 0 18px 0; }}
</style>
""", unsafe_allow_html=True)

# ------------------ HELPERS ------------------
def section(title, icon="✨"):
    st.markdown(
        f'<div class="section-header"><div class="bar"></div>'
        f'<div class="icon">{icon}</div><h3>{title}</h3></div>',
        unsafe_allow_html=True
    )

def metric_card(col, icon, label, value):
    col.markdown(f"""
    <div class="metric-card">
      <div class="row">
        <div class="icon">{icon}</div>
        <div>
          <div class="label">{label}</div>
          <div class="value">{value}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# ------------------ SIDEBAR BRAND ------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="logo">📊</div>
  <div>
    <div class="title">Sales IQ</div>
    <div class="sub">AI Forecast Dashboard</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="hero-title">📊 AI Sales Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload sales data, explore trends, forecast with ARIMA, and chat with AI.</div>', unsafe_allow_html=True)
st.markdown('<hr class="fancy"/>', unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded_file:
    st.info("⬆️ Upload a CSV file to begin")
    st.stop()

with st.spinner("📥 Reading your file..."):
    df = pd.read_csv(uploaded_file)

section("Data Preview", "🗂️")
st.dataframe(df.head(), use_container_width=True)

# ------------------ SETTINGS ------------------
st.sidebar.markdown("### ⚙️ Settings")
date_col    = st.sidebar.selectbox("📅 Date Column", df.columns)
sales_col   = st.sidebar.selectbox("💰 Sales Column", df.columns)
product_col = st.sidebar.selectbox("📦 Product Column", df.columns)

# ------------------ CLEANING ------------------
df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df = df.dropna(subset=[date_col, sales_col])

if df.empty:
    st.error("No valid data after cleaning")
    st.stop()

# ------------------ TIME SERIES ------------------
sales = df.groupby(date_col)[sales_col].sum().sort_index()

section("Sales Trend", "📈")
fig_trend = px.line(x=sales.index, y=sales.values, labels={"x":"Date","y":"Sales"})
fig_trend.update_traces(line=dict(color=PRIMARY, width=2.5))
st.plotly_chart(style_fig(fig_trend), use_container_width=True)

# Monthly aggregation
sales_monthly = sales.groupby(pd.Grouper(freq="ME")).sum()

section("Monthly Trend", "🗓️")
fig_month = px.area(x=sales_monthly.index, y=sales_monthly.values, labels={"x":"Month","y":"Sales"})
fig_month.update_traces(line=dict(color=ACCENT, width=2.5),
                        fillcolor="rgba(124,92,255,0.18)")
st.plotly_chart(style_fig(fig_month), use_container_width=True)

# ------------------ METRICS ------------------
section("Key Metrics", "🎯")
col1, col2, col3 = st.columns(3)
metric_card(col1, "💵", "Total Sales", f"{sales_monthly.sum():,.0f}")
metric_card(col2, "📊", "Avg Monthly", f"{sales_monthly.mean():,.0f}")
metric_card(col3, "🚀", "Max Monthly", f"{sales_monthly.max():,.0f}")

# ------------------ TOP PRODUCTS ------------------
section("Top Products", "🏆")
top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(5)
fig_top = px.bar(
    x=top_products.values, y=top_products.index.astype(str),
    orientation="h", labels={"x":"Sales","y":"Product"},
    color=top_products.values, color_continuous_scale=[[0, ACCENT], [1, PRIMARY]],
)
fig_top.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
st.plotly_chart(style_fig(fig_top), use_container_width=True)

# ------------------ FORECAST ------------------
section("Forecast", "🔮")
steps = st.slider("Months to Forecast", 1, 12, 6)

with st.spinner("🧠 Fitting ARIMA model..."):
    model = ARIMA(sales_monthly, order=(1,1,1))
    model_fit = model.fit()
    forecast_res = model_fit.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf = forecast_res.conf_int(alpha=0.05)

forecast_index = pd.date_range(
    start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
    periods=steps,
    freq=pd.offsets.MonthEnd()
)
forecast_df = pd.DataFrame({
    "Date":     forecast_index,
    "Forecast": forecast.values,
    "Lower":    conf.iloc[:, 0].values,
    "Upper":    conf.iloc[:, 1].values,
})

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown("**Forecast Table**")
    st.dataframe(
        forecast_df.style.format({"Forecast":"{:,.0f}","Lower":"{:,.0f}","Upper":"{:,.0f}"}),
        use_container_width=True, hide_index=True
    )
with c2:
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=sales_monthly.index, y=sales_monthly.values,
        name="Actual", line=dict(color=PRIMARY, width=2.5)
    ))
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_df["Date"]) + list(forecast_df["Date"][::-1]),
        y=list(forecast_df["Upper"]) + list(forecast_df["Lower"][::-1]),
        fill="toself", fillcolor="rgba(124,92,255,0.20)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="95% Confidence"
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecast"],
        name="Forecast", line=dict(color=ACCENT, width=2.5, dash="dash")
    ))
    st.plotly_chart(style_fig(fig_fc), use_container_width=True)

# ------------------ AI INSIGHTS ------------------
section("AI Insights", "🤖")
if client:
    if st.button("✨ Generate AI Analysis"):
        with st.spinner("🤖 AI is analyzing your data..."):
            prompt = f"""
You are a business analyst.
Analyze this sales data summary:
Total Sales: {sales_monthly.sum()}
Average Monthly Sales: {sales_monthly.mean()}
Max Monthly Sales: {sales_monthly.max()}
Top Products:
{top_products.to_string()}
Forecast (next months):
{forecast.values[:5].tolist()}

Give:
1. Business insights
2. Risks
3. Recommendations
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.replace("\n", "<br/>")
                st.markdown(
                    f'<div class="ai-card"><div class="head">🤖 AI Analyst</div>{content}</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"AI error: {e}")
else:
    st.warning("Add OPENAI_API_KEY in secrets to enable AI features")

# ------------------ AI CHAT ------------------
section("Ask AI About Your Data", "💬")
query = st.text_input("Ask something (e.g. 'What are top products?')")
if query and client:
    with st.spinner("💭 Thinking..."):
        try:
            context = f"""
Dataset Summary:
- Total sales: {sales_monthly.sum()}
- Products: {df[product_col].nunique()}
- Top products: {top_products.to_string()}
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": context + "\n\nQuestion: " + query}
                ]
            )
            content = response.choices[0].message.content.replace("\n", "<br/>")
            st.markdown(
                f'<div class="ai-card"><div class="head">💬 AI Response</div>{content}</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error: {e}")
