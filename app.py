import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import google.generativeai as genai

# ------------------ GEMINI SETUP ------------------
import google.generativeai as genai

model_ai = None

if "GEMINI_API_KEY" in st.secrets:

    genai.configure(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    model_ai = genai.GenerativeModel(
    "gemini-1.5-flash"
    )
# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------ COLORS ------------------
PRIMARY = "#00E5FF"
ACCENT = "#7C5CFF"
BG = "#0B1220"
TEXT = "#E6EEF8"
MUTED = "#8AA0BF"

# ------------------ CUSTOM CSS ------------------
st.markdown(f"""
<style>

.stApp {{
    background: {BG};
    color: {TEXT};
}}

.hero-title {{
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-sub {{
    color: {MUTED};
    font-size: 1rem;
    margin-bottom: 20px;
}}

.metric-card {{
    background: #111A2E;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}}

.metric-title {{
    color: {MUTED};
    font-size: 0.9rem;
}}

.metric-value {{
    color: {TEXT};
    font-size: 1.8rem;
    font-weight: 700;
}}

.ai-card {{
    background: rgba(124,92,255,0.08);
    padding: 20px;
    border-radius: 15px;
    border-left: 4px solid {ACCENT};
    margin-top: 10px;
}}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown(
    '<div class="hero-title">📊 AI Sales Forecast Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-sub">Upload CSV data, forecast trends, and analyze sales using Gemini AI.</div>',
    unsafe_allow_html=True
)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "📁 Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin")
    st.stop()

# ------------------ READ DATA ------------------
df = pd.read_csv(uploaded_file)

# ------------------ DATA PREVIEW ------------------
st.subheader("🗂️ Data Preview")
st.dataframe(df.head(), use_container_width=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Configuration")

date_col = st.sidebar.selectbox(
    "📅 Select Date Column",
    df.columns
)

sales_col = st.sidebar.selectbox(
    "💰 Select Sales Column",
    df.columns
)

product_col = st.sidebar.selectbox(
    "📦 Select Product Column",
    df.columns
)

# ------------------ CLEAN DATA ------------------
df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df[sales_col] = pd.to_numeric(
    df[sales_col],
    errors="coerce"
)

df = df.dropna(subset=[date_col, sales_col])

if df.empty:
    st.error("No valid data available after cleaning")
    st.stop()

# ------------------ TIME SERIES ------------------
sales = (
    df.groupby(date_col)[sales_col]
    .sum()
    .sort_index()
)

# ------------------ SALES TREND ------------------
st.subheader("📈 Sales Trend")

fig_sales = px.line(
    x=sales.index,
    y=sales.values,
    labels={
        "x": "Date",
        "y": "Sales"
    }
)

fig_sales.update_traces(
    line=dict(
        color=PRIMARY,
        width=3
    )
)

fig_sales.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)"
)

st.plotly_chart(
    fig_sales,
    use_container_width=True
)

# ------------------ MONTHLY SALES ------------------
sales_monthly = sales.groupby(
    pd.Grouper(freq="ME")
).sum()

st.subheader("🗓️ Monthly Sales Trend")

fig_month = px.area(
    x=sales_monthly.index,
    y=sales_monthly.values,
    labels={
        "x": "Month",
        "y": "Sales"
    }
)

fig_month.update_traces(
    line=dict(
        color=ACCENT,
        width=3
    ),
    fillcolor="rgba(124,92,255,0.2)"
)

fig_month.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)"
)

st.plotly_chart(
    fig_month,
    use_container_width=True
)

# ------------------ METRICS ------------------
st.subheader("🎯 Key Metrics")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Sales</div>
        <div class="metric-value">{sales_monthly.sum():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Monthly Sales</div>
        <div class="metric-value">{sales_monthly.mean():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Maximum Monthly Sales</div>
        <div class="metric-value">{sales_monthly.max():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------ TOP PRODUCTS ------------------
st.subheader("🏆 Top Products")

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
    labels={
        "x": "Sales",
        "y": "Product"
    },
    color=top_products.values,
    color_continuous_scale="Blues"
)

fig_top.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False
)

st.plotly_chart(
    fig_top,
    use_container_width=True
)

# ------------------ FORECAST ------------------
st.subheader("🔮 Sales Forecast")

if len(sales_monthly) < 3:
    st.error("Not enough data for forecasting")
    st.stop()

steps = st.slider(
    "Months to Forecast",
    1,
    12,
    6
)

with st.spinner("Training forecasting model..."):

    model = ARIMA(
        sales_monthly,
        order=(1,1,1)
    )

    model_fit = model.fit()

    forecast_res = model_fit.get_forecast(
        steps=steps
    )

    forecast = forecast_res.predicted_mean

forecast_index = pd.date_range(
    start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
    periods=steps,
    freq=pd.offsets.MonthEnd()
)

forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast": forecast.values
})

# ------------------ FORECAST TABLE ------------------
st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

# ------------------ FORECAST CHART ------------------
fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=sales_monthly.index,
        y=sales_monthly.values,
        mode="lines",
        name="Actual Sales",
        line=dict(color=PRIMARY, width=3)
    )
)

fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode="lines",
        name="Forecast",
        line=dict(
            color=ACCENT,
            width=3,
            dash="dash"
        )
    )
)

fig_forecast.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

# ------------------ GEMINI AI INSIGHTS ------------------
st.subheader("🤖 Gemini AI Insights")

if model_ai:

    if st.button("✨ Generate AI Analysis"):

        with st.spinner("Gemini AI analyzing data..."):

            prompt = f"""
You are an expert business analyst.

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

                response = model_ai.generate_content(
                    prompt
                )

                content = response.text.replace(
                    "\\n",
                    "<br>"
                )

                st.markdown(
                    f'''
                    <div class="ai-card">
                        <h3>🤖 Gemini AI Analyst</h3>
                        {content}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Gemini Error: {e}")

else:
    st.warning(
        "Add GEMINI_API_KEY in Streamlit secrets"
    )

# ------------------ GEMINI AI CHAT ------------------
st.subheader("💬 Ask Gemini About Your Data")

query = st.text_input(
    "Ask a question about your sales data"
)

if query and model_ai:

    with st.spinner("Gemini AI thinking..."):

        try:

            context = f"""
Dataset Summary:

Total Sales:
{sales_monthly.sum()}

Average Monthly Sales:
{sales_monthly.mean()}

Top Products:
{top_products.to_string()}

Forecast:
{forecast.values[:5].tolist()}
"""

            final_prompt = (
                context +
                "\\n\\nUser Question:\\n" +
                query
            )

            response = model_ai.generate_content(
                final_prompt
            )

            content = response.text.replace(
                "\\n",
                "<br>"
            )

            st.markdown(
                f'''
                <div class="ai-card">
                    <h3>💬 Gemini Response</h3>
                    {content}
                </div>
                ''',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Gemini Error: {e}")
