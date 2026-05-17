import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-Powered Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Sales Forecast Dashboard")
st.caption("Upload CSV • Analyze Sales • Forecast Trends • AI Insights")

# ---------------- GROQ SETUP ----------------
client = None

if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
h1, h2, h3 { color: white; }

[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    padding: 15px;
    border-radius: 12px;
}

.stButton>button {
    background: linear-gradient(90deg,#00C6FF,#0072FF);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.ai-box {
    background: #161B22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363D;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuration")

date_col = st.sidebar.selectbox("📅 Select Date Column", df.columns)
sales_col = st.sidebar.selectbox("💰 Select Sales Column", df.columns)
product_col = st.sidebar.selectbox("📦 Select Product Column", df.columns)

# ---------------- DATA CLEANING ----------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

df = df.dropna(subset=[date_col, sales_col])
df = df.sort_values(date_col)

if df.empty:
    st.error("No valid data after cleaning.")
    st.stop()

# ---------------- SALES SERIES ----------------
sales = df.groupby(date_col)[sales_col].sum()
sales.index = pd.to_datetime(sales.index)

sales_monthly = sales.resample("ME").sum()

# ---------------- SALES TREND ----------------
st.subheader("📈 Sales Trend")

sales_df = pd.DataFrame({
    "Date": sales.index,
    "Sales": sales.values
})

fig_sales = px.line(sales_df, x="Date", y="Sales", title="Sales Over Time")
fig_sales.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")

st.plotly_chart(fig_sales, use_container_width=True)

# ---------------- MONTHLY SALES ----------------
st.subheader("📅 Monthly Sales")

sales_monthly_df = pd.DataFrame({
    "Month": sales_monthly.index,
    "Sales": sales_monthly.values
})

fig_monthly = px.area(sales_monthly_df, x="Month", y="Sales", title="Monthly Sales Trend")
fig_monthly.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")

st.plotly_chart(fig_monthly, use_container_width=True)

# ---------------- METRICS ----------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Total Sales", f"{sales_monthly.sum():,.0f}")

with col2:
    st.metric("📈 Avg Monthly Sales", f"{sales_monthly.mean():,.0f}")

with col3:
    st.metric("🚀 Max Monthly Sales", f"{sales_monthly.max():,.0f}")

# ---------------- TOP PRODUCTS ----------------
st.subheader("🏆 Top Products")

top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)

top_products_df = pd.DataFrame({
    "Product": top_products.index.astype(str),
    "Sales": top_products.values
})

fig_products = px.bar(top_products_df, x="Product", y="Sales", color="Sales", title="Top Selling Products")
fig_products.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")

st.plotly_chart(fig_products, use_container_width=True)

# ---------------- FORECAST ----------------
st.subheader("🔮 Sales Forecast")

if len(sales_monthly) >= 3:

    steps = st.slider("Months to Forecast", 1, 12, 6)

    model = ARIMA(sales_monthly, order=(1, 1, 1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    forecast_index = pd.date_range(
        start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
        periods=steps,
        freq="ME"
    )

    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecast": forecast.values
    })

    st.dataframe(forecast_df, use_container_width=True)

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=sales_monthly.index,
        y=sales_monthly.values,
        mode="lines",
        name="Actual Sales"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode="lines",
        name="Forecast",
        line=dict(dash="dash")
    ))

    fig_forecast.update_layout(
        template="plotly_dark",
        title="Sales Forecast",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.warning("Need at least 3 months of data for forecasting.")

# ---------------- SQL SECTION ----------------
st.subheader("🧠 SQL Equivalent Queries")

st.code("""
SELECT SUM(sales) FROM sales_data;

SELECT product, SUM(sales)
FROM sales_data
GROUP BY product
ORDER BY SUM(sales) DESC
LIMIT 10;

SELECT DATE_TRUNC('month', order_date), SUM(sales)
FROM sales_data
GROUP BY 1;
""", language="sql")

# ---------------- AI INSIGHTS ENGINE ----------------
def generate_ai_insights(df, sales_monthly, top_products, product_col, sales_col):
    insights = []

    # Trend
    if len(sales_monthly) >= 2:
        growth = ((sales_monthly.iloc[-1] - sales_monthly.iloc[-2]) / sales_monthly.iloc[-2]) * 100

        if growth > 15:
            insights.append(f"📈 Strong upward sales trend (+{growth:.2f}%)")
        elif growth < -10:
            insights.append(f"📉 Significant drop in sales ({growth:.2f}%)")
        else:
            insights.append(f"📊 Stable sales trend ({growth:.2f}%)")

    # Dependency
    product_sales = df.groupby(product_col)[sales_col].sum()
    total_sales = product_sales.sum()
    top_share = (product_sales.max() / total_sales) * 100

    if top_share > 40:
        insights.append(f"⚠️ High dependency on top product ({top_share:.2f}%)")
    else:
        insights.append(f"✅ Balanced product distribution ({top_share:.2f}%)")

    # Volatility
    cv = (df[sales_col].std() / df[sales_col].mean()) * 100

    if cv > 30:
        insights.append(f"⚠️ High revenue volatility (CV: {cv:.2f}%)")
    else:
        insights.append(f"📊 Stable revenue pattern (CV: {cv:.2f}%)")

    # Top product
    insights.append(f"🏆 Top product: {top_products.index[0]}")

    return insights

# ---------------- AI INSIGHTS UI ----------------
st.subheader("🤖 AI Insights Engine")

if st.button("✨ Generate Insights"):
    insights = generate_ai_insights(df, sales_monthly, top_products, product_col, sales_col)

    for i in insights:
        st.markdown(f"- {i}")

# ---------------- GROQ AI CHAT ----------------
st.subheader("💬 Ask AI About Your Data")

query = st.text_input("Ask a question about your sales data")

if query and client:

    context = f"""
    Total Sales: {sales_monthly.sum()}
    Avg Sales: {sales_monthly.mean()}
    Top Product: {top_products.index[0]}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": context + "\n\nQuestion: " + query}
        ]
    )

    st.markdown(
        f"<div class='ai-box'>{response.choices[0].message.content}</div>",
        unsafe_allow_html=True
    )

elif query:
    st.warning("Add GROQ_API_KEY in Streamlit secrets.")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Streamlit • Plotly • ARIMA • Rule-Based AI Engine • Groq AI")
