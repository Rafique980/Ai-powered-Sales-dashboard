import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Sales Forecast Dashboard")
st.caption("Upload CSV • Analytics • Forecasting • AI Insights")

# ---------------- GROQ SETUP ----------------
client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuration")

date_col = st.sidebar.selectbox("Date Column", df.columns)
sales_col = st.sidebar.selectbox("Sales Column", df.columns)
product_col = st.sidebar.selectbox("Product Column", df.columns)

# ---------------- CLEANING ----------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df = df.dropna(subset=[date_col, sales_col])
df = df.sort_values(date_col)

# ---------------- SALES ----------------
sales = df.groupby(date_col)[sales_col].sum()
sales.index = pd.to_datetime(sales.index)
sales_monthly = sales.resample("ME").sum()

# ---------------- CHARTS ----------------
st.subheader("📈 Sales Trend")

trend_df = pd.DataFrame({"Date": sales.index, "Sales": sales.values})

fig = px.line(trend_df, x="Date", y="Sales", title="Sales Over Time")
fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# ---------------- METRICS ----------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{sales_monthly.sum():,.0f}")
col2.metric("Avg Monthly Sales", f"{sales_monthly.mean():,.0f}")
col3.metric("Max Monthly Sales", f"{sales_monthly.max():,.0f}")

# ---------------- TOP PRODUCTS ----------------
st.subheader("🏆 Top Products")

top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)

st.bar_chart(top_products)

# ---------------- FORECAST ----------------
st.subheader("🔮 Forecast")

if len(sales_monthly) >= 3:

    steps = st.slider("Forecast Months", 1, 12, 6)

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

    st.dataframe(forecast_df)

# ---------------- AI INSIGHTS ENGINE ----------------
def generate_ai_insights(df, sales_monthly, top_products, product_col, sales_col):

    insights = {"business": [], "risks": [], "recommendations": []}

    # ---- TREND ----
    if len(sales_monthly) >= 2:
        growth = ((sales_monthly.iloc[-1] - sales_monthly.iloc[-2]) / sales_monthly.iloc[-2]) * 100

        if growth > 15:
            insights["business"].append(f"Strong upward sales trend (+{growth:.2f}%)")
        elif growth < -10:
            insights["risks"].append(f"Significant sales drop detected ({growth:.2f}%)")
        else:
            insights["business"].append(f"Stable sales performance ({growth:.2f}%)")

    # ---- DEPENDENCY ----
    product_sales = df.groupby(product_col)[sales_col].sum()
    top_share = (product_sales.max() / product_sales.sum()) * 100

    if top_share > 40:
        insights["risks"].append(f"High dependency on single product ({top_share:.2f}%)")
        insights["recommendations"].append("Diversify product portfolio to reduce risk")

    else:
        insights["business"].append(f"Balanced product distribution ({top_share:.2f}%)")

    # ---- VOLATILITY ----
    cv = (df[sales_col].std() / df[sales_col].mean()) * 100

    if cv > 30:
        insights["risks"].append(f"High revenue volatility detected (CV {cv:.2f}%)")
        insights["recommendations"].append("Stabilize demand using promotions or pricing strategy")

    else:
        insights["business"].append("Stable revenue pattern observed")

    # ---- TOP PRODUCT ----
    insights["business"].append(f"Top product: {top_products.index[0]}")

    return insights

# ---------------- AI UI ----------------
st.subheader("🤖 AI Insights Engine")

if st.button("Generate AI Insights"):

    insights = generate_ai_insights(df, sales_monthly, top_products, product_col, sales_col)

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("📊 Business Insights"):
            for i in insights["business"]:
                st.write("•", i)

    with col2:
        with st.expander("⚠️ Risks"):
            for i in insights["risks"]:
                st.write("•", i)

    with col3:
        with st.expander("💡 Recommendations"):
            for i in insights["recommendations"]:
                st.write("•", i)

# ---------------- GROQ AI CHAT ----------------
st.subheader("💬 Ask AI About Data")

query = st.text_input("Ask a question")

if query and client:

    context = f"""
    Total Sales: {sales_monthly.sum()}
    Avg Sales: {sales_monthly.mean()}
    Top Product: {top_products.index[0]}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a business analyst."},
            {"role": "user", "content": context + "\n\nQuestion: " + query}
        ]
    )

    st.success(response.choices[0].message.content)

# ---------------- FOOTER ----------------
st.caption("Built with Streamlit • Plotly • ARIMA • AI Insights Engine • Groq")
