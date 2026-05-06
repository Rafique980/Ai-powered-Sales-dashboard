import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

st.title("📊 Production Sales Forecasting Dashboard")
st.caption("Stable version (no PandasAI, no deprecated pandas APIs)")

# ------------------ UPLOAD DATA ------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Raw Data")
st.dataframe(df.head())

# ------------------ COLUMN SELECTION ------------------
st.sidebar.header("⚙️ Settings")

date_col = st.sidebar.selectbox("Date Column", df.columns)
sales_col = st.sidebar.selectbox("Sales Column", df.columns)
product_col = st.sidebar.selectbox("Product Column", df.columns)

# ------------------ CLEANING ------------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

df = df.dropna(subset=[date_col, sales_col])

if df.empty:
    st.error("No valid data after cleaning")
    st.stop()

# ------------------ TIME SERIES ------------------
sales = df.groupby(date_col)[sales_col].sum().sort_index()

st.subheader("📈 Sales Trend")
st.line_chart(sales)

# ------------------ MONTHLY AGGREGATION (FIXED) ------------------
sales_monthly = sales.groupby(pd.Grouper(freq="ME")).sum()

st.subheader("📉 Monthly Sales Trend")
st.line_chart(sales_monthly)

# ------------------ METRICS ------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{sales_monthly.sum():,.0f}")
col2.metric("Avg Monthly Sales", f"{sales_monthly.mean():,.0f}")
col3.metric("Max Monthly Sales", f"{sales_monthly.max():,.0f}")

# ------------------ TOP PRODUCTS ------------------
st.subheader("🏆 Top 5 Products")

top_products = (
    df.groupby(product_col)[sales_col]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

fig1, ax1 = plt.subplots()
top_products.plot(kind="bar", ax=ax1)
ax1.set_title("Top 5 Products")
st.pyplot(fig1)

# ------------------ FORECASTING ------------------
st.subheader("🔮 Forecast")

if len(sales_monthly) < 3:
    st.warning("Not enough data for forecasting")
    st.stop()

steps = st.slider("Months to Forecast", 1, 12, 6)

# ARIMA MODEL
model = ARIMA(sales_monthly, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=steps)

# FIXED DATE INDEX (NO 'M' BUG)
forecast_index = pd.date_range(
    start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
    periods=steps,
    freq=pd.offsets.MonthEnd()
)

forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast": forecast.values
})

st.subheader("📅 Forecast Table")
st.dataframe(forecast_df)

# ------------------ VISUALIZATION ------------------
fig2, ax2 = plt.subplots()

ax2.plot(sales_monthly.index, sales_monthly.values, label="Actual")
ax2.plot(forecast_df["Date"], forecast_df["Forecast"], linestyle="--", label="Forecast")

ax2.legend()
ax2.set_title("Sales Forecast")

st.pyplot(fig2)

# ------------------ INSIGHTS ------------------
st.subheader("📌 Insights")

trend = (
    "increasing 📈"
    if sales_monthly.iloc[-1] > sales_monthly.iloc[0]
    else "stable ➖"
)

st.write(f"""
- Sales trend is **{trend}**
- Forecast helps in planning inventory and demand
- Top products drive majority of revenue
- Model used: ARIMA (1,1,1)
""")

# ------------------ SQL SNIPPETS ------------------
st.subheader("🧠 SQL Reference Queries")

st.code("""
-- Top products
SELECT product, SUM(sales)
FROM sales
GROUP BY product
ORDER BY SUM(sales) DESC
LIMIT 5;

-- Monthly trend
SELECT DATE_TRUNC('month', date), SUM(sales)
FROM sales
GROUP BY 1
ORDER BY 1;

-- Total sales
SELECT SUM(sales) FROM sales;
""", language="sql")
