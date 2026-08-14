import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Sales Forecasting Dashboard")
st.caption("Upload your data and analyze + forecast sales")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file")
    st.stop()

# ------------------ LOAD DATA ------------------
df = pd.read_csv(uploaded_file)

st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ------------------ COLUMN SELECTION ------------------
st.sidebar.header("⚙️ Configuration")

date_col = st.sidebar.selectbox("Select Date Column", df.columns)
sales_col = st.sidebar.selectbox("Select Sales Column", df.columns)
product_col = st.sidebar.selectbox("Select Product Column", df.columns)

# ------------------ DATA CLEANING ------------------
# FIXED DATE FORMAT
df[date_col] = pd.to_datetime(
    df[date_col],
    format='%d/%m/%Y',
    errors='coerce'
)

df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

st.write("Rows before cleaning:", len(df))

# Debug preview
st.write("Converted Date Sample:")
st.write(df[[date_col]].head())

df = df.dropna(subset=[date_col, sales_col])

st.write("Rows after cleaning:", len(df))

if df.empty:
    st.error("No valid data after cleaning")
    st.stop()

# ------------------ TIME SERIES ------------------
sales = df.groupby(date_col)[sales_col].sum()

# RAW DATA GRAPH
st.subheader("📈 Raw Sales Trend (All Data)")
st.line_chart(sales)

# MONTHLY DATA
sales_monthly = sales.resample('ME').sum()

st.subheader("📉 Monthly Sales Trend")
st.line_chart(sales_monthly)

st.divider()

# ------------------ METRICS ------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{int(sales_monthly.sum()):,}")
col2.metric("Avg Monthly Sales", f"{int(sales_monthly.mean()):,}")
col3.metric("Max Monthly Sales", f"{int(sales_monthly.max()):,}")

st.divider()

# ------------------ TOP PRODUCTS ------------------
st.subheader("🏆 Top 5 Products")

top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(5)

fig1, ax1 = plt.subplots()
top_products.plot(kind='bar', ax=ax1)
ax1.set_title("Top 5 Products")
plt.xticks(rotation=45)

st.pyplot(fig1)

st.write("📌 Insight: A few products contribute most of the revenue.")

st.divider()

# ------------------ PRODUCT COMPARISON ------------------
st.subheader("📊 Product Comparison")

product_sales = df.groupby(product_col)[sales_col].sum()

fig2, ax2 = plt.subplots()
product_sales.plot(kind='bar', ax=ax2)
ax2.set_title("Sales by Product")
plt.xticks(rotation=45)

st.pyplot(fig2)

st.write("📌 Insight: Sales vary significantly across products.")

st.divider()

# ------------------ FORECAST ------------------
st.subheader("🔮 Forecast Settings")

# SAFETY CHECK
if len(sales_monthly) < 3:
    st.error("Not enough data for forecasting")
    st.stop()

steps = st.slider("Months to forecast", 1, 12, 6)

model = ARIMA(sales_monthly, order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=steps)

forecast_df = pd.DataFrame({
    "Date": pd.date_range(
        start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
        periods=steps,
        freq='ME'
    ),
    "Predicted Sales": forecast.values
})

st.subheader("📅 Forecast Data")
st.dataframe(forecast_df)

# ------------------ FORECAST GRAPH ------------------
st.subheader("📊 Forecast Visualization")

fig3, ax3 = plt.subplots()

ax3.plot(sales_monthly.index, sales_monthly.values, label="Actual")
ax3.plot(forecast_df["Date"], forecast_df["Predicted Sales"], linestyle='--', label="Forecast")

ax3.legend()
ax3.set_title("Sales Forecast")

st.pyplot(fig3)

st.divider()

# ==============================================================================
# NEW SECTION: MONTHLY DEVIATION & ANOMALY ANALYSIS
# ==============================================================================
st.subheader("🚩 Forecast Deviations & Actionable Flags")
st.caption("Automated detection of significant forecasted monthly shifts with reasons and recommendations")

last_hist_sales = sales_monthly.iloc[-1]
avg_hist_sales = sales_monthly.mean()

def flag_monthly_deviations(row):
    pred = row["Predicted Sales"]
    pct_from_last = ((pred - last_hist_sales) / last_hist_sales) * 100
    
    if pct_from_last < -15:
        flag = "🚨 Significant Revenue Risk"
        reason = f"Predicted drop of {abs(pct_from_last):.1f}% compared to recent historical revenue."
        next_step = "Initiate targeted marketing campaigns and adjust inventory orders down to prevent holding costs."
    elif pct_from_last > 15:
        flag = "🚀 Growth Surge Opportunity"
        reason = f"Predicted increase of {pct_from_last:.1f}% above recent historical baseline."
        next_step = "Ramp up supply chain procurement and ensure adequate stock buffer for demand spike."
    else:
        flag = "🟢 Stable Forecast"
        reason = "Predicted revenue is within normal expected variance."
        next_step = "Maintain standard operations and routine inventory tracking."
        
    return pd.Series([flag, reason, next_step])

# Copy forecast dataframe for analysis
anomaly_df = forecast_df.copy()
anomaly_df["Month"] = anomaly_df["Date"].dt.strftime("%B %Y")
anomaly_df[["Status Flag", "Primary Reason", "Suggested Next Step"]] = anomaly_df.apply(flag_monthly_deviations, axis=1)

# Display table
st.dataframe(
    anomaly_df[["Month", "Predicted Sales", "Status Flag", "Primary Reason", "Suggested Next Step"]]
)

st.divider()

# ------------------ FINAL INSIGHTS ------------------
st.subheader("📌 Insights")

trend = "increasing 📈" if sales_monthly.iloc[-1] > sales_monthly.iloc[0] else "stable ➖"

st.write(f"""
- Sales trend appears **{trend}**
- Top-performing products drive majority of revenue
- Forecast suggests stable future performance
- Useful for business planning and decision-making
""")

# ------------------ SQL SECTION ------------------
st.subheader("🧠 SQL Equivalent Queries")

st.code("""
-- Top 5 Products by Sales
SELECT product, SUM(sales)
FROM sales_data
GROUP BY product
ORDER BY SUM(sales) DESC
LIMIT 5;

-- Monthly Sales Trend
SELECT DATE_TRUNC('month', order_date) AS month, SUM(sales)
FROM sales_data
GROUP BY month
ORDER BY month;

-- Total Sales
SELECT SUM(sales) FROM sales_data;

-- High Value Transactions
SELECT *
FROM sales_data
WHERE sales > 1000;
""", language="sql")

st.divider()
