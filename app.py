import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from openai import OpenAI

# ------------------ OPENAI SETUP ------------------
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------ PAGE ------------------
st.set_page_config(page_title="AI Sales Dashboard", layout="wide")

st.title("📊 AI Sales Forecast Dashboard")

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Data Preview")
st.dataframe(df.head())

# ------------------ SETTINGS ------------------
st.sidebar.header("Settings")

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

st.subheader("Sales Trend")
st.line_chart(sales)

# FIXED MONTHLY AGGREGATION
sales_monthly = sales.groupby(pd.Grouper(freq="ME")).sum()

st.subheader("Monthly Trend")
st.line_chart(sales_monthly)

# ------------------ METRICS ------------------
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{sales_monthly.sum():,.0f}")
col2.metric("Avg Monthly", f"{sales_monthly.mean():,.0f}")
col3.metric("Max Monthly", f"{sales_monthly.max():,.0f}")

# ------------------ TOP PRODUCTS ------------------
st.subheader("Top Products")

top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(5)

fig, ax = plt.subplots()
top_products.plot(kind="bar", ax=ax)
ax.set_title("Top 5 Products")
st.pyplot(fig)

# ------------------ FORECAST ------------------
st.subheader("Forecast")

steps = st.slider("Months to Forecast", 1, 12, 6)

model = ARIMA(sales_monthly, order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=steps)

forecast_index = pd.date_range(
    start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
    periods=steps,
    freq=pd.offsets.MonthEnd()
)

forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast": forecast.values
})

st.dataframe(forecast_df)

fig2, ax2 = plt.subplots()
ax2.plot(sales_monthly.index, sales_monthly.values, label="Actual")
ax2.plot(forecast_df["Date"], forecast_df["Forecast"], linestyle="--", label="Forecast")
ax2.legend()
st.pyplot(fig2)

# ------------------ 🤖 AI INSIGHTS ------------------
st.subheader("🤖 AI Insights")

if client:
    if st.button("Generate AI Analysis"):
        with st.spinner("AI is analyzing your data..."):

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

                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"AI error: {e}")

else:
    st.warning("Add OPENAI_API_KEY in secrets to enable AI features")

# ------------------ AI CHAT ------------------
st.subheader("💬 Ask AI About Your Data")

query = st.text_input("Ask something (e.g. 'What are top products?')")

if query and client:
    with st.spinner("Thinking..."):
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

            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")
