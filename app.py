import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

# ------------------ GROQ SETUP ------------------
try:
    from groq import Groq
    groq_client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")))
except Exception:
    groq_client = None

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="AI Sales Forecast Dashboard", layout="wide")

st.title("📊 AI Sales Forecast Dashboard")
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
df[date_col] = pd.to_datetime(
    df[date_col],
    format='%d/%m/%Y',
    errors='coerce'
)

df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

df = df.dropna(subset=[date_col, sales_col])

if df.empty:
    st.error("No valid data after cleaning")
    st.stop()

# ------------------ TIME SERIES ------------------
sales = df.groupby(date_col)[sales_col].sum()

st.subheader("📈 Sales Trend")
st.line_chart(sales)

sales_monthly = sales.resample('ME').sum()

st.subheader("📉 Monthly Sales")
st.line_chart(sales_monthly)

st.divider()

# ------------------ METRICS ------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"${int(sales_monthly.sum()):,}")
col2.metric("Avg Monthly Revenue", f"${int(sales_monthly.mean()):,}")
col3.metric("Max Monthly Revenue", f"${int(sales_monthly.max()):,}")

st.divider()

# ------------------ TOP PRODUCTS ------------------
st.subheader("🏆 Top Products")

top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)

fig1, ax1 = plt.subplots()
top_products.plot(kind='bar', ax=ax1)
ax1.set_title("Top Performing Products")
plt.xticks(rotation=45)

st.pyplot(fig1)

st.divider()

# ------------------ FORECAST ------------------
st.subheader("🔮 Sales Forecast")

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

st.dataframe(forecast_df)

fig3, ax3 = plt.subplots()
ax3.plot(sales_monthly.index, sales_monthly.values, label="Actual")
ax3.plot(forecast_df["Date"], forecast_df["Predicted Sales"], linestyle='--', label="Forecast")
ax3.legend()
ax3.set_title("Sales Forecast")
st.pyplot(fig3)

st.divider()

# ------------------ ACTIONABLE FLAGS ------------------
st.subheader("🚩 Forecast Deviations & Actionable Flags")

last_hist_sales = sales_monthly.iloc[-1]
avg_hist_sales = sales_monthly.mean()

def flag_monthly_deviations(df_forecast):
    flags, reasons, next_steps = [], [], []
    prev_val = last_hist_sales 
    
    for idx, row in df_forecast.iterrows():
        pred = row["Predicted Sales"]
        mom_change = ((pred - prev_val) / prev_val) * 100
        
        if mom_change < -0.5:
            flags.append("🚨 Revenue Risk")
            reasons.append(f"Projected drop of {abs(mom_change):.1f}% vs previous month.")
            next_steps.append("Launch promotional deals & optimize inventory to avoid excess holding costs.")
        elif mom_change > 0.5:
            flags.append("🚀 Growth Opportunity")
            reasons.append(f"Projected increase of {mom_change:.1f}% vs previous month.")
            next_steps.append("Ensure supply chain readiness & procure inventory stock for demand surge.")
        elif pred < avg_hist_sales * 0.95:
            flags.append("⚠️ Below Target Alert")
            reasons.append("Predicted sales fall significantly below historical average.")
            next_steps.append("Review product mix and trigger targeted marketing pushes.")
        else:
            flags.append("🟢 Stable Forecast")
            reasons.append("Predicted revenue is within expected steady performance range.")
            next_steps.append("Maintain standard operations and routine inventory tracking.")
            
        prev_val = pred
        
    return flags, reasons, next_steps

anomaly_df = forecast_df.copy()
anomaly_df["Month"] = anomaly_df["Date"].dt.strftime("%B %Y")
flags, reasons, next_steps = flag_monthly_deviations(anomaly_df)
anomaly_df["Status Flag"] = flags
anomaly_df["Primary Reason"] = reasons
anomaly_df["Suggested Next Step"] = next_steps

st.dataframe(
    anomaly_df[["Month", "Predicted Sales", "Status Flag", "Primary Reason", "Suggested Next Step"]],
    use_container_width=True
)

st.divider()

# ------------------ SQL EQUIVALENT QUERIES ------------------
st.subheader("🧠 SQL Equivalent Queries")

st.code("""
-- Top Products by Sales
SELECT product, SUM(sales)
FROM sales_data
GROUP BY product
ORDER BY SUM(sales) DESC
LIMIT 10;

-- Monthly Sales Trend
SELECT DATE_TRUNC('month', order_date) AS month, SUM(sales)
FROM sales_data
GROUP BY month
ORDER BY month;

-- Total Sales
SELECT SUM(sales) FROM sales_data;
""", language="sql")

st.divider()

# ------------------ AI INSIGHTS GENERATOR ------------------
st.subheader("📌 AI Insights")

if st.button("✨ Generate AI Insights"):
    if groq_client:
        with st.spinner("Analyzing data with Groq AI..."):
            data_summary = f"""
            Total Revenue: ${int(sales_monthly.sum()):,}
            Avg Monthly Revenue: ${int(sales_monthly.mean()):,}
            Top Product: {top_products.index[0]} (${int(top_products.iloc[0]):,})
            Forecasted Months: {steps}
            Next Month Projected Revenue: ${int(forecast_df.iloc[0]['Predicted Sales']):,}
            """
            prompt = f"Act as an executive business analyst. Generate detailed insights based on this data summary: {data_summary}. Structure your output with Key Findings, Risks, and Recommendations."
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error calling Groq API: {e}")
    else:
        st.info("💡 Pro Tip: Configure your GROQ_API_KEY in Streamlit Secrets to enable automated LLM executive summaries!")

st.divider()

# ------------------ ASK AI CHATBOT ------------------
st.subheader("🤖 Ask AI About Your Data")
st.caption("Ask any question regarding your sales dataset")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("e.g. Which month had highest revenue?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if groq_client:
        try:
            context = f"Dataset Total Revenue: ${int(sales_monthly.sum()):,}. Top Product: {top_products.index[0]}. Latest Monthly Revenue: ${int(sales_monthly.iloc[-1]):,}."
            system_msg = f"You are a helpful data analyst assistant. Context: {context}"
            
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_input}
                ]
            )
            bot_reply = response.choices[0].message.content
        except Exception as e:
            bot_reply = f"Couldn't connect to Groq API. Quick answer based on data: Total sales = ${int(sales_monthly.sum()):,} across {len(df)} rows."
    else:
        bot_reply = f"Total Revenue: ${int(sales_monthly.sum()):,}. Top Product: {top_products.index[0]}."

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)
