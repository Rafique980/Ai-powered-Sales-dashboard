import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sales Forecast Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- GROQ SETUP ----------------
client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- CACHE: LOAD DATA ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------------- CACHE: MONTHLY SALES ----------------
@st.cache_data
def get_monthly_sales(df, date_col, sales_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    df = df.sort_values(date_col)

    sales = df.groupby(date_col)[sales_col].sum()
    sales.index = pd.to_datetime(sales.index)

    return sales.resample("ME").sum(), df

# ---------------- CACHE: FORECAST ----------------
@st.cache_data
def forecast_sales(series, steps):
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# ---------------- UI HEADER ----------------
st.title("📊 AI Sales Forecast Dashboard")
st.caption("Upload CSV • Analyze Sales • Forecast Trends • AI Insights")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to continue.")
    st.stop()

df_raw = load_data(uploaded_file)

st.subheader("📄 Data Preview")
st.dataframe(df_raw.head(), use_container_width=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuration")

date_col = st.sidebar.selectbox("📅 Date Column", df_raw.columns)
sales_col = st.sidebar.selectbox("💰 Sales Column", df_raw.columns)
product_col = st.sidebar.selectbox("📦 Product Column", df_raw.columns)

# Apply processing once
sales_monthly, df = get_monthly_sales(df_raw, date_col, sales_col)

if sales_monthly.empty:
    st.error("No valid data after cleaning.")
    st.stop()

# ---------------- TREND ----------------
st.subheader("📈 Sales Trend")

trend_df = pd.DataFrame({
    "Date": sales_monthly.index,
    "Sales": sales_monthly.values
})

fig = px.line(trend_df, x="Date", y="Sales", title="Sales Over Time")
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ---------------- MONTHLY ----------------
st.subheader("📅 Monthly Sales")

fig2 = px.area(trend_df, x="Date", y="Sales", title="Monthly Sales Trend")
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- METRICS ----------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{sales_monthly.sum():,.0f}")
col2.metric("Avg Monthly", f"{sales_monthly.mean():,.0f}")
col3.metric("Max Monthly", f"{sales_monthly.max():,.0f}")

# ---------------- TOP PRODUCTS ----------------
st.subheader("🏆 Top Products")

top_products = (
    df.groupby(product_col)[sales_col]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_df = pd.DataFrame({
    "Product": top_products.index.astype(str),
    "Sales": top_products.values
})

fig3 = px.bar(top_df, x="Product", y="Sales", color="Sales", title="Top Products")
fig3.update_layout(template="plotly_dark")
st.plotly_chart(fig3, use_container_width=True)

# ---------------- FORECAST ----------------
st.subheader("🔮 Forecast")

if len(sales_monthly) < 3:
    st.warning("Need at least 3 months of data for forecasting.")
else:
    steps = st.slider("Months to Forecast", 1, 12, 6)

    forecast = forecast_sales(sales_monthly, steps)

    future_dates = pd.date_range(
        start=sales_monthly.index[-1] + pd.offsets.MonthEnd(1),
        periods=steps,
        freq="ME"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.values
    })

    st.dataframe(forecast_df, use_container_width=True)

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=sales_monthly.index,
        y=sales_monthly.values,
        name="Actual"
    ))

    fig4.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        name="Forecast",
        line=dict(dash="dash")
    ))

    fig4.update_layout(template="plotly_dark", title="Forecast")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------- AI INSIGHTS ----------------
st.subheader("🤖 AI Insights")

if client and st.button("Generate AI Analysis"):

    with st.spinner("AI analyzing..."):

        prompt = f"""
        Total Sales: {sales_monthly.sum()}
        Avg Monthly: {sales_monthly.mean()}
        Max Monthly: {sales_monthly.max()}

        Top Products:
        {top_products.to_string()}

        Give insights, risks, recommendations.
        """

        try:
            res = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a business analyst."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.markdown(
                f"<div class='ai-box'>{res.choices[0].message.content}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"AI Error: {e}")

elif not client:
    st.warning("Add GROQ_API_KEY in Streamlit secrets.")

# ---------------- CHAT ----------------
st.subheader("💬 Ask AI")

query = st.text_input("Ask about your data")

if query and client:

    context = f"""
    Total Sales: {sales_monthly.sum()}
    Avg: {sales_monthly.mean()}
    Top Product: {top_products.index[0]}
    """

    try:
        res = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": context + "\n\nQ: " + query}
            ]
        )

        st.markdown(
            f"<div class='ai-box'>{res.choices[0].message.content}</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"AI Error: {e}")

st.divider()
