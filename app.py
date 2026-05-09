import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- GROQ SETUP ----------------
client = None

if "GROQ_API_KEY" in st.secrets:
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

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

.stButton>button:hover {
    opacity: 0.9;
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

# ---------------- HEADER ----------------
st.title("📊 AI Sales Forecast Dashboard")
st.caption("Upload CSV • Analyze Sales • Forecast Trends • AI Insights")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📁 Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv(uploaded_file)

except Exception as e:
    st.error(f"CSV Error: {e}")
    st.stop()

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# ---------------- SIDEBAR ----------------
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

# ---------------- DATA CLEANING ----------------
df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df[sales_col] = pd.to_numeric(
    df[sales_col],
    errors="coerce"
)

df = df.dropna(
    subset=[date_col, sales_col]
)

if df.empty:
    st.error("No valid data after cleaning.")
    st.stop()

# ---------------- SORT DATA ----------------
df = df.sort_values(date_col)

# ---------------- SALES SERIES ----------------
sales = (
    df.groupby(date_col)[sales_col]
    .sum()
)

sales.index = pd.to_datetime(sales.index)

# ---------------- SALES TREND ----------------
st.subheader("📈 Sales Trend")

sales_df = pd.DataFrame({
    "Date": sales.index,
    "Sales": sales.values
})

fig_sales = px.line(
    sales_df,
    x="Date",
    y="Sales",
    title="Sales Over Time"
)

fig_sales.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(
    fig_sales,
    use_container_width=True
)

# ---------------- MONTHLY SALES ----------------
sales_monthly = (
    sales
    .resample("ME")
    .sum()
)

sales_monthly_df = pd.DataFrame({
    "Month": sales_monthly.index,
    "Sales": sales_monthly.values
})

st.subheader("📅 Monthly Sales")

fig_monthly = px.area(
    sales_monthly_df,
    x="Month",
    y="Sales",
    title="Monthly Sales Trend"
)

fig_monthly.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True
)

# ---------------- METRICS ----------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💰 Total Sales",
        f"{sales_monthly.sum():,.0f}"
    )

with col2:
    st.metric(
        "📈 Avg Monthly Sales",
        f"{sales_monthly.mean():,.0f}"
    )

with col3:
    st.metric(
        "🚀 Max Monthly Sales",
        f"{sales_monthly.max():,.0f}"
    )

# ---------------- TOP PRODUCTS ----------------
st.subheader("🏆 Top Products")

top_products = (
    df.groupby(product_col)[sales_col]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products_df = pd.DataFrame({
    "Product": top_products.index.astype(str),
    "Sales": top_products.values
})

fig_products = px.bar(
    top_products_df,
    x="Product",
    y="Sales",
    color="Sales",
    title="Top Selling Products"
)

fig_products.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)

# ---------------- FORECAST ----------------
st.subheader("🔮 Sales Forecast")

if len(sales_monthly) < 3:

    st.warning(
        "Need at least 3 months of data for forecasting."
    )

else:

    steps = st.slider(
        "Months to Forecast",
        1,
        12,
        6
    )

    try:

        model = ARIMA(
            sales_monthly,
            order=(1, 1, 1)
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(
            steps=steps
        )

        forecast_index = pd.date_range(
            start=sales_monthly.index[-1]
            + pd.offsets.MonthEnd(1),
            periods=steps,
            freq="ME"
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            "Forecast": forecast.values
        })

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        fig_forecast = go.Figure()

        fig_forecast.add_trace(
            go.Scatter(
                x=sales_monthly.index,
                y=sales_monthly.values,
                mode="lines",
                name="Actual Sales"
            )
        )

        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines",
                name="Forecast",
                line=dict(dash="dash")
            )
        )

        fig_forecast.update_layout(
            template="plotly_dark",
            title="Sales Forecast",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Forecast Error: {e}"
        )

# ---------------- Sql evqivalent queries ----------------
# ---------------- SQL SECTION ----------------
st.subheader("🧠 SQL Equivalent Queries")

st.code("""
-- Total Sales
SELECT SUM(sales) AS total_sales
FROM sales_data;

-- Average Monthly Sales
SELECT DATE_TRUNC('month', order_date) AS month,
       SUM(sales) AS monthly_sales
FROM sales_data
GROUP BY month
ORDER BY month;

-- Top Products
SELECT product,
       SUM(sales) AS total_sales
FROM sales_data
GROUP BY product
ORDER BY total_sales DESC
LIMIT 10;

-- Highest Sales Month
SELECT DATE_TRUNC('month', order_date) AS month,
       SUM(sales) AS monthly_sales
FROM sales_data
GROUP BY month
ORDER BY monthly_sales DESC
LIMIT 1;

-- High Value Transactions
SELECT *
FROM sales_data
WHERE sales > 1000;

-- Total Orders Per Product
SELECT product,
       COUNT(*) AS total_orders
FROM sales_data
GROUP BY product
ORDER BY total_orders DESC;
""", language="sql")
# ---------------- AI INSIGHTS ----------------
st.subheader("🤖 AI Insights")

if client:

    if st.button("✨ Generate AI Analysis"):

        with st.spinner("AI analyzing your data..."):

            prompt = f"""
            Analyze this sales dataset.

            Total Sales:
            {sales_monthly.sum()}

            Average Monthly Sales:
            {sales_monthly.mean()}

            Maximum Monthly Sales:
            {sales_monthly.max()}

            Top Products:
            {top_products.to_string()}

            Give:
            1. Business insights
            2. Risks
            3. Recommendations
            """

            try:

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert business analyst."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                ai_text = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.markdown(
                    f"""
                    <div class="ai-box">
                    {ai_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"AI Error: {e}"
                )

else:

    st.warning(
        "Add GROQ_API_KEY to Streamlit secrets."
    )

# ---------------- AI CHAT ----------------
st.subheader("💬 Ask AI About Your Data")

query = st.text_input(
    "Ask a question about your sales data"
)

if query:

    if client:

        with st.spinner("Thinking..."):

            context = f"""
            Dataset Summary:

            Total Sales:
            {sales_monthly.sum()}

            Average Monthly Sales:
            {sales_monthly.mean()}

            Best Product:
            {top_products.index[0]}

            Top Products:
            {top_products.to_string()}
            """

            try:

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a smart business data analyst."
                        },
                        {
                            "role": "user",
                            "content": context + "\n\nQuestion:\n" + query
                        }
                    ]
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.markdown(
                    f"""
                    <div class="ai-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"AI Error: {e}"
                )

    else:

        st.warning(
            "Add GROQ_API_KEY to Streamlit secrets."
        )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Built with Streamlit • Plotly • ARIMA • Groq AI"
)
