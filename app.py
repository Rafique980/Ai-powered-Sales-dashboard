import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Optional OpenAI (safe fallback if not configured)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard (ARIMA)")


# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Column selection
    columns = df.columns.tolist()

    date_col = st.selectbox("Select Date Column", columns)
    target_col = st.selectbox("Select Sales Column", columns)

    # -------------------------------
    # DATA PREPROCESSING
    # -------------------------------
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[[date_col, target_col]].dropna()

    df.set_index(date_col, inplace=True)

    st.subheader("Processed Data")
    st.line_chart(df[target_col])

    # -------------------------------
    # FORECAST SETTINGS
    # -------------------------------
    st.sidebar.header("Forecast Settings")

    p = st.sidebar.slider("ARIMA p", 0, 5, 1)
    d = st.sidebar.slider("ARIMA d", 0, 2, 1)
    q = st.sidebar.slider("ARIMA q", 0, 5, 1)

    steps = st.sidebar.slider("Forecast Days", 5, 60, 10)

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    if st.button("Run Forecast"):
        with st.spinner("Training ARIMA model..."):

            model = ARIMA(df[target_col], order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=steps)

            forecast_index = pd.date_range(
                start=df.index[-1],
                periods=steps + 1,
                freq="D"
            )[1:]

            forecast_df = pd.DataFrame({
                "Forecast": forecast
            }, index=forecast_index)

            # -------------------------------
            # PLOT
            # -------------------------------
            st.subheader("Forecast Results")

            fig, ax = plt.subplots()
            ax.plot(df[target_col], label="Actual")
            ax.plot(forecast_df["Forecast"], label="Forecast", linestyle="--")
            ax.legend()
            st.pyplot(fig)

            st.dataframe(forecast_df)

            # -------------------------------
            # INSIGHTS (OPTIONAL OPENAI)
            # -------------------------------
            st.subheader("AI Insights")

            if OPENAI_AVAILABLE and "OPENAI_API_KEY" in st.secrets:
                try:
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                    prompt = f"""
                    You are a data analyst.
                    Explain this sales forecast in simple terms:

                    - Last actual value: {df[target_col].iloc[-1]}
                    - Forecast values: {forecast.values[:5].tolist()}

                    Give business insights and risks.
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )

                    st.write(response.choices[0].message.content)

                except Exception as e:
                    st.warning(f"OpenAI insight failed: {e}")

            else:
                st.info("OpenAI not configured. Add API key in Streamlit secrets for AI insights.")
