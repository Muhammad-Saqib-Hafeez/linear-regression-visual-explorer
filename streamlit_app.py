# streamlit_app.py
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from regressio.data_handler import DataHandler
from regressio.model import RegressionModel

st.set_page_config(
    page_title="Linear Regression Demonstration",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def build_dataset(n_samples: int, noise: float, random_state: int):
    return DataHandler.generate_synthetic(n_samples=n_samples, noise=noise, random_state=random_state)

@st.cache_data
def train_model_from_df(df):
    X, y = DataHandler.get_features_targets(df)
    rm = RegressionModel()
    rm.train(X, y)
    return rm

st.markdown(
    """
    <h1 style="text-align:center; color:#1E2B45; margin-bottom:0;">
        Linear Regression Demonstration
    </h1>
    <p style="text-align:center; color:#6C757D; font-size:17px; margin-top:4px;">
        Explore, visualize, and understand how a simple linear regression model fits and predicts data.
    <hr style="margin-top:25px; margin-bottom:25px;">
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Experiment Parameters")
n_samples = st.sidebar.slider("Number of Samples", 10, 500, 100, step=10)
noise = st.sidebar.slider("Noise Level", 0.0, 50.0, 15.0, step=1.0)
random_state = st.sidebar.number_input("Random Seed", 0, 9999, 42)
user_x_raw = st.sidebar.number_input("Enter X value for Prediction", value=0.0, format="%.3f")
predict_button = st.sidebar.button("Generate Prediction")

st.sidebar.markdown("---")
st.sidebar.caption(
    """
    **Model Parameter Notes**
    - *Slope (Coefficient)* — Change in Y for each unit increase in X  
    - *Intercept* — Predicted Y when X = 0  
    - *R² Score* — Proportion of variance in Y explained by X
    """
)

df = build_dataset(int(n_samples), float(noise), int(random_state))
rm = train_model_from_df(df)
slope, intercept, r2 = rm.slope, rm.intercept, rm.r2

df_pred = df.copy()
df_pred["Y_pred"] = rm.model.predict(df_pred["X"].values.reshape(-1, 1))

chart_base = alt.Chart(df_pred).mark_circle(size=60, color="#1F77B4").encode(
    x=alt.X("X", title="X values"),
    y=alt.Y("Y", title="Actual Y values"),
    tooltip=["X", "Y"]
)

line = alt.Chart(df_pred).mark_line(color="#E4572E", strokeWidth=3).encode(
    x="X", y="Y_pred", tooltip=["X", "Y_pred"]
)

if predict_button:
    user_x = DataHandler.validate_user_x(user_x_raw)
    y_pred = float(rm.predict([user_x])[0])
    user_df = pd.DataFrame({"X": [user_x], "Y": [y_pred]})
    user_point = alt.Chart(user_df).mark_point(
        size=110, color="#2CA02C", shape="diamond"
    ).encode(
        x="X", y="Y", tooltip=["X", "Y"]
    )
    chart = chart_base + line + user_point
else:
    chart = chart_base + line

st.subheader("Regression Visualization")
st.altair_chart(chart.interactive().properties(height=480), use_container_width=True)

st.subheader("Model Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Slope (Coefficient)", f"{slope:.4f}")
col2.metric("Intercept", f"{intercept:.4f}")
col3.metric("R² Score", f"{r2:.4f}")

st.progress(min(max(r2, 0.0), 1.0), text=f"Model fit quality: {r2:.2%}")

if predict_button:
    st.markdown(
        f"""
        <div style="background-color:#F8F9FA; border-left:4px solid #0D6EFD; padding:1rem; margin-top:1rem;">
        <strong>Prediction Result:</strong><br>
        For X = <b>{user_x:.3f}</b>, the model predicts Y = <b>{y_pred:.3f}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Enter an X value in the sidebar and click *Generate Prediction* to compute Y.")

st.markdown("---")
st.subheader("Understanding the Linear Regression Model")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        **1. Concept Overview**  
        Linear regression models the relationship between two continuous variables, **X** and **Y**.  
        The model fits a straight line that minimizes the total squared error between predicted and actual values.  
        
        Mathematically:  
        \[
        Y = β₀ + β₁X + ε
        \]
        where:
        - \(β₀\): intercept  
        - \(β₁\): slope (coefficient)  
        - \(ε\): random error term
        """
    )

with c2:
    st.markdown(
        f"""
        **2. Interpretation of Results**  
        - Each unit increase in X changes Y by roughly **{slope:.2f}** units.  
        - When X = 0, predicted Y ≈ **{intercept:.2f}**.  
        - R² = **{r2:.2f}** → approximately **{r2*100:.1f}%** of Y’s variability is explained by X.  
        
        A higher R² means a stronger linear relationship.
        """
    )

st.markdown("---")
st.subheader("Dataset Overview")
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <hr style="margin-top:40px;">
    <p style="text-align:center; color:#6C757D; font-size:14px;">
    Developed for educational and explanatory purposes using Streamlit and scikit-learn.
    </p>
    """,
    unsafe_allow_html=True,
)
