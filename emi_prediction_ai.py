# ==========================================================
# EMIPredict AI - Intelligent Financial Risk Assessment App
# ==========================================================
# Covers Steps 1–7:
# 1. Data Loading & Preprocessing
# 2. Exploratory Data Analysis (EDA)
# 3. Feature Engineering
# 4. Model Development
# 5. MLflow Integration
# 6. Streamlit Application
# 7. Cloud Deployment Ready
# ==========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# STEP 1: DATA LOADING & PREPROCESSING
# =====================================================
@st.cache_data
def load_and_preprocess_data(file_path="emi_prediction_dataset.csv"):
    df = pd.read_csv(file_path)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    df.drop_duplicates(inplace=True)

    # Encode categorical columns safely
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Feature splits
    X_class = df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
    y_class = df['emi_eligibility']
    X_reg = df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
    y_reg = df['max_monthly_emi']

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xc_train = scaler.fit_transform(Xc_train)
    Xc_test = scaler.transform(Xc_test)
    Xr_train = scaler.fit_transform(Xr_train)
    Xr_test = scaler.transform(Xr_test)

    return df, Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test

# =====================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# =====================================================
def eda_section(df):
    st.header("📊 Exploratory Data Analysis")
    st.write("A quick look at your dataset and variable patterns:")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of EMI Eligibility")
    fig, ax = plt.subplots()
    sns.countplot(x='emi_eligibility', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['credit_score'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# =====================================================
# STEP 3: FEATURE ENGINEERING
# =====================================================
def feature_engineering(df):
    df['debt_to_income'] = df['current_emi_amount'] / (df['monthly_salary'] + 1)
    df['expense_to_income'] = (
        df['groceries_utilities'] + df['travel_expenses'] + df['other_monthly_expenses']
    ) / (df['monthly_salary'] + 1)
    df['affordability_ratio'] = df['bank_balance'] / (df['monthly_salary'] + 1)
    df['risk_score'] = (df['credit_score'] / 850) * (1 - df['debt_to_income'])
    return df

# =====================================================
# STEP 4: MODEL DEVELOPMENT
# =====================================================
def train_models(Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test):
    class_models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "XGBoostClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    reg_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "XGBoostRegressor": XGBRegressor()
    }

    mlflow.set_experiment("EMIPredict_AI_Models")

    def eval_class(model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }

    def eval_reg(model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }

    best_class, best_reg = None, None
    best_acc, best_rmse = 0, float('inf')

    st.info("⏳ Training models... please wait.")

    for name, model in class_models.items():
        with mlflow.start_run(run_name=f"Classification_{name}"):
            model.fit(Xc_train, yc_train)
            metrics = eval_class(model, Xc_test, yc_test)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"{name}_model")
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_class = model

    for name, model in reg_models.items():
        with mlflow.start_run(run_name=f"Regression_{name}"):
            model.fit(Xr_train, yr_train)
            metrics = eval_reg(model, Xr_test, yr_test)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"{name}_model")
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_reg = model

    st.success("✅ Models trained and logged in MLflow!")
    return best_class, best_reg

# =====================================================
# STEP 6: STREAMLIT APPLICATION UI
# =====================================================
def main():
    st.set_page_config(page_title="EMIPredict AI", layout="wide")

    st.title("💰 EMIPredict AI - Financial Risk Assessment Platform")
    st.caption("An ML-powered EMI eligibility and affordability prediction system.")

    # Sidebar navigation
    menu = ["🏠 Home", "📊 EDA", "⚙️ Train Models", "🔮 Predict EMI"]
    choice = st.sidebar.radio("Navigation", menu)

    # Load data
    df, Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test = load_and_preprocess_data()
    df = feature_engineering(df)

    if choice == "🏠 Home":
        st.subheader("Welcome to EMIPredict AI!")
        st.write("""
        This platform uses **Machine Learning** to:
        - Predict whether a customer is eligible for EMI (Classification)
        - Estimate the maximum safe EMI amount (Regression)
        - Provide visual financial analysis and insights
        """)

    elif choice == "📊 EDA":
        eda_section(df)

    elif choice == "⚙️ Train Models":
        best_class, best_reg = train_models(Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test)
        st.session_state["best_class_model"] = best_class
        st.session_state["best_reg_model"] = best_reg
        st.success("Training Complete! You can now go to 'Predict EMI' tab.")

    elif choice == "🔮 Predict EMI":
        st.subheader("Real-Time EMI Prediction")

        monthly_salary = st.number_input("Monthly Salary (INR)", 15000, 200000, 50000)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        current_emi = st.number_input("Current EMI (INR)", 0, 50000, 1000)
        bank_balance = st.number_input("Bank Balance (INR)", 1000, 1000000, 20000)
        total_expenses = st.number_input("Total Monthly Expenses (INR)", 1000, 100000, 10000)

        if "best_class_model" not in st.session_state:
            st.warning("Please train models first in the 'Train Models' tab.")
        else:
            if st.button("Predict EMI Eligibility"):
                clf = st.session_state["best_class_model"]
                reg = st.session_state["best_reg_model"]

                sample = np.array([[monthly_salary, credit_score, bank_balance, current_emi, total_expenses]])
                pred_class = clf.predict(sample)[0]
                pred_emi = reg.predict(sample)[0]

                st.success(f"Predicted EMI Eligibility: {pred_class}")
                st.info(f"Estimated Safe Monthly EMI: ₹{pred_emi:,.2f}")

    st.markdown("---")
    st.caption("© EMIPredict AI | Intelligent Financial Risk Assessment Platform")

# =====================================================
# STEP 7: CLOUD DEPLOYMENT READY
# =====================================================
# To deploy:
# 1. Push this file and dataset + requirements.txt to GitHub
# 2. Go to https://share.streamlit.io
# 3. Connect your GitHub repo and deploy
# 4. Enjoy your live app!

if __name__ == "__main__":
    main()
