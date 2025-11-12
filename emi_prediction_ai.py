# ==========================================================
# EMIPredict AI - Intelligent Financial Risk Assessment App
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

    # Encode categorical safely
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

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
# STEP 2: EXPLORATORY DATA ANALYSIS (15 GRAPHS)
# =====================================================
def eda_section(df):
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Visualizing trends, correlations, and risk indicators across 400K financial profiles.")

    st.subheader("Graph 1 - EMI Eligibility Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='emi_eligibility', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 2 - Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['credit_score'], bins=30, kde=True, color='blue', ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 3 - Monthly Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['monthly_salary'], bins=30, kde=True, color='green', ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 4 - EMI Scenario Count")
    fig, ax = plt.subplots()
    sns.countplot(x='emi_scenario', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Graph 5 - Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 6 - Age vs EMI Eligibility")
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_eligibility', y='age', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 7 - Salary vs EMI Eligibility")
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_eligibility', y='monthly_salary', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 8 - Credit Score vs EMI Eligibility")
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_eligibility', y='credit_score', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 9 - Existing Loan Status")
    fig, ax = plt.subplots()
    sns.countplot(x='existing_loans', hue='emi_eligibility', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 10 - Family Size Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['family_size'], bins=15, kde=True, color='purple', ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 11 - Expense Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x='emi_eligibility', y='groceries_utilities', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 12 - Gender vs EMI Eligibility")
    fig, ax = plt.subplots()
    sns.countplot(x='gender', hue='emi_eligibility', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Graph 13 - Education Level vs EMI Eligibility")
    fig, ax = plt.subplots()
    sns.countplot(x='education', hue='emi_eligibility', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Graph 14 - Loan Amount vs Scenario")
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_scenario', y='requested_amount', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Graph 15 - Tenure vs Scenario")
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_scenario', y='requested_tenure', data=df, ax=ax)
    plt.xticks(rotation=45)
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
# STEP 4 + 5: MODEL DEVELOPMENT + MLflow INTEGRATION
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

    st.success("Models trained and logged in MLflow!")
    st.info("To view MLflow dashboard, run in terminal: `mlflow ui` then open http://127.0.0.1:5000")
    return best_class, best_reg


# =====================================================
# STEP 6: STREAMLIT APPLICATION
# =====================================================
def main():
    st.set_page_config(page_title="EMIPredict AI", layout="wide")

    # Sidebar logo & navigation
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Loan_Icon.png", width=100)
    menu = ["Home", "EDA", "Train Models", "Predict EMI", "About"]
    choice = st.sidebar.radio("Navigation", menu)

    df, Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test = load_and_preprocess_data()
    df = feature_engineering(df)

    if choice == "Home":
        st.title("💰 EMIPredict AI")
        st.subheader("Intelligent Financial Risk Assessment Platform")
        st.markdown("""
        **EMIPredict AI** helps financial institutions, fintech firms, and individuals assess:
        - ✅ EMI eligibility classification  
        - 📈 Maximum safe EMI affordability  
        - 📊 Real-time financial risk scoring  
        - 🔍 Deep data analysis with 400K+ financial records  

        Powered by **Machine Learning, XGBoost, Random Forests, and MLflow tracking.**
        """)

    elif choice == "EDA":
        eda_section(df)

    elif choice == "Train Models":
        best_class, best_reg = train_models(Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test)
        st.session_state["best_class_model"] = best_class
        st.session_state["best_reg_model"] = best_reg
        st.success("Training Complete! Now go to 'Predict EMI' tab.")

    elif choice == "Predict EMI":
        st.header("Real-Time EMI Prediction")

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

    elif choice == "About":
        st.header("About EMIPredict AI")
        st.markdown("""
        **Project Overview:**  
        EMIPredict AI is a FinTech-focused ML platform that predicts EMI eligibility and affordability using real-world financial and demographic data.  
        It integrates **MLflow** for experiment tracking, supports dual ML pipelines (Classification + Regression), and visualizes patterns across risk segments.

        **Architecture Summary:**  
        - Machine Learning Models: Logistic Regression, Random Forest, XGBoost  
        - Feature Engineering: Financial ratios, affordability, risk scoring  
        - MLflow Integration: Automated experiment logging and model registry  
        - Deployment: Streamlit Cloud  
        """)

    st.markdown("---")
    st.caption("© 2025 EMIPredict AI | Intelligent Financial Risk Assessment Platform")


if __name__ == "__main__":
    main()
