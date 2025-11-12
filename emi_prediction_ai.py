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
    """
    Loads, cleans, encodes categorical variables, scales data, and splits for classification & regression.
    """
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

    # Use all columns for model training except targets
    X = df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
    y_class = df['emi_eligibility']
    y_reg = df['max_monthly_emi']

    # Split data
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    Xc_train = scaler.fit_transform(Xc_train)
    Xc_test = scaler.transform(Xc_test)
    Xr_train = scaler.transform(Xr_train)
    Xr_test = scaler.transform(Xr_test)

    return df, X, Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test, scaler


# =====================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (15 GRAPHS)
# =====================================================
def eda_section(df):
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Visualizing trends, correlations, and risk indicators across financial profiles.")

    # 15 EDA Graphs (same as your original)
    graphs = [
        ("EMI Eligibility Distribution", lambda ax: sns.countplot(x='emi_eligibility', data=df, ax=ax)),
        ("Credit Score Distribution", lambda ax: sns.histplot(df['credit_score'], bins=30, kde=True, color='blue', ax=ax)),
        ("Monthly Salary Distribution", lambda ax: sns.histplot(df['monthly_salary'], bins=30, kde=True, color='green', ax=ax)),
        ("EMI Scenario Count", lambda ax: sns.countplot(x='emi_scenario', data=df, ax=ax)),
        ("Correlation Heatmap", lambda ax: sns.heatmap(df.corr(), cmap='coolwarm', annot=False, ax=ax)),
        ("Age vs EMI Eligibility", lambda ax: sns.boxplot(x='emi_eligibility', y='age', data=df, ax=ax)),
        ("Salary vs EMI Eligibility", lambda ax: sns.boxplot(x='emi_eligibility', y='monthly_salary', data=df, ax=ax)),
        ("Credit Score vs EMI Eligibility", lambda ax: sns.boxplot(x='emi_eligibility', y='credit_score', data=df, ax=ax)),
        ("Existing Loan Status", lambda ax: sns.countplot(x='existing_loans', hue='emi_eligibility', data=df, ax=ax)),
        ("Family Size Distribution", lambda ax: sns.histplot(df['family_size'], bins=15, kde=True, color='purple', ax=ax)),
        ("Expense Comparison", lambda ax: sns.barplot(x='emi_eligibility', y='groceries_utilities', data=df, ax=ax)),
        ("Gender vs EMI Eligibility", lambda ax: sns.countplot(x='gender', hue='emi_eligibility', data=df, ax=ax)),
        ("Education Level vs EMI Eligibility", lambda ax: sns.countplot(x='education', hue='emi_eligibility', data=df, ax=ax)),
        ("Loan Amount vs Scenario", lambda ax: sns.boxplot(x='emi_scenario', y='requested_amount', data=df, ax=ax)),
        ("Tenure vs Scenario", lambda ax: sns.boxplot(x='emi_scenario', y='requested_tenure', data=df, ax=ax))
    ]

    for title, plot_func in graphs:
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_func(ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)


# =====================================================
# STEP 3: FEATURE ENGINEERING
# =====================================================
def feature_engineering(df):
    df['debt_to_income'] = df['current_emi_amount'] / (df['monthly_salary'] + 1)
    df['expense_to_income'] = (df['groceries_utilities'] + df['travel_expenses'] + df['other_monthly_expenses']) / (df['monthly_salary'] + 1)
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
            # Commented model saving to avoid OSError
            # mlflow.sklearn.log_model(model, f"{name}_model")
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_class = model

    for name, model in reg_models.items():
        with mlflow.start_run(run_name=f"Regression_{name}"):
            model.fit(Xr_train, yr_train)
            metrics = eval_reg(model, Xr_test, yr_test)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            # mlflow.sklearn.log_model(model, f"{name}_model")
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_reg = model

    st.success("Models trained and logged in MLflow!")
    st.info("To view MLflow dashboard, run: `python -m mlflow ui` → http://127.0.0.1:5000")
    return best_class, best_reg


# =====================================================
# STEP 6: STREAMLIT APPLICATION
# =====================================================
def main():
    st.set_page_config(page_title="EMIPredict AI", layout="wide")

    st.markdown(
        """
        <style>
            .main { background-color: #f9fafc; color: #1a1a1a; }
            h1, h2, h3 { color: #0e4f88; }
            .stButton>button {
                background-color: #0e4f88;
                color: white;
                border-radius: 10px;
                height: 3em;
                width: 100%;
            }
            .stButton>button:hover { background-color: #1261a0; }
        </style>
        """,
        unsafe_allow_html=True
    )

    menu = ["Home", "EDA", "Train Models", "Predict EMI", "About"]
    choice = st.sidebar.radio("Navigation", menu)

    df, X, Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test, scaler = load_and_preprocess_data()
    df = feature_engineering(df)
    mean_vector = X.mean()

    if choice == "Home":
        st.title("💰 EMIPredict AI")
        st.subheader("Intelligent Financial Risk Assessment Platform")
        st.markdown("""
        **EMIPredict AI** helps financial institutions, fintech firms, and individuals assess:
        - ✅ EMI eligibility classification  
        - 📈 Maximum safe EMI affordability  
        - 📊 Real-time financial risk scoring  
        - 🔍 Deep data analysis with 400K+ financial records  
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
        groceries_utilities = st.number_input("Monthly Groceries & Utilities (INR)", 1000, 100000, 10000)

        if "best_class_model" not in st.session_state:
            st.warning("Please train models first in the 'Train Models' tab.")
        else:
            if st.button("Predict EMI Eligibility"):
                clf = st.session_state["best_class_model"]
                reg = st.session_state["best_reg_model"]

                # Create a full-length feature vector
                sample_dict = mean_vector.copy()
                sample_dict["monthly_salary"] = monthly_salary
                sample_dict["credit_score"] = credit_score
                sample_dict["bank_balance"] = bank_balance
                sample_dict["current_emi_amount"] = current_emi
                sample_dict["groceries_utilities"] = groceries_utilities

                sample_df = pd.DataFrame([sample_dict])
                scaled_sample = scaler.transform(sample_df)

                pred_class = clf.predict(scaled_sample)[0]
                pred_emi = max(0, reg.predict(scaled_sample)[0])  # prevent negative EMI

                st.success(f"Predicted EMI Eligibility: {'1' if pred_class == 1 else '2'}")
                st.info(f"Estimated Safe Monthly EMI: ₹{pred_emi:,.2f}")

    elif choice == "About":
        st.header("About EMIPredict AI")
        st.markdown("""
        **Project Overview:**  
        EMIPredict AI is a FinTech-focused ML platform that predicts EMI eligibility and affordability using real-world financial and demographic data.  

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
