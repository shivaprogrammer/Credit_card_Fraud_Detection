import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)

# Optional SHAP
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False


# 1. Load Artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("Models/scaler.pkl")
    feature_names = joblib.load("Models/feature_names.pkl")
    default_model = joblib.load("Models/random_forest_smote_fraud_model.pkl")  # update if needed
    return scaler, feature_names, default_model

scaler, feature_names, model = load_artifacts()


# 2. Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Predict Fraud", "Compare Models"])


# 3. Home Page
if page == "Home":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("""
    This app detects fraudulent credit card transactions using machine learning.  
    
    **Features:**
    - Exploratory Data Analysis (EDA)  
    - Fraud prediction for single transactions or CSV batch  
    - Model explainability (SHAP)  
    - Model comparison (LogReg, RF, XGBoost)  
    - Adjustable fraud threshold slider
    """)


# 4. EDA Page
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    uploaded_file = st.file_uploader("Upload Dataset (creditcard.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Dataset Summary
        st.subheader("📋 Dataset Summary")
        st.write(f"**Total Transactions:** {len(df):,}")
        st.write(f"**Fraudulent Transactions:** {df['Class'].sum():,} ({(df['Class'].mean()*100):.3f}%)")
        st.write(f"**Non-Fraudulent Transactions:** {len(df) - df['Class'].sum():,}")
        st.write("**Descriptive Statistics (Amount column):**")
        st.dataframe(df.groupby("Class")["Amount"].describe().T)

        # Class Distribution
        fig, ax = plt.subplots()
        sns.countplot(x="Class", data=df, ax=ax)
        ax.set_title("Fraud (1) vs Non-Fraud (0) Distribution")
        st.pyplot(fig)

        # Pie Chart
        fig, ax = plt.subplots()
        df["Class"].value_counts().plot.pie(
            autopct='%1.2f%%', labels=["Non-Fraud","Fraud"], colors=["skyblue","red"], ax=ax
        )
        ax.set_ylabel("")
        ax.set_title("Fraud vs Non-Fraud (Pie Chart)")
        st.pyplot(fig)

        # Amount Distribution
        fig, ax = plt.subplots()
        sns.histplot(df["Amount"].dropna(), bins=100, kde=True, ax=ax)
        ax.set_title("Transaction Amount Distribution")
        st.pyplot(fig)

        # Log Scale Amount
        fig, ax = plt.subplots()
        sns.histplot(df["Amount"].dropna(), bins=100, log_scale=True, kde=True, ax=ax)
        ax.set_title("Transaction Amount Distribution (Log Scale)")
        st.pyplot(fig)

        # Amount by Class
        fig, ax = plt.subplots()
        sns.histplot(df[df["Class"]==0]["Amount"].dropna(), bins=50, color="blue", label="Non-Fraud", alpha=0.6, ax=ax)
        sns.histplot(df[df["Class"]==1]["Amount"].dropna(), bins=50, color="red", label="Fraud", alpha=0.6, ax=ax)
        ax.legend()
        ax.set_title("Amount Distribution by Class")
        st.pyplot(fig)

        # Time Distribution
        fig, ax = plt.subplots()
        sns.histplot(df[df["Class"]==0]["Time"].dropna(), bins=50, color="blue", label="Non-Fraud", alpha=0.6, ax=ax)
        sns.histplot(df[df["Class"]==1]["Time"].dropna(), bins=50, color="red", label="Fraud", alpha=0.6, ax=ax)
        ax.legend()
        ax.set_title("Transaction Time Distribution by Class")
        ax.set_xlabel("Time (seconds since first transaction)")
        st.pyplot(fig)

        # PCA Features
        fig, ax = plt.subplots(figsize=(10,6))
        for col in ["V1","V2","V3"]:
            sns.kdeplot(df[df["Class"]==0][col].dropna(), label=f"{col} Non-Fraud", fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(df[df["Class"]==1][col].dropna(), label=f"{col} Fraud", fill=True, alpha=0.3, ax=ax)
        ax.set_title("Distribution of PCA Features (V1–V3) by Class")
        ax.legend()
        st.pyplot(fig)

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10,6))
        corr = df.corr(numeric_only=True).fillna(0)
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.info("Please upload the dataset to see EDA.")


# 5. Prediction Page

elif page == "Predict Fraud":
    st.title("🤖 Fraud Prediction")

    option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

    # Manual entry
    if option == "Manual Entry":
        st.subheader("Enter Transaction Details")

        input_data = {}
        for col in feature_names:
            input_data[col] = st.number_input(f"{col}", value=0.0)

        threshold = st.slider("Set Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

        if st.button("Predict"):
            df_input = pd.DataFrame([input_data])
            df_input[["Time", "Amount"]] = scaler.transform(df_input[["Time", "Amount"]])

            prob = model.predict_proba(df_input)[0][1]
            pred = 1 if prob >= threshold else 0

            st.write(f"### Prediction: {'🚨 Fraud' if pred==1 else 'Not Fraud'}")
            st.write(f"Fraud Probability: **{prob:.4f}** (Threshold = {threshold})")

            if shap_available:
                explainer = shap.Explainer(model, df_input)
                shap_values = explainer(df_input)
                st.subheader("🔎 SHAP Explanation")

                try:
                    shap_single = shap_values[0, :, 1] if shap_values.values.ndim == 3 else shap_values[0]
                    st.pyplot(shap.plots.waterfall(shap_single, show=False))
                except Exception as e:
                    st.error(f"SHAP visualization error: {e}")

    # CSV upload
    elif option == "Upload CSV":
        st.subheader("Upload CSV for Batch Prediction")
        csv_file = st.file_uploader("Upload file", type=["csv"])

        if csv_file:
            df_input = pd.read_csv(csv_file)

            # Drop target column if present
            if "Class" in df_input.columns:
                df_input = df_input.drop(columns=["Class"])

            missing_cols = [c for c in feature_names if c not in df_input.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                df_input[["Time", "Amount"]] = scaler.transform(df_input[["Time", "Amount"]])
                probs = model.predict_proba(df_input)[:,1]

                # Threshold slider
                threshold = st.slider("Set Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)
                preds = (probs >= threshold).astype(int)

                df_input["Fraud_Pred"] = preds
                df_input["Fraud_Prob"] = probs

                st.write(f"### Predictions (Threshold = {threshold})")
                st.dataframe(df_input.head())

                csv_out = df_input.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions", data=csv_out, file_name="fraud_predictions.csv", mime="text/csv")


# 6. Compare Models Page

elif page == "Compare Models":
    st.title("📈 Compare Models (ROC & PR Curves)")

    uploaded_file = st.file_uploader("Upload Test Dataset (with 'Class' column)", type=["csv"])
    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)

        # Ensure features exist
        missing_cols = [c for c in feature_names if c not in df_test.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            # Keep y_test and drop 'Class' from features
            y_test = df_test["Class"]
            X_test = df_test.drop(columns=["Class"]).copy()

            X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

            # Load all models
            model_files = glob.glob("Models/*_fraud_model.pkl")
            metrics = []

            fig_roc, ax_roc = plt.subplots()
            fig_pr, ax_pr = plt.subplots()

            for mf in model_files:
                clf = joblib.load(mf)
                name = mf.replace("_fraud_model.pkl", "")

                probs = clf.predict_proba(X_test)[:,1]
                preds = clf.predict(X_test)

                auc = roc_auc_score(y_test, probs)
                pr_auc = average_precision_score(y_test, probs)

                fpr, tpr, _ = roc_curve(y_test, probs)
                prec, rec, _ = precision_recall_curve(y_test, probs)

                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
                ax_pr.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")

                report = classification_report(y_test, preds, output_dict=True)
                metrics.append({
                    "Model": name,
                    "ROC-AUC": auc,
                    "PR-AUC": pr_auc,
                    "Precision": report["1"]["precision"],
                    "Recall": report["1"]["recall"],
                    "F1-score": report["1"]["f1-score"]
                })

            # Plot ROC Curve
            ax_roc.plot([0,1],[0,1],'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curves")
            ax_roc.legend()
            st.pyplot(fig_roc)

            # Plot Precision-Recall Curve
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall Curves")
            ax_pr.legend()
            st.pyplot(fig_pr)

            # Show metrics table
            st.write("### Metrics Comparison")
            st.dataframe(pd.DataFrame(metrics).round(3))
    else:
        st.info("Please upload a labeled test dataset for model comparison.")
