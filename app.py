import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report

st.title("Machine Learning Classification App")

st.write("Upload test dataset and choose a model to see predictions.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Model selection
model_map = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

model_choice = st.selectbox("Select Model", list(model_map.keys()))

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Dataset must contain 'target' column")
    else:
        X = df.drop("target", axis=1)
        y = df["target"]

        model_path = model_map[model_choice]

        if os.path.exists(model_path):
            model = pickle.load(open(model_path, "rb"))

            preds = model.predict(X)

            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y, preds))

            st.subheader("Classification Report")
            st.text(classification_report(y, preds))

            st.success("Prediction complete")
        else:
            st.error("Model file not found. Train models first.")
