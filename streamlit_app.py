import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df = pd.read_csv("data/creditcard.csv")
st.title("Credit Card Fraud Detection System")


feature_names = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]
##Input Fields
st.subheader("Enter Transaction Features")

col1, col2, col3 = st.columns(3)

inputs = []

for i, feature in enumerate(feature_names):

    if i % 3 == 0:
        value = col1.number_input(f"{feature}", value=0.0, key=feature)
    elif i % 3 == 1:
        value = col2.number_input(f"{feature}", value=0.0, key=feature)
    else:
        value = col3.number_input(f"{feature}", value=0.0, key=feature)

    inputs.append(value)
# Prediction
if st.button("Predict Transaction"):

    data = np.array(inputs).reshape(1, -1)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    prob = model.predict_proba(data_scaled)

    fraud_prob = prob[0][1]
    legit_prob = prob[0][0]

    if prediction[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Legitimate Transaction")

    st.write("Fraud Probability:", round(fraud_prob, 4))
    st.write("Legitimate Probability:", round(legit_prob, 4))
    # Probability Graph
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots()
    labels = ["legitimate", "Fraud"]
    values = [legit_prob, fraud_prob]
    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Feature Importance Graph
    if hasattr(model, "feature_importances_"):

        st.subheader("Top Feature Importance")

        importance = model.feature_importances_

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()

        ax.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])

        ax.set_xlabel("Importance")

        st.pyplot(fig)
    # Dataset Distribution
    st.subheader("Dataset Fraud Distribution")

    fig, ax = plt.subplots()

    df["Class"].value_counts().plot(kind="bar", ax=ax)

    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Count")

    st.pyplot(fig)
