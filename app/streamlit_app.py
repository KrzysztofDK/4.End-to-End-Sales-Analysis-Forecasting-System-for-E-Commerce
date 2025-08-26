"""Module to create front of streamlit app."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from keras.models import load_model

from src.components.model_creation import BertModel

ANN_PATH = "models/classification_ann_model.h5"
PROPHET_PATH = "models/forecasting_prophet_model.pkl"
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ann_feature_names.pkl")
BERT_PATH = "models/sentiment_bert_model.pt"


@st.cache_resource
def load_ann():
    return load_model(ANN_PATH)


@st.cache_resource
def load_prophet():
    return joblib.load(PROPHET_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)


@st.cache_resource
def load_bert():
    model = BertModel(num_labels=3)
    model.load_state_dict(torch.load(BERT_PATH, map_location="cpu"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    return model, tokenizer


st.title("ML Model Playground")

model_choice = st.sidebar.selectbox(
    "Choose a model:",
    [
        "ANN (binary classification)",
        "Prophet (forecasting)",
        "BERTimbau (sentiment analysis)",
    ],
)

if model_choice == "ANN (binary classification)":
    st.header("Binary Classification - ANN")

    try:
        feature_names = load_features()
    except Exception as e:
        st.error(f"Could not load feature names: {e}")
        st.stop()

    st.write("Enter values for each feature:")

    inputs = {}
    with st.form("ann_input_form"):
        for feat in feature_names:
            inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")
        submitted = st.form_submit_button("Predict with ANN")

    if submitted:
        try:
            model = load_ann()
            x = np.array([[inputs[feat] for feat in feature_names]])
            pred = model.predict(x).ravel()[0]
            st.success(f"Probability of class 1: {pred:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")

elif model_choice == "Prophet (forecasting)":
    st.header("Time Series Forecasting - Prophet")
    st.write("Enter the number of days to forecast.")

    periods = st.number_input("Days ahead", min_value=1, max_value=365, value=30)
    if st.button("Forecast with Prophet"):
        try:
            model = load_prophet()
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            history_end = model.history["ds"].max()
            history = forecast[forecast["ds"] <= history_end]
            future_forecast = forecast[forecast["ds"] > history_end]

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.scatter(
                model.history["ds"],
                model.history["y"],
                color="black",
                s=10,
                label="Actual",
            )

            ax.plot(
                history["ds"],
                history["yhat"],
                color="blue",
                label="History",
            )

            ax.plot(
                future_forecast["ds"],
                future_forecast["yhat"],
                color="red",
                label="Forecast",
            )

            ax.legend()
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

elif model_choice == "BERTimbau (sentiment analysis)":
    st.header("Sentiment Analysis - BERTimbau")
    text = st.text_area("Enter Portuguese text", "Este produto é ótimo!")

    if st.button("Analyze sentiment"):
        try:
            model, tokenizer = load_bert()
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                labels = ["negative", "neutral", "positive"]
                result = {labels[i]: float(probs[i]) for i in range(len(labels))}
            st.json(result)
        except Exception as e:
            st.error(f"Error: {e}")
