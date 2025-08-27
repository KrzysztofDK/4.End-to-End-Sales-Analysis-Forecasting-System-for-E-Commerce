"""Module to create front of streamlit app."""

import sys
import os
import math
import re
from typing import List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gdown
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

BERT_URL = "https://drive.google.com/uc?id=1ML2YqBtSgmZOG8Iq4VvJYOur4wiuyGtT"
BERT_FILE = os.path.join("models", "sentiment_bert_model.pt")
if not os.path.exists(BERT_FILE):
    gdown.download(BERT_URL, BERT_FILE, quiet=False)

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
    state_dict = torch.load(BERT_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    return model, tokenizer


def extract_categories(feature_names: List[str], prefix: str) -> List[str]:
    """Return sorted unique categories for given one-hot prefix like 'customer_state'."""
    p = f"{prefix}_"
    vals = sorted({f[len(p) :] for f in feature_names if f.startswith(p)})
    return vals


_interval_re = re.compile(r"^\s*([\[\(])\s*(\d+)\s*[,;-]\s*(\d+)\s*([\]\)])\s*$")
_range_re = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$")
_single_re = re.compile(r"^\s*(\d+)\s*$")
_plus_re = re.compile(r"^\s*(\d+)\s*\+\s*$")
_ge_re = re.compile(r"^\s*[≥>=]\s*(\d+)\s*$")


def parse_bin_label(label: str) -> Optional[Tuple[float, float, bool, bool]]:
    """
    Parse bin label into (low, high, inc_low, inc_high).
    Supports: [a,b), (a,b], a-b, 'n', 'n+', '>=n'.
    Returns None if cannot parse.
    """
    s = str(label).strip()

    m = _interval_re.match(s)
    if m:
        left_br, lo, hi, right_br = m.groups()
        lo, hi = float(lo), float(hi)
        inc_lo = left_br == "["
        inc_hi = right_br == "]"
        return lo, hi, inc_lo, inc_hi

    m = _range_re.match(s)
    if m:
        lo, hi = map(float, m.groups())
        return lo, hi, True, True

    m = _single_re.match(s)
    if m:
        n = float(m.group(1))
        return n, n, True, True

    m = _plus_re.match(s) or _ge_re.match(s)
    if m:
        n = float(m.group(1))
        return n, float("inf"), True, True

    return None


def value_in_interval(
    x: float, lo: float, hi: float, inc_lo: bool, inc_hi: bool
) -> bool:
    if inc_lo:
        left_ok = x >= lo
    else:
        left_ok = x > lo
    if inc_hi:
        right_ok = x <= hi
    else:
        right_ok = x < hi
    return left_ok and right_ok


def pick_installments_bin(installments: int, bin_labels: List[str]) -> Optional[str]:
    """
    Given numeric installments and available 'installments_bins_*' labels,
    pick the label whose interval contains the value.
    """
    candidates = []
    for lab in bin_labels:
        parsed = parse_bin_label(lab)
        if parsed is None:
            continue
        lo, hi, inc_lo, inc_hi = parsed
        if value_in_interval(float(installments), lo, hi, inc_lo, inc_hi):
            width = hi - lo if hi != float("inf") else 1e9
            candidates.append((width, lab))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


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

    payment_types = extract_categories(feature_names, "payment_type")
    states = extract_categories(feature_names, "customer_state")
    categories = extract_categories(feature_names, "top20_categories_and_others")
    installments_bin_labels = extract_categories(feature_names, "installments_bins")

    if not payment_types:
        payment_types = ["boleto", "credit_card", "debit_card", "voucher"]
    if not states:
        st.warning("No states found in features. Provide a manual state code.")
        states = ["others"]
    if not categories:
        st.warning("No categories found in features. Falling back to 'others'.")
        categories = ["others"]

    st.write("Fill in the form below:")

    with st.form("ann_input_form"):
        n_distinct_categories = st.number_input(
            "Number of distinct categories", min_value=0, step=1
        )
        n_sellers = st.number_input("Number of sellers", min_value=0, step=1)
        n_items = st.number_input("Number of items", min_value=0, step=1)
        payment_value = st.number_input("Payment value", min_value=0.0, format="%.2f")
        total_freight_value = st.number_input(
            "Total freight value", min_value=0.0, format="%.2f"
        )

        dow = st.selectbox("Day of week (0=Mon ... 6=Sun)", list(range(7)))
        month = st.selectbox("Month (1-12)", list(range(1, 13)))

        payment_type = st.selectbox("Payment type", payment_types)
        state = st.selectbox("State", states)
        category = st.selectbox("Category", categories)

        installments_val = st.number_input(
            "Installments (0–24)", min_value=0, max_value=24, step=1
        )

        submitted = st.form_submit_button("Predict with ANN")

    if submitted:
        try:
            x = {feat: 0.0 for feat in feature_names}

            x["n_distinct_categories"] = float(n_distinct_categories)
            x["n_sellers"] = float(n_sellers)
            x["n_items"] = float(n_items)
            x["payment_value"] = float(payment_value)
            x["total_freight_value"] = float(total_freight_value)

            x["dow_sin"] = math.sin(2 * math.pi * dow / 7)
            x["dow_cos"] = math.cos(2 * math.pi * dow / 7)
            x["month_sin"] = math.sin(2 * math.pi * month / 12)
            x["month_cos"] = math.cos(2 * math.pi * month / 12)

            pt_feat = f"payment_type_{payment_type}"
            if pt_feat in x:
                x[pt_feat] = 1.0
            else:
                st.info(
                    f"Payment type '{payment_type}' not found in features; skipping one-hot."
                )

            st_feat = f"customer_state_{state}"
            if st_feat in x:
                x[st_feat] = 1.0
            else:
                if f"customer_state_others" in x:
                    x["customer_state_others"] = 1.0
                else:
                    st.info(f"State '{state}' not found in features; skipping one-hot.")

            cat_feat = f"top20_categories_and_others_{category}"
            if cat_feat in x:
                x[cat_feat] = 1.0
            else:
                if f"top20_categories_and_others_others" in x:
                    x["top20_categories_and_others_others"] = 1.0
                else:
                    st.info(
                        f"Category '{category}' not found in features; skipping one-hot."
                    )

            chosen_bin = None
            if installments_bin_labels:
                chosen_bin = pick_installments_bin(
                    int(installments_val), installments_bin_labels
                )
                if chosen_bin is None:
                    if str(installments_val) in installments_bin_labels:
                        chosen_bin = str(installments_val)

            if chosen_bin:
                inst_feat = f"installments_bins_{chosen_bin}"
                if inst_feat in x:
                    x[inst_feat] = 1.0
                else:
                    st.info(
                        f"Installments bin '{chosen_bin}' not found as feature; skipping one-hot."
                    )
            else:
                st.info(
                    "Could not match installments value to any bin; skipping one-hot."
                )

            x_arr = np.array([[x[feat] for feat in feature_names]])

            model = load_ann()
            pred = model.predict(x_arr).ravel()[0]

            st.success(
                f"Likelihood that a user will purchase again within 90 days: **{pred*100:.2f}%**"
            )
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
