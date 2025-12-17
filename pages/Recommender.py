import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import load_npz

# =========================
# LOAD MODEL (1 lần)
# =========================

@st.cache_resource
def load_item_sim():
    item_sim = load_npz("item_sim_top100.npz")
    return item_sim

@st.cache_resource
def load_model():
    model = joblib.load("item_knn_model.joblib")
    return model
# =========================
model = load_model()
user_index      = model["user_index"]
item_names      = model["item_names"]
pivot_sparse    = model["pivot_sparse"]
# =========================
similar_items = load_item_sim()


def recommend_for_user_norm(
    user_id,
    pivot_sparse,
    item_sim,
    user_index,
    item_names,
    top_n=10
):
    if user_id not in user_index:
        return []

    user_col = user_index[user_id]
    rated_items = pivot_sparse[:, user_col].nonzero()[0]
    if len(rated_items) == 0:
        return []

    ratings = pivot_sparse[rated_items, user_col].toarray().ravel()

    # Tử số: weighted sum
    numer = item_sim[:, rated_items] @ ratings

    # Mẫu số: tổng |sim|
    denom = np.abs(item_sim[:, rated_items]).sum(axis=1).A1 \
            if hasattr(item_sim[:, rated_items], "A1") else \
            np.asarray(np.abs(item_sim[:, rated_items]).sum(axis=1)).ravel()

    # Tránh chia cho 0
    scores = np.divide(
        numer.toarray().ravel() if hasattr(numer, "toarray") else numer,
        denom,
        out=np.zeros_like(denom),
        where=denom != 0
    )

    # Không recommend item đã xem
    scores[rated_items] = -np.inf

    # Top-N
    top_idx = np.argsort(scores)[::-1][:top_n]

    return [item_names[i] for i in top_idx]
res = recommend_for_user_norm(
    user_id=6,
    pivot_sparse=pivot_sparse,
    item_sim=similar_items,
    user_index=user_index,
    item_names=item_names,
    top_n=10
)
st.write(res)

    
