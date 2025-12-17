import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
import os
import requests

# hàm tải dữ liệu
@st.cache_data
def load_data():
    anime = pd.read_parquet("anime.parquet")
    rating = pd.read_parquet("rating.parquet")
    return rating, anime
# hàm tiền xử lý dữ liệu anime

@st.cache_resource
def load_model():
    return joblib.load("item_knn_model.joblib")
    

@st.cache_data
def preprocess_missing_values(anime):
    # Loại bỏ rating bị NaN và tạo bản copy an toàn
    anime = anime[~np.isnan(anime["rating"])].copy()
    # Sửa Missing bằng .loc để tránh SettingWithCopyWarning
    anime.loc[:, "genre"] = anime["genre"].fillna(anime["genre"].mode()[0])
    anime.loc[:, "type"] = anime["type"].fillna(anime["type"].mode()[0])
    # Tạo trường combined
    anime.loc[:, "combined"] = (
        anime["genre"].str.replace(",", " ", regex=False) + " " + anime["type"]
    )
    return anime

# # hàm loại bỏ đánh giá không hợp lệ
@st.cache_data
def delete_invalid_ratings(rating):
    rating_clean = rating[rating["rating"] != -1]
    return rating_clean

# # hàm loại bỏ dữ liệu trùng lặp
@st.cache_data
def preprocess_duplicate(anime,rating):
    anime_clean = anime.drop_duplicates(subset=["anime_id"])
    rating_clean = rating.drop_duplicates(subset=["user_id","anime_id"], keep="last")
    return anime_clean, rating_clean

# # hàm hợp dữ liệu rating và anime
@st.cache_data
def merge_data(rating_clean, anime_clean):
    merged = rating_clean.merge(anime_clean, on="anime_id", how="inner")
    merged = merged.rename(columns={
        "rating_x": "user_rating",
        "rating_y": "anime_avg_rating",
        "name": "anime_name",
        "genre": "anime_genre"
    })
    return merged

# # hàm xây dựng TF-IDF và ma trận cosine similarity  
@st.cache_resource
def build_tfidf(anime_cb):
    anime_cb = anime_cb.copy()
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(anime_cb["combined"])
    return tfidf, tfidf_matrix

# 
# @st.cache_resource
# def build_cf_model(rating):
#     # Chuẩn bị dữ liệu cho Collaborative Filtering
#     Y_data = rating[["user_id", "anime_id", "user_rating"]].to_numpy()
#     cf_model = CF(Y_data, k=20)
#     cf_model.fit()
#     return cf_model
# Hàm lấy ảnh từ tên anime qua Jikan API
def get_anime_image(anime_name):
    url = f"https://api.jikan.moe/v4/anime?q={anime_name}&limit=1"
    try:
        response = requests.get(url)
        data = response.json()
        if data['data']:
            return data['data'][0]['images']['jpg']['image_url']
    except:
        return None