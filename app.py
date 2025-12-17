import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_manager import load_data, load_model, preprocess_missing_values, delete_invalid_ratings, preprocess_duplicate, merge_data, build_tfidf    
import altair as alt
import pickle
import os
import joblib

st.set_page_config(page_title="Anime Analytics Dashboard", layout="wide")

# ============================
# 1. LOAD DATA
# ============================
rating, anime = load_data()
model = load_model()
# ============================
# 1. HEADER
# ============================
st.title("🎌 Anime Analytics Dashboard")
st.caption("✨ Phân tích, trực quan hóa và gợi ý anime dựa trên dữ liệu người dùng")
# ============================
# 2. LÀM SẠCH DỮ LIỆU
# ============================
st.header("🛠️ Làm sạch và chuẩn bị dữ liệu")
st.subheader("1. Missing Values")
colA, colB = st.columns(2)
with colA:
    st.subheader("🔍 Thiếu dữ liệu - Anime")
    missing_anime = anime.isna().sum()
    missing_anime = pd.DataFrame({"Tên cột": anime.columns, "Số lượng thiếu": missing_anime.values})
    st.dataframe(missing_anime, width="stretch")
with colB:
    st.subheader("🔍 Thiếu dữ liệu - Rating")
    missing_rating = rating.isna().sum()
    missing_rating = pd.DataFrame({"Tên cột": rating.columns, "Số lượng thiếu": missing_rating.values})
    st.dataframe(missing_rating, width="stretch")
## Xử lý dữ liệu
anime = preprocess_missing_values(anime)
after_missing = pd.DataFrame({"Tên cột": anime.columns, "Số lượng thiếu": anime.isna().sum().values})

st.subheader("⚙️ Sau khi xử lý Missing values")
st.dataframe(after_missing, width="stretch")

# #lọai bỏ dữ liệu trùng lặp
st.subheader("2. Loại bỏ dữ liệu trùng lặp")
before_dup_anime = len(anime)
anime_clean,rating_clean = preprocess_duplicate(anime,rating)
after_dup_anime = len(anime_clean)
st.success(f"✔ Đã loại {before_dup_anime - after_dup_anime} dòng trùng trong anime.")

# # Invalid Ratings
st.subheader("3. Chuẩn hóa dữ liệu  Rating")
st.write("Chuấn hóa Item-based mean centering trên dữ liệu Rating")
pivot_sparse = model["pivot_sparse"]   # csr_matrix (item × user)
item_names   = model["item_names"]     # list anime names
user_index   = model["user_index"]     # dict: user_id -> col index
knn          = model.get("knn", None)  # nếu bạn có lưu knn
st.success(f"✔ Dữ liệu Rating đã được chuẩn hóa.")
user_ids = list(user_index.keys())

rows, cols = pivot_sparse.nonzero()

if len(rows) > 0:
    sample_size = min(30, len(rows))
    sample_idx = np.random.choice(len(rows), sample_size, replace=False)

    nnz_data = {
        "Anime": [item_names[rows[i]] for i in sample_idx],
        "User":  [user_ids[cols[i]] for i in sample_idx],
        "Rating": pivot_sparse.data[sample_idx],
    }

    df_nnz = pd.DataFrame(nnz_data)
    st.dataframe(df_nnz, use_container_width=True)
else:
    st.warning("Sparse matrix has no non-zero entries!")

st.divider()


st.subheader("4. Vector hóa dữ liệu IF-IDF")
# # Tạo văn bản kết hợp (genre + type)
# # TF-IDF vectorizer
tfidf, tfidf_matrix = build_tfidf(anime_clean)
sample_tfidf = pd.DataFrame(
    tfidf_matrix[:10, :20].toarray(),
    columns=tfidf.get_feature_names_out()[:20],
    index=anime_clean["name"][:10]
)
st.dataframe(sample_tfidf)
# # # ============================
# # 3. GỘP DỮ LIỆU
# # ============================

# merged = merge_data(rating_clean, anime_clean)
# st.dataframe(merged.head(), width="stretch")

# # ============================
# # 4. DASHBOARD
# # ============================
st.header("📊 Phân tích & Trực quan hóa")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Phân bố Rating",
    "🏆 Top Anime",
    "🎭 Genre và Type",
    "🔥 Heatmap"
])

# ============================
# TAB 1: PHÂN BỐ RATING
# ============================
with tab1:
    st.subheader("📈 Histogram phân bố Rating anime")
    #histogram phân bố rating anime
    chart = (
        alt.Chart(anime_clean)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            alt.X("rating:Q", bin=alt.Bin(maxbins=30), title="Rating"),
            alt.Y("count():Q", title="Số lượng Anime"),
            tooltip=[alt.Tooltip("count():Q", title="Số lượng Anime")]
        )
        .properties(width="container", height=400)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    st.altair_chart(chart, width="stretch")
    
    
    

# # ============================
# # TAB 2: TOP ANIME
# # ============================
with tab2:
    st.subheader("🏆 Top Anime theo Rating trung bình")

    top_n = st.slider("Chọn số lượng top:", 5, 30, 15)

    top_anime = (
        anime_clean.sort_values("rating", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    order = top_anime["name"].tolist()
    st.dataframe(top_anime, width="stretch")
    
#     # Mỗi bar một màu
    top_anime["color_id"] = top_anime.index.astype(str)
    order = top_anime["name"].tolist()
    # Biểu đồ chính
    bars = (
    alt.Chart(top_anime)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("name:N", sort=order),
        y=alt.Y("rating:Q"),
        color=alt.Color("color_id:N", legend=None)
    )
)

    text = (
        alt.Chart(top_anime)
        .mark_text(align="center", baseline="bottom", dy=-4)
        .encode(
            x=alt.X("name:N", sort=order),
            y="rating:Q",
            text="rating:Q"
        )
    )

    # Layer + config
    final_chart = (
        (bars + text)
        .properties(width="container", height=450)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    st.altair_chart(final_chart, width="stretch")
    
    
    st.subheader("🏅 Top Anime theo Số lượng thành viên")
    top_members = (
        anime_clean.sort_values("members", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    st.dataframe(top_members, width="stretch")
    # Mỗi bar một màu
    top_members["color_id"] = top_members.index.astype(str)
    order_members = top_members["name"].tolist()
    # Biểu đồ chính
    bars_members = (
    alt.Chart(top_members)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("name:N", sort=order_members),
        y=alt.Y("members:Q"),
        color=alt.Color("color_id:N", legend=None)
    )
    )
    text_members = (
        alt.Chart(top_members)
        .mark_text(align="center", baseline="bottom", dy=-4)
        .encode(
            x=alt.X("name:N", sort=order_members),
            y="members:Q",
            text="members:Q"
        )
    )
    # Layer + config
    final_chart_members = (
        (bars_members + text_members)
        .properties(width="container", height=450)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    st.altair_chart(final_chart_members, width="stretch")
    

# # ============================
# # TAB 3: PHÂN TÍCH GENRE
# # ============================
with tab3:
    st.subheader("🎭 Phân tích type Anime")
    type_count = anime_clean["type"].value_counts().reset_index()
    type_count.columns = ["type", "count"]
    type_row = type_count.set_index("type").T
    st.dataframe(type_row, width="stretch")
    st.subheader("📊 Phân bố type Anime")
    #pie chart hiện thị tỉ lệ từng type
    chart_pie = (
        alt.Chart(type_count)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(field="type", type="nominal"),
            tooltip=[alt.Tooltip("type:N", title="Type"), alt.Tooltip("count:Q", title="Số lượng")]
        )
        .properties(width="container", height=400)
    )


    st.altair_chart(chart_pie, width="stretch")
    st.subheader("📊 Phân bố type Anime (Altair Bar Chart)")
    chart_bar_type = (
        alt.Chart(type_count)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("type:N", sort="-y", title="Type"),
            y=alt.Y("count:Q", title="Tần suất"),
            color=alt.Color("type:N", legend=None)
        )
        .properties(width="container", height=400)
    )
    text = (
        alt.Chart(type_count)
        .mark_text(align="center", baseline="bottom", dy=-4)
        .encode(
            x=alt.X("type:N", sort="-y"),
            y="count:Q",
            text="count:Q"
        )
    )
    chart_bar_type = chart_bar_type + text
    st.altair_chart(chart_bar_type, width="stretch")
    st.subheader("🎭 Tần suất thể loại Anime")
    # Tách từng genre
    genre_exploded = anime_clean["genre"].dropna().str.split(", ").explode()
    # Đếm tần suất
    genre_count = genre_exploded.value_counts().reset_index()
    genre_count.columns = ["genre", "count"]
    # Chuyển thành format hàng ngang
    genre_row = genre_count.set_index("genre").T

    st.dataframe(genre_row, width="stretch")
    st.subheader("📊 Phân bố thể loại Anime (Altair Bar Chart)")

    chart_bar = (
        alt.Chart(genre_count)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("genre:N", sort="-y", title="Thể loại"),
            y=alt.Y("count:Q", title="Tần suất"),
            color=alt.Color("genre:N", legend=None)
        )
        .properties(width="container", height=400)
    )

    st.altair_chart(chart_bar, width="stretch")
    st.subheader("☁️ WordCloud thể loại Anime")
    # Tạo WordCloud
    genre_text = " ".join(genre_exploded.tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(genre_text)
    # Hiển thị WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    


# # ============================
# # TAB 4: HEATMAP
# # ============================
@st.cache_data(show_spinner=False)
def load_joined_data():
    return (
        pd.read_parquet("joined.parquet")
        .rename(columns={
            "rating_x": "user_rating",
            "rating_y": "anime_avg_rating",
            "name": "anime_name"
        })
    )
df_corr = load_joined_data()[
    ["user_rating", "anime_avg_rating", "members"]
]

# df_corr = (
#     df_corr
#     .replace(-1, np.nan)
#     .dropna()
#     .sample(n=min(30_000, len(df_corr)), random_state=42)
# )
# with tab4:
#     joined = load_joined_data()

#     df_corr = (
#         joined[["user_rating", "anime_avg_rating", "members"]]
#         .replace(-1, np.nan)
#         .dropna()
#         .sample(n=min(30_000, len(joined)), random_state=42)
#     )

#     corr = df_corr.corr().reset_index().melt(
#         id_vars="index",
#         var_name="Variable",
#         value_name="Correlation"
#     ).rename(columns={"index": "Feature"})

#     st.subheader("🔥 Ma trận tương quan dữ liệu")
#     left, center, right = st.columns([1, 3, 1])

#     with center:
#         heatmap = (
#             alt.Chart(corr)
#             .mark_rect(cornerRadius=6)
#             .encode(
#                 x=alt.X("Variable:N", title=None),
#                 y=alt.Y("Feature:N", title=None),
#                 color=alt.Color(
#                     "Correlation:Q",
#                     scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
#                     legend=alt.Legend(title="Corr")
#                 ),
#                 tooltip=[
#                     alt.Tooltip("Feature:N"),
#                     alt.Tooltip("Variable:N"),
#                     alt.Tooltip("Correlation:Q", format=".2f")
#                 ]
#             )
#             .properties(width=700, height=600)
#         )

#         text = (
#             alt.Chart(corr)
#             .mark_text(fontSize=30, fontWeight="bolder")
#             .encode(
#                 x="Variable:N",
#                 y="Feature:N",
#                 text=alt.Text("Correlation:Q", format=".2f"),
#                 color=alt.condition(
#                     "datum.Correlation > 0.5 || datum.Correlation < -0.5",
#                     alt.value("white"),
#                     alt.value("black")
#                 )
#             )
#         )

#         st.altair_chart(heatmap + text, use_container_width=False)


# with tab4:
#     joined = load_joined_data()
#     corr = joined[["user_rating", "anime_avg_rating", "members"]].corr()
#     st.subheader("🔥 Heatmap ma trận tương quan")
#     fig, ax = plt.subplots(figsize=(5, 3))
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     st.pyplot(fig)
