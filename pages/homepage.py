import streamlit as st
import pandas as pd
from datetime import datetime
import os
import requests
st.title("🎌 Anime Streaming Platform")
from scipy.sparse import load_npz

from controller.LogController import log_action
from pages.Recommender import load_model, recommend_for_user_norm,load_item_sim
user_id = st.session_state.get("user_id", None)

# ==========================

# Nếu chưa login
if user_id is None:
    input_id = st.text_input("Nhập User ID để tiếp tục:", key="input_user_id")
    if st.button("Đăng nhập"):
        if input_id.strip() == "":
            st.error("User ID không được để trống.")
        else:
            # Lưu vào session state
            st.session_state["user_id"] = input_id.strip()
            # Rerun lại trang
            st.rerun()
    # Ngừng chạy toàn bộ các phần bên dưới
    st.stop()
# Sau khi login → lấy user_id từ session
user_id = st.session_state["user_id"]
# ==========================
# LOAD CLEAN DATA
# ==========================
ANIME_FILE = "anime_preprocessed.parquet"
LOG_FILE = "logs.csv"

if "page" not in st.session_state:
    st.session_state.page = 1
col1, col2 = st.columns([4, 1])  # col1 rộng hơn col2
with col1:
    st.header(f"👋 Xin chào, User {user_id} !")
with col2:
    if st.button("🚪 Đăng xuất"):
        st.session_state.clear()
        st.rerun()

anime = pd.read_parquet(ANIME_FILE)
st.subheader("🔥 Recommended For You")
model = load_model()
similar_items = load_item_sim()
recommendations = recommend_for_user_norm(
    user_id=int(user_id),
    pivot_sparse=model["pivot_sparse"],
    item_sim=similar_items,
    user_index=model["user_index"],
    item_names=model["item_names"],
    top_n=10
)
if len(recommendations) == 0:
    st.write("Chưa có đề xuất cho bạn. Hãy đánh giá một số anime để nhận đề xuất nhé!")
else:
    rec_anime = anime[anime["name"].isin(recommendations)]
    
    # Chia thành các chunks 5 item mỗi lần
    for i in range(0, len(rec_anime), 5):
        cols = st.columns(5)
        chunk = rec_anime.iloc[i:i+5]
        for j, (_, row) in enumerate(chunk.iterrows()):
            with cols[j]:
                st.markdown(
                f"""
                <div style="
                    border:1px solid #ccc; 
                    border-radius:10px; 
                    padding:10px; 
                    text-align:center; 
                    box-shadow: 2px 2px 5px #eee;
                    min-height: 210px; 
                    display:flex;
                    flex-direction:column;
                    justify-content:space-between;
                ">
                    <div>
                        <b>{row['name']}</b><br>
                        Rating: {row['rating']} ⭐<br>
                        Episodes: {row['episodes']}<br><br>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
                if st.button(
                        "▶ Xem phim",
                        key=f"btn_{row['anime_id']}",
                        use_container_width=True
                    ):
                        st.session_state.selected_movie = row["anime_id"]
                        st.rerun()
    










st.subheader("🎥 Phim anime mới cập nhập")

# ==========================
# STATE
# ==========================
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# ==========================
# UI – LIST VIEW
# ==========================

# Hàm lấy ảnh từ tên anime qua Jikan API

def show_movie_list():
    # Chọn số phim mỗi trang
    movies_per_page = st.selectbox(
        "Số phim mỗi trang:", [10, 20, 30, 40, 50], index=1
    )

    total_movies = len(anime)
    total_pages = (total_movies - 1) // movies_per_page + 1

    # Đảm bảo page nằm trong phạm vi
    current_page = st.session_state.get("page", 1)
    current_page = max(1, min(current_page, total_pages))
    st.session_state.page = current_page

    # Lấy data của trang hiện tại
    start = (current_page - 1) * movies_per_page
    end = start + movies_per_page
    current_movies = anime.iloc[start:end]

    # In trạng thái trang
    st.write(f"Trang {current_page}/{total_pages}")

    # Hiển thị dạng grid 5 cột
    cols = st.columns(5)
    for i, row in current_movies.iterrows():
        col = cols[i % 5]
        with col:
            st.markdown(
                f"""
                <div style="
                    border:1px solid #ccc; 
                    border-radius:10px; 
                    padding:10px; 
                    text-align:center; 
                    box-shadow: 2px 2px 5px #eee;
                    min-height: 250px; 
                    display:flex;
                    flex-direction:column;
                    justify-content:space-between;
                ">
                    <div>
                        <b>{row['name']}</b><br>
                        Rating: {row['rating']} ⭐<br>
                        Episodes: {row['episodes']}<br><br>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            if st.button(
                        "▶ Xem phim",
                        key=f"btn_{row['anime_id']}",
                        use_container_width=True
                    ):
                        st.session_state.selected_movie = row["anime_id"]
                        st.rerun()

    # ================================
    # 🚀 PAGINATION DẠNG SỐ
    # ================================
   
    st.write("---")
    st.subheader("Trang")

    pagination = st.container()
    with pagination:
        cols = st.columns(10)

        # First page <<
        if cols[0].button("⏮"):
            st.session_state.page = 1
            st.rerun()

        # Previous page <
        if cols[1].button("◀"):
            if current_page > 1:
                st.session_state.page -= 1
                st.rerun()

        # Hiển thị 5 trang xung quanh current
        page_range = 5
        start_page = max(1, current_page - page_range // 2)
        end_page = min(total_pages, start_page + page_range - 1)

        btn_index = 2
        for p in range(start_page, end_page + 1):
            if p == current_page:
                if cols[btn_index].button(f"[{p}]"):
                    pass  # không làm gì
            else:
                if cols[btn_index].button(str(p)):
                    st.session_state.page = p
                    st.rerun()
            btn_index += 1

        # Next page >
        if cols[7].button("▶"):
            if current_page < total_pages:
                st.session_state.page += 1
                st.rerun()

        # Last page >>
        if cols[8].button("⏭"):
            st.session_state.page = total_pages
            st.rerun()

# ==========================
# UI – WATCH PAGE
# ==========================
def show_movie_detail(anime_id):
    movie = anime[anime["anime_id"] == anime_id].iloc[0]

    st.title(f"🎬 {movie['name']}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write(f"**Thể loại:** {movie.get('genre', 'N/A')}")
        st.write(f"**Rating:** ⭐ {movie.get('rating', 'N/A')}")
        st.write(f"**Số tập:** {movie.get('episodes', 'N/A')}")

        # Các nút hành động
        if st.button("📺 Watch Now"):
            log_action(user_id, anime_id, "watch")
            st.success("Đã lưu vào lịch sử xem!")

        rating = st.radio(
            "⭐ Đánh giá anime",
            options=list(range(1, 11)),
            horizontal=True
        )
        if st.button("⭐ Rate"):
            log_action(user_id, anime_id, f"rate_{rating}")
            st.success(f"Đã đánh giá {rating} sao!")
          
      
        if st.button("❤️ Favorite"):
            log_action(user_id, anime_id, "favorite")
            st.success("Đã thêm vào danh sách yêu thích!")

        if st.button("👆 Click"):
            log_action(user_id, anime_id, "click")
            st.success("Đã ghi click!")

        if st.button("⬅️ Quay lại Danh sách"):
            st.session_state.selected_movie = None
            st.rerun()

    with col2:
        st.subheader("Mô tả phim")
        st.write(movie.get("description", "Chưa có mô tả cho anime này."))

       


# ==========================
# MAIN ROUTER
# ==========================
if st.session_state.selected_movie is None:
    show_movie_list()
else:
    show_movie_detail(st.session_state.selected_movie)
