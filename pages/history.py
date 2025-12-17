import streamlit as st
import pandas as pd
import os 
from datetime import datetime
from controller.LogController import log_action, load_log_file
st.title("History")
history_df = load_log_file()
st.subheader("📜 User Interaction History")
# Hiển thị file log
st.write("📄 Current Logs:")
st.dataframe(history_df.sort_values(by="timestamp", ascending=False))