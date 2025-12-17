import pandas as pd
from datetime import datetime
import os
import requests

LOG_FILE = "logs.csv"
def load_log_file():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["user_id", "anime_id", "action", "timestamp"]).to_csv(LOG_FILE, index=False)
    return pd.read_csv(LOG_FILE)
def log_action(user_id, anime_id, action):
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    new_row = pd.DataFrame([{
        "user_id": user_id,
        "anime_id": anime_id,
        "action": action,
        "timestamp": timestamp
    }])
    new_row.to_csv(LOG_FILE, mode="a", header=False, index=False)