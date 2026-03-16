import os
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv

# Absolute path to the .env file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

class DatabaseManager:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL")
        
        # Debugging: This will tell us if the file is even being read
        if not db_url:
            print(f"\n[DEBUG] Checking file: {ENV_PATH}")
            if os.path.exists(ENV_PATH):
                print("[DEBUG] File exists, but DATABASE_URL is missing or empty inside it.")
            else:
                print("[DEBUG] The .env file does not exist at this path.")
            raise ValueError("DATABASE_URL not found.")

        # NeonDB requires the 'postgresql' prefix, not 'postgres'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)

        self.engine = create_engine(
            db_url, 
            pool_pre_ping=True, 
            connect_args={'sslmode': 'require'}
        )

    def load_data(self, query: str) -> pd.DataFrame:
        return pd.read_sql(query, self.engine)

    def save_metric(self, table_name: str, data: dict):
        df = pd.DataFrame([data])
        df.to_sql(table_name, self.engine, if_exists='append', index=False)

db = DatabaseManager()