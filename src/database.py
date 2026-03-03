from sqlalchemy import create_engine
from config.settings import DATABASE_URL
import pandas as pd

class DatabaseManager:
    def __init__(self):
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL is missing in .env")
        self.engine = create_engine(DATABASE_URL)

    def load_data(self, query: str) -> pd.DataFrame:
        """Loads data safely into Pandas."""
        return pd.read_sql(query, self.engine)

    def save_metric(self, table_name: str, data: dict):
        """Saves a single row dictionary to the database."""
        df = pd.DataFrame([data])
        df.to_sql(table_name, self.engine, if_exists='append', index=False)

# Singleton instance
db = DatabaseManager()