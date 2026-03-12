from sqlalchemy import create_engine
from config.settings import DATABASE_URL
import pandas as pd

class DatabaseManager:
    def __init__(self):
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL is missing in .env")
        
        # FIX: Added pool_pre_ping=True and sslmode=require
        # This forces Python to check if Neon closed the connection, and automatically
        # reconnects instead of crashing with an "EOF detected" error.
        self.engine = create_engine(
            DATABASE_URL, 
            pool_pre_ping=True, 
            connect_args={'sslmode': 'require'}
        )

    def load_data(self, query: str) -> pd.DataFrame:
        """Loads data safely into Pandas."""
        return pd.read_sql(query, self.engine)

    def save_metric(self, table_name: str, data: dict):
        """Saves a single row dictionary to the database."""
        df = pd.DataFrame([data])
        df.to_sql(table_name, self.engine, if_exists='append', index=False)

# Singleton instance
db = DatabaseManager()