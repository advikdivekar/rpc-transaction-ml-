import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
)

query = """
SELECT rpc_id, latency_ms
FROM rpc_metrics
WHERE failure_flag = 0
"""

df = pd.read_sql(query, engine)

print(df.groupby("rpc_id")["latency_ms"].mean())