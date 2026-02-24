from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
)

query = """
SELECT rpc_id, AVG(latency_ms) AS recent_avg
FROM rpc_metrics
WHERE failure_flag = 0
AND timestamp >= UTC_TIMESTAMP() - INTERVAL 2 MINUTE
GROUP BY rpc_id
ORDER BY recent_avg ASC
"""

df = pd.read_sql(query, engine)

if not df.empty:
    best_rpc = df.iloc[0]["rpc_id"]
    print("Best RPC right now:", best_rpc)
else:
    print("No recent data found.")