import sqlite3
import pandas as pd

df = pd.read_csv("data/cars.csv")

conn = sqlite3.connect("data/cars.db")

df.to_sql(
    "cars",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("SQLite 저장 완료: data/cars.db")
print(f"저장된 데이터 수: {len(df)}")