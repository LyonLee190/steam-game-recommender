import pandas as pd
import sqlite3

# Read CSV
df = pd.read_csv("steam_games.csv")

# Connect directly to SQLite (no SQLAlchemy needed)
conn = sqlite3.connect("games.db")
df.to_sql('games', conn, if_exists='replace', index=False)
conn.close()

print("Done! Database created: games.db")