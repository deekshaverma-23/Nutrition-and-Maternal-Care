import sqlite3

conn = sqlite3.connect("chat.db")
cur = conn.cursor()

cur.execute(
    "ALTER TABLE messages ADD COLUMN audio_path TEXT;"
)

conn.commit()
conn.close()

print("✅ audio_path column added")
