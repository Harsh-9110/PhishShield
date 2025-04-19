import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Read the connection string
conn_string = os.getenv("DATABASE_URL")

try:
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Connected to Neon DB successfully!")
    print("Current time from DB:", result)

    cursor.close()
    conn.close()
except Exception as e:
    print("Error connecting to Neon DB:", e)
