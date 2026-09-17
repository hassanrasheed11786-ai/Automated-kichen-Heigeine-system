import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    # 'YOUR_PASSWORD' ko apne PostgreSQL password se badal dein
    connection = psycopg2.connect(
        user="postgres", 
        password="hr789", 
        host="localhost", 
        port="5432"
    )
    
    # Database create karne ke liye autocommit on karna zaroori hai
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = connection.cursor()
    
    # Naya database banane ki SQL command
    cursor.execute("CREATE DATABASE hygiene_db;")
    
    print("✅ Database 'hygiene_db' successfully create ho gaya hai!")
    
    cursor.close()
    connection.close()

except psycopg2.errors.DuplicateDatabase:
    print("⚠️ Database 'hygiene_db' pehle se mojood hai!")
except Exception as e:
    print("❌ Error:", e)