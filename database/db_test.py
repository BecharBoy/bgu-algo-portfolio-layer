import psycopg2
from psycopg2 import OperationalError


def track_db_connection():
    # Insert the credentials you just set up on the server
    db_config = {
        "dbname": "my_traders_db",  # Default database name
        "user": "my_traders_liran",  # Default admin user
        "password": "12345678",
        "host": "192.168.1.160",
        "port": "5432"
    }

    try:
        print(f"Attempting to connect to {db_config['host']}...")

        # Try to establish the connection
        connection = psycopg2.connect(**db_config)

        print("✅ Connection successful! The server is fully open and responding.")

        # Close it down cleanly since this is just a test
        connection.close()

    except OperationalError as e:
        # This block catches and tracks the exact failure reason
        print("❌ Connection failed.")
        print("--- Error Tracker Details ---")
        print(e)


if __name__ == "__main__":
    track_db_connection()