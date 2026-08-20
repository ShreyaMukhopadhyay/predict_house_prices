import os
import dotenv

home_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv(os.path.join(home_folder, ".env"))

PG_HOST = os.getenv("POSTGRES_HOST")
PG_PORT = os.getenv("POSTGRES_PORT")
PG_USER = os.getenv("POSTGRES_USERNAME")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
