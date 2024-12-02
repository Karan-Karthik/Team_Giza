import psycopg
import logging
from database.credentials import DB_USER, DB_PASS, DB_NAME

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()          # Log to console
    ]
)
logger = logging.getLogger()

def get_connection():
    """
    Establish a connection to the database with provided credentials.
    """
    try:
        # Database connection details
        db_host = "pinniped.postgres.database.azure.com"
        db_name = DB_NAME

        logger.info(
            f"Attempting to connect to the database at {db_host} with user {DB_USER}"
        )

        # Connect to the database using imported credentials
        connection = psycopg.connect(
            host=db_host,
            dbname=db_name,
            user=DB_USER,
            password=DB_PASS
        )
        logger.info("Database connection established successfully.")
        return connection
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise
