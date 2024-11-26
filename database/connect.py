import psycopg
import logging

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
        # Static credentials
        db_host = "pinniped.postgres.database.azure.com"
        db_name = "njacimov"
        db_user = "njacimov"
        db_pass = "Vj2A0rxBtk"

        logger.info(
            f"Attempting to connect to the database at {db_host} with user {db_user}")

        # Connect to the database
        connection = psycopg.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_pass
        )
        logger.info("Database connection established successfully.")
        return connection
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise
