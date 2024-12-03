import sys
import psycopg
import pandas as pd
import logging
from tqdm import tqdm
from database.credentials import DB_USER, DB_PASS, DB_NAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BATCH_SIZE = 1000


def get_connection():
    """Create a database connection."""
    return psycopg.connect(
        host="pinniped.postgres.database.azure.com",
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )


def transform_quality_data(df, year):
    """Transform the hospital quality data."""
    df = df[["Facility ID", "Facility Name",
             "Hospital Type", "Hospital Ownership",
             "Emergency Services", "Hospital overall rating"]].copy()

    df.rename(columns={
        "Facility ID": 'hospital_pk',
        "Facility Name": 'facility_name',
        'Hospital Type': 'type_of_hospital',
        'Hospital Ownership': 'type_of_ownership',
        "Emergency Services": 'emergency_services',
        "Hospital overall rating": 'overall_quality_rating'
    }, inplace=True)

    df['hospital_pk'] = df['hospital_pk'].astype(str)
    df = df[df['hospital_pk'].str.len() == 6]

    df['emergency_services'] = df['emergency_services'].apply(
        lambda x: True if str(x).strip().lower() == 'yes' else False)
    df['rating_year'] = year

    return df


def get_existing_hospitals(conn):
    """Get existing hospitals from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT hospital_pk FROM hospital")
        return {r[0] for r in cur.fetchall()}


def batch_insert(cursor, query, values, batch_size=BATCH_SIZE, desc=""):
    """Insert data in batches."""
    for i in tqdm(range(0, len(values), batch_size), desc=desc):
        batch = values[i:i + batch_size]
        cursor.executemany(query, batch)


def load_quality_data(file_path, year):
    """Load quality CSV and insert data into tables."""
    df = pd.read_csv(file_path)
    df = transform_quality_data(df, year)

    with get_connection() as conn:
        existing = get_existing_hospitals(conn)
        df = df[df['hospital_pk'].isin(existing)]

        with conn.cursor() as cur:
            with conn.transaction():
                quality_query = """
                INSERT INTO hospital_quality (
                    hospital_pk, facility_name, type_of_hospital,
                    type_of_ownership, emergency_services,
                    overall_quality_rating, rating_year
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hospital_pk, rating_year) DO UPDATE SET
                    facility_name = EXCLUDED.facility_name,
                    type_of_hospital = EXCLUDED.type_of_hospital,
                    type_of_ownership = EXCLUDED.type_of_ownership,
                    emergency_services = EXCLUDED.emergency_services,
                    overall_quality_rating = EXCLUDED.overall_quality_rating
                """
                quality_values = df[['hospital_pk',
                                     'facility_name', 'type_of_hospital',
                                     'type_of_ownership', 'emergency_services',
                                     'overall_quality_rating',
                                     'rating_year'
                                     ]].drop_duplicates().values.tolist()

                batch_insert(cur, quality_query, quality_values,
                             desc="Inserting into hospital_quality")

                logging.info(
                    f"Loaded {len(quality_values)} hospital quality records.")


def main():
    """Main function to run script."""
    if len(sys.argv) != 3:
        print("Usage: python load-quality.py <rating_year> <csv_file>")
        sys.exit(1)

    rating_year = int(sys.argv[1])
    file_path = sys.argv[2]

    try:
        load_quality_data(file_path, rating_year)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
