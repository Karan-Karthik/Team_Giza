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


def get_connection():
    """Create a database connection."""
    return psycopg.connect(
        host="pinniped.postgres.database.azure.com", dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def transform_quality_data(df, year):
    """Transform the hospital quality data."""
    df = df[["Facility ID", "Facility Name",
             "Hospital Type", "Hospital Ownership",
            "Emergency Services", "Hospital overall rating"]].copy()

    df.rename(columns={"Facility ID": 'hospital_pk'}, inplace=True)
    df.rename(columns={"Facility Name": 'facility_name'}, inplace=True)
    df.rename(columns={'Hospital Type': 'type_of_hospital'}, inplace=True)
    df.rename(columns={'Hospital Ownership': 'type_of_ownership'},
              inplace=True)
    df.rename(columns={"Emergency Services": 'emergency_services'},
              inplace=True)
    df.rename(columns={"Hospital overall rating": 'overall_quality_rating'},
              inplace=True)

    df['hospital_pk'] = df['hospital_pk'].astype(str)
    df = df[df['hospital_pk'].str.len() == 6]

    df['emergency_services'] = df['emergency_services'].apply(
        lambda x: True if str(x).strip().lower() == 'yes' else False)
    df['rating_year'] = year

    return df


def get_existing_hospitals(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT hospital_pk FROM hospital")
        return {r[0] for r in cur.fetchall()}


def load_quality_data(file_path, year):
    """Load quality csv and insert data into tables."""
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
                    hospital_pk = EXCLUDED.hospital_pk,
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

                for value in tqdm(
                        quality_values,
                        desc="Inserting into hospital_quality",
                        unit="record"):
                    try:
                        cur.executemany(quality_query, [value])
                    except Exception as e:
                        logging.error(f"Error occurred: {value}. Error: {e}")
                        raise

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
