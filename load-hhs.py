import sys
import psycopg
import pandas as pd
import logging
from tqdm import tqdm
from database.credentials import DB_USER, DB_PASS, DB_NAME

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BATCH_SIZE = 1000


def get_connection():
    """Create a database connection."""
    return psycopg.connect(
        host="pinniped.postgres.database.azure.com",
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )


def transform_hhs_data(df):
    """Transform the HHS data."""
    df = df[['hospital_pk', 'hospital_name', 'address',
             'city', 'state', 'zip', 'fips_code',
             'collection_week', 'all_adult_hospital_beds_7_day_avg',
             'all_pediatric_inpatient_beds_7_day_avg',
             'all_adult_hospital_inpatient_bed_occupied_7_day_avg',
             'all_pediatric_inpatient_bed_occupied_7_day_avg',
             'total_icu_beds_7_day_avg', 'icu_beds_used_7_day_avg',
             'inpatient_beds_used_covid_7_day_avg',
             'staffed_icu_adult_patients_confirmed_covid_7_day_avg']].copy()

    df['hospital_pk'] = df['hospital_pk'].astype(str)
    df = df[df['hospital_pk'].str.len() == 6]
    df.replace(-999999, None, inplace=True)

    numerics = df.select_dtypes(include=['number']).columns
    df[numerics] = df[numerics].fillna(0)
    df['collection_week'] = pd.to_datetime(
        df['collection_week'], errors='coerce')

    return df


def batch_insert(cursor, query, values, batch_size=BATCH_SIZE, desc=""):
    """Insert data in batches."""
    for i in tqdm(range(0, len(values), batch_size), desc=desc):
        batch = values[i:i + batch_size]
        cursor.executemany(query, batch)


def load_hhs_data(file_path):
    """Load HHS CSV and insert data into tables."""
    df = pd.read_csv(file_path)
    df = transform_hhs_data(df)

    with get_connection() as conn:
        with conn.cursor() as cur:
            with conn.transaction():
                # Insert into Hospital table
                hospital_query = """
                INSERT INTO Hospital (
                    hospital_pk, hospital_name, address, city, state, 
                    zip, fips_code
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hospital_pk) DO UPDATE SET
                    hospital_name = EXCLUDED.hospital_name,
                    address = EXCLUDED.address,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    zip = EXCLUDED.zip,
                    fips_code = EXCLUDED.fips_code
                """
                hospital_values = df[['hospital_pk',
                                      'hospital_name',
                                      'address', 'city', 'state', 'zip',
                                      'fips_code']].drop_duplicates().values.tolist()

                batch_insert(cur, hospital_query, hospital_values,
                             desc="Inserting into Hospital Table")

                logging.info(
                    f"Loaded {len(hospital_values)} hospital records.")

                # Insert into weekly_hospital_stats table
                stats_query = """
                INSERT INTO weekly_hospital_stats (
                    hospital_pk, collection_week,
                    all_adult_hospital_beds_7_day_avg,
                    all_pediatric_inpatient_beds_7_day_avg,
                    all_adult_hospital_inpatient_bed_occupied_7_day_avg,
                    all_pediatric_inpatient_bed_occupied_7_day_avg,
                    total_icu_beds_7_day_avg,
                    icu_beds_used_7_day_avg,
                    inpatient_beds_used_covid_7_day_avg,
                    staffed_icu_adult_patients_confirmed_covid_7_day_avg
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hospital_pk, collection_week) DO UPDATE SET
                    all_adult_hospital_beds_7_day_avg = EXCLUDED.all_adult_hospital_beds_7_day_avg,
                    all_pediatric_inpatient_beds_7_day_avg = EXCLUDED.all_pediatric_inpatient_beds_7_day_avg,
                    all_adult_hospital_inpatient_bed_occupied_7_day_avg = EXCLUDED.all_adult_hospital_inpatient_bed_occupied_7_day_avg,
                    all_pediatric_inpatient_bed_occupied_7_day_avg = EXCLUDED.all_pediatric_inpatient_bed_occupied_7_day_avg,
                    total_icu_beds_7_day_avg = EXCLUDED.total_icu_beds_7_day_avg,
                    icu_beds_used_7_day_avg = EXCLUDED.icu_beds_used_7_day_avg,
                    inpatient_beds_used_covid_7_day_avg = EXCLUDED.inpatient_beds_used_covid_7_day_avg,
                    staffed_icu_adult_patients_confirmed_covid_7_day_avg = EXCLUDED.staffed_icu_adult_patients_confirmed_covid_7_day_avg
                """
                stats_values = df[['hospital_pk', 'collection_week',
                                   'all_adult_hospital_beds_7_day_avg',
                                   'all_pediatric_inpatient_beds_7_day_avg',
                                   'all_adult_hospital_inpatient_bed_occupied_7_day_avg',
                                   'all_pediatric_inpatient_bed_occupied_7_day_avg',
                                   'total_icu_beds_7_day_avg',
                                   'icu_beds_used_7_day_avg',
                                   'inpatient_beds_used_covid_7_day_avg',
                                   'staffed_icu_adult_patients_confirmed_covid_7_day_avg']].values.tolist()

                batch_insert(cur, stats_query, stats_values,
                             desc="Inserting into Weekly Hospital Stats")

                logging.info(f"Loaded {len(stats_values)} weekly stat records.")


def main():
    """Main function to run script."""
    if len(sys.argv) != 2:
        print("Usage: python load-hhs.py <csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        load_hhs_data(file_path)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
