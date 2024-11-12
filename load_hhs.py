import psycopg
import sys
import pandas as pd


# load csv data and create strings to execute commands based on file name
full_df = pd.read_csv(sys.argv[1], encoding = 'utf-8')
hhs = "hhs"
general = "General"

# transformations for hhs csv
if hhs in sys.argv[1]:
    # choose columns that are needed for sql tables
    df = full_df[['hospital_pk', 'hospital_name', 'address', 'city', 'state', 'zip', 'fips_code',
                  'collection_week', 'all_adult_hospital_beds_7_day_avg',
                  'all_adult_hospital_inpatient_bed_occupied_7_day_avg',
                  'all_pediatric_inpatient_bed_occupied_7_day_avg', 'total_icu_beds_7_day_avg',
                  'icu_beds_used_7_day_avg',
                  'inpatient_beds_used_covid_7_day_avg',
                  'staffed_icu_adult_patients_confirmed_covid_7_day_avg']].copy()
    
    # rename columns to match sql tables
    df.rename(columns = {'hospital_pk':'hospital_id'}, inplace = True)

    # replace -999999 placeholders with None in numeric columns
    df.replace(-999999, None, inplace = True)

    # select numeric columns and replace all NA with 0
    numerics = df.select_dtypes(include=['number']).columns
    df[numerics] = df.fillna(0, inplace=True)

    # convert collection_week from object to datetime
    df['collection_week'] = pd.to_datetime(df['collection_week'])

    # convert hospital_id to str to account for non-numeric data
    df['hospital_id'] = df['hospital_id'].astype(str)

# transformations for general csv
if general in sys.argv[1]:
    # choose columns that are needed for sql tables
    df = full_df[['Facility ID', 'Facility Name', 'Address', 'City', 'County Name',
                  'State','ZIP Code', 'Hospital Type',
                  'Hospital Ownership', 'Emergency Services',
                  'Hospital overall rating']].copy()
    
    # rename columns to match sql tables
    df.rename(columns = {'Facility ID':'hospital_id', 'Facility Name':'hospital_name', 'Address':'address',
                         'City':'city', 'County Name':'county_name', 'State':'state',
                         'ZIP Code':'zip', 'Hospital Type':'hospital_type',
                         'Hospital Ownership':'hospital_ownership',
                         'Emergency Services':'emergency_services',
                         'Hospital overall rating':'overall_quality_rating'}, inplace = True)

    # convert emergency_services column to boolean. Unsure if this is needed her or if this can be done with insert
    #df.loc[:,'emergency_services'] = df['emergency_services'].apply(lambda x: True if x == 'Yes' else False)

    #df.loc[:,'emergency_services'] = df['emergency_services'].astype(bool)

# connect to server
conn = psycopg.connect(
    host="pinniped.postgres.database.azure.com", dbname="dbname",
    user="yourusername", password="yourpassword"
)

# create cursor
cur = conn.cursor()

# create counter for number of rows inserted from 0
num_rows_inserted = 0

# make a new transaction
with conn.transaction():
    for row in df:
        try:
            # make a new SAVEPOINT -- like a save in a video game
            with conn.transaction():
                # perhaps a bunch of reformatting and data manipulation goes here
                cur.execute("")
                # now insert the data
                cur.executemany("INSERT INTO foo ...", ...)
        except Exception as e:
            # if an exception/error happens in this block, Postgres goes back to
            # the last savepoint upon exiting the `with` block
            print("insert failed")
            # add additional logging, error handling here
        else:
            # no exception happened, so we continue without reverting the savepoint
            num_rows_inserted += 1

# now we commit the entire transaction
conn.commit()