# Team_Giza

## Overview
Team Giza is meant to maintain a database on hospital quality and information. This repository contains scripts and resources for creating a database pipeline for US Department of Health and Human Services (HHS) and Centers for Medicare and Medicaid Services (CMS) data. This includes SQL schema, data loading scripts, data processing functions, and an automated dashboard script that provides insights into the data. The data is available on HealthData.gov and CMS.gov.

## Setup
Pull the repository to your platform of choice, then update the credentials.py file in the repository with the following strucure:

```
DB_NAME = "YOUR_DATABASE_NAME"
DB_USER = "YOUR_USERNAME"
DB_PASSWORD = "YOUR_PASSWORD"
```

## Usage
Use the jupyter notebook final_scheme.ipynb to create SQL table to initialize the database with the necessary colunmns.

There are two python scripts to load datasets from HHS and CMS. Each python script is named according to the dataset that it is meant to load. The script can be run by calling the name of the csv file as an argument.

### Example
```
python load-hhs.py 2022-01-04-hhs-data.csv
```

```
python load-quality.py 2021-07-01 Hospital_General_Information-2021-07.csv
```

Columns and data types are manipulated for insertion into SQL tables. 

Each python script will insert the data with executemany(), print a summary of how much data has been read from the CSV and loaded into the database, and print an error message for any rows that fail to be inserted. This message identifyies which row failed and gives information about it, then stops the script. If an error occurs, no data is loaded into the database. 

Some data is provided under the Data folder.

There are several python scripts meant for data processing to create a dashboard for insights into the data. These are not necessary to run. The dashboard can be viewed by the following command.

```
streamlit run app.py
```
The dashboard allows the user to select the region of interest and time period (week) of interest. 

## Notes
- The database host is configured as `pinniped.postgres.database.azure.com`.
- All scripts include docstrings for readability and editing.
