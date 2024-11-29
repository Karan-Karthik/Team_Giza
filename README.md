# Team_Giza

## Overview
Team Giza is meant to maintain a database on hospital quality and information. This repository provides SQL table schema to hold data from the US Department of Health and Human Services (HHS) about hospitals throughout the US, with weekly updates on how many patients have been admitted with COVID, how many hospital beds are currently available, how many staff are vaccinated, and related factors. The data is available on HealthData.gov. The tables also hold data on hospital quality ratings from Centers for Medicare and Medicaid Services (CMS). 

Two python scripts are provided to load csv files from both datasets into the SQL tables.

## Usage
Each python script is named according to the dataset that it is meant to load. The script can be run by calling the name of the csv file as an argument (ex. python load-hhs.py 2022-01-04-hhs-data.csv). Columns and data types are manipulated for insertion into SQL tables. 

Each python script will insert the data with executemany(), print a summary of how much data has been read from the CSV and loaded into the database, and print an error message for any rows that fail to be inserted. This message identifyies which row failed and gives information about it, then stops the script. If an error occurs, no data is loaded into the database. 

Data is provided under the Data folder.
