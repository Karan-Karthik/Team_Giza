import pandas as pd
import numpy as np
import streamlit as st
from utils.utils import US_REGIONS


def validate_and_clean_bed_data(df):
    """
    Validate and clean hospital bed data to ensure logical consistency.

    Args:
        df (pandas.DataFrame): Input DataFrame containing bed-related columns
            Required columns: total_adult_beds, occupied_adult_beds,
                            total_pediatric_beds, occupied_pediatric_beds,
                            covid_beds

    Returns:
        pandas.DataFrame: Cleaned DataFrame with validated bed data where:
            - Rows with negative or invalid values are dropped
            - Occupied beds do not exceed total beds
            - COVID beds do not exceed total occupied beds
    """
    df = df.copy()
    bed_columns = [
        'total_adult_beds', 'occupied_adult_beds',
        'total_pediatric_beds', 'occupied_pediatric_beds',
        'covid_beds'
    ]

    print("Initial DataFrame:")
    print(df)

    # Ensure all bed-related columns are numeric
    for col in bed_columns:
        if col in df.columns:
            # Convert to numeric, invalid values become NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("After converting to numeric:")
    print(df)

    # Drop rows with NaN values in any bed-related column
    df = df.dropna(subset=bed_columns)
    print("After dropping NaN values:")
    print(df)

    # Drop rows with negative values in any bed-related column
    for col in bed_columns:
        df = df[df[col] >= 0]

    print("After dropping negative values:")
    print(df)

    # Ensure occupied adult beds don't exceed total adult beds
    mask = df['occupied_adult_beds'] > df['total_adult_beds']
    if mask.any():
        df.loc[mask, 'occupied_adult_beds'] = df.loc[mask, 'total_adult_beds']

    print("After adjusting occupied adult beds:")
    print(df)

    # Ensure occupied pediatric beds don't exceed total pediatric beds
    mask = df['occupied_pediatric_beds'] > df['total_pediatric_beds']
    if mask.any():
        df.loc[
            mask, 'occupied_pediatric_beds'
               ] = df.loc[
                   mask, 'total_pediatric_beds'
                   ]

    print("After adjusting occupied pediatric beds:")
    print(df)

    # Ensure COVID beds don't exceed total occupied beds
    df['total_occupied'] = df[
        'occupied_adult_beds'
        ] + df[
            'occupied_pediatric_beds'
                                    ]
    mask = df['covid_beds'] > df['total_occupied']
    if mask.any():
        df.loc[mask, 'covid_beds'] = df.loc[mask, 'total_occupied']

    print("Final DataFrame:")
    print(df)

    return df


def calculate_occupancy_stats(df):
    """
    Calculate hospital occupancy statistics with data validation.

    Args:
        df (pandas.DataFrame): Input DataFrame with bed-related columns
            Required columns: total_adult_beds, total_pediatric_beds,
                            occupied_adult_beds, occupied_pediatric_beds

    Returns:
        pandas.DataFrame: DataFrame with additional calculated columns:
            - total_beds: Sum of adult and pediatric beds
            - occupied_beds: Sum of occupied adult and pediatric beds
            - occupancy_rate: Percentage of total beds that are occupied
            - covid_utilization: Percentage of occupied beds used for COVID
    """
    try:
        df = validate_and_clean_bed_data(df)
        df['total_beds'] = df['total_adult_beds'].fillna(
            0) + df['total_pediatric_beds'].fillna(0)
        df['occupied_beds'] = df['occupied_adult_beds'].fillna(
            0) + df['occupied_pediatric_beds'].fillna(0)
        df['occupancy_rate'] = np.where(
            df['total_beds'] > 0,
            (df['occupied_beds'] / df['total_beds'] * 100),
            0
        )
        df['covid_percentage'] = np.where(
            df['occupied_beds'] > 0,
            (df['covid_beds'].fillna(0) / df['occupied_beds'] * 100),
            0
        )
        df['occupancy_rate'] = df['occupancy_rate'].clip(0, 100)
        df['covid_percentage'] = df['covid_percentage'].clip(0, 100)
        return df
    except Exception as e:
        st.error(f"Error calculating occupancy stats: {str(e)}")
        return df


def get_region_stats(df):
    """
    Convert state-level statistics to regional statistics.

    Args:
        df (pandas.DataFrame): Input DataFrame with state-level statistics
            Required columns: state, plus any numeric columns to be aggregated

    Returns:
        pandas.DataFrame: DataFrame with statistics aggregated by region,
            handling NaN values appropriately
    """
    region_stats = pd.DataFrame()
    for region, states in US_REGIONS.items():
        region_data = df[df['state'].isin(states)].copy()
        region_data = region_data.replace([np.inf, -np.inf], np.nan)
        numeric_cols = region_data.select_dtypes(include=[np.number]).columns
        region_sums = region_data[numeric_cols].sum().to_frame().T
        region_sums['region'] = region
        region_stats = pd.concat([region_stats, region_sums])
    return region_stats
