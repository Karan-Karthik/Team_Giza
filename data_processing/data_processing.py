"""
Data Processing Module for Hospital COVID-19 Dashboard

This module provides functions for processing and validating hospital data,
including bed occupancy calculations and regional statistics aggregation.
It ensures data consistency and handles edge cases such as missing values
and logical constraints (e.g., occupied beds cannot exceed total beds).

Functions:
    validate_and_clean_bed_data: Validates and cleans hospital bed data
    calculate_occupancy_stats: Calculates occupancy statistics with validation
    get_region_stats: Aggregates state-level statistics to regional level
"""

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
            - All metrics are non-negative
            - Occupied beds do not exceed total beds
            - COVID beds do not exceed total occupied beds

    Raises:
        Warning: If any data inconsistencies are found and adjusted
    """
    df = df.copy()
    bed_columns = [
        'total_adult_beds', 'occupied_adult_beds',
        'total_pediatric_beds', 'occupied_pediatric_beds',
        'covid_beds'
    ]
    for col in bed_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0)
    # Ensure occupied adult beds don't exceed total adult beds
    mask = df['occupied_adult_beds'] > df['total_adult_beds']
    if mask.any():
        st.warning(
            f"Found {
                mask.sum()} instances where occupied adult beds exceeded total beds. Adjusting values.")
        df.loc[mask, 'occupied_adult_beds'] = df.loc[mask, 'total_adult_beds']
    # Ensure occupied pediatric beds don't exceed total pediatric beds
    mask = df['occupied_pediatric_beds'] > df['total_pediatric_beds']
    if mask.any():
        st.warning(
            f"Found {
                mask.sum()} instances where occupied pediatric beds exceeded total beds. Adjusting values.")
        df.loc[mask, 'occupied_pediatric_beds'] = df.loc[mask,
                                                         'total_pediatric_beds']
    # Ensure COVID beds don't exceed total occupied beds
    df['total_occupied'] = df['occupied_adult_beds'] + \
        df['occupied_pediatric_beds']
    mask = df['covid_beds'] > df['total_occupied']
    if mask.any():
        st.warning(
            f"Found {
                mask.sum()} instances where COVID beds exceeded total occupied beds. Adjusting values.")
        df.loc[mask, 'covid_beds'] = df.loc[mask, 'total_occupied']
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
