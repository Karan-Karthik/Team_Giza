import pandas as pd
import numpy as np
from data_processing.data_processing import (
    validate_and_clean_bed_data,
    calculate_occupancy_stats,
    get_region_stats
    )


# Sample data for testing
sample_data = {
    "total_adult_beds": [10, 20, 30, None],
    "occupied_adult_beds": [5, 25, 15, 5],
    "total_pediatric_beds": [5, 15, 25, None],
    "occupied_pediatric_beds": [3, 10, 20, 2],
    "covid_beds": [2, 20, 10, 1],
    "state": ["NY", "CA", "TX", "FL"]
}


US_REGIONS = {
    "Northeast": ["NY", "MA", "PA"],
    "West": ["CA", "OR", "WA"],
    "South": ["TX", "FL", "GA"]
}


df_sample = pd.DataFrame(sample_data)


def test_validate_and_clean_bed_data_logical_constraints():
    df = validate_and_clean_bed_data(df_sample)
    assert (df['occupied_adult_beds'] <= df['total_adult_beds']).all(), "Occupied adult beds exceed total adult beds"
    assert (df['occupied_pediatric_beds'] <= df['total_pediatric_beds']).all(), "Occupied pediatric beds exceed total pediatric beds"
    assert (df['covid_beds'] <= (df['occupied_adult_beds'] + df['occupied_pediatric_beds'])).all(), "COVID beds exceed total occupied beds"


def test_validate_and_clean_bed_data_negative_values():
    df_sample_negative = df_sample.copy()
    # Introduce a negative value
    df_sample_negative.loc[0, "total_adult_beds"] = -10
    original_row_count = len(df_sample_negative)

    # Count rows with NaN values in the relevant columns
    bed_columns = [
        'total_adult_beds', 'occupied_adult_beds',
        'total_pediatric_beds', 'occupied_pediatric_beds',
        'covid_beds'
    ]
    nan_row_count = df_sample_negative[bed_columns].isna().any(axis=1).sum()

    # Call the updated function
    df = validate_and_clean_bed_data(df_sample_negative)

    # Check that rows with negative or NaN values were dropped
    expected_row_count = original_row_count - 1 - nan_row_count
    assert len(df) == expected_row_count, "Rows with negative or NaN values were not correctly dropped"

    # Validate that no negative values remain in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert (df[numeric_cols] >= 0).all().all(), "Data contains negative values after cleaning"


def test_calculate_occupancy_stats_occupancy_rate():
    df = calculate_occupancy_stats(df_sample)
    expected_occupancy_rate = ((5 + 3) / (10 + 5)) * 100
    calculated_rate = df.loc[0, 'occupancy_rate']
    assert abs(calculated_rate - expected_occupancy_rate) < 1e-5, f"Incorrect occupancy rate calculation: {calculated_rate} != {expected_occupancy_rate}"


def test_calculate_occupancy_stats_covid_utilization():
    df = calculate_occupancy_stats(df_sample)
    expected_covid_percentage = (2 / (5 + 3)) * 100
    calculated_percentage = df.loc[0, 'covid_percentage']
    assert abs(calculated_percentage - expected_covid_percentage) < 1e-5, f"Incorrect COVID utilization calculation: {calculated_percentage} != {expected_covid_percentage}"


def test_calculate_occupancy_stats_edge_cases():
    df_sample_edge = df_sample.copy()
    df_sample_edge.loc[0, "total_adult_beds"] = 0
    df_sample_edge.loc[0, "total_pediatric_beds"] = 0
    df = calculate_occupancy_stats(df_sample_edge)
    assert df.loc[0, 'occupancy_rate'] == 0, "Occupancy rate should be zero for zero total beds"
    assert df.loc[0, 'covid_percentage'] == 0, "COVID percentage should be zero for zero occupied beds"


def test_get_region_stats_aggregation():
    df = calculate_occupancy_stats(df_sample)
    region_stats = get_region_stats(df)
    northeast_data = df[df['state'].isin(US_REGIONS['Northeast'])]
    expected_total_beds = northeast_data['total_adult_beds'].sum() + northeast_data['total_pediatric_beds'].sum()
    assert region_stats.loc[region_stats['region'] == 'Northeast', 'total_beds'].iloc[0] == expected_total_beds, "Aggregation for total beds is incorrect"


def test_get_region_stats_missing_values():
    df_sample_missing = df_sample.copy()
    df_sample_missing.loc[0, "total_adult_beds"] = np.nan
    region_stats = get_region_stats(df_sample_missing)
    assert not region_stats.isna().any().any(), "Missing values not handled in regional aggregation"


def test_full_pipeline():
    df_cleaned = validate_and_clean_bed_data(df_sample)
    df_stats = calculate_occupancy_stats(df_cleaned)
    region_stats = get_region_stats(df_stats)
    assert not region_stats.isna().any().any(), "Full pipeline produced missing values"
    assert 'occupancy_rate' in df_stats.columns, "Occupancy rate not calculated in the pipeline"
