"""
Database Queries Module for Hospital COVID-19 Dashboard

This module provides a collection of SQL queries and functions to retrieve
and process hospital data from the database. It includes queries for various
dashboard components such as metrics, comparisons, and trend analysis.

The module interfaces with a PostgreSQL database containing hospital data
including bed utilization, COVID cases, and quality metrics. Each function
returns data in a pandas DataFrame format suitable for dashboard visualization.

Functions:
    get_available_dates: Retrieve available collection dates
    get_available_states_and_regions: Get list of states and regions
    get_metrics: Retrieve metrics for selected states
    get_state_metrics_for_map: Get state-level metrics for mapping
    get_week_over_week_comparison: Compare hospital data week-over-week
    get_bed_availability_comparison: Compare bed availability over 4 weeks
    get_quality_rating_analysis: Analyze bed utilization by quality rating
    get_state_covid_map_data: Get state COVID data for mapping
    get_hospitals_with_significant_changes: Find hospitals with major changes
    get_non_reporting_hospitals: Find non-reporting hospitals
    get_trend_data: Get temporal trend data
    get_top_states_by_covid: Get states with highest COVID cases
"""

import pandas as pd


def get_available_dates(conn):
    """
    Get list of available data collection dates from the database.

    Args:
        conn: Database connection object

    Returns:
        list: Available dates in descending order
    """
    """Get list of available dates from the database."""
    query = """
    SELECT DISTINCT DATE(collection_week) as week
    FROM weekly_hospital_stats
    ORDER BY week DESC
    """
    dates_df = pd.read_sql_query(query, conn)
    return dates_df['week'].tolist()


def get_available_states_and_regions(conn):
    """
    Get list of available states and regions for filtering.

    Args:
        conn: Database connection object

    Returns:
        dict: Dictionary containing:
            - 'Regions': List of US regions
            - 'States': List of available states
    """
    """Get list of available states and add regions as options."""
    query = """
    SELECT DISTINCT state
    FROM hospital
    WHERE state IS NOT NULL
    ORDER BY state
    """
    states = pd.read_sql_query(query, conn)['state'].tolist()
    return {
        'Regions': [
            'Northeast',
            'Midwest',
            'South',
            'West'],
        'States': states}


def get_metrics(conn, selected_date, states):
    """
    Get hospital metrics for selected states on a specific date.

    Args:
        conn: Database connection object
        selected_date (str): Date to retrieve metrics for
        states (list): List of state codes to include

    Returns:
        pandas.DataFrame: DataFrame containing:
            - state: State code
            - total_beds: Total available beds
            - occupied_beds: Number of occupied beds
            - covid_beds: Beds used for COVID patients
            - reporting_hospitals: Number of hospitals reporting
            - occupancy_rate: Percentage of beds occupied
    """
    """Get metrics for selected states."""
    query = """
    WITH base_stats AS (
        SELECT
            h.state,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 'NaN'::numeric), 0), 0)) as covid_beds,
            COUNT(DISTINCT w.hospital_pk) as reporting_hospitals
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE DATE(w.collection_week) = %s
        AND h.state = ANY(%s)
        GROUP BY h.state
    )
    SELECT
        state,
        total_beds,
        LEAST(occupied_beds, total_beds) as occupied_beds,
        LEAST(covid_beds, LEAST(occupied_beds, total_beds)) as covid_beds,
        reporting_hospitals,
        CASE
            WHEN total_beds > 0 THEN
                LEAST(ROUND((LEAST(occupied_beds, total_beds)::float / NULLIF(total_beds, 0) * 100)::numeric, 1), 100)
            ELSE 0
        END as occupancy_rate,
        CASE
            WHEN LEAST(occupied_beds, total_beds) > 0 THEN
                LEAST(ROUND((LEAST(covid_beds, LEAST(occupied_beds, total_beds))::float /
                    NULLIF(LEAST(occupied_beds, total_beds), 0) * 100)::numeric, 1), 100)
            ELSE 0
        END as covid_percentage
    FROM base_stats
    """
    df = pd.read_sql_query(query, conn, params=[selected_date, states])
    return df


def get_state_metrics_for_map(conn, selected_date):
    """
    Get state-level metrics for mapping.

    Args:
        conn: Database connection object
        selected_date (str): Date to retrieve metrics for

    Returns:
        pandas.DataFrame: DataFrame containing:
            - state: State code
            - covid_cases: Number of COVID cases
            - occupancy_rate: Percentage of beds occupied
    """
    """Get state-level metrics for mapping."""
    query = """
    WITH state_stats AS (
        SELECT
            h.state,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 'NaN'::numeric), 0), 0)) as covid_cases
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE DATE(w.collection_week) = %s
        AND h.state IS NOT NULL
        GROUP BY h.state
    )
    SELECT
        state,
        covid_cases,
        CAST(
            CASE
                WHEN total_beds > 0
                THEN LEAST((LEAST(occupied_beds, total_beds)::float / total_beds * 100), 100)
                ELSE 0
            END AS DECIMAL(5,1)
        ) as occupancy_rate
    FROM state_stats
    ORDER BY state
    """
    return pd.read_sql_query(query, conn, params=[selected_date])


def get_week_over_week_comparison(conn, selected_date):
    """
    Compare hospital data week-over-week.

    Args:
        conn: Database connection object
        selected_date (str): Date to compare

    Returns:
        pandas.DataFrame: DataFrame containing:
            - week: Date
            - hospital_count: Number of hospitals reporting
            - prev_week_count: Number of hospitals reporting in previous week
            - percent_change: Percentage change in hospital count
    """
    """Get week-over-week comparison of hospital records."""
    query = """
    WITH weekly_counts AS (
        SELECT
            DATE(collection_week) as week,
            COUNT(DISTINCT hospital_pk) as hospital_count
        FROM weekly_hospital_stats
        WHERE collection_week <= %s
        AND collection_week >= %s - INTERVAL '5 weeks'
        GROUP BY week
        ORDER BY week DESC
        LIMIT 5
    )
    SELECT
        week,
        hospital_count,
        LAG(hospital_count) OVER (ORDER BY week DESC) as prev_week_count,
        (hospital_count - LAG(hospital_count) OVER (ORDER BY week DESC))::float /
        NULLIF(LAG(hospital_count) OVER (ORDER BY week DESC), 0) * 100 as percent_change
    FROM weekly_counts
    """
    return pd.read_sql_query(
        query, conn, params=[
            selected_date, selected_date])


def get_bed_availability_comparison(conn, selected_date):
    """
    Compare bed availability over 4 weeks.

    Args:
        conn: Database connection object
        selected_date (str): Date to compare

    Returns:
        pandas.DataFrame: DataFrame containing:
            - week: Date
            - total_adult_beds: Total adult beds available
            - occupied_adult_beds: Number of occupied adult beds
            - available_adult_beds: Number of available adult beds
            - total_pediatric_beds: Total pediatric beds available
            - occupied_pediatric_beds: Number of occupied pediatric beds
            - available_pediatric_beds: Number of available pediatric beds
            - covid_beds: Number of beds used for COVID patients
    """
    """Get 4-week comparison of bed availability."""
    query = """
    WITH weekly_beds AS (
        SELECT
            DATE(collection_week) as week,
            SUM(GREATEST(COALESCE(NULLIF(all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_adult_beds,
            SUM(GREATEST(COALESCE(NULLIF(all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_adult_beds,
            SUM(GREATEST(COALESCE(NULLIF(all_pediatric_inpatient_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_pediatric_beds,
            SUM(GREATEST(COALESCE(NULLIF(all_pediatric_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_pediatric_beds,
            SUM(GREATEST(COALESCE(NULLIF(inpatient_beds_used_covid_7_day_avg, 'NaN'::numeric), 0), 0)) as covid_beds
        FROM weekly_hospital_stats
        WHERE DATE(collection_week) <= %s
        AND DATE(collection_week) >= %s - INTERVAL '4 weeks'
        GROUP BY week
        ORDER BY week DESC
        LIMIT 5
    )
    SELECT
        week,
        total_adult_beds,
        occupied_adult_beds,
        total_adult_beds - occupied_adult_beds as available_adult_beds,
        total_pediatric_beds,
        occupied_pediatric_beds,
        total_pediatric_beds - occupied_pediatric_beds as available_pediatric_beds,
        covid_beds
    FROM weekly_beds
    """
    return pd.read_sql_query(
        query, conn, params=[
            selected_date, selected_date])


def get_quality_rating_analysis(conn, selected_date):
    """
    Analyze bed utilization by quality rating.

    Args:
        conn: Database connection object
        selected_date (str): Date to analyze

    Returns:
        pandas.DataFrame: DataFrame containing:
            - overall_quality_rating: Overall quality rating
            - total_beds: Total beds available
            - occupied_beds: Number of occupied beds
            - occupancy_rate: Percentage of beds occupied
    """
    """Get bed utilization by hospital quality rating."""
    query = """
    WITH current_stats AS (
        SELECT
            hq.overall_quality_rating,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_beds
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        JOIN hospital_quality hq ON h.hospital_pk = hq.hospital_pk
        WHERE DATE(w.collection_week) = %s
        AND hq.overall_quality_rating IS NOT NULL
        AND hq.overall_quality_rating != 'Not Available'
        GROUP BY hq.overall_quality_rating
    )
    SELECT
        overall_quality_rating,
        total_beds,
        LEAST(occupied_beds, total_beds) as occupied_beds,
        CAST(
            CASE
                WHEN total_beds > 0
                THEN LEAST((LEAST(occupied_beds, total_beds)::float / total_beds * 100), 100)
                ELSE 0
            END AS DECIMAL(5,1)
        ) as occupancy_rate
    FROM current_stats
    ORDER BY overall_quality_rating::int
    """
    return pd.read_sql_query(query, conn, params=[selected_date])


def get_state_covid_map_data(conn, selected_date):
    """
    Get state COVID data for mapping.

    Args:
        conn: Database connection object
        selected_date (str): Date to retrieve data

    Returns:
        pandas.DataFrame: DataFrame containing:
            - state_fips: State FIPS code
            - state: State name
            - covid_cases: Number of COVID cases
            - rank: Rank of state by COVID cases
    """
    """Get state-level COVID case data for mapping."""
    query = """
    WITH state_stats AS (
        SELECT
            SUBSTRING(h.fips_code, 1, 2) as state_fips,
            h.state,
            SUM(GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 0), 0), 0)) as covid_cases
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE DATE(w.collection_week) = %s
        AND h.state IS NOT NULL
        GROUP BY state_fips, state
    )
    SELECT
        state_fips,
        state,
        covid_cases,
        RANK() OVER (ORDER BY covid_cases DESC) as rank
    FROM state_stats
    ORDER BY covid_cases DESC
    """
    return pd.read_sql_query(query, conn, params=[selected_date])


def get_hospitals_with_significant_changes(conn, selected_date):
    """
    Find hospitals with major changes in COVID cases.

    Args:
        conn: Database connection object
        selected_date (str): Date to compare

    Returns:
        pandas.DataFrame: DataFrame containing:
            - hospital_name: Hospital name
            - city: City
            - state: State
            - current_cases: Current number of COVID cases
            - previous_cases: Previous number of COVID cases
            - absolute_change: Absolute change in COVID cases
            - percent_change: Percentage change in COVID cases
    """
    """Get hospitals with significant changes in COVID cases."""
    query = """
    WITH current_week AS (
        SELECT
            w.hospital_pk,
            h.hospital_name,
            h.city,
            h.state,
            GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 0), 0), 0) as covid_cases
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE DATE(w.collection_week) = %s
    ),
    prev_week AS (
        SELECT
            w.hospital_pk,
            GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 0), 0), 0) as covid_cases
        FROM weekly_hospital_stats w
        WHERE DATE(w.collection_week) = %s - INTERVAL '1 week'
    )
    SELECT
        c.hospital_name,
        c.city,
        c.state,
        c.covid_cases as current_cases,
        p.covid_cases as previous_cases,
        c.covid_cases - p.covid_cases as absolute_change,
        CASE
            WHEN p.covid_cases > 0
            THEN ((c.covid_cases - p.covid_cases)::float / p.covid_cases * 100)
            ELSE NULL
        END as percent_change
    FROM current_week c
    LEFT JOIN prev_week p ON c.hospital_pk = p.hospital_pk
    WHERE ABS(c.covid_cases - p.covid_cases) >= 5
    ORDER BY ABS(c.covid_cases - p.covid_cases) DESC
    LIMIT 10
    """
    return pd.read_sql_query(
        query, conn, params=[
            selected_date, selected_date])


def get_non_reporting_hospitals(conn, selected_date):
    """
    Find non-reporting hospitals.

    Args:
        conn: Database connection object
        selected_date (str): Date to compare

    Returns:
        pandas.DataFrame: DataFrame containing:
            - hospital_name: Hospital name
            - city: City
            - state: State
            - last_report_date: Last report date
            - days_since_report: Days since last report
    """
    """Get hospitals that haven't reported data recently."""
    query = """
    WITH last_report AS (
        SELECT
            h.hospital_pk,
            h.hospital_name,
            h.city,
            h.state,
            MAX(DATE(w.collection_week)) as last_report_date
        FROM hospital h
        LEFT JOIN weekly_hospital_stats w ON h.hospital_pk = w.hospital_pk
        GROUP BY h.hospital_pk, h.hospital_name, h.city, h.state
    )
    SELECT
        hospital_name,
        city,
        state,
        last_report_date,
        %s::date - last_report_date as days_since_report
    FROM last_report
    WHERE last_report_date < %s::date - INTERVAL '2 weeks'
    OR last_report_date IS NULL
    ORDER BY last_report_date DESC NULLS LAST
    LIMIT 10
    """
    return pd.read_sql_query(
        query, conn, params=[
            selected_date, selected_date])


def get_trend_data(conn, states):
    """
    Get temporal trend data.

    Args:
        conn: Database connection object
        states (list): List of state codes

    Returns:
        pandas.DataFrame: DataFrame containing:
            - week: Date
            - state: State code
            - total_beds: Total beds available
            - occupied_beds: Number of occupied beds
            - covid_beds: Number of beds used for COVID patients
            - occupancy_rate: Percentage of beds occupied
            - covid_percentage: Percentage of beds used for COVID patients
    """
    """Get trend data for selected states."""
    query = """
    WITH weekly_stats AS (
        SELECT
            w.collection_week::date as week,
            h.state,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0) +
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 'NaN'::numeric), 0), 0)) as covid_beds
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE h.state = ANY(%s)
        GROUP BY week, h.state
        ORDER BY week DESC
    )
    SELECT
        week::text,
        state,
        total_beds,
        LEAST(occupied_beds, total_beds) as occupied_beds,
        LEAST(covid_beds, LEAST(occupied_beds, total_beds)) as covid_beds,
        CASE
            WHEN total_beds > 0 THEN
                LEAST(ROUND((LEAST(occupied_beds, total_beds)::float / NULLIF(total_beds, 0) * 100)::numeric, 1), 100)
            ELSE 0
        END as occupancy_rate,
        CASE
            WHEN LEAST(occupied_beds, total_beds) > 0 THEN
                LEAST(ROUND((LEAST(covid_beds, LEAST(occupied_beds, total_beds))::float /
                    NULLIF(LEAST(occupied_beds, total_beds), 0) * 100)::numeric, 1), 100)
            ELSE 0
        END as covid_percentage
    FROM weekly_stats
    """
    df = pd.read_sql_query(query, conn, params=[states])
    df['week'] = pd.to_datetime(df['week'])
    return df


def get_top_states_by_covid(conn, selected_date):
    """
    Get states with highest COVID cases.

    Args:
        conn: Database connection object
        selected_date (str): Date to compare

    Returns:
        pandas.DataFrame: DataFrame containing:
            - state: State code
            - covid_cases: Number of COVID cases
    """
    """Get top 5 states by COVID cases."""
    query = """
    WITH state_stats AS (
        SELECT
            h.state,
            SUM(GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 0), 0), 0)) as covid_cases
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE DATE(w.collection_week) = %s
        AND h.state IS NOT NULL
        GROUP BY h.state
    )
    SELECT
        state,
        ROUND(covid_cases) as covid_cases
    FROM state_stats
    WHERE covid_cases > 0
    ORDER BY covid_cases DESC
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn, params=[selected_date])
    if not df.empty:
        df['covid_cases'] = df['covid_cases'].map('{:,.0f}'.format)
    return df
