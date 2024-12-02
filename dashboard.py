import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg
import numpy as np
from plotly.subplots import make_subplots

# Database connection function
def get_connection():
    return psycopg.connect(
        host="pinniped.postgres.database.azure.com",
        dbname="njacimov",
        user="njacimov",
        password="Vj2A0rxBtk"
    )

# Define US regions
US_REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

def get_region_for_state(state):
    """Map state to US region"""
    region_mapping = {
        # Northeast
        'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
        'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
        'PA': 'Northeast',
        # Midwest
        'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
        'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
        'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
        # South
        'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
        'SC': 'South', 'VA': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South',
        'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South',
        'TX': 'South', 'DC': 'South',
        # West
        'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
        'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
        'HI': 'West', 'OR': 'West', 'WA': 'West'
    }
    return region_mapping.get(state, 'Other')

def get_region_stats(df):
    """Convert state-level statistics to regional statistics with proper handling of NaN values"""
    region_stats = pd.DataFrame()
    
    for region, states in US_REGIONS.items():
        region_data = df[df['state'].isin(states)].copy()
        # Replace infinite values with NaN
        region_data = region_data.replace([np.inf, -np.inf], np.nan)
        # Sum only numeric columns
        numeric_cols = region_data.select_dtypes(include=[np.number]).columns
        region_sums = region_data[numeric_cols].sum().to_frame().T
        region_sums['region'] = region
        region_stats = pd.concat([region_stats, region_sums])
    
    return region_stats

def validate_and_clean_bed_data(df):
    """
    Validate and clean bed data to ensure logical consistency:
    - Occupied beds cannot exceed total beds
    - All metrics must be non-negative
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure all bed counts are non-negative
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
        st.warning(f"Found {mask.sum()} instances where occupied adult beds exceeded total beds. Adjusting values.")
        df.loc[mask, 'occupied_adult_beds'] = df.loc[mask, 'total_adult_beds']
    
    # Ensure occupied pediatric beds don't exceed total pediatric beds
    mask = df['occupied_pediatric_beds'] > df['total_pediatric_beds']
    if mask.any():
        st.warning(f"Found {mask.sum()} instances where occupied pediatric beds exceeded total beds. Adjusting values.")
        df.loc[mask, 'occupied_pediatric_beds'] = df.loc[mask, 'total_pediatric_beds']
    
    # Ensure COVID beds don't exceed total occupied beds
    df['total_occupied'] = df['occupied_adult_beds'] + df['occupied_pediatric_beds']
    mask = df['covid_beds'] > df['total_occupied']
    if mask.any():
        st.warning(f"Found {mask.sum()} instances where COVID beds exceeded total occupied beds. Adjusting values.")
        df.loc[mask, 'covid_beds'] = df.loc[mask, 'total_occupied']
    
    return df

def calculate_occupancy_stats(df):
    """Calculate occupancy statistics with validation"""
    try:
        # Clean and validate the data first
        df = validate_and_clean_bed_data(df)
        
        # Calculate total beds and occupied beds
        df['total_beds'] = df['total_adult_beds'].fillna(0) + df['total_pediatric_beds'].fillna(0)
        df['occupied_beds'] = df['occupied_adult_beds'].fillna(0) + df['occupied_pediatric_beds'].fillna(0)
        
        # Calculate occupancy rate
        df['occupancy_rate'] = np.where(
            df['total_beds'] > 0,
            (df['occupied_beds'] / df['total_beds'] * 100),
            0
        )
        
        # Calculate COVID percentage
        df['covid_percentage'] = np.where(
            df['occupied_beds'] > 0,
            (df['covid_beds'].fillna(0) / df['occupied_beds'] * 100),
            0
        )
        
        # Ensure rates are between 0 and 100
        df['occupancy_rate'] = df['occupancy_rate'].clip(0, 100)
        df['covid_percentage'] = df['covid_percentage'].clip(0, 100)
        
        return df
    except Exception as e:
        st.error(f"Error calculating occupancy stats: {str(e)}")
        return df

def get_week_over_week_comparison(conn, selected_date):
    """Get week-over-week comparison of hospital records"""
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
    return pd.read_sql_query(query, conn, params=[selected_date, selected_date])

def get_bed_availability_comparison(conn, selected_date):
    """Get 4-week comparison of bed availability"""
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
    return pd.read_sql_query(query, conn, params=[selected_date, selected_date])

def get_quality_rating_analysis(conn, selected_date):
    """Get bed utilization by hospital quality rating"""
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
    """Get state-level COVID case data for mapping"""
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
    """Get hospitals with significant changes in COVID cases"""
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
    return pd.read_sql_query(query, conn, params=[selected_date, selected_date])

def get_non_reporting_hospitals(conn, selected_date):
    """Get hospitals that haven't reported data recently"""
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
    return pd.read_sql_query(query, conn, params=[selected_date, selected_date])

def get_week_over_week_comparison(conn, selected_date):
    """Get week-over-week comparison of hospital records"""
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
    return pd.read_sql_query(query, conn, params=[selected_date, selected_date])

def get_bed_availability_comparison(conn, selected_date):
    """Get 4-week comparison of bed availability"""
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
    return pd.read_sql_query(query, conn, params=[selected_date, selected_date])

def get_quality_rating_analysis(conn, selected_date):
    """Get bed utilization by hospital quality rating"""
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

def get_trend_data(conn, view_type='States'):
    """Get trend data for hospital metrics by state/region"""
    base_query = """
    WITH weekly_stats AS (
        SELECT 
            DATE(w.collection_week) as week,
            h.state,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.inpatient_beds_used_covid_7_day_avg, 'NaN'::numeric), 0), 0)) as covid_beds
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE h.state IS NOT NULL
        GROUP BY week, h.state
        ORDER BY week DESC, h.state
    )
    SELECT 
        week,
        state,
        total_beds,
        LEAST(occupied_beds, total_beds) as occupied_beds,
        LEAST(covid_beds, LEAST(occupied_beds, total_beds)) as covid_beds,
        CAST(
            CASE 
                WHEN total_beds > 0 
                THEN LEAST((LEAST(occupied_beds, total_beds)::float / total_beds * 100), 100)
                ELSE 0 
            END AS DECIMAL(5,1)
        ) as occupancy_rate,
        CAST(
            CASE 
                WHEN LEAST(occupied_beds, total_beds) > 0 
                THEN LEAST((LEAST(covid_beds, LEAST(occupied_beds, total_beds))::float / 
                          LEAST(occupied_beds, total_beds) * 100), 100)
                ELSE 0 
            END AS DECIMAL(5,1)
        ) as covid_percentage
    FROM weekly_stats
    """
    
    df = pd.read_sql_query(base_query, conn)
    
    if view_type == 'US Regions':
        # Add region column
        df['region'] = df['state'].apply(get_region_for_state)
        
        # Group by region and week
        region_df = df.groupby(['week', 'region']).agg({
            'total_beds': 'sum',
            'occupied_beds': 'sum',
            'covid_beds': 'sum'
        }).reset_index()
        
        # Recalculate percentages for regions
        region_df['occupancy_rate'] = (region_df['occupied_beds'] / region_df['total_beds'] * 100).clip(0, 100).round(1)
        region_df['covid_percentage'] = (region_df['covid_beds'] / region_df['occupied_beds'] * 100).clip(0, 100).round(1)
        
        return region_df
    else:
        return df[df['state'].isin(selected_states)] if 'selected_states' in locals() else df

def get_current_metrics(conn, selected_date, view_type='States'):
    """Get current overall metrics"""
    if view_type == 'US Regions':
        query = """
        WITH state_stats AS (
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
            AND h.state IS NOT NULL
            GROUP BY h.state
        )
        SELECT 
            SUM(total_beds) as total_beds,
            SUM(LEAST(occupied_beds, total_beds)) as occupied_beds,
            SUM(LEAST(covid_beds, LEAST(occupied_beds, total_beds))) as covid_beds,
            SUM(reporting_hospitals) as reporting_hospitals,
            CAST(
                CASE 
                    WHEN SUM(total_beds) > 0 
                    THEN LEAST((SUM(LEAST(occupied_beds, total_beds))::float / SUM(total_beds) * 100), 100)
                    ELSE 0 
                END AS DECIMAL(5,1)
            ) as occupancy_rate,
            CAST(
                CASE 
                    WHEN SUM(LEAST(occupied_beds, total_beds)) > 0 
                    THEN LEAST((SUM(LEAST(covid_beds, LEAST(occupied_beds, total_beds)))::float / 
                              SUM(LEAST(occupied_beds, total_beds)) * 100), 100)
                    ELSE 0 
                END AS DECIMAL(5,1)
            ) as covid_percentage
        FROM state_stats
        """
    else:
        query = """
        WITH state_stats AS (
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
            AND h.state IS NOT NULL
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
                    LEAST(ROUND((LEAST(occupied_beds, total_beds)::float / total_beds * 100)::numeric, 1), 100)
                ELSE 0 
            END as occupancy_rate,
            CASE 
                WHEN LEAST(occupied_beds, total_beds) > 0 THEN 
                    LEAST(ROUND((LEAST(covid_beds, LEAST(occupied_beds, total_beds))::float / 
                        LEAST(occupied_beds, total_beds) * 100)::numeric, 1), 100)
                ELSE 0 
            END as covid_percentage
        FROM state_stats
        ORDER BY state
        """
    
    df = pd.read_sql_query(query, conn, params=[selected_date])
    
    if view_type == 'US Regions':
        return df
    else:
        return df[df['state'].isin(selected_states)] if 'selected_states' in locals() else df

def get_trend_data(conn, selections):
    """Get trend data for selected states/regions"""
    states = get_states_for_selection(selections)
    
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
    
    try:
        # Convert week column to datetime
        df['week'] = pd.to_datetime(df['week'])
    except Exception as e:
        st.error(f"Error in get_trend_data: {str(e)}")
        st.write("Week column type:", df['week'].dtype)
        st.write("First few weeks:", df['week'].head())
        return df
    
    df['region'] = df['state'].map(lambda x: next((region for region, states in US_REGIONS.items() if x in states), None))
    
    return df

def plot_trend_data(trend_data, view_type):
    """Plot trend data for COVID percentage"""
    # Ensure week column is datetime
    if not pd.api.types.is_datetime64_any_dtype(trend_data['week']):
        try:
            trend_data['week'] = pd.to_datetime(trend_data['week'])
        except Exception as e:
            st.error(f"Error converting dates: {str(e)}")
            st.write("Week column type:", trend_data['week'].dtype)
            st.write("First few weeks:", trend_data['week'].head())
            return

    # Sort by week
    trend_data = trend_data.sort_values('week')
    
    # Create figure
    fig = go.Figure()

    # Format dates for x-axis
    date_strings = [d.strftime('%Y-%m-%d') if isinstance(d, (datetime, pd.Timestamp)) else str(d) 
                   for d in trend_data['week'].unique()]

    # Plot COVID percentage for each state
    for state in trend_data['state'].unique():
        state_data = trend_data[trend_data['state'] == state]
        fig.add_trace(
            go.Scatter(
                x=state_data['week'],
                y=state_data['covid_percentage'],
                name=state,
                mode='lines',
                showlegend=True
            )
        )

    # Update layout
    fig.update_layout(
        height=600,
        title="COVID % of Occupied Beds by State",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, b=40),
        xaxis=dict(
            title="Week",
            tickmode='array',
            ticktext=date_strings,
            tickvals=trend_data['week'].unique(),
            tickangle=45,
            nticks=10
        ),
        yaxis=dict(
            title="COVID % of Occupied Beds",
            rangemode='tozero',
            autorange=True
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def get_available_dates(conn):
    """Get list of available dates from the database"""
    query = """
    SELECT DISTINCT DATE(collection_week) as week
    FROM weekly_hospital_stats
    ORDER BY week DESC
    """
    dates_df = pd.read_sql_query(query, conn)
    return dates_df['week'].tolist()

def get_state_metrics_for_map(conn, selected_date):
    """Get state-level metrics for mapping"""
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

def get_available_states_and_regions(conn):
    """Get list of available states and add regions as options"""
    query = """
    SELECT DISTINCT state 
    FROM hospital 
    WHERE state IS NOT NULL 
    ORDER BY state
    """
    states = pd.read_sql_query(query, conn)['state'].tolist()
    
    # Add regions as options at the top
    regions = ['Northeast', 'Midwest', 'South', 'West']
    return {'Regions': regions, 'States': states}

def get_states_for_selection(selections):
    """Get list of states for the selected states/regions"""
    states = []
    for selection in selections:
        if selection in ['Northeast', 'Midwest', 'South', 'West']:
            states.extend(US_REGIONS[selection])
        else:
            states.append(selection)
    return list(set(states))  # Remove duplicates

def get_metrics(conn, selected_date, selections):
    """Get metrics for selected states/regions"""
    states = get_states_for_selection(selections)
    
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
    
    # Add region column
    df['region'] = df['state'].map(lambda x: next((region for region, states in US_REGIONS.items() if x in states), None))
    
    return df

def get_trend_data(conn, selections):
    """Get trend data for selected states/regions"""
    states = get_states_for_selection(selections)
    
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
    df['region'] = df['state'].map(lambda x: next((region for region, states in US_REGIONS.items() if x in states), None))
    
    return df

def get_top_states_by_covid(conn, selected_date):
    """Get top 5 states by COVID cases"""
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
    WHERE covid_cases > 0  -- Exclude states with 0 or NULL cases
    ORDER BY covid_cases DESC
    LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn, params=[selected_date])
    if not df.empty:
        df['covid_cases'] = df['covid_cases'].map('{:,.0f}'.format)
    return df

def get_quality_metrics(conn, selected_date):
    """Get quality rating statistics"""
    query = """
    WITH quality_stats AS (
        SELECT 
            h.quality_rating,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_beds_7_day_avg, 'NaN'::numeric), 0), 0) + 
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_beds_7_day_avg, 'NaN'::numeric), 0), 0)) as total_beds,
            SUM(GREATEST(COALESCE(NULLIF(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0) + 
                GREATEST(COALESCE(NULLIF(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 'NaN'::numeric), 0), 0)) as occupied_beds
        FROM weekly_hospital_stats w
        JOIN hospital h ON w.hospital_pk = h.hospital_pk
        WHERE DATE(w.collection_week) = %s
        AND h.quality_rating IS NOT NULL
        GROUP BY h.quality_rating
    )
    SELECT 
        quality_rating,
        ROUND(total_beds::numeric, 1) as total_beds,
        ROUND(LEAST(occupied_beds, total_beds)::numeric, 1) as occupied_beds,
        ROUND(
            CASE 
                WHEN total_beds > 0 THEN 
                    LEAST((LEAST(occupied_beds, total_beds) / NULLIF(total_beds, 0) * 100), 100)
                ELSE 0 
            END::numeric, 1
        ) as occupancy_rate
    FROM quality_stats
    ORDER BY quality_rating
    """
    
    df = pd.read_sql_query(query, conn, params=[selected_date])
    if not df.empty:
        # Format numbers with commas
        df['total_beds'] = df['total_beds'].map('{:,.1f}'.format)
        df['occupied_beds'] = df['occupied_beds'].map('{:,.1f}'.format)
        df['occupancy_rate'] = df['occupancy_rate'].map('{:.1f}%'.format)
    return df

def main():
    try:
        # Page setup
        st.set_page_config(page_title="Hospital COVID-19 Dashboard", layout="wide")
        st.title("Hospital COVID-19 Analysis Dashboard")

        # Get available dates from database
        with get_connection() as conn:
            available_dates = get_available_dates(conn)

        if not available_dates:
            st.error("No data available. Please check the database connection.")
            return

        # Date selector with only available dates
        selected_date = st.selectbox('Select Analysis Date', available_dates)

        # Sidebar for state/region selection
        st.sidebar.header("Filters")
        
        with get_connection() as conn:
            # Get available states and regions
            available_selections = get_available_states_and_regions(conn)
            
            # Allow selecting multiple states/regions
            selected_areas = st.sidebar.multiselect(
                'Select States and/or Regions',
                options=['Northeast', 'Midwest', 'South', 'West'] + available_selections['States'],
                default=['Northeast'],
                help="You can select multiple states and/or regions"
            )
            
            if not selected_areas:
                st.warning("Please select at least one state or region to view data.")
                return

        # Current Week Summary
        st.header("Current Week Summary")
        with get_connection() as conn:
            current_metrics = get_metrics(conn, selected_date, selected_areas)
            
            if current_metrics.empty:
                st.warning(f"No data available for {selected_date}")
                return

            # Calculate totals
            total_metrics = pd.DataFrame([{
                'reporting_hospitals': current_metrics['reporting_hospitals'].sum(),
                'total_beds': current_metrics['total_beds'].sum(),
                'occupied_beds': current_metrics['occupied_beds'].sum(),
                'covid_beds': current_metrics['covid_beds'].sum()
            }])
            
            total_metrics['occupancy_rate'] = np.where(
                total_metrics['total_beds'] > 0,
                np.minimum(total_metrics['occupied_beds'] / total_metrics['total_beds'] * 100, 100),
                0
            )
            
            total_metrics['covid_percentage'] = np.where(
                total_metrics['occupied_beds'] > 0,
                np.minimum(total_metrics['covid_beds'] / total_metrics['occupied_beds'] * 100, 100),
                0
            )

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reporting Hospitals", 
                       f"{int(total_metrics['reporting_hospitals'].iloc[0]):,}")
            col2.metric("Overall Bed Occupancy", 
                       f"{total_metrics['occupancy_rate'].iloc[0]:.1f}%")
            col3.metric("COVID % of Occupied Beds", 
                       f"{total_metrics['covid_percentage'].iloc[0]:.1f}%")

            # Geographic comparison visualization
            st.header("Geographic Comparison")
            
            # Prepare data for visualization
            plot_data = current_metrics.copy()
            
            # Create the bar chart
            comparison_fig = px.bar(plot_data,
                                  x='state',
                                  y=['occupancy_rate', 'covid_percentage'],
                                  title='Hospital Metrics by State',
                                  labels={
                                      'occupancy_rate': 'Occupancy Rate (%)',
                                      'covid_percentage': 'COVID % of Occupied Beds',
                                      'value': 'Percentage',
                                      'variable': 'Metric',
                                      'state': 'State'
                                  },
                                  barmode='group')
            
            st.plotly_chart(comparison_fig, use_container_width=True)

        # State Maps Section
        st.header("State-Level Analysis")
        
        with get_connection() as conn:
            state_metrics = get_state_metrics_for_map(conn, selected_date)
            
            if not state_metrics.empty:
                # COVID Cases Map
                fig_covid = px.choropleth(
                    state_metrics,
                    locations='state',
                    locationmode="USA-states",
                    color='covid_cases',
                    scope="usa",
                    title="COVID-19 Cases by State",
                    color_continuous_scale="Viridis",
                    labels={'covid_cases': 'COVID-19 Cases'}
                )
                fig_covid.update_layout(
                    title_x=0.5,
                    geo=dict(scope='usa'),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_covid, use_container_width=True)
                
                # Bed Utilization Map
                fig_beds = px.choropleth(
                    state_metrics,
                    locations='state',
                    locationmode="USA-states",
                    color='occupancy_rate',
                    scope="usa",
                    title="Hospital Bed Utilization by State (%)",
                    color_continuous_scale="RdYlBu_r",
                    range_color=[0, 100],
                    labels={'occupancy_rate': 'Occupancy Rate (%)'}
                )
                fig_beds.update_layout(
                    title_x=0.5,
                    geo=dict(scope='usa'),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_beds, use_container_width=True)
            else:
                st.warning("No state-level data available for the selected date.")

        # Week-over-week comparison
        st.header("Weekly Reporting Summary")
        with get_connection() as conn:
            wow_comparison = get_week_over_week_comparison(conn, selected_date)
        
            if not wow_comparison.empty:
                current_count = wow_comparison.iloc[0]['hospital_count']
                prev_count = wow_comparison.iloc[0]['prev_week_count']
                percent_change = wow_comparison.iloc[0]['percent_change']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Week Reporting Hospitals", 
                           f"{int(current_count):,}",
                           f"{percent_change:.1f}%" if not pd.isna(percent_change) else "N/A")
                
                # Create week-over-week comparison chart
                wow_fig = go.Figure()
                wow_fig.add_trace(go.Bar(
                    x=wow_comparison['week'],
                    y=wow_comparison['hospital_count'],
                    name='Reporting Hospitals'
                ))
                wow_fig.update_layout(
                    title='Number of Reporting Hospitals by Week',
                    xaxis_title='Week',
                    yaxis_title='Number of Hospitals',
                    showlegend=True
                )
                st.plotly_chart(wow_fig, use_container_width=True)
        
        # Bed availability comparison
        st.header("Bed Availability Trends")
        with get_connection() as conn:
            bed_comparison = get_bed_availability_comparison(conn, selected_date)
        
            if not bed_comparison.empty:
                # Create bed availability table
                st.subheader("4-Week Bed Availability Summary")
                formatted_bed_table = bed_comparison.copy()
                for col in formatted_bed_table.columns:
                    if col != 'week':
                        formatted_bed_table[col] = formatted_bed_table[col].map('{:,.0f}'.format)
                st.dataframe(formatted_bed_table)
                
                # Create bed trend visualization
                bed_trend_fig = go.Figure()
                
                # Total beds trend
                bed_trend_fig.add_trace(go.Scatter(
                    x=bed_comparison['week'],
                    y=bed_comparison['occupied_adult_beds'] + bed_comparison['occupied_pediatric_beds'],
                    name='Total Occupied Beds',
                    mode='lines+markers'
                ))
                
                # COVID beds trend
                bed_trend_fig.add_trace(go.Scatter(
                    x=bed_comparison['week'],
                    y=bed_comparison['covid_beds'],
                    name='COVID Beds',
                    mode='lines+markers'
                ))
                
                bed_trend_fig.update_layout(
                    title='Hospital Bed Utilization Trend',
                    xaxis_title='Week',
                    yaxis_title='Number of Beds',
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(bed_trend_fig, use_container_width=True)
        
        # Hospital quality rating analysis
        st.header("Hospital Quality Analysis")
        with get_connection() as conn:
            quality_stats = get_quality_rating_analysis(conn, selected_date)
        
            if not quality_stats.empty:
                quality_fig = px.bar(
                    quality_stats,
                    x='overall_quality_rating',
                    y='occupancy_rate',
                    title='Hospital Occupancy Rate by Quality Rating',
                    labels={
                        'overall_quality_rating': 'Hospital Quality Rating (Stars)',
                        'occupancy_rate': 'Occupancy Rate (%)'
                    }
                )
                quality_fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(quality_fig, use_container_width=True)
                
                # Display quality stats table
                st.subheader("Detailed Quality Rating Statistics")
                formatted_quality_table = quality_stats.copy()
                formatted_quality_table['occupancy_rate'] = formatted_quality_table['occupancy_rate'].map('{:.1f}%'.format)
                formatted_quality_table = formatted_quality_table.rename(columns={
                    'overall_quality_rating': 'Quality Rating',
                    'total_beds': 'Total Beds',
                    'occupied_beds': 'Occupied Beds',
                    'occupancy_rate': 'Occupancy Rate'
                })
                st.dataframe(formatted_quality_table)

        # Trend Analysis
        st.header("Trend Analysis")
        with get_connection() as conn:
            trend_data = get_trend_data(conn, selected_areas)
            
            if not trend_data.empty:
                plot_trend_data(trend_data, 'States')
            else:
                st.warning("No trend data available.")

        # Additional Analyses Section
        st.header("Additional Insights")
        
        # State-level COVID Map
        st.subheader("COVID Cases by State")
        with get_connection() as conn:
            state_map_data = get_state_covid_map_data(conn, selected_date)
            
            if not state_map_data.empty:
                # Create choropleth map
                fig = px.choropleth(
                    state_map_data,
                    locations='state',
                    locationmode="USA-states",
                    color='covid_cases',
                    scope="usa",
                    color_continuous_scale="Viridis",
                    title=f'COVID Cases by State ({selected_date})',
                    labels={'covid_cases': 'COVID Cases'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top 5 states table
                st.subheader("Top 5 States by COVID Cases")
                top_states = get_top_states_by_covid(conn, selected_date)
                st.dataframe(top_states)
        
        # Hospitals with Significant Changes
        st.subheader("Hospitals with Significant COVID Case Changes")
        with get_connection() as conn:
            significant_changes = get_hospitals_with_significant_changes(conn, selected_date)
            
            if not significant_changes.empty:
                formatted_changes = significant_changes.copy()
                formatted_changes['current_cases'] = formatted_changes['current_cases'].map('{:,.0f}'.format)
                formatted_changes['previous_cases'] = formatted_changes['previous_cases'].map('{:,.0f}'.format)
                formatted_changes['absolute_change'] = formatted_changes['absolute_change'].map('{:+,.0f}'.format)
                formatted_changes['percent_change'] = formatted_changes['percent_change'].map(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "N/A")
                st.dataframe(formatted_changes)
        
        # Non-reporting Hospitals
        st.subheader("Recently Non-reporting Hospitals")
        with get_connection() as conn:
            non_reporting = get_non_reporting_hospitals(conn, selected_date)
            
            if not non_reporting.empty:
                formatted_non_reporting = non_reporting.copy()
                formatted_non_reporting['days_since_report'] = formatted_non_reporting['days_since_report'].map(lambda x: f"{x.days} days" if pd.notnull(x) else "Never reported")
                st.dataframe(formatted_non_reporting)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your database connection and try again.")

if __name__ == "__main__":
    main()
