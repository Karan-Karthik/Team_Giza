import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg
import numpy as np

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

def main():
    try:
        # Page setup
        st.set_page_config(page_title="Hospital COVID-19 Dashboard", layout="wide")
        st.title("Hospital COVID-19 Analysis Dashboard")

        # Get available dates from database
        with get_connection() as conn:
            date_query = """
            SELECT DISTINCT DATE(collection_week)
            FROM weekly_hospital_stats
            WHERE collection_week IS NOT NULL
            ORDER BY DATE(collection_week) DESC
            """
            available_dates = pd.read_sql_query(date_query, conn)['date'].tolist()

        if not available_dates:
            st.error("No data available. Please check the database connection.")
            return

        # Date selector with only available dates
        selected_date = st.date_input(
            "Select Analysis Date",
            value=available_dates[0],
            min_value=available_dates[-1],
            max_value=available_dates[0]
        )

        # Sidebar for state/region selection
        st.sidebar.header("Filters")
        with get_connection() as conn:
            state_query = """
            SELECT DISTINCT state 
            FROM hospital 
            WHERE state IS NOT NULL 
            ORDER BY state
            """
            states = pd.read_sql_query(state_query, conn)['state'].tolist()
            
            view_type = st.sidebar.radio(
                "Select View Type",
                ["US Regions", "States"]
            )
            
            if view_type == "States":
                selected_states = st.sidebar.multiselect(
                    "Select States",
                    states,
                    default=states[:5]  # Default to first 5 states
                )
                if not selected_states:
                    st.warning("Please select at least one state to view data.")
                    return
            else:
                selected_states = states

        # Current Week Summary
        st.header("Current Week Summary")
        with get_connection() as conn:
            current_query = """
            WITH raw_stats AS (
                SELECT 
                    h.state,
                    COUNT(DISTINCT w.hospital_pk) as reporting_hospitals,
                    SUM(GREATEST(COALESCE(w.all_adult_hospital_beds_7_day_avg, 0), 0)) as total_adult_beds,
                    SUM(GREATEST(COALESCE(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 0), 0)) as occupied_adult_beds,
                    SUM(GREATEST(COALESCE(w.all_pediatric_inpatient_beds_7_day_avg, 0), 0)) as total_pediatric_beds,
                    SUM(GREATEST(COALESCE(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 0), 0)) as occupied_pediatric_beds,
                    SUM(GREATEST(COALESCE(w.inpatient_beds_used_covid_7_day_avg, 0), 0)) as covid_beds
                FROM weekly_hospital_stats w
                JOIN hospital h ON w.hospital_pk = h.hospital_pk
                WHERE DATE(w.collection_week) = %s
                AND h.state IS NOT NULL
                GROUP BY h.state
            )
            SELECT 
                *,
                LEAST(
                    occupied_adult_beds,
                    total_adult_beds
                ) as adjusted_occupied_adult_beds,
                LEAST(
                    occupied_pediatric_beds,
                    total_pediatric_beds
                ) as adjusted_occupied_pediatric_beds
            FROM raw_stats
            """
            current_stats = pd.read_sql_query(current_query, conn, params=[selected_date])
            
            if current_stats.empty:
                st.warning(f"No data available for {selected_date}")
                return

            # Use the adjusted occupied beds
            current_stats['occupied_adult_beds'] = current_stats['adjusted_occupied_adult_beds']
            current_stats['occupied_pediatric_beds'] = current_stats['adjusted_occupied_pediatric_beds']
            current_stats = calculate_occupancy_stats(current_stats)

            if view_type == "US Regions":
                current_stats = get_region_stats(current_stats)
            else:
                current_stats = current_stats[current_stats['state'].isin(selected_states)]

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reporting Hospitals", 
                       f"{int(current_stats['reporting_hospitals'].sum()):,}")
            col2.metric("Overall Bed Occupancy", 
                       f"{current_stats['occupancy_rate'].mean():.1f}%")
            col3.metric("COVID % of Occupied Beds", 
                       f"{current_stats['covid_percentage'].mean():.1f}%")

            # Geographic comparison visualization
            st.header("Geographic Comparison")
            comparison_fig = px.bar(current_stats,
                                  x='region' if view_type == "US Regions" else 'state',
                                  y=['occupancy_rate', 'covid_percentage'],
                                  title=f'Hospital Metrics by {"Region" if view_type == "US Regions" else "State"}',
                                  labels={
                                      'occupancy_rate': 'Occupancy Rate (%)',
                                      'covid_percentage': 'COVID % of Occupied Beds',
                                      'value': 'Percentage',
                                      'variable': 'Metric'
                                  },
                                  barmode='group')
            st.plotly_chart(comparison_fig, use_container_width=True)

        # Trend Analysis
        st.header("Trend Analysis")
        with get_connection() as conn:
            trend_query = """
            WITH raw_weekly_stats AS (
                SELECT 
                    DATE(w.collection_week) as week,
                    h.state,
                    SUM(GREATEST(COALESCE(w.all_adult_hospital_beds_7_day_avg, 0), 0)) as total_adult_beds,
                    SUM(GREATEST(COALESCE(w.all_adult_hospital_inpatient_bed_occupied_7_day_avg, 0), 0)) as occupied_adult_beds,
                    SUM(GREATEST(COALESCE(w.all_pediatric_inpatient_beds_7_day_avg, 0), 0)) as total_pediatric_beds,
                    SUM(GREATEST(COALESCE(w.all_pediatric_inpatient_bed_occupied_7_day_avg, 0), 0)) as occupied_pediatric_beds,
                    SUM(GREATEST(COALESCE(w.inpatient_beds_used_covid_7_day_avg, 0), 0)) as covid_beds
                FROM weekly_hospital_stats w
                JOIN hospital h ON w.hospital_pk = h.hospital_pk
                WHERE w.collection_week <= %s
                AND w.collection_week >= %s - INTERVAL '12 weeks'
                AND h.state IS NOT NULL
                GROUP BY week, h.state
            ),
            adjusted_stats AS (
                SELECT 
                    week,
                    state,
                    total_adult_beds,
                    total_pediatric_beds,
                    LEAST(occupied_adult_beds, total_adult_beds) as occupied_adult_beds,
                    LEAST(occupied_pediatric_beds, total_pediatric_beds) as occupied_pediatric_beds,
                    covid_beds
                FROM raw_weekly_stats
            )
            SELECT 
                *,
                CASE 
                    WHEN (total_adult_beds + total_pediatric_beds) > 0 
                    THEN LEAST(
                        ((occupied_adult_beds + occupied_pediatric_beds)::float / 
                        (total_adult_beds + total_pediatric_beds) * 100),
                        100
                    )
                    ELSE 0 
                END as occupancy_rate,
                CASE 
                    WHEN (occupied_adult_beds + occupied_pediatric_beds) > 0 
                    THEN LEAST(
                        (covid_beds::float / 
                        (occupied_adult_beds + occupied_pediatric_beds) * 100),
                        100
                    )
                    ELSE 0 
                END as covid_percentage
            FROM adjusted_stats
            ORDER BY week, state
            """
            trend_data = pd.read_sql_query(
                trend_query, 
                conn, 
                params=[selected_date, selected_date]
            )

            if view_type == "US Regions":
                region_trends = []
                for week in trend_data['week'].unique():
                    week_data = trend_data[trend_data['week'] == week]
                    region_data = get_region_stats(week_data)
                    region_data['week'] = week
                    region_trends.append(region_data)
                trend_data = pd.concat(region_trends)
                group_col = 'region'
            else:
                trend_data = trend_data[trend_data['state'].isin(selected_states)]
                group_col = 'state'

            if not trend_data.empty:
                # Create trend visualization
                trend_fig = go.Figure()
                
                for group in trend_data[group_col].unique():
                    group_data = trend_data[trend_data[group_col] == group]
                    
                    trend_fig.add_trace(go.Scatter(
                        x=group_data['week'],
                        y=group_data['occupancy_rate'],
                        name=f'{group} - Occupancy Rate',
                        mode='lines+markers'
                    ))
                
                trend_fig.update_layout(
                    title=f'Hospital Occupancy Trends by {"Region" if view_type == "US Regions" else "State"}',
                    xaxis_title='Week',
                    yaxis_title='Occupancy Rate (%)',
                    showlegend=True,
                    yaxis_range=[0, 100],
                    hovermode='x unified'
                )
                
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.warning("No trend data available for the selected criteria.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your database connection and try again.")

if __name__ == "__main__":
    main()
