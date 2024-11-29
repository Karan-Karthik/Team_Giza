"""
Hospital COVID-19 Dashboard Application

This module implements a Streamlit-based dashboard for
visualizing and analyzing hospital COVID-19 data across
different states and regions. It provides various
visualizations and metrics including bed occupancy,
COVID cases, quality ratings, and temporal trends.

The dashboard features:
- Date-based filtering
- State/region selection
- Current week summary statistics
- Geographic comparisons
- State-level analysis
- Weekly reporting trends
- Bed availability analysis
- Hospital quality analysis
- COVID case mapping

Dependencies:
    - streamlit
    - plotly.express
    - pandas
    - numpy
    - custom database connection and query modules
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from database.connect import get_connection
from database.queries import (
    get_available_dates, get_available_states_and_regions,
    get_metrics, get_state_metrics_for_map, get_week_over_week_comparison,
    get_bed_availability_comparison, get_quality_rating_analysis,
    get_state_covid_map_data, get_hospitals_with_significant_changes,
    get_non_reporting_hospitals, get_trend_data, get_top_states_by_covid
)

from utils.utils import get_states_for_selection
from visuals.visualizations import (
    plot_comparison_chart, plot_state_maps, plot_weekly_reporting,
    plot_bed_trends, plot_quality_chart, plot_trend_data
)


def main():
    """
    Main application function that sets up and runs the Streamlit dashboard.

    The function performs the following operations:
    1. Sets up the page configuration and title
    2. Retrieves and validates available dates
    3. Creates filters for date and state/region selection
    4. Generates various sections of the dashboard:
        - Current week summary
        - Geographic comparison
        - State-level analysis
        - Weekly reporting summary
        - Bed availability trends
        - Hospital quality analysis
        - Trend analysis
        - Additional insights

    Returns:
        None
    """
    try:
        # Page setup
        st.set_page_config(
            page_title="Hospital COVID-19 Dashboard", layout="wide")
        st.title("Hospital COVID-19 Analysis Dashboard")

        # Get available dates from database
        with get_connection() as conn:
            available_dates = get_available_dates(conn)

        if not available_dates:
            st.error(
                "No data available. Please check the database connection."
                )
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
                options=['Northeast', 'Midwest', 'South', 'West'] +
                available_selections['States'],
                default=['Northeast'],
                help="You can select multiple states and/or regions"
            )

            if not selected_areas:
                st.warning(
                    "Please select at least one state or region to view data.")
                return

        # Current Week Summary
        st.header("Current Week Summary")
        with get_connection() as conn:
            states = get_states_for_selection(selected_areas)
            current_metrics = get_metrics(conn, selected_date, states)

            if current_metrics.empty:
                st.warning(f"No data available for {selected_date}")
                return

            # Calculate totals
            total_metrics = pd.DataFrame([{
                'reporting_hospitals': current_metrics[
                    'reporting_hospitals'
                    ].sum(),
                'total_beds': current_metrics['total_beds'].sum(),
                'occupied_beds': current_metrics['occupied_beds'].sum(),
                'covid_beds': current_metrics['covid_beds'].sum()
            }])

            total_metrics['occupancy_rate'] = np.where(
                total_metrics['total_beds'] > 0,
                np.minimum(total_metrics['occupied_beds'] /
                           total_metrics['total_beds'] * 100, 100),
                0
            )

            total_metrics['covid_percentage'] = np.where(
                total_metrics['occupied_beds'] > 0,
                np.minimum(total_metrics['covid_beds'] /
                           total_metrics['occupied_beds'] * 100, 100),
                0
            )

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Total Reporting Hospitals",
                f"{int(total_metrics['reporting_hospitals'].iloc[0]):,}"
                        )
            col2.metric("Overall Bed Occupancy",
                        f"{total_metrics['occupancy_rate'].iloc[0]:.1f}%")
            col3.metric("COVID % of Occupied Beds",
                        f"{total_metrics['covid_percentage'].iloc[0]:.1f}%")

            # Geographic comparison visualization
            st.header("Geographic Comparison")

            # Prepare data for visualization
            plot_data = current_metrics.copy()
            plot_comparison_chart(plot_data)

        # State Maps Section
        st.header("State-Level Analysis")

        with get_connection() as conn:
            state_metrics = get_state_metrics_for_map(conn, selected_date)

            if not state_metrics.empty:
                plot_state_maps(state_metrics, selected_date)
            else:
                st.warning(
                    "No state-level data available for the selected date.")

        # Week-over-week comparison
        st.header("Weekly Reporting Summary")
        with get_connection() as conn:
            wow_comparison = get_week_over_week_comparison(conn, selected_date)

            if not wow_comparison.empty:
                current_count = wow_comparison.iloc[0]['hospital_count']
                percent_change = wow_comparison.iloc[0]['percent_change']

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Current Week Reporting Hospitals",
                    f"{int(current_count):,}",
                    f"{percent_change:.1f}%" if not pd.isna(percent_change) else "N/A"
                    )

                plot_weekly_reporting(wow_comparison)

        # Bed availability comparison
        st.header("Bed Availability Trends")
        with get_connection() as conn:
            bed_comparison = get_bed_availability_comparison(
                conn, selected_date)

            if not bed_comparison.empty:
                # Create bed availability table
                st.subheader("4-Week Bed Availability Summary")
                formatted_bed_table = bed_comparison.copy()
                for col in formatted_bed_table.columns:
                    if col != 'week':
                        formatted_bed_table[col] = formatted_bed_table[col].map(
                            '{:,.0f}'.format)
                st.dataframe(formatted_bed_table)

                plot_bed_trends(bed_comparison)

        # Hospital quality rating analysis
        st.header("Hospital Quality Analysis")
        with get_connection() as conn:
            quality_stats = get_quality_rating_analysis(conn, selected_date)

            if not quality_stats.empty:
                plot_quality_chart(quality_stats)

                # Display quality stats table
                st.subheader("Detailed Quality Rating Statistics")
                formatted_quality_table = quality_stats.copy()
                formatted_quality_table['occupancy_rate'] = formatted_quality_table['occupancy_rate'].map(
                    '{:.1f}%'.format)
                formatted_quality_table = formatted_quality_table.rename(
                    columns={
                        'overall_quality_rating': 'Quality Rating',
                        'total_beds': 'Total Beds',
                        'occupied_beds': 'Occupied Beds',
                        'occupancy_rate': 'Occupancy Rate'})
                st.dataframe(formatted_quality_table)

        # Trend Analysis
        st.header("Trend Analysis")
        with get_connection() as conn:
            trend_data = get_trend_data(conn, states)

            if not trend_data.empty:
                plot_trend_data(trend_data)
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
            significant_changes = get_hospitals_with_significant_changes(
                conn, selected_date)

            if not significant_changes.empty:
                formatted_changes = significant_changes.copy()
                formatted_changes['current_cases'] = formatted_changes['current_cases'].map(
                    '{:,.0f}'.format)
                formatted_changes['previous_cases'] = formatted_changes['previous_cases'].map(
                    '{:,.0f}'.format)
                formatted_changes['absolute_change'] = formatted_changes['absolute_change'].map(
                    '{:+,.0f}'.format)
                formatted_changes['percent_change'] = formatted_changes['percent_change'].map(
                    lambda x: f"{x:+.1f}%" if pd.notnull(x) else "N/A")
                st.dataframe(formatted_changes)


        # Non-reporting Hospitals
        st.subheader("Recently Non-reporting Hospitals")
        with get_connection() as conn:
            non_reporting = get_non_reporting_hospitals(conn, selected_date)

            if not non_reporting.empty:
                formatted_non_reporting = non_reporting.copy()
                formatted_non_reporting['days_since_report'] = formatted_non_reporting['days_since_report'].map(
                    lambda x: f"{x} days" if pd.notnull(x) else "Never reported"
                )
                st.dataframe(formatted_non_reporting)


    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your database connection and try again.")


if __name__ == "__main__":
    main()
