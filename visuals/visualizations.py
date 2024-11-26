# visualizations.py
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def plot_comparison_chart(data):
    """Plot comparison bar chart for occupancy and COVID percentage."""
    fig = px.bar(data,
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
    st.plotly_chart(fig, use_container_width=True)


def plot_state_maps(state_metrics, selected_date):
    """Plot state-level maps for COVID cases and bed utilization."""
    # COVID Cases Map
    fig_covid = px.choropleth(
        state_metrics,
        locations='state',
        locationmode="USA-states",
        color='covid_cases',
        scope="usa",
        title=f"COVID-19 Cases by State ({selected_date})",
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
        title=f"Hospital Bed Utilization by State (%) ({selected_date})",
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


def plot_weekly_reporting(wow_comparison):
    """Plot week-over-week reporting hospital counts."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=wow_comparison['week'],
        y=wow_comparison['hospital_count'],
        name='Reporting Hospitals'
    ))
    fig.update_layout(
        title='Number of Reporting Hospitals by Week',
        xaxis_title='Week',
        yaxis_title='Number of Hospitals',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_bed_trends(bed_comparison):
    """Plot bed availability trends over weeks."""
    fig = go.Figure()
    # Total occupied beds trend
    fig.add_trace(go.Scatter(
        x=bed_comparison['week'],
        y=bed_comparison['occupied_adult_beds'] +
        bed_comparison['occupied_pediatric_beds'],
        name='Total Occupied Beds',
        mode='lines+markers'
    ))
    # COVID beds trend
    fig.add_trace(go.Scatter(
        x=bed_comparison['week'],
        y=bed_comparison['covid_beds'],
        name='COVID Beds',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Hospital Bed Utilization Trend',
        xaxis_title='Week',
        yaxis_title='Number of Beds',
        showlegend=True,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_quality_chart(quality_stats):
    """Plot hospital occupancy rate by quality rating."""
    fig = px.bar(
        quality_stats,
        x='overall_quality_rating',
        y='occupancy_rate',
        title='Hospital Occupancy Rate by Quality Rating',
        labels={
            'overall_quality_rating': 'Hospital Quality Rating (Stars)',
            'occupancy_rate': 'Occupancy Rate (%)'
        }
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def plot_trend_data(trend_data):
    """Plot occupancy and COVID percentage trends."""
    trend_fig = px.line(
        trend_data,
        x='week',
        y='occupancy_rate',
        color='state',
        title='Hospital Occupancy Rate Over Time',
        labels={
            'occupancy_rate': 'Occupancy Rate (%)',
            'week': 'Week',
            'state': 'State'
        }
    )
    st.plotly_chart(trend_fig, use_container_width=True)
