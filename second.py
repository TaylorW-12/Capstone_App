import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.image("same_color.png", width=40000)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
    }
    
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Define metric categories
pass_stats = ["attempts", "completions", "passing_yards", "passing_tds", "interceptions",
              "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
              "passing_air_yards", "passing_yards_after_catch",
              "passing_first_downs", "passing_2pt_conversions"]

rush_stats = ["carries", "rushing_yards", "rushing_tds",
              "rushing_fumbles", "rushing_fumbles_lost", "rushing_first_downs"]

rec_stats = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
             'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch', 
             'receiving_first_downs', 'receiving_2pt_conversions', 'racr', 'target_share', 
             'air_yards_share', 'wopr']

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("df_merged.csv")
    return df

def calculate_travel_impact(df, season, entity_type, entity_name, metrics):
    """
    Calculate how traveling over average distance impacts stats
    entity_type: 'team' or 'player'
    """
    # Filter data
    season_df = df[df['season'] == season].copy()
    
    # Calculate average travel distance for the season
    if 'travel_distance' in season_df.columns:
        avg_distance = season_df['travel_distance'].mean()
        season_df['miles_over_avg'] = season_df['travel_distance'] - avg_distance
    else:
        st.warning("'travel_distance' column not found in dataset")
        return None, None
    
    # Filter by entity
    entity_col = 'team' if entity_type == 'team' else 'player_display_name'
    entity_df = season_df[season_df[entity_col] == entity_name].copy()
    
    if len(entity_df) == 0:
        return None, None
    
    # Create bins for miles over average
    entity_df['distance_bin'] = pd.cut(entity_df['miles_over_avg'], 
                                        bins=[-np.inf, -500, -100, 100, 500, np.inf],
                                        labels=['<-500', '-500 to -100', '-100 to 100', 
                                               '100 to 500', '>500'])
    
    # Calculate average stats per distance bin
    results = []
    for metric in metrics:
        if metric in entity_df.columns:
            grouped = entity_df.groupby('distance_bin')[metric].agg(['mean', 'count']).reset_index()
            grouped['metric'] = metric
            results.append(grouped)
    
    if results:
        impact_df = pd.concat(results, ignore_index=True)
        return impact_df, entity_df
    
    return None, None

def create_travel_impact_chart(impact_df, metric):
    """Create line chart showing metric change by travel distance"""
    metric_data = impact_df[impact_df['metric'] == metric]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metric_data['distance_bin'].astype(str),
        y=metric_data['mean'],
        mode='lines+markers',
        name=metric,
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} vs Travel Distance',
        xaxis_title='Miles Over Average',
        yaxis_title=f'Average {metric.replace("_", " ").title()}',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_scatter_plot(entity_df, metric):
    """Create scatter plot of metric vs miles over average"""
    fig = px.scatter(
        entity_df,
        x='miles_over_avg',
        y=metric,
        trendline='ols',
        title=f'{metric.replace("_", " ").title()} vs Miles Over Average',
        labels={
            'miles_over_avg': 'Miles Over Season Average',
            metric: metric.replace("_", " ").title()
        },
        height=400
    )
    
    fig.update_traces(marker=dict(size=8, color='#4ecdc4'))
    
    return fig

def main():
    # NFL data
    df = load_data()
     
    # SIDEBAR FILTERS
    st.sidebar.title("🔍 Filters")

    # Initialize session state for filters
    if 'selected_season' not in st.session_state:
        st.session_state.selected_season = None
    if 'selected_team' not in st.session_state:
        st.session_state.selected_team = None
    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None
    
    # Season filter (always available)
    seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.selectbox(
        "📅 Select Season",
        options=['All'] + seasons,
        key='season_select'
    )
    st.session_state.selected_season = selected_season
    
    # Team filter (filtered by season)
    if selected_season != 'All':
        filtered_teams = df[df['season'] == selected_season]['team'].unique()
    else:
        filtered_teams = df['team'].unique()
    
    teams = sorted(filtered_teams)
    selected_team = st.sidebar.selectbox(
        "🏈 Select Team",
        options=['All'] + teams,
        key='team_select'
    )
    st.session_state.selected_team = selected_team
    
    # Player filter (filtered by season and team)
    filtered_df = df.copy()
    
    if selected_season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
    
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    
    players = sorted(filtered_df['player_display_name'].unique())
    selected_player = st.sidebar.selectbox(
        "👤 Select Player",
        options=['All'] + players,
        key='player_select'
    )
    st.session_state.selected_player = selected_player
    
    # Metric Selection
    st.sidebar.divider()
    st.sidebar.subheader("📊 Select Metrics (Max 3)")
    
    # Organize metrics by category with expanders
    selected_metrics = []
    
    with st.sidebar.expander("🎯 Passing Stats"):
        available_pass = [stat for stat in pass_stats if stat in df.columns]
        pass_selected = st.multiselect("Select passing metrics:", available_pass, key='pass_metrics')
        selected_metrics.extend(pass_selected)
    
    with st.sidebar.expander("🏃 Rushing Stats"):
        available_rush = [stat for stat in rush_stats if stat in df.columns]
        rush_selected = st.multiselect("Select rushing metrics:", available_rush, key='rush_metrics')
        selected_metrics.extend(rush_selected)
    
    with st.sidebar.expander("🤲 Receiving Stats"):
        available_rec = [stat for stat in rec_stats if stat in df.columns]
        rec_selected = st.multiselect("Select receiving metrics:", available_rec, key='rec_metrics')
        selected_metrics.extend(rec_selected)
    
    # Enforce max 3 metrics
    if len(selected_metrics) > 3:
        st.sidebar.error("⚠️ Please select a maximum of 3 metrics")
        selected_metrics = selected_metrics[:3]
    
    st.sidebar.info(f"**Selected: {len(selected_metrics)}/3 metrics**")
    
    st.sidebar.divider()
    st.sidebar.info(f"**Active Filters:**\n\n Season: `{selected_season}`\n\n Team: `{selected_team}`\n\n Player: `{selected_player}`")
    
    # Apply all filters to show final results
    result_df = df.copy()
    
    if selected_season != 'All':
        result_df = result_df[result_df['season'] == selected_season]
    
    if selected_team != 'All':
        result_df = result_df[result_df['team'] == selected_team]
    
    if selected_player != 'All':
        result_df = result_df[result_df['player_display_name'] == selected_player]
    
    # MAIN CONTENT AREA
    st.markdown('<h1 class="main-header">NFL Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(result_df))
    with col2:
        st.metric("Unique Players", result_df['player_display_name'].nunique())
    with col3:
        st.metric("Unique Teams", result_df['team'].nunique())
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["📊 Data View", "✈️ Travel Impact Analysis"])
    
    with tab1:
        st.subheader("Filtered Data")
        if selected_metrics:
            display_cols = ['season', 'team', 'player_display_name'] + selected_metrics
            if 'travel_distance' in result_df.columns:
                display_cols.append('travel_distance')
            display_cols = [col for col in display_cols if col in result_df.columns]
            st.dataframe(result_df[display_cols], use_container_width=True, height=400)
        else:
            st.dataframe(result_df, use_container_width=True, height=400)
    
    with tab2:
        st.subheader("✈️ Travel Distance Impact on Performance")
        
        if selected_season == 'All':
            st.warning("⚠️ Please select a specific season to analyze travel impact")
        elif selected_team == 'All' and selected_player == 'All':
            st.warning("⚠️ Please select a specific team or player to analyze travel impact")
        elif not selected_metrics:
            st.warning("⚠️ Please select at least one metric to analyze")
        else:
            # Determine entity type and name
            if selected_player != 'All':
                entity_type = 'player'
                entity_name = selected_player
                entity_display = f"Player: {selected_player}"
            else:
                entity_type = 'team'
                entity_name = selected_team
                entity_display = f"Team: {selected_team}"
            
            st.info(f"**Analysis for {entity_display} in {selected_season} season**")
            
            # Calculate travel impact
            impact_df, entity_df = calculate_travel_impact(
                df, selected_season, entity_type, entity_name, selected_metrics
            )
            
            if impact_df is not None and entity_df is not None:
                # Show average travel distance
                avg_season_distance = df[df['season'] == selected_season]['travel_distance'].mean()
                avg_entity_distance = entity_df['travel_distance'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Season Avg Distance", f"{avg_season_distance:.0f} miles")
                with col2:
                    st.metric(f"{entity_display} Avg Distance", f"{avg_entity_distance:.0f} miles")
                with col3:
                    diff = avg_entity_distance - avg_season_distance
                    st.metric("Difference", f"{diff:+.0f} miles")
                
                st.divider()
                
                # Create visualizations for each metric
                for metric in selected_metrics:
                    st.subheader(f"📈 {metric.replace('_', ' ').title()}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Binned chart
                        fig1 = create_travel_impact_chart(impact_df, metric)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Scatter plot with trendline
                        fig2 = create_scatter_plot(entity_df, metric)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Calculate correlation
                    if metric in entity_df.columns and 'miles_over_avg' in entity_df.columns:
                        corr = entity_df[[metric, 'miles_over_avg']].corr().iloc[0, 1]
                        
                        if abs(corr) > 0.3:
                            direction = "increases" if corr > 0 else "decreases"
                            strength = "strong" if abs(corr) > 0.6 else "moderate"
                            st.info(f"📊 **{strength.title()} correlation detected:** {metric.replace('_', ' ').title()} {direction} by approximately **{abs(corr)*100:.1f}%** correlation as travel distance increases")
                        else:
                            st.info(f"📊 **Weak correlation:** Travel distance has minimal impact on {metric.replace('_', ' ').title()}")
                    
                    st.divider()
            else:
                st.error("Unable to calculate travel impact. Please check your data.")
    
if __name__ == "__main__":
    main()

# Header
#st.markdown('<h1 class="main-header">🤖 State of Large Language Models</h1>', unsafe_allow_html=True)
#st.markdown("---")

# Sidebar filters


# Main content tabs
tab1, tab2, tab3 = st.tabs(["Overall Analysis", "Model Simulation", "Insights"])

with tab1:
    st.header("Overall Analysis")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Models Tracked", 
            #len(perf_data),
            delta="2 new this month"
        )
    
    with col2:
        #avg_mmlu = perf_data['MMLU Score'].mean()
        st.metric(
            "Average MMLU Score", 
            #"{avg_mmlu:.1f}",
            delta="5.2% vs last quarter"
        )
    
    with col3:
        #top_model = perf_data.loc[perf_data['MMLU Score'].idxmax(), 'Model']
        st.metric(
            "Leading Model", 
            #top_model,
            delta="GPT-4 maintains lead"
        )
    
    with col4:
        #min_cost = perf_data['Cost per 1M tokens'].min()
        st.metric(
            "Lowest Cost", 
            #f"${min_cost}",
            delta="-60% cost reduction"
        )
    
    st.markdown("---")
    
    # Filter data based on selection
    #filtered_data = perf_data[perf_data['Model'].isin(selected_models)]
    
    # Performance scatter plot
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_scatter = px.scatter(
            #filtered_data,
            x='MMLU Score',
            y='HumanEval Score',
            size='Parameters (B)',
            color='Cost per 1M tokens',
            hover_name='Model',
            title="Model Performance vs Cost",
            labels={'MMLU Score': 'MMLU Score (%)', 'HumanEval Score': 'HumanEval Score (%)'},
            color_continuous_scale='Viridis'
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Model ranking
        st.subheader("Top Performers")
        #ranking_metric = metric_type
        #if ranking_metric == "Cost per 1M tokens":
            #top_models = filtered_data.nsmallest(5, ranking_metric)
       # else:
            #top_models = filtered_data.nlargest(5, ranking_metric)
        
#         for i, (_, model) in enumerate(top_models.iterrows(), 1):
#             value = model[ranking_metric]
#             if ranking_metric == "Cost per 1M tokens":
#                 st.write(f"{i}. **{model['Model']}** - ${value}")
#             else:
#                 st.write(f"{i}. **{model['Model']}** - {value}")

# with tab2:
#     st.header("Model Simulation")
    

#     # Key metrics row
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             "Total Models Tracked", 
#             len(perf_data),
#             delta="2 new this month"
#         )
    
#     with col2:
#         avg_mmlu = perf_data['MMLU Score'].mean()
#         st.metric(
#             "Average MMLU Score", 
#             f"{avg_mmlu:.1f}",
#             delta="5.2% vs last quarter"
#         )
    
#     with col3:
#         top_model = perf_data.loc[perf_data['MMLU Score'].idxmax(), 'Model']
#         st.metric(
#             "Leading Model", 
#             top_model,
#             delta="GPT-4 maintains lead"
#         )
    
#     with col4:
#         min_cost = perf_data['Cost per 1M tokens'].min()
#         st.metric(
#             "Lowest Cost", 
#             f"${min_cost}",
#             delta="-60% cost reduction"
#         )
    
#     st.markdown("---")

#     # Usage trends
#     col1, col2 = st.columns(2)
    
#     with col1:
#         fig_api = px.line(
#             usage_data,
#             x='Date',
#             y='API Calls (Millions)',
#             title="API Usage Growth",
#             markers=True
#         )
#         fig_api.update_layout(height=400)
#         st.plotly_chart(fig_api, use_container_width=True)
    
#     with col2:
#         fig_users = px.line(
#             usage_data,
#             x='Date',
#             y='Active Users (Thousands)',
#             title="Active User Growth",
#             markers=True,
#             color_discrete_sequence=['#ff6b6b']
#         )
#         fig_users.update_layout(height=400)
#         st.plotly_chart(fig_users, use_container_width=True)
    
#     # Market size projection
#     st.subheader("Market Projections")
    
#     future_dates = pd.date_range(start='2024-04-01', end='2025-12-01', freq='M')
#     # Simple exponential growth projection
#     last_value = usage_data['API Calls (Millions)'].iloc[-1]
#     growth_rate = 1.15  # 15% monthly growth
#     projections = [last_value * (growth_rate ** i) for i in range(1, len(future_dates) + 1)]
    
#     future_data = pd.DataFrame({
#         'Date': future_dates,
#         'Projected API Calls (Millions)': projections
#     })
    
#     # Combine historical and projected data
#     combined_data = pd.concat([
#         usage_data[['Date', 'API Calls (Millions)']].rename(columns={'API Calls (Millions)': 'Value'}),
#         future_data[['Date', 'Projected API Calls (Millions)']].rename(columns={'Projected API Calls (Millions)': 'Value'})
#     ])
#     combined_data['Type'] = ['Historical'] * len(usage_data) + ['Projected'] * len(future_data)
    
#     fig_projection = px.line(
#         combined_data,
#         x='Date',
#         y='Value',
#         color='Type',
#         title="API Usage: Historical vs Projected",
#         labels={'Value': 'API Calls (Millions)'}
#     )
#     st.plotly_chart(fig_projection, use_container_width=True)

with tab3:
    st.header("Key Insights & Analysis")
    
    # Insights cards
    insights = [
        {
            "title": "🚀 Performance Leaders",
            "content": "GPT-4 and Claude-3 continue to dominate benchmark scores, with both achieving 80%+ on MMLU evaluations."
        },
        {
            "title": "💰 Cost Efficiency",
            "content": "Open-source models like Mistral-7B offer 60x cost savings compared to premium models while maintaining reasonable performance."
        },
        {
            "title": "📈 Rapid Growth",
            "content": "API usage has grown 10x in the past year, with software development leading adoption at 85% penetration."
        },
        {
            "title": "🔮 Future Outlook",
            "content": "Multimodal capabilities and reduced inference costs are driving the next wave of LLM applications."
        }
    ]
    
    # for insight in insights:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{insight['title']}</h3>
    #         <p>{insight['content']}</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    #     st.markdown("<br>", unsafe_allow_html=True)
    
    # # Trend analysis
    # if show_trends:
    #     st.subheader("Trend Analysis")
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         st.write("**Key Trends Observed:**")
    #         st.write("• Model performance plateauing on traditional benchmarks")
    #         st.write("• Shift focus to specialized and multimodal capabilities")
    #         st.write("• Increased emphasis on cost-efficiency and speed")
    #         st.write("• Growing enterprise adoption across all sectors")
        
    #     with col2:
    #         st.write("**Emerging Patterns:**")
    #         st.write("• Open-source models closing the performance gap")
    #         st.write("• Real-time applications driving infrastructure demands")
    #         st.write("• Regulatory considerations shaping development")
    #         st.write("• Fine-tuning becoming standard practice")

# Footer
st.markdown("---")
