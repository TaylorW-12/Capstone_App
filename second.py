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
    # Remove travel_distance_home (always 0) and use travel_distance_away for travel_distance
    if 'travel_distance_away' in df.columns:
        df['travel_distance'] = df['travel_distance_away']
    return df

@st.cache_data
def train_rf_model(df, metrics, target_flag):
    """
    Train Random Forest model to predict impact of flag on selected metrics
    target_flag: 'isaway', 'is_thursday', 'intl', or 'extended_away_games'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    # Prepare data
    model_df = df.copy()
    
    # Filter to rows with valid flag data
    model_df = model_df[model_df[target_flag].notna()].copy()
    
    if len(model_df) == 0:
        return None, None, None
    
    # Select features (metrics)
    available_metrics = [m for m in metrics if m in model_df.columns]
    if not available_metrics:
        return None, None, None
    
    X = model_df[available_metrics].fillna(0)
    y = model_df[target_flag].astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=99)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.DataFrame({
        'metric': available_metrics,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    metrics_dict = {
        'accuracy': accuracy,
        'f1_score': f1,
        'n_samples': len(model_df)
    }
    
    return importances, metrics_dict, model

@st.cache_data
def calculate_flag_impact(df, metrics, flag):
    """
    Calculate average difference in metrics when flag=1 vs flag=0
    """
    results = []
    
    for metric in metrics:
        if metric not in df.columns or flag not in df.columns:
            continue
        
        flag_1 = df[df[flag] == 1][metric].mean()
        flag_0 = df[df[flag] == 0][metric].mean()
        
        if pd.notna(flag_1) and pd.notna(flag_0):
            difference = flag_1 - flag_0
            pct_change = (difference / flag_0 * 100) if flag_0 != 0 else 0
            
            results.append({
                'metric': metric,
                'flag_0_avg': flag_0,
                'flag_1_avg': flag_1,
                'difference': difference,
                'pct_change': pct_change,
                'direction': 'up' if difference > 0 else 'down'
            })
    
    return pd.DataFrame(results)

def calculate_travel_impact(df, season, entity_type, entity_name, metrics, home_away='both', use_team_avg=False):
    """
    Calculate how traveling over average distance impacts stats
    entity_type: 'team' or 'player'
    home_away: 'away' or 'both'
    use_team_avg: if True, compare to team's own average instead of season average
    """
    # Filter data
    season_df = df[df['season'] == season].copy()
    
    # Determine which travel distance column to use
    if home_away == 'away' and 'travel_distance' in season_df.columns:
        season_df['travel_distance'] = season_df['travel_distance']
    elif home_away == 'both':
        # Use travel_distance (which is travel_distance_away)
        if 'travel_distance' not in season_df.columns:
            st.warning("No travel distance column found in dataset")
            return None, None, None
    else:
        st.warning(f"Travel distance column for '{home_away}' not found in dataset")
        return None, None, None
    
    # Filter by entity
    entity_col = 'team' if entity_type == 'team' else 'player_display_name'
    entity_df = season_df[season_df[entity_col] == entity_name].copy()
    
    if len(entity_df) == 0:
        return None, None, None
    
    # Sort by date if available, otherwise by index
    if 'game_date' in entity_df.columns:
        entity_df = entity_df.sort_values('game_date')
    elif 'week' in entity_df.columns:
        entity_df = entity_df.sort_values('week')
    
    # Calculate average - either team's own average or season average
    if use_team_avg:
        avg_distance = entity_df['travel_distance'].mean()
    else:
        avg_distance = season_df['travel_distance'].mean()
    
    # Calculate miles over average
    entity_df['miles_over_avg'] = entity_df['travel_distance'] - avg_distance
    
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
        return impact_df, entity_df, avg_distance
    
    return None, None, avg_distance

def create_moving_average_chart(entity_df, metric, window=3):
    """Create line chart showing metric with moving average"""
    
    # Calculate moving average
    entity_df_sorted = entity_df.sort_values('week' if 'week' in entity_df.columns else entity_df.index.name or 'index')
    entity_df_sorted[f'{metric}_ma'] = entity_df_sorted[metric].rolling(window=window, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=entity_df_sorted.index if 'week' not in entity_df_sorted.columns else entity_df_sorted['week'],
        y=entity_df_sorted[metric],
        mode='lines+markers',
        name='Actual',
        line=dict(color='lightblue', width=2),
        marker=dict(size=6)
    ))
    
    # Moving average
    fig.add_trace(go.Scatter(
        x=entity_df_sorted.index if 'week' not in entity_df_sorted.columns else entity_df_sorted['week'],
        y=entity_df_sorted[f'{metric}_ma'],
        mode='lines',
        name=f'{window}-Game Moving Avg',
        line=dict(color='#ff6b6b', width=3)
    ))
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} - Season Progression',
        xaxis_title='Week' if 'week' in entity_df_sorted.columns else 'Game',
        yaxis_title=metric.replace("_", " ").title(),
        height=400,
        hovermode='x unified'
    )
    
    return fig
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

def create_correlation_heatmap(entity_df, metrics):
    """Create correlation heatmap for selected metrics and travel distance"""
    # Prepare data for correlation
    corr_cols = ['miles_over_avg', 'travel_distance'] + [m for m in metrics if m in entity_df.columns]
    corr_data = entity_df[corr_cols].copy()
    
    # Calculate correlation matrix
    corr_matrix = corr_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix: Metrics vs Travel Distance',
        xaxis_title='',
        yaxis_title='',
        height=500,
        xaxis={'tickangle': -45}
    )
    
    return fig, corr_matrix

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
        "Select Season",
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
        "Select Team",
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
    st.sidebar.subheader("Select Metrics (Max 3)")
    
    # Organize metrics by category with expanders
    selected_metrics = []
    
    with st.sidebar.expander("Passing Stats"):
        available_pass = [stat for stat in pass_stats if stat in df.columns]
        pass_selected = st.multiselect("Select passing metrics:", available_pass, key='pass_metrics')
        selected_metrics.extend(pass_selected)
    
    with st.sidebar.expander("Rushing Stats"):
        available_rush = [stat for stat in rush_stats if stat in df.columns]
        rush_selected = st.multiselect("Select rushing metrics:", available_rush, key='rush_metrics')
        selected_metrics.extend(rush_selected)
    
    with st.sidebar.expander("Receiving Stats"):
        available_rec = [stat for stat in rec_stats if stat in df.columns]
        rec_selected = st.multiselect("Select receiving metrics:", available_rec, key='rec_metrics')
        selected_metrics.extend(rec_selected)
    
    # Enforce max 3 metrics
    if len(selected_metrics) > 3:
        st.sidebar.error("Please select a maximum of 3 metrics")
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
    tab1, tab2, tab3 = st.tabs(["📊 Data View", "✈️ Travel Impact Analysis", "🔬 Flag Impact Analysis"])
    
    with tab1:
        st.subheader("Filtered Data")
        if selected_metrics:
            display_cols = ['season', 'team', 'player_display_name'] + selected_metrics
            # Add travel distance column if it exists
            if 'travel_distance' in result_df.columns:
                display_cols.append('travel_distance')
            display_cols = [col for col in display_cols if col in result_df.columns]
            st.dataframe(result_df[display_cols], use_container_width=True, height=400)
        else:
            st.dataframe(result_df, use_container_width=True, height=400)
    
    with tab2:
        st.subheader("✈️ Travel Distance Impact on Performance")
        
        # Add home/away filter
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            home_away_filter = st.selectbox(
                "Game Location:",
                options=['Both', 'Away Games Only'],
                key='home_away_filter'
            )
        
        with col_filter2:
            comparison_type = st.selectbox(
                "Compare To:",
                options=['Season Average', 'Team\'s Own Average'],
                key='comparison_type'
            )
        
        with col_filter3:
            ma_window = st.slider(
                "Moving Avg Window:",
                min_value=2,
                max_value=10,
                value=3,
                key='ma_window'
            )
        
        # Map selection to parameter
        home_away_map = {
            'Both': 'both',
            'Away Games Only': 'away'
        }
        home_away_param = home_away_map[home_away_filter]
        use_team_avg = (comparison_type == 'Team\'s Own Average')
        
        if selected_season == 'All':
            st.warning("Please select a specific season to analyze travel impact")
        elif selected_team == 'All' and selected_player == 'All':
            st.warning("Please select a specific team or player to analyze travel impact")
        elif not selected_metrics:
            st.warning("Please select at least one metric to analyze")
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
            
            st.info(f"**Analysis for {entity_display} in {selected_season} season ({home_away_filter})**")
            
            # Calculate travel impact
            impact_df, entity_df, avg_distance = calculate_travel_impact(
                df, selected_season, entity_type, entity_name, selected_metrics, home_away_param, use_team_avg
            )
            
            if impact_df is not None and entity_df is not None:
                # Show average travel distance based on filter
                season_df_temp = df[df['season'] == selected_season].copy()
                
                if use_team_avg:
                    # Use the avg_distance from the function (team's own average)
                    comparison_distance = avg_distance
                    comparison_label = f"{entity_display}'s Avg"
                else:
                    # Calculate season average
                    if home_away_param == 'both' or home_away_param == 'away':
                        comparison_distance = season_df_temp['travel_distance'].mean()
                    comparison_label = "Season Avg"
                
                avg_entity_distance = entity_df['travel_distance'].mean()
                
                distance_label = "Travel Distance"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{comparison_label} {distance_label}", f"{comparison_distance:.0f} miles")
                with col2:
                    st.metric(f"{entity_display} Avg", f"{avg_entity_distance:.0f} miles")
                with col3:
                    diff = avg_entity_distance - comparison_distance
                    st.metric("Difference", f"{diff:+.0f} miles")
                
                st.divider()
                
                # Add correlation heatmap at the top
                st.subheader("Correlation Analysis")
                st.write("This heatmap shows how strongly each metric correlates with travel distance and with each other.")
                
                fig_corr, corr_matrix = create_correlation_heatmap(entity_df, selected_metrics)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Display key correlations
                st.write("**Key Insights:**")
                cols_insight = st.columns(len(selected_metrics))
                for idx, metric in enumerate(selected_metrics):
                    if metric in corr_matrix.index:
                        corr_with_miles = corr_matrix.loc[metric, 'miles_over_avg']
                        with cols_insight[idx]:
                            if abs(corr_with_miles) > 0.5:
                                strength = "Strong"
                                color = "🔴" if corr_with_miles < 0 else "🟢"
                            elif abs(corr_with_miles) > 0.3:
                                strength = "Moderate"
                                color = "🟠"
                            else:
                                strength = "Weak"
                                color = "⚪"
                            
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=f"{corr_with_miles:.2f}",
                                delta=f"{color} {strength}"
                            )
                
                st.divider()
                
                # Create visualizations for each metric
                for metric in selected_metrics:
                    st.subheader(f"{metric.replace('_', ' ').title()}")
                    
                    # Create tabs for different views
                    tab_a, tab_b, tab_c = st.tabs(["📊 Distance Impact", "📈 Season Trend", "🔍 Scatter Plot"])
                    
                    with tab_a:
                        # Binned chart
                        fig1 = create_travel_impact_chart(impact_df, metric)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with tab_b:
                        # Moving average chart
                        fig_ma = create_moving_average_chart(entity_df, metric, ma_window)
                        st.plotly_chart(fig_ma, use_container_width=True)
                    
                    with tab_c:
                        # Scatter plot with trendline
                        fig2 = create_scatter_plot(entity_df, metric)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Calculate correlation
                    if metric in entity_df.columns and 'miles_over_avg' in entity_df.columns:
                        corr = entity_df[[metric, 'miles_over_avg']].corr().iloc[0, 1]
                        
                        if abs(corr) > 0.3:
                            direction = "increases" if corr > 0 else "decreases"
                            strength = "strong" if abs(corr) > 0.6 else "moderate"
                            st.info(f" **{strength.title()} correlation detected:** {metric.replace('_', ' ').title()} {direction} by approximately **{abs(corr)*100:.1f}%** correlation as travel distance increases")
                        else:
                            st.info(f"**Weak correlation:** Travel distance has minimal impact on {metric.replace('_', ' ').title()}")
                    
                    st.divider()
            else:
                st.error("Unable to calculate travel impact. Please check your data.")
    
    with tab3:
        st.subheader("Flag Impact Analysis")
        st.write("Analyze which metrics are most affected by away games, Thursday games, extended away games, and international games.")
        
        # Flag selection
        col1, col2 = st.columns(2)
        with col1:
            flag_options = {
                'Away Games': 'isaway',
                'Thursday Games': 'is_thursday',
                'Extended Away Games': 'extended_away_games',
                'International Games': 'intl'
            }
            selected_flag_name = st.selectbox(
                "Select Flag to Analyze:",
                options=list(flag_options.keys()),
                key='flag_select'
            )
            selected_flag = flag_options[selected_flag_name]
        
        with col2:
            analysis_method = st.selectbox(
                "Analysis Method:",
                options=['Statistical Comparison', 'Random Forest Model'],
                key='analysis_method'
            )
        
        if not selected_metrics:
            st.warning("Please select at least one metric from the sidebar to analyze")
        elif selected_flag not in result_df.columns:
            st.warning(f"Flag '{selected_flag_name}' not found in dataset")
        else:
            st.info(f"**Analyzing impact of {selected_flag_name} on selected metrics**")
            
            if analysis_method == 'Statistical Comparison':
                # Calculate statistical impact
                impact_results = calculate_flag_impact(result_df, selected_metrics, selected_flag)
                
                if not impact_results.empty:
                    st.subheader("Metric Impact Summary")
                    
                    # Display metric cards
                    cols = st.columns(min(len(selected_metrics), 3))
                    for idx, row in impact_results.iterrows():
                        col_idx = idx % 3
                        with cols[col_idx]:
                            direction_emoji = "📈" if row['direction'] == 'up' else "📉"
                            color = "normal" if abs(row['pct_change']) < 5 else "inverse"
                            
                            st.metric(
                                label=f"{direction_emoji} {row['metric'].replace('_', ' ').title()}",
                                value=f"{row['pct_change']:.1f}%",
                                delta=f"{row['difference']:.2f}",
                                delta_color=color
                            )
                            
                            with st.expander("Details"):
                                st.write(f"**When {selected_flag_name} = No:** {row['flag_0_avg']:.2f}")
                                st.write(f"**When {selected_flag_name} = Yes:** {row['flag_1_avg']:.2f}")
                                st.write(f"**Difference:** {row['difference']:.2f}")
                    
                    st.divider()
                    
                    # Visualization
                    st.subheader("Visual Comparison")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Flag = 0',
                        x=impact_results['metric'],
                        y=impact_results['flag_0_avg'],
                        marker_color='#4ecdc4'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Flag = 1',
                        x=impact_results['metric'],
                        y=impact_results['flag_1_avg'],
                        marker_color='#ff6b6b'
                    ))
                    
                    fig.update_layout(
                        barmode='group',
                        title=f'Metric Averages: {selected_flag_name} Impact',
                        xaxis_title='Metrics',
                        yaxis_title='Average Value',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("Detailed Results")
                    display_impact = impact_results.copy()
                    display_impact['metric'] = display_impact['metric'].str.replace('_', ' ').str.title()
                    display_impact = display_impact.round(2)
                    st.dataframe(display_impact, use_container_width=True)
                
                else:
                    st.warning("No results to display. Check your data and selected metrics.")
            
            else:  # Random Forest Model
                st.subheader("Random Forest Feature Importance")
                
                with st.spinner("Training Random Forest model..."):
                    importances, metrics_dict, model = train_rf_model(result_df, selected_metrics, selected_flag)
                
                if importances is not None:
                    # Model performance
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Accuracy", f"{metrics_dict['accuracy']:.2%}")
                    with col2:
                        st.metric("F1 Score", f"{metrics_dict['f1_score']:.2f}")
                    with col3:
                        st.metric("Training Samples", f"{metrics_dict['n_samples']:,}")
                    
                    st.divider()
                    
                    # Feature importance chart
                    st.subheader("Feature Importance Rankings")
                    st.write("Higher importance = metric is more predictive of the flag condition")
                    
                    fig = go.Figure(go.Bar(
                        x=importances['importance'],
                        y=importances['metric'],
                        orientation='h',
                        marker_color='#ff6b6b'
                    ))
                    
                    fig.update_layout(
                        title=f'Which Metrics Best Predict {selected_flag_name}?',
                        xaxis_title='Importance Score',
                        yaxis_title='Metric',
                        height=max(400, len(selected_metrics) * 50),
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    top_metric = importances.iloc[0]
                    st.info(f"**Key Insight:** `{top_metric['metric'].replace('_', ' ').title()}` is the most important metric "
                           f"for predicting {selected_flag_name} (importance: {top_metric['importance']:.3f})")
                    
                    # Detailed table
                    st.subheader("Feature Importance Table")
                    display_imp = importances.copy()
                    display_imp['metric'] = display_imp['metric'].str.replace('_', ' ').str.title()
                    display_imp['importance'] = display_imp['importance'].round(4)
                    st.dataframe(display_imp, use_container_width=True)
                
                else:
                    st.error("Unable to train model. Check your data and selected metrics.")
    
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
