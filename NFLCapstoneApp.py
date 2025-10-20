import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
    

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

performance_ratios = [
    'snap_share', 'pass_usage', 'pass_pct_of_offense', 'pass_air_yard_pct',
    'pass_yards_after_catch_pct', 'pass_average_air_yards', 'rusher_usage',
    'rusher_fumble_pct', 'rusher_yards_per_carry', 'receiver_usage',
    'receiver_efficiency', 'receiver_yac_pct', 'receiver_yards_per_reception',
    'receiver_yac_to_air_yards'
]

# Descriptions for performance ratios
performance_ratio_descriptions = {
    'snap_share': 'Player On-Field % (offensive_snaps / team_offensive_snaps)',
    'pass_usage': 'QB Pass Utilization (attempts / offensive_snaps)',
    'pass_pct_of_offense': 'Offense Pass Utilization (attempts / team_offensive_snaps)',
    'pass_air_yard_pct': 'Reliance on QB (passing_air_yards / passing_yards)',
    'pass_yards_after_catch_pct': 'Reliance on Receiver (passing_yards_after_catch / passing_yards)',
    'pass_average_air_yards': 'Average Air Yards Thrown',
    'rusher_usage': '% of RB snaps that were Rushes (carries / offensive_snaps)',
    'rusher_fumble_pct': 'Rusher Fumble % (rushing_fumbles / carries)',
    'rusher_yards_per_carry': 'Rusher Yards per Carry (rushing_yards / carries)',
    'receiver_usage': 'Receiver Utilization (targets / offensive_snaps)',
    'receiver_efficiency': 'Receiver Catch % (receptions / targets)',
    'receiver_yac_pct': '% of receiving yards after catch',
    'receiver_yards_per_reception': 'Yards per Reception (receiving_yards / receptions)',
    'receiver_yac_to_air_yards': 'YAC to Air Yards Ratio'
}

# NFL team names
nfl_team_names = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders"
}

# NFL team logos
nfl_logos = {
    "ARI": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "BUF": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "CAR": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "GB":  "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "JAX": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "KC":  "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "LV":  "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "LA": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "LAR": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "NE":  "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "NO":  "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "NYG": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "NYJ": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "SF":  "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "TB":  "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "TEN": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png"
}

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
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("df_merged.csv")
    df['team_full_name'] = df['team'].map(nfl_team_names)
    # Remove travel_distance_home (always 0) and use travel_distance_away for travel_distance
    if 'travel_distance_away' in df.columns:
        df['travel_distance'] = df['travel_distance_away']
    
    # Calculate performance ratios if they don't exist
    if 'snap_share' not in df.columns and 'offensive_snaps' in df.columns and 'team_offensive_snaps' in df.columns:
        df['snap_share'] = df['offensive_snaps'] / df['team_offensive_snaps']
    
    if 'pass_usage' not in df.columns and 'attempts' in df.columns and 'offensive_snaps' in df.columns:
        df['pass_usage'] = df['attempts'] / df['offensive_snaps']
    
    if 'pass_pct_of_offense' not in df.columns and 'attempts' in df.columns and 'team_offensive_snaps' in df.columns:
        df['pass_pct_of_offense'] = df['attempts'] / df['team_offensive_snaps']
    
    if 'pass_air_yard_pct' not in df.columns and 'passing_air_yards' in df.columns and 'passing_yards' in df.columns:
        df['pass_air_yard_pct'] = df['passing_air_yards'] / df['passing_yards']
    
    if 'pass_yards_after_catch_pct' not in df.columns and 'passing_yards_after_catch' in df.columns and 'passing_yards' in df.columns:
        df['pass_yards_after_catch_pct'] = df['passing_yards_after_catch'] / df['passing_yards']
    
    if 'pass_average_air_yards' not in df.columns and 'air_yards_completion' in df.columns and 'air_yards_incompletion' in df.columns:
        df['pass_average_air_yards'] = (df['air_yards_completion'] + df['air_yards_incompletion']) / 2
    
    if 'rusher_usage' not in df.columns and 'carries' in df.columns and 'offensive_snaps' in df.columns:
        df['rusher_usage'] = df['carries'] / df['offensive_snaps']
    
    if 'rusher_fumble_pct' not in df.columns and 'rushing_fumbles' in df.columns and 'carries' in df.columns:
        df['rusher_fumble_pct'] = df['rushing_fumbles'] / df['carries']
    
    if 'rusher_yards_per_carry' not in df.columns and 'rushing_yards' in df.columns and 'carries' in df.columns:
        df['rusher_yards_per_carry'] = df['rushing_yards'] / df['carries']
    
    if 'receiver_usage' not in df.columns and 'targets' in df.columns and 'offensive_snaps' in df.columns:
        df['receiver_usage'] = df['targets'] / df['offensive_snaps']
    
    if 'receiver_efficiency' not in df.columns and 'receptions' in df.columns and 'targets' in df.columns:
        df['receiver_efficiency'] = df['receptions'] / df['targets']
    
    if 'receiver_yac_pct' not in df.columns and 'receiving_yards_after_catch' in df.columns and 'receiving_yards' in df.columns:
        df['receiver_yac_pct'] = df['receiving_yards_after_catch'] / df['receiving_yards']
    
    if 'receiver_yards_per_reception' not in df.columns and 'receiving_yards' in df.columns and 'receptions' in df.columns:
        df['receiver_yards_per_reception'] = df['receiving_yards'] / df['receptions']
    
    if 'receiver_yac_to_air_yards' not in df.columns and 'receiving_yards_after_catch' in df.columns and 'receiving_air_yards' in df.columns:
        df['receiver_yac_to_air_yards'] = df['receiving_yards_after_catch'] / df['receiving_air_yards']
    
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

@st.cache_data
def train_rf_model(df, metrics, target_flag):
    """
    Train Random Forest model to predict impact of flag on selected metrics
    target_flag: 'isaway', 'is_thursday', 'intl', or 'extended_away_games'
    """
    
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
def create_flag_correlation_matrix(df, entity_type, entity_name, season):
    """Create correlation matrix for flags and performance metrics"""
    # Filter data
    filtered_df = df.copy()
    
    if season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    entity_col = 'team' if entity_type == 'team' else 'player_display_name'
    filtered_df = filtered_df[filtered_df[entity_col] == entity_name]
    
    if len(filtered_df) == 0:
        return None
    
    # Select flags and performance metrics
    flags = ['isaway', 'is_thursday', 'intl', 'extended_away_games']
    
    # Performance metrics from your data
    performance_metrics = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions',
        'sacks', 'sack_yards', 'carries', 'rushing_yards', 'rushing_tds',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
        'passing_air_yards', 'passing_yards_after_catch', 'receiving_air_yards',
        'receiving_yards_after_catch', 'racr', 'target_share', 'air_yards_share', 'wopr',
        'offensive_snaps', 'lead_changes', 'travel_distance'
    ]
    
    # Add calculated ratios
    if 'snap_share' in filtered_df.columns:
        performance_metrics.append('snap_share')
    if 'receiver_efficiency' in filtered_df.columns:
        performance_metrics.append('receiver_efficiency')
    
    # Filter to available columns
    available_flags = [f for f in flags if f in filtered_df.columns]
    available_metrics = [m for m in performance_metrics if m in filtered_df.columns]
    
    if not available_flags or not available_metrics:
        return None
    
    # Create correlation matrix
    corr_cols = available_flags + available_metrics
    corr_data = filtered_df[corr_cols].copy()
    
    # Calculate correlation
    corr_matrix = corr_data.corr()
    
    # Extract flag correlations with metrics
    flag_correlations = corr_matrix.loc[available_metrics, available_flags]
    
    return flag_correlations, available_flags, available_metrics

def create_flag_impact_comparison(df, entity_type, entity_name, season, metric):
    """Compare metric values across different flag conditions"""
    # Filter data
    filtered_df = df.copy()
    
    if season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    entity_col = 'team' if entity_type == 'team' else 'player_display_name'
    filtered_df = filtered_df[filtered_df[entity_col] == entity_name]
    
    if len(filtered_df) == 0 or metric not in filtered_df.columns:
        return None
    
    flags = ['isaway', 'is_thursday', 'intl', 'extended_away_games']
    results = []
    
    for flag in flags:
        if flag in filtered_df.columns:
            flag_1 = filtered_df[filtered_df[flag] == 1][metric].mean()
            flag_0 = filtered_df[filtered_df[flag] == 0][metric].mean()
            
            if pd.notna(flag_1) and pd.notna(flag_0):
                pct_change = ((flag_1 - flag_0) / flag_0 * 100) if flag_0 != 0 else 0
                results.append({
                    'flag': flag.replace('_', ' ').title(),
                    'when_true': flag_1,
                    'when_false': flag_0,
                    'pct_change': pct_change
                })
    
    return pd.DataFrame(results) if results else None

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

def create_flag_impact_comparison(df, entity_type, entity_name, season, metric):
    """Compare metric values across different flag conditions"""
    # Filter data
    filtered_df = df.copy()
    
    if season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    # Handle entity filtering based on type
    if entity_type == 'team':
        # For teams, need to match against the abbreviated team name, not full name
        team_abbr = None
        for abbr, full_name in nfl_team_names.items():
            if full_name == entity_name:
                team_abbr = abbr
                break
        
        if team_abbr:
            filtered_df = filtered_df[filtered_df['team'] == team_abbr]
        else:
            # If not found in mapping, try direct match
            filtered_df = filtered_df[filtered_df['team'] == entity_name]
    else:
        # For players, use player_display_name
        filtered_df = filtered_df[filtered_df['player_display_name'] == entity_name]
    
    if len(filtered_df) == 0 or metric not in filtered_df.columns:
        return None
    
    flags = ['isaway', 'is_thursday', 'intl', 'extended_away_games']
    results = []
    
    for flag in flags:
        if flag in filtered_df.columns:
            flag_1 = filtered_df[filtered_df[flag] == 1][metric].mean()
            flag_0 = filtered_df[filtered_df[flag] == 0][metric].mean()
            
            if pd.notna(flag_1) and pd.notna(flag_0):
                pct_change = ((flag_1 - flag_0) / flag_0 * 100) if flag_0 != 0 else 0
                results.append({
                    'flag': flag.replace('_', ' ').title(),
                    'when_true': flag_1,
                    'when_false': flag_0,
                    'pct_change': pct_change
                })
    
    return pd.DataFrame(results) if results else None
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
        "Select Player",
        options=['All'] + players,
        key='player_select'
    )
    st.session_state.selected_player = selected_player
    st.sidebar.info(f"**Active Filters:**\n\n Season: `{selected_season}`\n\n Team: `{selected_team}`\n\n Player: `{selected_player}`")
    
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
    
    with st.sidebar.expander("Player Performance Ratios"):
        available_ratios = [stat for stat in performance_ratios if stat in df.columns]
        
        # Show descriptions in a nice format
        st.caption("These metrics show efficiency and utilization rates")
        
        ratio_selected = st.multiselect(
            "Select performance ratios:",
            available_ratios,
            key='ratio_metrics',
            format_func=lambda x: f"{x.replace('_', ' ').title()}"
        )
        
        # Show descriptions for selected ratios
        if ratio_selected:
            st.markdown("**Selected Ratio Descriptions:**")
            for ratio in ratio_selected:
                if ratio in performance_ratio_descriptions:
                    st.caption(f"• **{ratio.replace('_', ' ').title()}**: {performance_ratio_descriptions[ratio]}")
        
        selected_metrics.extend(ratio_selected)
    
    # Enforce max 3 metrics
    if len(selected_metrics) > 3:
        st.sidebar.error("Please select a maximum of 3 metrics")
        selected_metrics = selected_metrics[:3]
    
    st.sidebar.info(f"**Selected: {len(selected_metrics)}/3 metrics**")
    
    st.sidebar.divider()
   
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
    tab1, tab2, tab3, tab4 = st.tabs(["Data View", "Travel Impact Analysis", "Flag Impact Analysis", "Team/Player Insights"])
    
    with tab1:
        st.subheader("Team Performance Summary")
        
        # Check if we need aggregated view (default) or filtered view
        show_aggregated = (selected_season == 'All' and selected_team == 'All' and selected_player == 'All')
        
        if show_aggregated:
            st.write("**Aggregated team statistics across all seasons**")
            
            # Create aggregated dataframe - one row per team
            team_agg = df.groupby('team').agg({
                'is_thursday': 'sum',  # Total Thursday games
                'intl': 'sum',  # Total international games
                'isaway': 'sum',  # Total away games
                'travel_distance': 'mean',  # Average travel distance
                'season': 'nunique'  # Number of seasons for averaging
            }).reset_index()
            
            # Calculate averages per season
            team_agg['Avg Thursday Games/Season'] = (team_agg['is_thursday'] / team_agg['season']).round(2)
            team_agg['Avg Intl Games/Season'] = (team_agg['intl'] / team_agg['season']).round(2)
            team_agg['Avg Away Games/Season'] = (team_agg['isaway'] / team_agg['season']).round(2)
            team_agg['Avg Travel Distance'] = team_agg['travel_distance'].round(0)
            
            # Add logo URLs
            team_agg['Logo'] = team_agg['team'].map(nfl_logos)
            
            # Select and rename columns for display
            display_df = team_agg[[
                'Logo', 'team', 
                'Avg Thursday Games/Season', 
                'Avg Intl Games/Season', 
                'Avg Away Games/Season',
                'Avg Travel Distance'
            ]].copy()
            
            display_df = display_df.rename(columns={'team': 'Team'})
            
            # Sort by team name
            display_df = display_df.sort_values('Team').reset_index(drop=True)
            
            # Display with custom column configuration
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600,
                column_config={
                    "Logo": st.column_config.ImageColumn("", width="small"),
                    "Team": st.column_config.TextColumn("Team", width="small"),
                    "Avg Thursday Games/Season": st.column_config.NumberColumn(
                        "Avg Thursday Games/Season", 
                        format="%.2f"
                    ),
                    "Avg Intl Games/Season": st.column_config.NumberColumn(
                        "Avg Intl Games/Season", 
                        format="%.2f"
                    ),
                    "Avg Away Games/Season": st.column_config.NumberColumn(
                        "Avg Away Games/Season", 
                        format="%.2f"
                    ),
                    "Avg Travel Distance": st.column_config.NumberColumn(
                        "Avg Travel Distance (miles)", 
                        format="%.0f"
                    )
                },
                hide_index=True
            )
            
            # Summary stats
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Teams", len(display_df))
            with col2:
                st.metric("Avg Thursday Games", f"{display_df['Avg Thursday Games/Season'].mean():.2f}")
            with col3:
                st.metric("Avg Intl Games", f"{display_df['Avg Intl Games/Season'].mean():.2f}")
            with col4:
                st.metric("Avg Travel Distance", f"{display_df['Avg Travel Distance'].mean():,.0f} mi")
        
        else:
            st.write("**Filtered data view**")
            
            if selected_metrics:
                display_cols = ['season', 'team', 'player_display_name'] + selected_metrics
                # Add additional columns
                additional_cols = ['attempts', 'is_thursday', 'intl', 'travel_distance']
                for col in additional_cols:
                    if col in result_df.columns and col not in display_cols:
                        display_cols.append(col)
                
                display_cols = [col for col in display_cols if col in result_df.columns]
                st.dataframe(result_df[display_cols], use_container_width=True, height=400)
            else:
                # Show default columns when no metrics selected
                default_cols = ['season', 'team', 'player_display_name', 
                               'attempts', 'is_thursday', 'intl', 'travel_distance']
                display_cols = [col for col in default_cols if col in result_df.columns]
                st.dataframe(result_df[display_cols], use_container_width=True, height=400)
    
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
                    tab_a, tab_b, tab_c = st.tabs(["Distance Impact", "Season Trend", "Scatter Plot"])
                    
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
                            st.info(f"**{strength.title()} correlation detected:** {metric.replace('_', ' ').title()} {direction} by approximately **{abs(corr)*100:.1f}%** correlation as travel distance increases")
                        else:
                            st.info(f"**Weak correlation:** Travel distance has minimal impact on {metric.replace('_', ' ').title()}")
                    
                    st.divider()
            else:
                st.error("Unable to calculate travel impact. Please check your data.")
    
    with tab3:
        st.subheader("🔬 Model Analysis")
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
                    st.subheader("📊 Metric Impact Summary")
                    
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
                st.write("This model identifies which metrics are most predictive of the selected flag condition.")
                
                # Get all available metrics for RF model (not just selected ones)
                all_numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                # Remove non-feature columns
                exclude_cols = ['season', 'week', 'away_score', 'home_score', 'result', 'total', 
                            'isaway', 'is_thursday', 'extended_away_games', 'intl', 'is_international',
                            'overtime', 'div_game', 'lead_changes', 'gsis', 'old_game_id', 'jersey_number',
                            'season_type', 'opponent_team', 'depth_chart_position', 'football_name', 
                            'recent_team', 'status', 'status_description_abbr', 'game_type', 'player_name', 
                            'position', 'game_id', 'gameday', 'weekday', 'location', 'stadium', 
                            'away_rest', 'home_rest', 'week_after_intl', 'defensive_snaps', 
                            'team_defensive_snaps', 'special_team_snaps', 'team_special_team_snaps',
                            'sack_fumbles_lost', 'passing_first_downs', 'passing_2pt_conversions',
                            'rushing_first_downs', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 
                            'receiving_first_downs', 'receiving_2pt_conversions', 'yards_after_catch',
                            'player_name_flat', 'travel_distance_home', 'travel_distance_away']
                rf_metrics = [col for col in all_numeric_cols if col not in exclude_cols]
                
                with st.spinner("Training Random Forest model on all available metrics..."):
                    importances, metrics_dict, model = train_rf_model(result_df, rf_metrics, selected_flag)
                
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
                    
                    # Get top 20 metrics
                    top_20 = importances.head(20).copy()
                    
                    # Check which selected metrics are in top 20 and which are not
                    selected_in_top20 = [m for m in selected_metrics if m in top_20['metric'].values]
                    selected_not_in_top20 = [m for m in selected_metrics if m not in top_20['metric'].values]
                    
                    # Create display dataframe
                    if selected_not_in_top20:
                        # Add selected metrics not in top 20
                        additional_metrics = importances[importances['metric'].isin(selected_not_in_top20)]
                        display_importances = pd.concat([top_20, additional_metrics], ignore_index=True)
                    else:
                        display_importances = top_20.copy()
                    
                    # Create color array based on whether metric is user-selected
                    colors = []
                    for metric in display_importances['metric']:
                        if metric in selected_metrics:
                            colors.append('#00cc96')  # Green for user-selected
                        else:
                            colors.append('#ff6b6b')  # Red for default top 20
                    
                    # Feature importance chart
                    st.subheader("📊 Feature Importance Rankings")
                    
                    # Legend explanation
                    col_leg1, col_leg2 = st.columns(2)
                    with col_leg1:
                        st.markdown("🔴 **Red bars**: Top 20 most predictive metrics")
                    with col_leg2:
                        st.markdown("🟢 **Green bars**: Your selected metrics")
                    
                    if selected_not_in_top20:
                        st.info(f"**Note:** {len(selected_not_in_top20)} of your selected metrics are outside the top 20 and shown below")
                    
                    fig = go.Figure(go.Bar(
                        x=display_importances['importance'],
                        y=display_importances['metric'],
                        orientation='h',
                        marker_color=colors,
                        text=display_importances['importance'].round(3),
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f'Which Metrics Best Predict {selected_flag_name}?',
                        xaxis_title='Importance Score',
                        yaxis_title='Metric',
                        height=max(500, len(display_importances) * 25),
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show rankings for selected metrics
                    st.subheader("Your Selected Metrics Rankings")
                    cols_rank = st.columns(min(len(selected_metrics), 3))
                    
                    for idx, metric in enumerate(selected_metrics):
                        metric_row = importances[importances['metric'] == metric]
                        if not metric_row.empty:
                            rank = importances.index[importances['metric'] == metric].tolist()[0] + 1
                            importance_val = metric_row['importance'].values[0]
                            
                            with cols_rank[idx % 3]:
                                if rank <= 20:
                                    rank_emoji = "🏆" if rank <= 5 else "⭐"
                                    rank_color = "off"
                                else:
                                    rank_emoji = "📍"
                                    rank_color = "inverse"
                                
                                st.metric(
                                    label=f"{rank_emoji} {metric.replace('_', ' ').title()}",
                                    value=f"Rank #{rank}",
                                    delta=f"Importance: {importance_val:.4f}",
                                    delta_color=rank_color
                                )
                    
                    # Interpretation
                    st.divider()
                    top_metric = importances.iloc[0]
                    st.success(f"**Key Insight:** `{top_metric['metric'].replace('_', ' ').title()}` is the #1 most important metric "
                            f"for predicting {selected_flag_name} (importance: {top_metric['importance']:.4f})")
                    
                    # Show if any selected metrics are highly predictive
                    highly_predictive_selected = [m for m in selected_metrics if m in importances.head(5)['metric'].values]
                    if highly_predictive_selected:
                        st.success(f"**Your selection includes top predictors:** {', '.join([m.replace('_', ' ').title() for m in highly_predictive_selected])}")
                    
                    # Detailed table with highlighting
                    st.subheader("📋 Complete Feature Importance Table")
                    
                    # Add rank column
                    display_table = importances.copy()
                    display_table['rank'] = range(1, len(display_table) + 1)
                    display_table['is_selected'] = display_table['metric'].isin(selected_metrics)
                    display_table['metric'] = display_table['metric'].str.replace('_', ' ').str.title()
                    display_table['importance'] = display_table['importance'].round(4)
                    
                    # Reorder columns
                    display_table = display_table[['rank', 'metric', 'importance', 'is_selected']]
                    
                    # Style the dataframe
                    st.dataframe(
                        display_table,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", format="#%d"),
                            "metric": st.column_config.TextColumn("Metric"),
                            "importance": st.column_config.NumberColumn("Importance", format="%.4f"),
                            "is_selected": st.column_config.CheckboxColumn("Your Selection")
                        }
                    )
                
                else:
                    st.error("Unable to train model. Check your data and selected metrics.")
                
    with tab4:
        st.subheader("Team/Player Performance Insights")
        st.write("Analyze how away games, Thursday games, international games, and extended away trips impact performance metrics.")
        
        # Entity selection
        col1, col2 = st.columns(2)
        with col1:
            insight_entity = st.radio(
                "Analyze by:",
                options=['Team', 'Player'],
                key='insight_entity',
                horizontal=True
            )
        
        with col2:
            if insight_entity == 'Team':
                if selected_team == 'All':
                    st.info("Select a team from the sidebar")
                    entity_selected = None
                else:
                    entity_selected = selected_team
                    st.success(f"**{selected_team}**")
            else:
                if selected_player == 'All':
                    st.info("Select a player from the sidebar")
                    entity_selected = None
                else:
                    entity_selected = selected_player
                    st.success(f"**{selected_player}**")
        
        if entity_selected:
            st.divider()
            
            # Get correlation matrix
            corr_result = create_flag_correlation_matrix(
                df, 
                insight_entity.lower(), 
                entity_selected,
                selected_season
            )
            
            if corr_result is not None:
                flag_correlations, available_flags, available_metrics = corr_result
                
                # Display correlation heatmap
                st.subheader("Flag Impact Correlation Matrix")
                st.write("Shows how strongly each flag condition correlates with performance metrics. "
                        "Red = negative impact, Blue = positive impact.")
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=flag_correlations.values,
                    x=[f.replace('_', ' ').title() for f in available_flags],
                    y=[m.replace('_', ' ').title() for m in available_metrics],
                    colorscale='RdBu',
                    zmid=0,
                    text=flag_correlations.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title="Correlation"),
                    hovertemplate='<b>%{y}</b><br>%{x}<br>Correlation: %{z:.3f}<extra></extra>'
                ))
                
                fig_corr.update_layout(
                    height=max(400, len(available_metrics) * 20),
                    xaxis_title='Game Condition',
                    yaxis_title='Performance Metric',
                    yaxis={'tickfont': {'size': 9}}
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.divider()
                
                # Key insights section
                st.subheader("Key Insights")
                
                # Find strongest correlations for each flag
                for flag in available_flags:
                    flag_display = flag.replace('_', ' ').title()
                    correlations = flag_correlations[flag].abs().sort_values(ascending=False).head(5)
                    
                    with st.expander(f"{flag_display} - Top 5 Impacted Metrics"):
                        for idx, (metric, corr_val) in enumerate(correlations.items(), 1):
                            actual_corr = flag_correlations.loc[metric, flag]
                            direction = "📈 Positive" if actual_corr > 0 else "📉 Negative"
                            strength = "Strong" if abs(actual_corr) > 0.5 else "Moderate" if abs(actual_corr) > 0.3 else "Weak"
                            
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            with col_a:
                                st.write(f"**{idx}. {metric.replace('_', ' ').title()}**")
                            with col_b:
                                st.write(f"{direction}")
                            with col_c:
                                st.write(f"{strength} ({actual_corr:.3f})")
                
                st.divider()
                
                # Metric-specific comparison
                st.subheader("Detailed Metric Analysis")
                st.write("Compare how a specific metric performs under different game conditions.")
                
                # Metric selector
                available_display_metrics = [m for m in available_metrics if m in df.columns]
                if available_display_metrics:
                    selected_insight_metric = st.selectbox(
                        "Select a metric to analyze:",
                        options=available_display_metrics,
                        format_func=lambda x: x.replace('_', ' ').title(),
                        key='insight_metric'
                    )
                    
                    # Get comparison data
                    comparison_df = create_flag_impact_comparison(
                        df,
                        insight_entity.lower(),
                        entity_selected,
                        selected_season,
                        selected_insight_metric
                    )
                    
                    if comparison_df is not None and len(comparison_df) > 0:
                        # Metric cards showing impact
                        st.write(f"**{selected_insight_metric.replace('_', ' ').title()} Performance by Game Condition**")
                        
                        cols = st.columns(len(comparison_df))
                        for idx, row in comparison_df.iterrows():
                            with cols[idx]:
                                impact_emoji = "🔴" if row['pct_change'] < 0 else "🟢" if row['pct_change'] > 0 else "⚪"
                                
                                st.metric(
                                    label=f"{impact_emoji} {row['flag']}",
                                    value=f"{row['when_true']:.2f}",
                                    delta=f"{row['pct_change']:.1f}% vs baseline",
                                    delta_color="normal" if abs(row['pct_change']) < 5 else "inverse"
                                )
                                
                                with st.expander("Details"):
                                    st.write(f"**When condition is TRUE:** {row['when_true']:.2f}")
                                    st.write(f"**When condition is FALSE:** {row['when_false']:.2f}")
                                    st.write(f"**Difference:** {(row['when_true'] - row['when_false']):.2f}")
                        
                        st.divider()
                        
                        # Visual comparison
                        fig_comparison = go.Figure()
                        
                        fig_comparison.add_trace(go.Bar(
                            name='Condition False',
                            x=comparison_df['flag'],
                            y=comparison_df['when_false'],
                            marker_color='#4ecdc4'
                        ))
                        
                        fig_comparison.add_trace(go.Bar(
                            name='Condition True',
                            x=comparison_df['flag'],
                            y=comparison_df['when_true'],
                            marker_color='#ff6b6b'
                        ))
                        
                        fig_comparison.update_layout(
                            barmode='group',
                            title=f'{selected_insight_metric.replace("_", " ").title()} by Game Condition',
                            xaxis_title='Game Condition',
                            yaxis_title=selected_insight_metric.replace("_", " ").title(),
                            height=400
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Summary insights
                        st.subheader("Summary")
                        
                        worst_impact = comparison_df.loc[comparison_df['pct_change'].idxmin()]
                        best_impact = comparison_df.loc[comparison_df['pct_change'].idxmax()]
                        
                        col_sum1, col_sum2 = st.columns(2)
                        with col_sum1:
                            st.error(f"**Biggest Negative Impact:** {worst_impact['flag']} "
                                   f"({worst_impact['pct_change']:.1f}% decrease)")
                        with col_sum2:
                            st.success(f"**Best Performance:** {best_impact['flag']} "
                                     f"({best_impact['pct_change']:.1f}% change)")
                
                st.divider()
                
                # Team scatter plot analysis
                st.subheader("Team Performance Scatter Analysis")
                st.write("Compare all teams' completion percentage vs attempts across different game conditions.")
                
                # Filter selection for scatter
                scatter_flag = st.selectbox(
                    "Select game condition to analyze:",
                    options=['All Games', 'Away Games', 'Thursday Games', 'International Games', 'Extended Away Games'],
                    key='scatter_flag'
                )
                
                # Map selection to flag column
                flag_map = {
                    'All Games': None,
                    'Away Games': 'isaway',
                    'Thursday Games': 'is_thursday',
                    'International Games': 'intl',
                    'Extended Away Games': 'extended_away_games'
                }
                
                selected_scatter_flag = flag_map[scatter_flag]
                
                # Prepare data for scatter plot
                scatter_df = df.copy()
                
                if selected_season != 'All':
                    scatter_df = scatter_df[scatter_df['season'] == selected_season]
                
                # Filter by flag if selected
                if selected_scatter_flag:
                    scatter_df = scatter_df[scatter_df[selected_scatter_flag] == 1]
                
                # Calculate team-level aggregates
                if 'attempts' in scatter_df.columns and 'completions' in scatter_df.columns:
                    team_stats = scatter_df.groupby('team').agg({
                        'attempts': 'sum',
                        'completions': 'sum',
                        'passing_yards': 'sum' if 'passing_yards' in scatter_df.columns else 'count'
                    }).reset_index()
                    
                    # Calculate completion percentage
                    team_stats['completion_pct'] = (team_stats['completions'] / team_stats['attempts'] * 100).round(2)
                    
                    # Filter out teams with very few attempts (noise)
                    team_stats = team_stats[team_stats['attempts'] >= 10]
                    
                    if len(team_stats) > 0:
                        # Highlight selected team if applicable
                        if entity_selected and insight_entity == 'Team':
                            team_stats['is_selected'] = team_stats['team'] == entity_selected
                        else:
                            team_stats['is_selected'] = False
                        
                        # Add logo URLs
                        team_stats['logo_url'] = team_stats['team'].map(nfl_logos)
                        
                        # Create scatter plot with team logos
                        fig_scatter = go.Figure()
                        
                        # Add scatter points with logos as custom markers
                        for idx, row in team_stats.iterrows():
                            color = '#ff6b6b' if row['is_selected'] else '#4ecdc4'
                            size = 15 if row['is_selected'] else 10
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=[row['attempts']],
                                y=[row['completion_pct']],
                                mode='markers',
                                marker=dict(
                                    size=size,
                                    color=color,
                                    line=dict(width=2, color='white')
                                ),
                                name=row['team'],
                                text=row['team'],
                                hovertemplate='<b>%{text}</b><br>' +
                                            'Attempts: %{x}<br>' +
                                            'Completion %: %{y:.1f}%<br>' +
                                            '<extra></extra>',
                                showlegend=False
                            ))
                            
                            # Add team logo as image annotation
                            if pd.notna(row['logo_url']):
                                fig_scatter.add_layout_image(
                                    dict(
                                        source=row['logo_url'],
                                        xref="x",
                                        yref="y",
                                        x=row['attempts'],
                                        y=row['completion_pct'],
                                        sizex=max(team_stats['attempts']) * 0.05,
                                        sizey=3,
                                        xanchor="center",
                                        yanchor="middle",
                                        layer="above"
                                    )
                                )
                        
                        # Add trendline
                        from sklearn.linear_model import LinearRegression
                        X = team_stats['attempts'].values.reshape(-1, 1)
                        y = team_stats['completion_pct'].values
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=team_stats['attempts'],
                            y=y_pred,
                            mode='lines',
                            name='Trendline',
                            line=dict(color='rgba(255, 107, 107, 0.5)', dash='dash', width=2),
                            showlegend=True
                        ))
                        
                        fig_scatter.update_layout(
                            title=f'Team Completion % vs Attempts - {scatter_flag}',
                            xaxis_title='Total Attempts',
                            yaxis_title='Completion Percentage (%)',
                            height=600,
                            hovermode='closest',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Stats summary
                        col_scatter1, col_scatter2, col_scatter3, col_scatter4 = st.columns(4)
                        with col_scatter1:
                            st.metric("Teams Analyzed", len(team_stats))
                        with col_scatter2:
                            st.metric("Avg Completion %", f"{team_stats['completion_pct'].mean():.1f}%")
                        with col_scatter3:
                            st.metric("Highest Comp %", f"{team_stats['completion_pct'].max():.1f}%")
                        with col_scatter4:
                            st.metric("Lowest Comp %", f"{team_stats['completion_pct'].min():.1f}%")
                        
                        # Show top and bottom performers
                        col_perf1, col_perf2 = st.columns(2)
                        
                        with col_perf1:
                            st.write("**Top 5 Completion %**")
                            top_5 = team_stats.nlargest(5, 'completion_pct')[['team', 'completion_pct', 'attempts']]
                            top_5_display = top_5.copy()
                            top_5_display.columns = ['Team', 'Comp %', 'Attempts']
                            st.dataframe(top_5_display, hide_index=True, use_container_width=True)
                        
                        with col_perf2:
                            st.write("**Bottom 5 Completion %**")
                            bottom_5 = team_stats.nsmallest(5, 'completion_pct')[['team', 'completion_pct', 'attempts']]
                            bottom_5_display = bottom_5.copy()
                            bottom_5_display.columns = ['Team', 'Comp %', 'Attempts']
                            st.dataframe(bottom_5_display, hide_index=True, use_container_width=True)
                    else:
                        st.warning("Insufficient data for scatter plot analysis with current filters.")
                else:
                    st.warning("Required columns (attempts, completions) not found in dataset.")
                    
            else:
                st.warning("Insufficient data for correlation analysis. Try selecting a different season or entity.")
        else:
            st.info("Select a team or player from the sidebar to view performance insights")
    
if __name__ == "__main__":
    main()