import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LinearRegression

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

# Define position groups and their relevant metrics
POSITION_GROUPS = {
    'QB': {
        'name': 'Quarterbacks',
        'stats': ["attempts", "completions", "passing_yards", "passing_tds", "interceptions",
                  "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
                  "passing_air_yards", "passing_yards_after_catch",
                  "passing_first_downs", "passing_2pt_conversions"],
        'ratios': ['snap_share', 'pass_usage', 'pass_pct_of_offense', 'pass_air_yard_pct',
                   'pass_yards_after_catch_pct', 'pass_average_air_yards']
    },
    'RB': {
        'name': 'Running Backs',
        'stats': ["carries", "rushing_yards", "rushing_tds", "rushing_fumbles", 
                  "rushing_fumbles_lost", "rushing_first_downs",
                  'receptions', 'targets', 'receiving_yards', 'receiving_tds', 
                  'receiving_air_yards', 'receiving_yards_after_catch', 
                  'receiving_first_downs', 'target_share', 'air_yards_share'],
        'ratios': ['snap_share', 'rusher_usage', 'rusher_fumble_pct', 'rusher_yards_per_carry',
                   'receiver_usage', 'receiver_efficiency', 'receiver_yac_pct', 
                   'receiver_yards_per_reception']
    },
    'WR': {
        'name': 'Wide Receivers',
        'stats': ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
                  'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch', 
                  'receiving_first_downs', 'receiving_2pt_conversions', 'racr', 'target_share', 
                  'air_yards_share', 'wopr'],
        'ratios': ['snap_share', 'receiver_usage', 'receiver_efficiency', 'receiver_yac_pct', 
                   'receiver_yards_per_reception', 'receiver_yac_to_air_yards']
    },
    'TE': {
        'name': 'Tight Ends',
        'stats': ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
                  'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch', 
                  'receiving_first_downs', 'receiving_2pt_conversions', 'racr', 'target_share', 
                  'air_yards_share', 'wopr'],
        'ratios': ['snap_share', 'receiver_usage', 'receiver_efficiency', 'receiver_yac_pct', 
                   'receiver_yards_per_reception', 'receiver_yac_to_air_yards']
    }
}

# NFL stadium coordinates
nfl_stadium_coords = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "stadium": "State Farm Stadium"},
    "ATL": {"lat": 33.7555, "lon": -84.4008, "stadium": "Mercedes-Benz Stadium"},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "stadium": "M&T Bank Stadium"},
    "BUF": {"lat": 42.7738, "lon": -78.7870, "stadium": "Highmark Stadium"},
    "CAR": {"lat": 35.2258, "lon": -80.8528, "stadium": "Bank of America Stadium"},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "stadium": "Soldier Field"},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "stadium": "Paycor Stadium"},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "stadium": "Cleveland Browns Stadium"},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "stadium": "AT&T Stadium"},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "stadium": "Empower Field at Mile High"},
    "DET": {"lat": 42.3400, "lon": -83.0456, "stadium": "Ford Field"},
    "GB": {"lat": 44.5013, "lon": -88.0622, "stadium": "Lambeau Field"},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "stadium": "NRG Stadium"},
    "IND": {"lat": 39.7601, "lon": -86.1639, "stadium": "Lucas Oil Stadium"},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "stadium": "EverBank Stadium"},
    "KC": {"lat": 39.0489, "lon": -94.4839, "stadium": "GEHA Field at Arrowhead Stadium"},
    "LV": {"lat": 36.0908, "lon": -115.1833, "stadium": "Allegiant Stadium"},
    "LAC": {"lat": 33.9534, "lon": -118.3390, "stadium": "SoFi Stadium"},
    "LA": {"lat": 33.9534, "lon": -118.3390, "stadium": "SoFi Stadium"},
    "LAR": {"lat": 33.9534, "lon": -118.3390, "stadium": "SoFi Stadium"},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "stadium": "Hard Rock Stadium"},
    "MIN": {"lat": 44.9738, "lon": -93.2575, "stadium": "U.S. Bank Stadium"},
    "NE": {"lat": 42.0909, "lon": -71.2643, "stadium": "Gillette Stadium"},
    "NO": {"lat": 29.9511, "lon": -90.0812, "stadium": "Caesars Superdome"},
    "NYG": {"lat": 40.8128, "lon": -74.0742, "stadium": "MetLife Stadium"},
    "NYJ": {"lat": 40.8128, "lon": -74.0742, "stadium": "MetLife Stadium"},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "stadium": "Lincoln Financial Field"},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "stadium": "Acrisure Stadium"},
    "SF": {"lat": 37.4032, "lon": -121.9696, "stadium": "Levi's Stadium"},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "stadium": "Lumen Field"},
    "TB": {"lat": 27.9759, "lon": -82.5033, "stadium": "Raymond James Stadium"},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "stadium": "Nissan Stadium"},
    "WAS": {"lat": 38.9076, "lon": -77.0169, "stadium": "FedExField"},
    "LON": {"lat": 51.5074, "lon": -0.1278, "stadium": "Wembley Stadium / Tottenham Hotspur Stadium (London)"},
    "MEX": {"lat": 19.4326, "lon": -99.1332, "stadium": "Estadio Azteca (Mexico City)"},
    "GER": {"lat": 50.0685, "lon": 8.6454, "stadium": "Deutsche Bank Park (Frankfurt)"},
    "MUN": {"lat": 48.2188, "lon": 11.6247, "stadium": "Allianz Arena (Munich)"}
}

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
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
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
    "GB": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "JAX": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "KC": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "LV": "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "LA": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "LAR": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "NE": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "NO": "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "NYG": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "NYJ": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "SF": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "TB": "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "TEN": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png"
}

def create_team_travel_map(df, selected_season, selected_team):
    """Create interactive travel map for a team's season"""
    
    # Convert team full name to abbreviation if needed
    team_abbr = selected_team
    for abbr, full_name in nfl_team_names.items():
        if full_name == selected_team:
            team_abbr = abbr
            break
    
    # Filter data for the selected team and season
    team_data = df[
        (df['season'] == selected_season) & 
        (df['team'] == team_abbr)
    ].sort_values('week').copy()
    
    if len(team_data) == 0:
        return None
    
    # Check if there are any international games
    has_international = False
    if 'is_international' in team_data.columns:
        has_international = (team_data['is_international'] == 1).any()
    
    # Get unique weeks and assign colors
    weeks_filtered = sorted(team_data['week'].unique())
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    color_map = {week: colors[i % len(colors)] for i, week in enumerate(weeks_filtered)}
    
    # Create map figure
    fig = go.Figure()
    
    prev_lat, prev_lon = None, None
    route_counter = 0
    
    for week in weeks_filtered:
        week_data = team_data[team_data['week'] == week].iloc[0]
        
        # Get coordinates for this week's game
        if 'opponent_team' in week_data and pd.notna(week_data['opponent_team']):
            opponent = week_data['opponent_team']
        else:
            opponent = team_abbr
        
        # Determine location based on isaway flag
        if 'isaway' in week_data and week_data['isaway'] == 1:
            # Away game - use opponent's stadium
            game_team = opponent if opponent in nfl_stadium_coords else team_abbr
        else:
            # Home game - use team's stadium
            game_team = team_abbr
        
        if game_team not in nfl_stadium_coords:
            continue
            
        lat = nfl_stadium_coords[game_team]['lat']
        lon = nfl_stadium_coords[game_team]['lon']
        stadium = nfl_stadium_coords[game_team]['stadium']
        
        # Create travel line if there's a previous location
        if prev_lat is not None and prev_lon is not None:
            fig.add_trace(go.Scattergeo(
                lon=[prev_lon, lon],
                lat=[prev_lat, lat],
                mode="lines",
                line=dict(width=3, color=color_map[week]),
                name=f"Week {week}",
                showlegend=True,
                hoverinfo="skip"
            ))
            route_counter += 1
        
        # Determine marker properties
        marker_color = color_map[week]
        marker_size = 10
        marker_symbol = "circle"
        
        # Special markers for special game conditions
        if 'is_thursday' in week_data and week_data['is_thursday'] == 1:
            marker_symbol = "star"
            marker_size = 14
            marker_color = "gold"
        elif 'is_international' in week_data and week_data['is_international'] == 1:
            marker_symbol = "diamond"
            marker_size = 14
            marker_color = "purple"
        
        # Build hover text
        hover_text = f"<b>Week {week}</b><br>Stadium: {stadium}<br>"
        
        if 'isaway' in week_data:
            location_type = "Away" if week_data['isaway'] == 1 else "Home"
            hover_text += f"Location: {location_type}<br>"
        
        if 'travel_distance' in week_data and pd.notna(week_data['travel_distance']) and week_data['travel_distance'] > 0:
            hover_text += f"Travel Distance: {week_data['travel_distance']:.0f} miles<br>"
        
        if 'is_thursday' in week_data and week_data['is_thursday'] == 1:
            hover_text += "<b>⭐ Thursday Game</b><br>"
        
        if 'is_international' in week_data and week_data['is_international'] == 1:
            hover_text += "<b>◆ International Game</b><br>"
        
        # Add marker
        fig.add_trace(go.Scattergeo(
            lon=[lon],
            lat=[lat],
            mode="markers",
            marker=dict(
                size=marker_size,
                symbol=marker_symbol,
                color=marker_color,
                line=dict(width=1, color='white')
            ),
            hoverinfo="text",
            text=hover_text,
            showlegend=False
        ))
        
        prev_lat, prev_lon = lat, lon
    
    # Update map layout - use different projection based on international games
    if has_international:
        # World view for international games
        geo_config = dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(229, 229, 229)",
            coastlinecolor="rgb(204, 204, 204)",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            showcountries=True,
            countrycolor="rgb(204, 204, 204)",
            bgcolor="rgb(243, 243, 243)",
            center=dict(lat=35, lon=-40),  # Center between US and Europe
            projection=dict(scale=1.5)
        )
    else:
        # USA-only view for domestic games
        geo_config = dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(229, 229, 229)",
            coastlinecolor="rgb(204, 204, 204)",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            subunitcolor="rgb(217, 217, 217)",
            bgcolor="rgb(243, 243, 243)"
        )
    
    fig.update_layout(
        title=dict(
            text=f"{selected_team} - {selected_season} Season Travel Map" + 
                 (" (includes international games)" if has_international else ""),
            font=dict(size=16),
            x=0.5,
            xanchor="center"
        ),
        geo=geo_config,
        height=500,
        margin={"r": 120, "t": 50, "l": 0, "b": 0},
        legend=dict(
            title=dict(text="<b>Weeks</b>", font=dict(size=10)),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=9),
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle"
        )
    )
    
    return fig

def create_season_trend_with_moving_avg(df, selected_team, selected_position_group, selected_season, metric, moving_avg_window):
    """Create season progression chart with moving average and game condition markers"""
    
    filtered_df = df.copy()
    
    if selected_season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
    
    # Apply team filter
    if selected_team != 'All':
        team_abbr = None
        for abbr, full_name in nfl_team_names.items():
            if full_name == selected_team:
                team_abbr = abbr
                break
        if team_abbr:
            filtered_df = filtered_df[filtered_df['team'] == team_abbr]
        else:
            filtered_df = filtered_df[filtered_df['team'] == selected_team]
    
    # Apply position group filter
    if selected_position_group != 'All':
        filtered_df = filtered_df[filtered_df['position_group'] == selected_position_group]
    
    if len(filtered_df) == 0 or metric not in filtered_df.columns:
        return None
    
    # Only show game condition markers if a specific team is selected
    # (otherwise players could be from different teams with different game conditions)
    show_game_conditions = (selected_team != 'All')
    
    # Sort by week and aggregate by week
    agg_dict = {metric: 'mean'}
    if show_game_conditions:
        agg_dict.update({
            'isaway': 'max',
            'is_thursday': 'max',
            'is_international': 'max'
        })
    
    weekly_data = filtered_df.groupby('week').agg(agg_dict).reset_index()
    weekly_data = weekly_data.sort_values('week')
    
    # Calculate moving average
    weekly_data[f'{metric}_ma'] = weekly_data[metric].rolling(window=moving_avg_window, min_periods=1).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add actual values line
    fig.add_trace(go.Scatter(
        x=weekly_data['week'],
        y=weekly_data[metric],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#4ecdc4', width=2),
        marker=dict(size=6)
    ))
    
    # Add moving average line
    fig.add_trace(go.Scatter(
        x=weekly_data['week'],
        y=weekly_data[f'{metric}_ma'],
        mode='lines',
        name=f'{moving_avg_window}-Game Moving Avg',
        line=dict(color='#ff6b6b', width=3)
    ))
    
    # Add markers for game conditions only if team filter is active
    if show_game_conditions:
        for idx, row in weekly_data.iterrows():
            markers = []
            if row['is_thursday'] == 1:
                markers.append('Thursday')
            if row['isaway'] == 1:
                markers.append('Away')
            if row['is_international'] == 1:
                markers.append('International')
            
            if markers:
                fig.add_trace(go.Scatter(
                    x=[row['week']],
                    y=[row[metric]],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        symbol='star' if 'Thursday' in markers else 'diamond' if 'International' in markers else 'circle',
                        color='gold' if 'Thursday' in markers else 'purple' if 'International' in markers else '#00cc96',
                        line=dict(width=2, color='white')
                    ),
                    text=['★' if 'Thursday' in markers else '◆' if 'International' in markers else ''],
                    textposition='top center',
                    textfont=dict(size=12, color='white'),
                    name=', '.join(markers),
                    showlegend=False,
                    hovertemplate=f'<b>Week {row["week"]}</b><br>{", ".join(markers)}<br>{metric}: {row[metric]:.2f}<extra></extra>'
                ))
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} - Season Progression',
        xaxis_title='Week',
        yaxis_title=metric.replace('_', ' ').title(),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_game_condition_breakdown(df, entity_type, entity_name, selected_season, metric):
    """Create breakdown showing metric performance by game conditions over season"""
    
    filtered_df = df.copy()
    
    if selected_season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
    
    # Handle both team and position group filtering
    if entity_type == 'team':
        team_abbr = None
        for abbr, full_name in nfl_team_names.items():
            if full_name == entity_name:
                team_abbr = abbr
                break
        if team_abbr:
            filtered_df = filtered_df[filtered_df['team'] == team_abbr]
        else:
            filtered_df = filtered_df[filtered_df['team'] == entity_name]
    else:  # position_group
        filtered_df = filtered_df[filtered_df['position_group'] == entity_name]
    
    if len(filtered_df) == 0 or metric not in filtered_df.columns:
        return None, None
    
    # Sort by week
    filtered_df = filtered_df.sort_values('week')
    
    # Create condition labels
    filtered_df['condition'] = 'Regular'
    filtered_df.loc[filtered_df['isaway'] == 1, 'condition'] = 'Away'
    filtered_df.loc[filtered_df['is_thursday'] == 1, 'condition'] = 'Thursday'
    filtered_df.loc[filtered_df['is_international'] == 1, 'condition'] = 'International'
    
    # Calculate averages by condition
    condition_avg = filtered_df.groupby('condition')[metric].mean().reset_index()
    condition_avg = condition_avg.sort_values(metric, ascending=False)
    
    # Create scatter plot with conditions highlighted
    fig_scatter = go.Figure()
    
    # Plot each condition with different marker
    for condition in filtered_df['condition'].unique():
        condition_data = filtered_df[filtered_df['condition'] == condition]
        
        marker_style = {
            'Regular': dict(symbol='circle', color='#4ecdc4', size=8),
            'Away': dict(symbol='circle', color='#00cc96', size=10),
            'Thursday': dict(symbol='star', color='gold', size=12),
            'International': dict(symbol='diamond', color='purple', size=12)
        }
        
        fig_scatter.add_trace(go.Scatter(
            x=condition_data['week'],
            y=condition_data[metric],
            mode='markers',
            name=condition,
            marker=marker_style.get(condition, dict(symbol='circle', color='gray', size=8)),
            hovertemplate=f'<b>Week %{{x}}</b><br>{condition}<br>{metric}: %{{y:.2f}}<extra></extra>'
        ))
    
    fig_scatter.update_layout(
        title=f'{metric.replace("_", " ").title()} with Game Conditions Highlighted',
        xaxis_title='Week',
        yaxis_title=metric.replace('_', ' ').title(),
        height=400,
        hovermode='closest'
    )
    
    return fig_scatter, condition_avg

def format_metric_name(metric):
    """Convert metric names from snake_case to Title Case"""
    return metric.replace('_', ' ').title()

def get_position_group(position):
    """Map a position to its position group"""
    if pd.isna(position):
        return None
    position = str(position).upper()
    if position in ['QB']:
        return 'QB'
    elif position in ['RB', 'FB', 'HB']:
        return 'RB'
    elif position in ['WR']:
        return 'WR'
    elif position in ['TE']:
        return 'TE'
    return None

def get_position_metrics(position_group, df):
    """Get available metrics for a position group"""
    if position_group not in POSITION_GROUPS:
        return [], []
    
    config = POSITION_GROUPS[position_group]
    available_stats = [s for s in config['stats'] if s in df.columns]
    available_ratios = [r for r in config['ratios'] if r in df.columns]
    
    return available_stats, available_ratios

@st.cache_data
def load_data():
    df = pd.read_csv("df_merged.csv")
    df['team_full_name'] = df['team'].map(nfl_team_names)
    
    if 'travel_distance_away' in df.columns:
        df['travel_distance'] = df['travel_distance_away']
    
    # Map positions to position groups
    if 'position' in df.columns:
        df['position_group'] = df['position'].apply(get_position_group)
    
    # Calculate performance ratios
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
    if 'pass_average_air_yards' not in df.columns and 'passing_air_yards' in df.columns and 'attempts' in df.columns:
        df['pass_average_air_yards'] = df['passing_air_yards'] / df['attempts']
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
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

@st.cache_data
def calculate_flag_impact(df, metrics, flag):
    results = []
    for metric in metrics:
        if metric not in df.columns or flag not in df.columns:
            continue
        
        metric_data = df[df[metric].notna()]
        if len(metric_data) == 0:
            continue
        
        flag_1 = metric_data[metric_data[flag] == 1][metric].mean()
        flag_0 = metric_data[metric_data[flag] == 0][metric].mean()
        
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

@st.cache_data
def train_rf_model(df, metrics, target_flag):
    model_df = df[df[target_flag].notna()].copy()
    
    if len(model_df) == 0:
        return None, None, None
    
    # Filter to only include metrics that have data
    available_metrics = []
    for m in metrics:
        if m in model_df.columns:
            non_na_count = model_df[m].notna().sum()
            if non_na_count > 10:
                available_metrics.append(m)
    
    if not available_metrics:
        return None, None, None
    
    X = model_df[available_metrics].fillna(0)
    y = model_df[target_flag].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
    
    model = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=99)
    model.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        'metric': available_metrics,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
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
    filtered_df = df.copy()
    
    if season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    # Handle both team and position group filtering
    if entity_type == 'team':
        # Convert full team name to abbreviation if needed
        team_abbr = None
        for abbr, full_name in nfl_team_names.items():
            if full_name == entity_name:
                team_abbr = abbr
                break
        if team_abbr:
            filtered_df = filtered_df[filtered_df['team'] == team_abbr]
        else:
            filtered_df = filtered_df[filtered_df['team'] == entity_name]
    else:  # position_group
        filtered_df = filtered_df[filtered_df['position_group'] == entity_name]
    
    if len(filtered_df) == 0:
        return None
    
    flags = ['isaway', 'is_thursday', 'is_international']
    
    # Get metrics based on entity type
    if entity_type == 'position_group':
        stats, ratios = get_position_metrics(entity_name, filtered_df)
        performance_metrics = stats + ratios + ['offensive_snaps', 'lead_changes', 'travel_distance']
    else:
        # For teams, use all available metrics
        performance_metrics = [
            'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions',
            'sacks', 'sack_yards', 'carries', 'rushing_yards', 'rushing_tds',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds',
            'passing_air_yards', 'passing_yards_after_catch', 'receiving_air_yards',
            'receiving_yards_after_catch', 'racr', 'target_share', 'air_yards_share', 'wopr',
            'offensive_snaps', 'lead_changes', 'travel_distance'
        ]
    
    available_flags = [f for f in flags if f in filtered_df.columns]
    
    # Filter to only include metrics that have sufficient non-null data
    available_metrics = []
    for m in performance_metrics:
        if m in filtered_df.columns:
            non_null_count = filtered_df[m].notna().sum()
            if non_null_count > 10:
                available_metrics.append(m)
    
    if not available_flags or not available_metrics:
        return None
    
    corr_cols = available_flags + available_metrics
    corr_data = filtered_df[corr_cols].copy()
    
    # Calculate correlation and drop any metrics that result in all NaN correlations
    corr_matrix = corr_data.corr()
    flag_correlations = corr_matrix.loc[available_metrics, available_flags]
    
    # Remove rows that have all NaN values
    flag_correlations = flag_correlations.dropna(how='all')
    flag_correlations = flag_correlations.dropna(how='any')
    
    if len(flag_correlations) == 0:
        return None
    
    # Update available_metrics to only include those that remain
    available_metrics = list(flag_correlations.index)
    
    return flag_correlations, available_flags, available_metrics

def create_flag_impact_comparison(df, entity_type, entity_name, season, metric):
    filtered_df = df.copy()
    
    if season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    # Handle both team and position group filtering
    if entity_type == 'team':
        team_abbr = None
        for abbr, full_name in nfl_team_names.items():
            if full_name == entity_name:
                team_abbr = abbr
                break
        if team_abbr:
            filtered_df = filtered_df[filtered_df['team'] == team_abbr]
        else:
            filtered_df = filtered_df[filtered_df['team'] == entity_name]
    else:  # position_group
        filtered_df = filtered_df[filtered_df['position_group'] == entity_name]
    
    if len(filtered_df) == 0 or metric not in filtered_df.columns:
        return None
    
    flags = ['isaway', 'is_thursday', 'is_international']
    results = []
    
    for flag in flags:
        if flag in filtered_df.columns:
            metric_data = filtered_df[filtered_df[metric].notna()]
            flag_1 = metric_data[metric_data[flag] == 1][metric].mean()
            flag_0 = metric_data[metric_data[flag] == 0][metric].mean()
            
            if pd.notna(flag_1) and pd.notna(flag_0):
                pct_change = ((flag_1 - flag_0) / flag_0 * 100) if flag_0 != 0 else 0
                results.append({
                    'flag': flag.replace('_', ' ').title(),
                    'when_true': flag_1,
                    'when_false': flag_0,
                    'pct_change': pct_change
                })
    
    return pd.DataFrame(results) if results else None


def main():
    df = load_data()
     
    # SIDEBAR FILTERS
    st.sidebar.title("Filters")

    if 'selected_season' not in st.session_state:
        st.session_state.selected_season = None
    if 'selected_team' not in st.session_state:
        st.session_state.selected_team = None
    if 'selected_position_group' not in st.session_state:
        st.session_state.selected_position_group = None
    
    seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.selectbox("Select Season", options=['All'] + seasons, key='season_select')
    st.session_state.selected_season = selected_season
    
    if selected_season != 'All':
        filtered_teams = df[df['season'] == selected_season]['team'].unique()
    else:
        filtered_teams = df['team'].unique()
    
    teams = sorted(filtered_teams)
    selected_team = st.sidebar.selectbox("Select Team", options=['All'] + teams, key='team_select')
    st.session_state.selected_team = selected_team
    
    # Position Group Filter
    position_groups = ['All'] + list(POSITION_GROUPS.keys())
    selected_position_group = st.sidebar.selectbox(
        "Select Position Group", 
        options=position_groups, 
        format_func=lambda x: x if x == 'All' else f"{x} - {POSITION_GROUPS[x]['name']}",
        key='position_group_select'
    )
    st.session_state.selected_position_group = selected_position_group
    
    filtered_df = df.copy()
    if selected_season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    if selected_position_group != 'All':
        filtered_df = filtered_df[filtered_df['position_group'] == selected_position_group]

    # Moving Average Window Filter
    moving_avg_window = st.sidebar.slider(
        "Moving Average Window (Games)",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        help="Number of games to use for calculating moving average trends"
    )
    st.session_state.moving_avg_window = moving_avg_window
    
    st.sidebar.info(f"**Active Filters:**\n\n Season: `{selected_season}`\n\n Team: `{selected_team}`\n\n Position: `{selected_position_group}`\n\n Moving Average Window: `{moving_avg_window}`")
    
    # Metric Selection - filtered by position group
    st.sidebar.divider()
    st.sidebar.subheader("Select Metrics (Max 3)")
    
    selected_metrics = []
    
    if selected_position_group != 'All':
        stats, ratios = get_position_metrics(selected_position_group, df)
        
        # Only show stats expander if there are stats available
        if stats:
            with st.sidebar.expander(f"{selected_position_group} Stats", expanded=True):
                st.caption(f"Primary statistics for {POSITION_GROUPS[selected_position_group]['name']}")
                stats_selected = st.multiselect("Select statistics:", stats, key='position_stats',format_func=format_metric_name)
                selected_metrics.extend(stats_selected)
        
        # Only show ratios expander if there are ratios available
        if ratios:
            with st.sidebar.expander(f"{selected_position_group} Performance Ratios"):
                st.caption("Efficiency and utilization metrics")
                ratios_selected = st.multiselect("Select ratios:", ratios, key='position_ratios',format_func=format_metric_name)
                if ratios_selected:
                    st.markdown("**Selected Ratio Descriptions:**")
                    for ratio in ratios_selected:
                        if ratio in performance_ratio_descriptions:
                            st.caption(f"• **{ratio.replace('_', ' ').title()}**: {performance_ratio_descriptions[ratio]}")
                selected_metrics.extend(ratios_selected)
    else:
        st.sidebar.info("Select a position group to view available metrics")
    
    if len(selected_metrics) > 3:
        st.sidebar.error("Please select a maximum of 3 metrics")
        selected_metrics = selected_metrics[:3]
    
    st.sidebar.info(f"**Selected: {len(selected_metrics)}/3 metrics**")
    st.sidebar.divider()
   
    result_df = filtered_df.copy()
    
    # MAIN CONTENT
    st.markdown('<h1 class="main-header">NFL Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(result_df))
    with col2:
        st.metric("Unique Players", result_df['player_display_name'].nunique() if 'player_display_name' in result_df.columns else 0)
    with col3:
        st.metric("Unique Teams", result_df['team'].nunique())
    
    tab1, tab2, tab3 = st.tabs(["Data View", "Model Analysis", "Team/Position Insights"])
    
    with tab1:
        st.subheader("Team Performance Summary")
        
        show_aggregated = (selected_season == 'All' and selected_team == 'All' and selected_position_group == 'All')
        
        if show_aggregated:
            st.write("**Aggregated team statistics across all seasons**")
            team_agg = df.groupby('team').agg({'game_id': 'nunique', 'season': 'nunique'}).reset_index()
            team_agg['Total Thursday Games'] = df[df['is_thursday'] == 1].groupby('team')['game_id'].nunique().reindex(team_agg['team'], fill_value=0).values
            team_agg['Total International Games'] = df[df['is_international'] == 1].groupby('team')['game_id'].nunique().reindex(team_agg['team'], fill_value=0).values
            team_agg['Total Away Games'] = df[df['isaway'] == 1].groupby('team')['game_id'].nunique().reindex(team_agg['team'], fill_value=0).values
            team_agg['Avg Travel Distance'] = np.ceil(df.groupby('team')['travel_distance'].mean().reindex(team_agg['team']).values)
            team_agg['Logo'] = team_agg['team'].map(nfl_logos)
            
            display_df = team_agg[['Logo', 'team', 'Total Thursday Games', 'Total International Games', 'Total Away Games', 'Avg Travel Distance']].copy()
            display_df = display_df.rename(columns={'team': 'Team'})
            display_df = display_df.sort_values('Team').reset_index(drop=True)
            
            st.dataframe(display_df, use_container_width=True, height=600,
                        column_config={
                            "Logo": st.column_config.ImageColumn("", width="small"),
                            "Team": st.column_config.TextColumn("Team", width="small"),
                            "Total Thursday Games": st.column_config.NumberColumn("Total Thursday Games", format="%.0f"),
                            "Total International Games": st.column_config.NumberColumn("Total International Games", format="%.0f"),
                            "Total Away Games": st.column_config.NumberColumn("Total Away Games", format="%.0f"),
                            "Avg Travel Distance": st.column_config.NumberColumn("Avg Travel Distance (miles)", format="%.0f")
                        }, hide_index=True)
            
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Teams", len(display_df))
            with col2:
                st.metric("Avg Thursday Games", f"{display_df['Total Thursday Games'].mean():.2f}")
            with col3:
                st.metric("Avg International Games", f"{display_df['Total International Games'].mean():.2f}")
            with col4:
                st.metric("Avg Travel Distance", f"{display_df['Avg Travel Distance'].mean():,.0f} mi")
        else:
            st.write("**Filtered data view**")
            if selected_metrics:
                display_cols = ['season', 'team', 'player_display_name', 'position_group'] + selected_metrics
                additional_cols = ['week', 'is_thursday', 'is_international', 'travel_distance']
                for col in additional_cols:
                    if col in result_df.columns and col not in display_cols:
                        display_cols.append(col)
                display_cols = [col for col in display_cols if col in result_df.columns]
                st.dataframe(result_df[display_cols], use_container_width=True, height=400)
            else:
                default_cols = ['week','season', 'team', 'player_display_name', 'position_group', 'is_thursday', 'is_international', 'travel_distance']
                display_cols = [col for col in default_cols if col in result_df.columns]
                st.dataframe(result_df[display_cols], use_container_width=True, height=400)
        
        # Travel Map Section - Show when specific team and season are selected
        if selected_team != 'All' and selected_season != 'All':
            st.divider()
            st.subheader("Team Travel Map")
            st.write("Visualize the team's travel throughout the season. Special markers indicate Thursday games (⭐ gold) and international games (◆ purple).")
            
            travel_fig = create_team_travel_map(df, selected_season, selected_team)
            
            if travel_fig is not None:
                st.plotly_chart(travel_fig, use_container_width=True)
                
                # Add travel statistics
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                
                # Get team abbreviation
                team_abbr = selected_team
                for abbr, full_name in nfl_team_names.items():
                    if full_name == selected_team:
                        team_abbr = abbr
                        break
                
                season_data = df[
                    (df['season'] == selected_season) & 
                    (df['team'] == team_abbr)
                ]
                
                if len(season_data) > 0 and 'travel_distance' in season_data.columns:
                    total_distance = season_data['travel_distance'].sum()
                    avg_distance = season_data[season_data['travel_distance'] > 0]['travel_distance'].mean()
                    max_distance = season_data['travel_distance'].max()
                    away_games = season_data['isaway'].sum() if 'isaway' in season_data.columns else 0
                    
                    with col_t1:
                        st.metric("Total Distance", f"{total_distance:,.0f} mi")
                    with col_t2:
                        st.metric("Avg Trip Distance", f"{avg_distance:,.0f} mi" if pd.notna(avg_distance) else "N/A")
                    with col_t3:
                        st.metric("Longest Trip", f"{max_distance:,.0f} mi" if pd.notna(max_distance) else "N/A")
                    with col_t4:
                        st.metric("Away Games", f"{away_games:.0f}")
                    
                    # Additional insights
                    if 'is_thursday' in season_data.columns:
                        thursday_games = season_data['is_thursday'].sum()
                        if thursday_games > 0:
                            st.info(f"⭐ **{thursday_games:.0f} Thursday game(s)** this season")
                    
                    if 'is_international' in season_data.columns:
                        intl_games = season_data['is_international'].sum()
                        if intl_games > 0:
                            st.info(f"◆ **{intl_games:.0f} international game(s)** this season")
            else:
                st.info("No travel data available for the selected team and season.")
        elif selected_team != 'All' and selected_season == 'All':
            st.info("**Tip:** Select a specific season to view the team's travel map")
        elif selected_team == 'All':
            st.info("**Tip:** Select a specific team and season to view the travel map")            
    
    with tab2:
        st.subheader("Model Analysis")
        st.write("Analyze which metrics are most affected by away games, Thursday games, and international games.")
        
        col1, col2 = st.columns(2)
        with col1:
            flag_options = {'Away Games': 'isaway', 'Thursday Games': 'is_thursday', 'International Games': 'is_international'}
            selected_flag_name = st.selectbox("Select Flag to Analyze:", options=list(flag_options.keys()), key='flag_select')
            selected_flag = flag_options[selected_flag_name]
        
        with col2:
            analysis_method = st.selectbox("Analysis Method:", options=['Statistical Comparison', 'Random Forest Model'], key='analysis_method')
        
        if not selected_metrics:
            st.warning("Please select at least one metric from the sidebar to analyze")
        elif selected_flag not in result_df.columns:
            st.warning(f"Flag '{selected_flag_name}' not found in dataset")
        else:
            entity_description = "selected filters"
            if selected_position_group != 'All':
                entity_description = POSITION_GROUPS[selected_position_group]['name']
            if selected_team != 'All':
                entity_description = f"{selected_team} {entity_description}" if selected_position_group != 'All' else selected_team
            
            st.info(f"**Analyzing impact of {selected_flag_name} on {entity_description}**")
            
            if analysis_method == 'Statistical Comparison':
                impact_results = calculate_flag_impact(result_df, selected_metrics, selected_flag)
                
                if not impact_results.empty:
                    st.subheader("Metric Impact Summary")
                    
                    cols = st.columns(min(len(selected_metrics), 3))
                    for idx, row in impact_results.iterrows():
                        col_idx = idx % 3
                        with cols[col_idx]:
                            direction_emoji = "📈" if row['direction'] == 'up' else "📉"
                            color = "normal" if abs(row['pct_change']) < 5 else "inverse"
                            st.metric(label=f"{direction_emoji} {row['metric'].replace('_', ' ').title()}",
                                    value=f"{row['pct_change']:.1f}%", delta=f"{row['difference']:.2f}", delta_color=color)
                            with st.expander("Details"):
                                st.write(f"**When {selected_flag_name} = No:** {row['flag_0_avg']:.2f}")
                                st.write(f"**When {selected_flag_name} = Yes:** {row['flag_1_avg']:.2f}")
                                st.write(f"**Difference:** {row['difference']:.2f}")
                    
                    st.divider()
                    st.subheader("Visual Comparison")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Flag = 0', x=impact_results['metric'], y=impact_results['flag_0_avg'], marker_color='#4ecdc4'))
                    fig.add_trace(go.Bar(name='Flag = 1', x=impact_results['metric'], y=impact_results['flag_1_avg'], marker_color='#ff6b6b'))
                    fig.update_layout(barmode='group', title=f'Metric Averages: {selected_flag_name} Impact',
                                    xaxis_title='Metrics', yaxis_title='Average Value', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Detailed Results")
                    display_impact = impact_results.copy()
                    display_impact['metric'] = display_impact['metric'].str.replace('_', ' ').str.title()
                    display_impact = display_impact.round(2)
                    st.dataframe(display_impact, use_container_width=True)
                else:
                    st.warning("No results to display. Check your data and selected metrics.")
            
            else:  # Random Forest Model
                st.subheader("Random Forest Feature Importance")
                st.write(f"This model identifies which metrics are most predictive of the selected flag condition.")
                
                # Determine which metrics to use based on filters
                if selected_position_group != 'All':
                    stats, ratios = get_position_metrics(selected_position_group, result_df)
                    rf_metrics = stats + ratios
                    universal_metrics = ['offensive_snaps', 'lead_changes', 'travel_distance']
                    rf_metrics.extend([m for m in universal_metrics if m in result_df.columns])
                else:
                    # Use all numeric columns
                    all_numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                    exclude_cols = ['season', 'week', 'away_score', 'home_score', 'result', 'total', 'isaway', 'is_thursday',
                                  'extended_away_games', 'is_international', 'overtime', 'div_game', 'lead_changes', 'gsis',
                                  'old_game_id', 'jersey_number', 'season_type', 'game_id']
                    rf_metrics = [col for col in all_numeric_cols if col not in exclude_cols]
                
                with st.spinner(f"Training Random Forest model..."):
                    importances, metrics_dict, model = train_rf_model(result_df, rf_metrics, selected_flag)
                
                if importances is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Accuracy", f"{metrics_dict['accuracy']:.2%}")
                    with col2:
                        st.metric("F1 Score", f"{metrics_dict['f1_score']:.2f}")
                    with col3:
                        st.metric("Training Samples", f"{metrics_dict['n_samples']:,}")
                    
                    st.divider()
                    
                    top_20 = importances.head(20).copy()
                    selected_in_top20 = [m for m in selected_metrics if m in top_20['metric'].values]
                    selected_not_in_top20 = [m for m in selected_metrics if m not in top_20['metric'].values]
                    
                    if selected_not_in_top20:
                        additional_metrics = importances[importances['metric'].isin(selected_not_in_top20)]
                        display_importances = pd.concat([top_20, additional_metrics], ignore_index=True)
                    else:
                        display_importances = top_20.copy()
                    
                    colors = ['#00cc96' if metric in selected_metrics else '#ff6b6b' for metric in display_importances['metric']]
                    
                    # Format metric names for display (capitalize and remove underscores)
                    display_importances['metric_display'] = display_importances['metric'].str.replace('_', ' ').str.title()
                    
                    st.subheader("Feature Importance Rankings")
                    
                    col_leg1, col_leg2 = st.columns(2)
                    with col_leg1:
                        st.markdown("🔴 **Red bars**: Top 20 most predictive metrics")
                    with col_leg2:
                        st.markdown("🟢 **Green bars**: Your selected metrics")
                    
                    if selected_not_in_top20:
                        st.info(f"**Note:** {len(selected_not_in_top20)} of your selected metrics are outside the top 20 and shown below")
                    
                    fig = go.Figure(go.Bar(x=display_importances['importance'], y=display_importances['metric_display'],
                                          orientation='h', marker_color=colors, text=display_importances['importance'].round(3),
                                          textposition='auto', 
                                          hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
                                          customdata=display_importances['metric']))
                    fig.update_layout(title=f'Which Metrics Best Predict {selected_flag_name}?',
                                    xaxis_title='Importance Score', yaxis_title='Metric',
                                    height=max(500, len(display_importances) * 25),
                                    yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if selected_metrics:
                        st.subheader("Your Selected Metrics Rankings")
                        cols_rank = st.columns(min(len(selected_metrics), 3))
                        
                        for idx, metric in enumerate(selected_metrics):
                            metric_row = importances[importances['metric'] == metric]
                            if not metric_row.empty:
                                rank = importances.index[importances['metric'] == metric].tolist()[0] + 1
                                importance_val = metric_row['importance'].values[0]
                                with cols_rank[idx % 3]:
                                    rank_emoji = "🏆" if rank <= 5 else "⭐" if rank <= 20 else "📍"
                                    rank_color = "off" if rank <= 20 else "inverse"
                                    st.metric(label=f"{rank_emoji} {metric.replace('_', ' ').title()}", value=f"Rank #{rank}",
                                            delta=f"Importance: {importance_val:.4f}", delta_color=rank_color)
                    
                    st.divider()
                    top_metric = importances.iloc[0]
                    st.success(f"**Key Insight:** `{top_metric['metric'].replace('_', ' ').title()}` is the #1 most important metric "
                             f"for predicting {selected_flag_name} (importance: {top_metric['importance']:.4f})")
                    
                    if selected_metrics:
                        highly_predictive_selected = [m for m in selected_metrics if m in importances.head(5)['metric'].values]
                        if highly_predictive_selected:
                            st.success(f"**Your selection includes top predictors:** {', '.join([m.replace('_', ' ').title() for m in highly_predictive_selected])}")
                    
                    st.subheader("Complete Feature Importance Table")
                    display_table = importances.copy()
                    display_table['rank'] = range(1, len(display_table) + 1)
                    display_table['is_selected'] = display_table['metric'].isin(selected_metrics)
                    display_table['metric'] = display_table['metric'].str.replace('_', ' ').str.title()
                    display_table['importance'] = display_table['importance'].round(4)
                    display_table = display_table[['rank', 'metric', 'importance', 'is_selected']]
                    
                    st.dataframe(display_table, use_container_width=True, height=400,
                               column_config={
                                   "rank": st.column_config.NumberColumn("Rank", format="#%d"),
                                   "metric": st.column_config.TextColumn("Metric"),
                                   "importance": st.column_config.NumberColumn("Importance", format="%.4f"),
                                   "is_selected": st.column_config.CheckboxColumn("Your Selection")
                               })
                else:
                    st.error("Unable to train model. Check your data and selected metrics.")
    
    with tab3:
        st.subheader("Team/Position Performance Insights")
        st.write("Analyze performance trends using the global filters. Select a team and/or position group from the sidebar to view insights.")
        
        # Check what filters are active
        has_team_filter = selected_team != 'All'
        has_position_filter = selected_position_group != 'All'
        
        # Display current filter context
        filter_context = []
        if has_team_filter:
            filter_context.append(f"**Team:** {selected_team}")
        if has_position_filter:
            filter_context.append(f"**Position:** {POSITION_GROUPS[selected_position_group]['name']}")
        if selected_season != 'All':
            filter_context.append(f"**Season:** {selected_season}")
        
        if filter_context:
            st.info("**Current Analysis Context:** " + " | ".join(filter_context))
        else:
            st.info("**Tip:** Use the sidebar filters to narrow your analysis to specific teams, positions, or seasons")
        
        # Only show insights if at least team or position is selected
        if has_team_filter or has_position_filter:
            st.divider()
            
            # Get available metrics based on filters
            if has_position_filter:
                stats, ratios = get_position_metrics(selected_position_group, df)
                available_metrics_list = stats + ratios + ['offensive_snaps', 'lead_changes', 'travel_distance']
                available_metrics_list = [m for m in available_metrics_list if m in df.columns]
            else:
                # For teams without position filter, use common metrics
                available_metrics_list = [
                    'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions',
                    'sacks', 'sack_yards', 'carries', 'rushing_yards', 'rushing_tds',
                    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                    'offensive_snaps', 'lead_changes', 'travel_distance'
                ]
                available_metrics_list = [m for m in available_metrics_list if m in df.columns]
            
            if len(available_metrics_list) > 0:
                # Check if user has selected metrics from sidebar
                if selected_metrics:
                    # Use the first selected metric as default
                    default_metric = selected_metrics[0]
                    
                    col_metric1, col_metric2 = st.columns([3, 1])
                    
                    with col_metric1:
                        selected_insight_metric = st.selectbox(
                            "Select a metric to analyze:", 
                            options=selected_metrics,
                            format_func=lambda x: x.replace('_', ' ').title(), 
                            key='insight_metric',
                            index=0
                        )
                    
                    with col_metric2:
                        # Show description if it's a performance ratio
                        if selected_insight_metric in performance_ratio_descriptions:
                            with st.popover("ℹ️ Metric Info"):
                                st.caption("**Description:**")
                                st.write(performance_ratio_descriptions[selected_insight_metric])
                    
                    st.caption(f"📊 Analyzing from your selected metrics in the sidebar. Selected: {len(selected_metrics)}/3")
                else:
                    # No metrics selected - show all available metrics
                    col_metric1, col_metric2 = st.columns([3, 1])
                    
                    with col_metric1:
                        selected_insight_metric = st.selectbox(
                            "Select a metric to analyze:", 
                            options=available_metrics_list,
                            format_func=lambda x: x.replace('_', ' ').title(), 
                            key='insight_metric'
                        )
                    
                    with col_metric2:
                        # Show description if it's a performance ratio
                        if selected_insight_metric in performance_ratio_descriptions:
                            with st.popover("ℹ️ Metric Info"):
                                st.caption("**Description:**")
                                st.write(performance_ratio_descriptions[selected_insight_metric])
                    
                    st.info("💡 **Tip:** Select metrics from the sidebar to limit your choices here to just those metrics")
                
                st.divider()
                
                # Season Trend with Moving Average
                st.subheader(f"Season Trend with {moving_avg_window}-Game Moving Average")
                
                if selected_season == 'All':
                    # Show all seasons
                    available_seasons = sorted(df['season'].unique())
                    
                    for season in available_seasons:
                        with st.expander(f"Season {season}", expanded=(season == available_seasons[-1])):
                            trend_fig = create_season_trend_with_moving_avg(
                                df, selected_team, selected_position_group, season, 
                                selected_insight_metric, moving_avg_window
                            )
                            
                            if trend_fig is not None:
                                st.plotly_chart(trend_fig, use_container_width=True)
                            else:
                                st.warning(f"Insufficient data for season {season}.")
                    
                    # Only show legend if team filter is active
                    if has_team_filter:
                        st.info("**Legend:** ⭐ = Thursday Game | ◆ = International Game | Regular markers = Away Games")
                else:
                    # Show specific season
                    trend_fig = create_season_trend_with_moving_avg(
                        df, selected_team, selected_position_group, selected_season, 
                        selected_insight_metric, moving_avg_window
                    )
                    
                    if trend_fig is not None:
                        st.plotly_chart(trend_fig, use_container_width=True)
                        # Only show legend if team filter is active
                        if has_team_filter:
                            st.info("**Legend:** ⭐ = Thursday Game | ◆ = International Game | Regular markers = Away Games")
                    else:
                        st.warning("Insufficient data for trend analysis with current filters.")
                
                st.divider()
                
                # Team scatter plot
                st.subheader("Team Performance Scatter Analysis")
                st.write("Compare all teams' completion percentage vs attempts across different game conditions.")
                
                scatter_flag = st.selectbox("Select game condition to analyze:",
                                          options=['All Games', 'Away Games', 'Thursday Games', 'International Games'],
                                          key='scatter_flag')
                
                flag_map = {'All Games': None, 'Away Games': 'isaway', 'Thursday Games': 'is_thursday',
                           'International Games': 'is_international'}
                selected_scatter_flag = flag_map[scatter_flag]
                
                # Scatter plot can work with all seasons or specific season
                if selected_season == 'All':
                    # Show all seasons in expandable sections
                    available_seasons = sorted(df['season'].unique())
                    
                    for season in available_seasons:
                        with st.expander(f"Season {season} Scatter Analysis", expanded=(season == available_seasons[-1])):
                            scatter_df = df[df['season'] == season].copy()
                            
                            # Apply team filter if set
                            if has_team_filter:
                                team_abbr = None
                                for abbr, full_name in nfl_team_names.items():
                                    if full_name == selected_team:
                                        team_abbr = abbr
                                        break
                                if team_abbr:
                                    scatter_df = scatter_df[scatter_df['team'] == team_abbr]
                            
                            # Apply position filter if set
                            if has_position_filter:
                                scatter_df = scatter_df[scatter_df['position_group'] == selected_position_group]
                            
                            if selected_scatter_flag:
                                scatter_df = scatter_df[scatter_df[selected_scatter_flag] == 1]
                            
                            if 'attempts' in scatter_df.columns and 'completions' in scatter_df.columns:
                                team_stats = scatter_df.groupby('team').agg({
                                    'attempts': 'sum', 'completions': 'sum',
                                    'passing_yards': 'sum' if 'passing_yards' in scatter_df.columns else 'count'
                                }).reset_index()
                                
                                team_stats['completion_pct'] = (team_stats['completions'] / team_stats['attempts'] * 100).round(2)
                                team_stats = team_stats[team_stats['attempts'] >= 10]
                                
                                if len(team_stats) > 0:
                                    if has_team_filter:
                                        team_abbr = None
                                        for abbr, full_name in nfl_team_names.items():
                                            if full_name == selected_team:
                                                team_abbr = abbr
                                                break
                                        team_stats['is_selected'] = team_stats['team'] == (team_abbr if team_abbr else selected_team)
                                    else:
                                        team_stats['is_selected'] = False
                                    
                                    team_stats['logo_url'] = team_stats['team'].map(nfl_logos)
                                    
                                    fig_scatter = go.Figure()
                                    
                                    for idx, row in team_stats.iterrows():
                                        color = '#ff6b6b' if row['is_selected'] else '#4ecdc4'
                                        size = 15 if row['is_selected'] else 10
                                        fig_scatter.add_trace(go.Scatter(
                                            x=[row['attempts']], y=[row['completion_pct']], mode='markers',
                                            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
                                            name=row['team'], text=row['team'],
                                            hovertemplate='<b>%{text}</b><br>Attempts: %{x}<br>Completion %: %{y:.1f}%<br><extra></extra>',
                                            showlegend=False))
                                        
                                        if pd.notna(row['logo_url']):
                                            fig_scatter.add_layout_image(dict(
                                                source=row['logo_url'], xref="x", yref="y",
                                                x=row['attempts'], y=row['completion_pct'],
                                                sizex=max(team_stats['attempts']) * 0.05, sizey=3,
                                                xanchor="center", yanchor="middle", layer="above"))
                                    
                                    if len(team_stats) > 1:
                                        X = team_stats['attempts'].values.reshape(-1, 1)
                                        y_vals = team_stats['completion_pct'].values
                                        model = LinearRegression()
                                        model.fit(X, y_vals)
                                        y_pred = model.predict(X)
                                        
                                        fig_scatter.add_trace(go.Scatter(
                                            x=team_stats['attempts'], y=y_pred, mode='lines', name='Trendline',
                                            line=dict(color='rgba(255, 107, 107, 0.5)', dash='dash', width=2), showlegend=True))
                                    
                                    fig_scatter.update_layout(title=f'Team Completion % vs Attempts - {scatter_flag} ({season})',
                                                            xaxis_title='Total Attempts', yaxis_title='Completion Percentage (%)',
                                                            height=500, hovermode='closest', showlegend=True)
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                    
                                    col_scatter1, col_scatter2, col_scatter3, col_scatter4 = st.columns(4)
                                    with col_scatter1:
                                        st.metric("Teams Analyzed", len(team_stats))
                                    with col_scatter2:
                                        st.metric("Avg Completion %", f"{team_stats['completion_pct'].mean():.1f}%")
                                    with col_scatter3:
                                        st.metric("Highest Comp %", f"{team_stats['completion_pct'].max():.1f}%")
                                    with col_scatter4:
                                        st.metric("Lowest Comp %", f"{team_stats['completion_pct'].min():.1f}%")
                                else:
                                    st.warning(f"Insufficient data for season {season}.")
                else:
                    # Show specific season
                    scatter_df = df.copy()
                    if selected_season != 'All':
                        scatter_df = scatter_df[scatter_df['season'] == selected_season]
                    
                    # Apply team filter if set
                    if has_team_filter:
                        team_abbr = None
                        for abbr, full_name in nfl_team_names.items():
                            if full_name == selected_team:
                                team_abbr = abbr
                                break
                        if team_abbr:
                            scatter_df = scatter_df[scatter_df['team'] == team_abbr]
                    
                    # Apply position filter if set
                    if has_position_filter:
                        scatter_df = scatter_df[scatter_df['position_group'] == selected_position_group]
                    
                    if selected_scatter_flag:
                        scatter_df = scatter_df[scatter_df[selected_scatter_flag] == 1]
                    
                    if 'attempts' in scatter_df.columns and 'completions' in scatter_df.columns:
                        team_stats = scatter_df.groupby('team').agg({
                            'attempts': 'sum', 'completions': 'sum',
                            'passing_yards': 'sum' if 'passing_yards' in scatter_df.columns else 'count'
                        }).reset_index()
                        
                        team_stats['completion_pct'] = (team_stats['completions'] / team_stats['attempts'] * 100).round(2)
                        team_stats = team_stats[team_stats['attempts'] >= 10]
                        
                        if len(team_stats) > 0:
                            if has_team_filter:
                                team_abbr = None
                                for abbr, full_name in nfl_team_names.items():
                                    if full_name == selected_team:
                                        team_abbr = abbr
                                        break
                                team_stats['is_selected'] = team_stats['team'] == (team_abbr if team_abbr else selected_team)
                            else:
                                team_stats['is_selected'] = False
                            
                            team_stats['logo_url'] = team_stats['team'].map(nfl_logos)
                            
                            fig_scatter = go.Figure()
                            
                            for idx, row in team_stats.iterrows():
                                color = '#ff6b6b' if row['is_selected'] else '#4ecdc4'
                                size = 15 if row['is_selected'] else 10
                                fig_scatter.add_trace(go.Scatter(
                                    x=[row['attempts']], y=[row['completion_pct']], mode='markers',
                                    marker=dict(size=size, color=color, line=dict(width=2, color='white')),
                                    name=row['team'], text=row['team'],
                                    hovertemplate='<b>%{text}</b><br>Attempts: %{x}<br>Completion %: %{y:.1f}%<br><extra></extra>',
                                    showlegend=False))
                                
                                if pd.notna(row['logo_url']):
                                    fig_scatter.add_layout_image(dict(
                                        source=row['logo_url'], xref="x", yref="y",
                                        x=row['attempts'], y=row['completion_pct'],
                                        sizex=max(team_stats['attempts']) * 0.05, sizey=3,
                                        xanchor="center", yanchor="middle", layer="above"))
                            
                            if len(team_stats) > 1:
                                X = team_stats['attempts'].values.reshape(-1, 1)
                                y_vals = team_stats['completion_pct'].values
                                model = LinearRegression()
                                model.fit(X, y_vals)
                                y_pred = model.predict(X)
                                
                                fig_scatter.add_trace(go.Scatter(
                                    x=team_stats['attempts'], y=y_pred, mode='lines', name='Trendline',
                                    line=dict(color='rgba(255, 107, 107, 0.5)', dash='dash', width=2), showlegend=True))
                            
                            fig_scatter.update_layout(title=f'Team Completion % vs Attempts - {scatter_flag}',
                                                    xaxis_title='Total Attempts', yaxis_title='Completion Percentage (%)',
                                                    height=600, hovermode='closest', showlegend=True)
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            col_scatter1, col_scatter2, col_scatter3, col_scatter4 = st.columns(4)
                            with col_scatter1:
                                st.metric("Teams Analyzed", len(team_stats))
                            with col_scatter2:
                                st.metric("Avg Completion %", f"{team_stats['completion_pct'].mean():.1f}%")
                            with col_scatter3:
                                st.metric("Highest Comp %", f"{team_stats['completion_pct'].max():.1f}%")
                            with col_scatter4:
                                st.metric("Lowest Comp %", f"{team_stats['completion_pct'].min():.1f}%")
                            
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
                st.warning("No metrics available for the selected filters.")
        else:
            st.info("💡 **Get started:** Select a team and/or position group from the sidebar to view performance insights and trends.")
        

if __name__ == "__main__":
    main()
