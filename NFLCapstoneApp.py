import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# Define position groups and their relevant metrics
POSITION_GROUPS = {
    'QB': {
        'name': 'Quarterbacks',
        'stats': ["attempts", "completions", "passing_yards", "passing_tds", "interceptions"],
        'ratios': ['snap_share', 'pass_usage']
    },
    'RB': {
        'name': 'Running Backs',
        'stats': ["carries", "rushing_yards", "rushing_tds"],
        'ratios': ['snap_share', 'rusher_usage', 'rusher_yards_per_carry']
    },
    'WR': {
        'name': 'Wide Receivers',
        'stats': ['receptions', 'targets', 'receiving_yards', 'receiving_tds'],
        'ratios': ['snap_share', 'receiver_usage', 'receiver_efficiency']
    },
    'TE': {
        'name': 'Tight Ends',
        'stats': ['receptions', 'targets', 'receiving_yards', 'receiving_tds'],
        'ratios': ['snap_share', 'receiver_usage', 'receiver_efficiency']
    }
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
    
    if 'position' in df.columns:
        df['position_group'] = df['position'].apply(get_position_group)
    
    # Calculate performance ratios
    if 'snap_share' not in df.columns and 'offensive_snaps' in df.columns and 'team_offensive_snaps' in df.columns:
        df['snap_share'] = df['offensive_snaps'] / df['team_offensive_snaps']
    if 'pass_usage' not in df.columns and 'attempts' in df.columns and 'offensive_snaps' in df.columns:
        df['pass_usage'] = df['attempts'] / df['offensive_snaps']
    if 'rusher_usage' not in df.columns and 'carries' in df.columns and 'offensive_snaps' in df.columns:
        df['rusher_usage'] = df['carries'] / df['offensive_snaps']
    if 'rusher_yards_per_carry' not in df.columns and 'rushing_yards' in df.columns and 'carries' in df.columns:
        df['rusher_yards_per_carry'] = df['rushing_yards'] / df['carries']
    if 'receiver_usage' not in df.columns and 'targets' in df.columns and 'offensive_snaps' in df.columns:
        df['receiver_usage'] = df['targets'] / df['offensive_snaps']
    if 'receiver_efficiency' not in df.columns and 'receptions' in df.columns and 'targets' in df.columns:
        df['receiver_efficiency'] = df['receptions'] / df['targets']
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

@st.cache_data
def train_advanced_rf_model(df, target_col, selected_position_group='All'):
    """
    Train an advanced Random Forest model based on your Python script logic
    """
    # Filter by position if specified
    if selected_position_group != 'All':
        df_model = df[df['position_group'] == selected_position_group].copy()
    else:
        df_model = df[df['position_group'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
    
    if len(df_model) < 50:
        return None, None, None, None
    
    # Remove ID columns and select numeric features
    id_cols = ['player_id', 'season', 'week', 'game_id', 'player_name', 'player_display_name']
    X_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in X_cols if c not in id_cols and c != target_col]
    
    # Ensure target exists
    if target_col not in df_model.columns:
        return None, None, None, None
    
    X = df_model[X_cols].fillna(0)
    y = df_model[target_col]
    
    # Drop rows where target is NaN
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(y) < 50 or y.nunique() < 2:
        return None, None, None, None
    
    # Determine if classification or regression
    is_binary = (y.nunique() <= 2) and set(y.unique()).issubset({0, 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=20007, 
        stratify=y if is_binary else None
    )
    
    # Train model
    if is_binary:
        model = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=20007)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_type': 'Classification',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'n_samples': len(y_test)
        }
    else:
        model = RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=20007)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'model_type': 'Regression',
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_test, y_pred),
            'n_samples': len(y_test)
        }
    
    # Feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, metrics, importances, is_binary

def main():
    df = load_data()
     
    # SIDEBAR FILTERS
    st.sidebar.title("Filters")
    
    seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.selectbox("Select Season", options=['All'] + seasons, key='season_select')
    
    if selected_season != 'All':
        filtered_teams = df[df['season'] == selected_season]['team'].unique()
    else:
        filtered_teams = df['team'].unique()
    
    teams = sorted(filtered_teams)
    selected_team = st.sidebar.selectbox("Select Team", options=['All'] + teams, key='team_select')
    
    # Position Group Filter
    position_groups = ['All'] + list(POSITION_GROUPS.keys())
    selected_position_group = st.sidebar.selectbox(
        "Select Position Group", 
        options=position_groups, 
        format_func=lambda x: x if x == 'All' else f"{x} - {POSITION_GROUPS[x]['name']}",
        key='position_group_select'
    )
    
    filtered_df = df.copy()
    if selected_season != 'All':
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
    if selected_position_group != 'All':
        filtered_df = filtered_df[filtered_df['position_group'] == selected_position_group]
    
    result_df = filtered_df.copy()
    
    # MAIN CONTENT
    st.markdown('<h1 class="main-header">NFL Analytics Dashboard - Advanced RF Model</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(result_df))
    with col2:
        st.metric("Unique Players", result_df['player_display_name'].nunique() if 'player_display_name' in result_df.columns else 0)
    with col3:
        st.metric("Unique Teams", result_df['team'].nunique())
    
    tab1, tab2 = st.tabs(["Data View", "Advanced RF Model Analysis"])
    
    with tab1:
        st.subheader("Filtered Data View")
        
        if selected_position_group != 'All':
            stats, ratios = get_position_metrics(selected_position_group, result_df)
            display_cols = ['season', 'week', 'team', 'player_display_name', 'position_group'] + stats[:5] + ratios[:3]
        else:
            display_cols = ['season', 'week', 'team', 'player_display_name', 'position_group']
            
        display_cols = [col for col in display_cols if col in result_df.columns]
        st.dataframe(result_df[display_cols].head(100), use_container_width=True, height=400)
    
    with tab2:
        st.subheader("Advanced Random Forest Model")
        st.write("Train a Random Forest model using your advanced pipeline methodology")
        
        # Target selection
        col1, col2 = st.columns(2)
        
        with col1:
            # Get numeric columns for target selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_targets = ['player_id', 'season', 'week', 'isaway', 'is_thursday', 'is_international']
            target_options = [c for c in numeric_cols if c not in exclude_targets]
            
            selected_target = st.selectbox(
                "Select Target Variable:",
                options=target_options,
                format_func=lambda x: x.replace('_', ' ').title(),
                key='target_select'
            )
        
        with col2:
            st.info(f"**Position Filter**: {selected_position_group}\n\n**Model will be trained on {selected_position_group} data**" if selected_position_group != 'All' else "**Model will be trained on all offensive positions**")
        
        if st.button("Train Advanced RF Model", type="primary"):
            with st.spinner(f"Training Random Forest model for {selected_target}..."):
                model, metrics, importances, is_binary = train_advanced_rf_model(
                    df, selected_target, selected_position_group
                )
                
                if model is None:
                    st.error("Unable to train model. Check your data and filters.")
                else:
                    st.success(f"✅ Model trained successfully ({metrics['model_type']})!")
                    
                    # Display metrics
                    st.divider()
                    st.subheader("Model Performance Metrics")
                    
                    if metrics['model_type'] == 'Classification':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.4f}")
                        with col4:
                            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col3:
                            st.metric("MAE", f"{metrics['mae']:.4f}")
                        with col4:
                            st.metric("Test Samples", f"{metrics['n_samples']:,}")
                    
                    # Feature importances
                    st.divider()
                    st.subheader("Top 20 Feature Importances")
                    
                    top_20 = importances.head(20).copy()
                    top_20['feature_display'] = top_20['feature'].str.replace('_', ' ').str.title()
                    
                    fig = go.Figure(go.Bar(
                        x=top_20['importance'],
                        y=top_20['feature_display'],
                        orientation='h',
                        marker_color='#ff6b6b',
                        text=top_20['importance'].round(4),
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f'Top 20 Feature Importances for {selected_target.replace("_", " ").title()}',
                        xaxis_title='Importance Score',
                        yaxis_title='Feature',
                        height=600,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Full importances table
                    st.subheader("Complete Feature Importance Table")
                    display_importances = importances.copy()
                    display_importances['rank'] = range(1, len(display_importances) + 1)
                    display_importances['feature'] = display_importances['feature'].str.replace('_', ' ').str.title()
                    display_importances['importance'] = display_importances['importance'].round(4)
                    display_importances = display_importances[['rank', 'feature', 'importance']]
                    
                    st.dataframe(
                        display_importances,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", format="#%d"),
                            "feature": st.column_config.TextColumn("Feature"),
                            "importance": st.column_config.NumberColumn("Importance", format="%.4f")
                        }
                    )
                    
                    # Insights
                    st.divider()
                    st.subheader("Model Insights")
                    
                    top_feature = importances.iloc[0]
                    st.success(f"🏆 **Most Important Feature**: `{top_feature['feature'].replace('_', ' ').title()}` (importance: {top_feature['importance']:.4f})")
                    
                    # Check for scheduling factors in top features
                    scheduling_features = ['isaway', 'is_thursday', 'is_international', 'travel_distance', 'rest_days']
                    top_10_features = importances.head(10)['feature'].tolist()
                    scheduling_in_top10 = [f for f in scheduling_features if f in top_10_features]
                    
                    if scheduling_in_top10:
                        st.info(f"📊 **Scheduling factors in top 10**: {', '.join([f.replace('_', ' ').title() for f in scheduling_in_top10])}")
                    else:
                        st.info("📊 **Note**: No scheduling factors (away games, Thursday, international, travel) appear in the top 10 features")

if __name__ == "__main__":
    main()
