import streamlit as st
import pandas as pd
import plotly.express as px  

# Load data
df = pd.read_csv("NFL2014_2024.csv")

# Remove 2020 and 2021
df = df[~df["season"].isin([2020, 2021])]


st.title("NFL Analysis on International, Thursday Night, and Away Games (2014–2024)")
## Is it a Away game
df['is_away']=(df['posteam']==df['away_team']).astype(int)

# Sidebar Filters
teams = df['posteam'].dropna().unique()
metrics = ['pass_attempts', 'passing_yards', 'rushing_yards','points per game']
seasons = sorted(df['season'].dropna().unique())

selected_team = st.sidebar.selectbox("Select Team", sorted(teams))  
selected_metric = st.sidebar.selectbox("Select Game Metric", metrics)
selected_season = st.sidebar.selectbox("Select Season", seasons)

# Filter data
filtered_df = df[
    (df['posteam'] == selected_team) &
    (df['season'] == selected_season)
]
st.sidebar.markdown("""**Note:** 
International Data Begins in 2005 and there are 14 Thursday night teams a season (some teams can get multiple and some can get none) """)
# International Games 
st.subheader("International Games")
intl_games = filtered_df[filtered_df['is_international'] == 1]

if not intl_games.empty:
    fig = px.bar(
        filtered_df,
        x='game_id',
        y=selected_metric,
        title=f"{selected_team} – {selected_metric.replace('_', ' ').title()} in International Games",
        labels={'game_id': 'Game ID', selected_metric: selected_metric.replace('_', ' ').title()},
        color='is_international'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No international games found for this team and season.")

#  Thursday Night Games
st.subheader("Thursday Night Games")
thursday_games = filtered_df[filtered_df['is_thursday'] == 1]

if not thursday_games.empty:
    fig2 = px.bar(
        filtered_df,
        x='game_id',
        y=selected_metric,
        title=f"{selected_team} – {selected_metric.replace('_', ' ').title()} on Thursday Night",
        labels={'game_id': 'Game ID', selected_metric: selected_metric.replace('_', ' ').title()},
        color='is_thursday'
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No Thursday Night games found for this team and season.")

# Plot 3: Away Games
st.subheader("Away Games")
away_games = filtered_df[filtered_df['is_away'] == 1]

if not away_games.empty:
    fig3 = px.bar(
        filtered_df,
        x='game_id',
        y=selected_metric,
        title=f"{selected_team} – {selected_metric.replace('_', ' ').title()} in Away Games",
        labels={'game_id': 'Game ID', selected_metric: selected_metric.replace('_', ' ').title()},
        color='is_away'
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No away games found for this team and season.")

st.subheader("Points per Game vs. Thursday Night Games (League-wide)")
df['points_scored'] = df.apply(
    lambda row: row['home_score'] if row['posteam'] == row['home_team'] else row['away_score'],
    axis=1
)
# Group by team and season to get average points and Thursday games
scatter_data = df.groupby(['posteam', 'season']).agg(
    points_per_game=('points_scored', 'mean'),
    thursday_games=('is_thursday', 'sum')
).reset_index()

# Create scatter plot with Plotly
fig4 = px.scatter(
    scatter_data,
    x='thursday_games',
    y='points_per_game',
    color='posteam',
    #trendline="ols", # regression line
    title="Points per Game vs. Thursday Night Games (All Teams)",
    labels={
        'thursday_games': 'Thursday Night Games',
        'points_per_game': 'Points per Game'
    },
    hover_data=['posteam', 'season']
)

# Highlight selected team & season
highlight = scatter_data[
    (scatter_data['posteam'] == selected_team) & (scatter_data['season'] == selected_season)
]

if not highlight.empty:
    fig4.add_scatter(
        x=highlight['thursday_games'],
        y=highlight['points_per_game'],
        mode='markers+text',
        marker=dict(size=12, color='black', symbol='star'),
        text=[f"{selected_team} {selected_season}"],
        textposition="top center",
        name="Selected Team"
    )

st.plotly_chart(fig4, use_container_width=True)

# Points per Game vs. International Games 

st.subheader("Points per Game vs. International Games (League-wide)")

# Group by team and season for international games
intl_data = df.groupby(['posteam', 'season']).agg(
    points_per_game=('points_scored', 'mean'),
    intl_games=('is_international', 'sum')
).reset_index()

fig5 = px.scatter(
    intl_data,
    x='intl_games',
    y='points_per_game',
    color='posteam',
    #trendline="ols", # linear regression 
    title="Points per Game vs. International Games (All Teams)",
    labels={
        'intl_games': 'International Games',
        'points_per_game': 'Points per Game'
    },
    hover_data=['posteam', 'season']
)

highlight_intl = intl_data[
    (intl_data['posteam'] == selected_team) & (intl_data['season'] == selected_season)
]

if not highlight_intl.empty:
    fig5.add_scatter(
        x=highlight_intl['intl_games'],
        y=highlight_intl['points_per_game'],
        mode='markers+text',
        marker=dict(size=12, color='black', symbol='star'),
        text=[f"{selected_team} {selected_season}"],
        textposition="top center",
        name="Selected Team"
    )

st.plotly_chart(fig5, use_container_width=True)




st.subheader("Points per Game vs. Away Games (League-wide)")

# Create a new column to count when team is away
df['is_away_game'] = df['posteam'] != df['home_team']

away_data = df.groupby(['posteam', 'season']).agg(
    points_per_game=('points_scored', 'mean'),
    away_games=('is_away_game', 'sum')
).reset_index()

fig6 = px.scatter(
    away_data,
    x='away_games',
    y='points_per_game',
    color='posteam',
    #trendline="ols", # linear regression
    title="Points per Game vs. Away Games (All Teams)",
    labels={
        'away_games': 'Away Games',
        'points_per_game': 'Points per Game'
    },
    hover_data=['posteam', 'season']
)

highlight_away = away_data[
    (away_data['posteam'] == selected_team) & (away_data['season'] == selected_season)
]

if not highlight_away.empty:
    fig6.add_scatter(
        x=highlight_away['away_games'],
        y=highlight_away['points_per_game'],
        mode='markers+text',
        marker=dict(size=12, color='black', symbol='star'),
        text=[f"{selected_team} {selected_season}"],
        textposition="top center",
        name="Selected Team"
    )

st.plotly_chart(fig6, use_container_width=True)