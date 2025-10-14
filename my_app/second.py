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

# Generate sample data
@st.cache_data
def load_data():
    # LLM Performance Data
    models = ['GPT-4', 'Claude-3', 'Gemini-Pro', 'LLaMA-2', 'PaLM-2', 'GPT-3.5', 'Mistral-7B']
    performance_data = pd.DataFrame({
        'Model': models,
        'Parameters (B)': [1000, 500, 540, 70, 540, 175, 7],
        'MMLU Score': [86.4, 84.9, 83.7, 68.9, 78.3, 70.0, 64.2],
        'HumanEval Score': [67.0, 65.8, 74.4, 29.9, 44.3, 48.1, 40.2],
        'Cost per 1M tokens': [30, 25, 20, 5, 15, 2, 0.5],
        'Release Date': ['2023-03', '2024-03', '2023-12', '2023-07', '2023-05', '2022-11', '2023-09']
    })
    
    # Usage trends over time
    dates = pd.date_range(start='2023-01-01', end='2024-03-01', freq='M')
    usage_data = pd.DataFrame({
        'Date': dates,
        'API Calls (Millions)': np.cumsum(np.random.randint(10, 100, len(dates))),
        'Active Users (Thousands)': np.cumsum(np.random.randint(5, 50, len(dates))),
        'New Applications': np.random.randint(50, 500, len(dates))
    })
    
    # Market segments
    segments = ['Software Development', 'Content Creation', 'Customer Service', 
               'Research', 'Education', 'Healthcare', 'Finance']
    segment_data = pd.DataFrame({
        'Segment': segments,
        'Adoption Rate (%)': [85, 78, 72, 68, 64, 45, 38],
        'Growth Rate (%)': [120, 95, 88, 75, 82, 65, 55]
    })
    
    return performance_data, usage_data, segment_data

# Load data
perf_data, usage_data, segment_data = load_data()

# Header
#st.markdown('<h1 class="main-header">🤖 State of Large Language Models</h1>', unsafe_allow_html=True)
#st.markdown("---")

# Sidebar filters
st.sidebar.title("🔍 Filters & Settings")
selected_models = st.sidebar.multiselect(
    "Select Models",
    options=perf_data['Model'].tolist(),
    default=perf_data['Model'].tolist()[:4]
)

metric_type = st.sidebar.selectbox(
    "Primary Metric",
    ["MMLU Score", "HumanEval Score", "Parameters (B)", "Cost per 1M tokens"]
)

show_trends = st.sidebar.checkbox("Show Trend Analysis", value=True)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["", "📈 Market Trends", "🏢 Industry Adoption", "🔮 Insights"])

with tab1:
    st.header("Model Performance Comparison")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Models Tracked", 
            len(perf_data),
            delta="2 new this month"
        )
    
    with col2:
        avg_mmlu = perf_data['MMLU Score'].mean()
        st.metric(
            "Average MMLU Score", 
            f"{avg_mmlu:.1f}",
            delta="5.2% vs last quarter"
        )
    
    with col3:
        top_model = perf_data.loc[perf_data['MMLU Score'].idxmax(), 'Model']
        st.metric(
            "Leading Model", 
            top_model,
            delta="GPT-4 maintains lead"
        )
    
    with col4:
        min_cost = perf_data['Cost per 1M tokens'].min()
        st.metric(
            "Lowest Cost", 
            f"${min_cost}",
            delta="-60% cost reduction"
        )
    
    st.markdown("---")
    
    # Filter data based on selection
    filtered_data = perf_data[perf_data['Model'].isin(selected_models)]
    
    # Performance scatter plot
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_scatter = px.scatter(
            filtered_data,
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
        ranking_metric = metric_type
        if ranking_metric == "Cost per 1M tokens":
            top_models = filtered_data.nsmallest(5, ranking_metric)
        else:
            top_models = filtered_data.nlargest(5, ranking_metric)
        
        for i, (_, model) in enumerate(top_models.iterrows(), 1):
            value = model[ranking_metric]
            if ranking_metric == "Cost per 1M tokens":
                st.write(f"{i}. **{model['Model']}** - ${value}")
            else:
                st.write(f"{i}. **{model['Model']}** - {value}")

with tab2:
    st.header("Market Trends & Growth")
    
    # Usage trends
    col1, col2 = st.columns(2)
    
    with col1:
        fig_api = px.line(
            usage_data,
            x='Date',
            y='API Calls (Millions)',
            title="API Usage Growth",
            markers=True
        )
        fig_api.update_layout(height=400)
        st.plotly_chart(fig_api, use_container_width=True)
    
    with col2:
        fig_users = px.line(
            usage_data,
            x='Date',
            y='Active Users (Thousands)',
            title="Active User Growth",
            markers=True,
            color_discrete_sequence=['#ff6b6b']
        )
        fig_users.update_layout(height=400)
        st.plotly_chart(fig_users, use_container_width=True)
    
    # Market size projection
    st.subheader("Market Projections")
    
    future_dates = pd.date_range(start='2024-04-01', end='2025-12-01', freq='M')
    # Simple exponential growth projection
    last_value = usage_data['API Calls (Millions)'].iloc[-1]
    growth_rate = 1.15  # 15% monthly growth
    projections = [last_value * (growth_rate ** i) for i in range(1, len(future_dates) + 1)]
    
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Projected API Calls (Millions)': projections
    })
    
    # Combine historical and projected data
    combined_data = pd.concat([
        usage_data[['Date', 'API Calls (Millions)']].rename(columns={'API Calls (Millions)': 'Value'}),
        future_data[['Date', 'Projected API Calls (Millions)']].rename(columns={'Projected API Calls (Millions)': 'Value'})
    ])
    combined_data['Type'] = ['Historical'] * len(usage_data) + ['Projected'] * len(future_data)
    
    fig_projection = px.line(
        combined_data,
        x='Date',
        y='Value',
        color='Type',
        title="API Usage: Historical vs Projected",
        labels={'Value': 'API Calls (Millions)'}
    )
    st.plotly_chart(fig_projection, use_container_width=True)

with tab3:
    st.header("Industry Adoption Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Adoption rate by segment
        fig_adoption = px.bar(
            segment_data,
            x='Adoption Rate (%)',
            y='Segment',
            orientation='h',
            title="LLM Adoption Rate by Industry",
            color='Adoption Rate (%)',
            color_continuous_scale='Blues'
        )
        fig_adoption.update_layout(height=500)
        st.plotly_chart(fig_adoption, use_container_width=True)
    
    with col2:
        # Growth rate comparison
        fig_growth = px.scatter(
            segment_data,
            x='Adoption Rate (%)',
            y='Growth Rate (%)',
            size='Adoption Rate (%)',
            hover_name='Segment',
            title="Adoption vs Growth Rate",
            labels={'Growth Rate (%)': 'YoY Growth Rate (%)'}
        )
        fig_growth.update_layout(height=500)
        st.plotly_chart(fig_growth, use_container_width=True)
    
    # Detailed breakdown
    st.subheader("Segment Analysis")
    
    for _, segment in segment_data.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{segment['Segment']}**")
        with col2:
            st.write(f"Adoption: {segment['Adoption Rate (%)']}%")
        with col3:
            st.write(f"Growth: {segment['Growth Rate (%)']}%")

with tab4:
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
    
    for insight in insights:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{insight['title']}</h3>
            <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Trend analysis
    if show_trends:
        st.subheader("Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Key Trends Observed:**")
            st.write("• Model performance plateauing on traditional benchmarks")
            st.write("• Shift focus to specialized and multimodal capabilities")
            st.write("• Increased emphasis on cost-efficiency and speed")
            st.write("• Growing enterprise adoption across all sectors")
        
        with col2:
            st.write("**Emerging Patterns:**")
            st.write("• Open-source models closing the performance gap")
            st.write("• Real-time applications driving infrastructure demands")
            st.write("• Regulatory considerations shaping development")
            st.write("• Fine-tuning becoming standard practice")

# Footer
st.markdown("---")
