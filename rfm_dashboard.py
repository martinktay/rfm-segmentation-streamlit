import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Luxury Fashion RFM Segmentation Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .segment-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the RFM dataset."""
    try:
        df = pd.read_csv('rfm_enriched.csv')
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def calculate_rfm(df):
    """
    Calculate RFM metrics for each customer.

    Args:
        df: DataFrame with transaction data

    Returns:
        DataFrame with RFM metrics per customer
    """
    # Get the most recent date for recency calculation
    max_date = df['purchase_date'].max()

    # Calculate RFM metrics
    rfm = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (max_date - x.max()).days,  # Recency
        'customer_id': 'count',  # Frequency
        'purchase_amount': 'sum'  # Monetary
    }).rename(columns={
        'purchase_date': 'recency',
        'customer_id': 'frequency',
        'purchase_amount': 'monetary'
    })

    # Reset index to make customer_id a column
    rfm = rfm.reset_index()

    return rfm


def normalize_rfm(rfm_df):
    """
    Normalize RFM values for clustering.

    Args:
        rfm_df: DataFrame with RFM metrics

    Returns:
        Normalized DataFrame and scaler
    """
    # Create a copy for normalization
    rfm_normalized = rfm_df[['recency', 'frequency', 'monetary']].copy()

    # Apply log transformation to handle skewness
    rfm_normalized['recency'] = np.log1p(rfm_normalized['recency'])
    rfm_normalized['frequency'] = np.log1p(rfm_normalized['frequency'])
    rfm_normalized['monetary'] = np.log1p(rfm_normalized['monetary'])

    # Standardize the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_normalized)

    return rfm_scaled, scaler, rfm_normalized


def find_optimal_clusters(rfm_scaled, max_clusters=10):
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Args:
        rfm_scaled: Scaled RFM data
        max_clusters: Maximum number of clusters to test

    Returns:
        Optimal number of clusters
    """
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)

        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette_scores.append(
                silhouette_score(rfm_scaled, kmeans.labels_))
        else:
            silhouette_scores.append(0)

    # Find elbow point (simplified method)
    elbow_idx = np.argmin(np.diff(inertias, 2)) + 2

    # Find best silhouette score
    best_silhouette_idx = np.argmax(silhouette_scores) + 1

    # Choose the optimal number of clusters
    optimal_k = min(elbow_idx, best_silhouette_idx)

    return optimal_k, inertias, silhouette_scores, K_range


def perform_clustering(rfm_scaled, n_clusters):
    """
    Perform K-means clustering on RFM data.

    Args:
        rfm_scaled: Scaled RFM data
        n_clusters: Number of clusters

    Returns:
        KMeans model and cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled)

    return kmeans, cluster_labels


def assign_segment_names(cluster_labels, rfm_df):
    """
    Assign meaningful names to clusters based on RFM characteristics.

    Args:
        cluster_labels: Cluster labels from K-means
        rfm_df: DataFrame with RFM metrics

    Returns:
        List of segment names
    """
    # Create a DataFrame with cluster labels
    cluster_df = rfm_df.copy()
    cluster_df['cluster'] = cluster_labels

    # Calculate cluster centroids
    cluster_centers = cluster_df.groupby(
        'cluster')[['recency', 'frequency', 'monetary']].mean()

    # Sort clusters by monetary value (descending)
    cluster_centers_sorted = cluster_centers.sort_values(
        'monetary', ascending=False)

    # Define segment names based on RFM characteristics with more flexible thresholds
    segment_names = []
    for i, (cluster_id, center) in enumerate(cluster_centers_sorted.iterrows()):
        recency, frequency, monetary = center['recency'], center['frequency'], center['monetary']

        # Luxury fashion-specific naming logic
        if recency < 50 and frequency > 4 and monetary > 600:
            name = "VIP Fashionistas"
        elif recency < 80 and frequency > 3 and monetary > 400:
            name = "Loyal Luxury Buyers"
        elif recency > 100 and frequency > 2 and monetary > 200:
            name = "At-Risk Premium Customers"
        elif recency < 40 and frequency < 3 and monetary < 300:
            name = "New Luxury Prospects"
        elif recency > 120 and frequency < 2 and monetary < 150:
            name = "Dormant High-End Customers"
        elif monetary > 500:
            name = "High-Value Fashion Enthusiasts"
        elif frequency > 4:
            name = "Frequent Luxury Shoppers"
        elif recency < 60:
            name = "Recent Fashion Buyers"
        else:
            name = f"Premium Segment {i+1}"

        segment_names.append((cluster_id, name))

    # Create mapping from cluster ID to segment name
    segment_mapping = dict(segment_names)

    return [segment_mapping[label] for label in cluster_labels]


def create_rfm_visualizations(rfm_df, cluster_labels, segment_names):
    """
    Create RFM analysis visualizations.

    Args:
        rfm_df: DataFrame with RFM metrics
        cluster_labels: Cluster labels
        segment_names: Segment names

    Returns:
        Dictionary of plotly figures
    """
    # Add cluster and segment information
    rfm_viz = rfm_df.copy()
    rfm_viz['cluster'] = cluster_labels
    rfm_viz['segment'] = segment_names

    # 1. RFM Distribution with enhanced colors
    fig_rfm_dist = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Recency Distribution',
                        'Frequency Distribution', 'Monetary Distribution'),
        specs=[[{"type": "histogram"}, {
            "type": "histogram"}, {"type": "histogram"}]]
    )

    fig_rfm_dist.add_trace(go.Histogram(
        x=rfm_viz['recency'], name='Recency', nbinsx=30,
        marker_color='#2E8B57'), row=1, col=1)
    fig_rfm_dist.add_trace(go.Histogram(
        x=rfm_viz['frequency'], name='Frequency', nbinsx=30,
        marker_color='#FF6B6B'), row=1, col=2)
    fig_rfm_dist.add_trace(go.Histogram(
        x=rfm_viz['monetary'], name='Monetary', nbinsx=30,
        marker_color='#4ECDC4'), row=1, col=3)

    fig_rfm_dist.update_layout(
        height=400, showlegend=False, title_text="RFM Metrics Distribution",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # 2. Segment Distribution with enhanced colors
    segment_counts = rfm_viz['segment'].value_counts()
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_segments.update_layout(
        height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # 3. RFM Scatter Plot (3D) with enhanced colors
    fig_3d = px.scatter_3d(
        rfm_viz,
        x='recency',
        y='frequency',
        z='monetary',
        color='segment',
        title="3D RFM Scatter Plot by Segment",
        labels={
            'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary (£)'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_3d.update_layout(
        height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # 4. Segment Characteristics Heatmap with enhanced colors
    segment_means = rfm_viz.groupby(
        'segment')[['recency', 'frequency', 'monetary']].mean()

    fig_heatmap = px.imshow(
        segment_means.T,
        text_auto=True,
        aspect="auto",
        title="Segment Characteristics Heatmap",
        labels=dict(x="Segment", y="RFM Metric", color="Value"),
        color_continuous_scale='viridis'
    )
    fig_heatmap.update_layout(
        height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # 5. Monetary vs Frequency by Segment with enhanced colors
    fig_scatter = px.scatter(
        rfm_viz,
        x='frequency',
        y='monetary',
        color='segment',
        size='recency',
        hover_data=['customer_id'],
        title="Monetary vs Frequency by Segment (Size = Recency)",
        labels={'frequency': 'Frequency',
                'monetary': 'Monetary (£)', 'recency': 'Recency (days)'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_scatter.update_layout(
        height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return {
        'rfm_distribution': fig_rfm_dist,
        'segment_distribution': fig_segments,
        'rfm_3d': fig_3d,
        'segment_heatmap': fig_heatmap,
        'monetary_frequency': fig_scatter
    }


def create_behavioral_analysis(original_df, rfm_df, cluster_labels, segment_names):
    """
    Create comprehensive behavioral analysis with multi-dimensional insights.

    Args:
        original_df: Original transaction data
        rfm_df: DataFrame with RFM metrics
        cluster_labels: Cluster labels
        segment_names: Segment names

    Returns:
        Dictionary with behavioral analysis visualizations and insights
    """
    # Combine RFM data with original data
    rfm_with_segments = rfm_df.copy()
    rfm_with_segments['cluster'] = cluster_labels
    rfm_with_segments['segment'] = segment_names

    # Merge with original data
    customer_segments = rfm_with_segments[['customer_id', 'segment']]
    enriched_data = original_df.merge(
        customer_segments, on='customer_id', how='left')

    # 1. Channel Performance Analysis
    channel_stats = enriched_data.groupby('channel').agg({
        'purchase_amount': ['count', 'mean', 'sum'],
        'customer_id': 'nunique'
    }).round(2)
    channel_stats.columns = ['Transaction_Count',
                             'Avg_Amount', 'Total_Revenue', 'Unique_Customers']

    fig_channel = px.bar(
        channel_stats,
        x=channel_stats.index,
        y=['Transaction_Count', 'Unique_Customers'],
        title="Channel Performance Analysis",
        color_discrete_sequence=['#2E8B57', '#FF6B6B'],
        barmode='group'
    )
    fig_channel.update_layout(
        height=400, xaxis_title="Channel", yaxis_title="Count")

    # 2. Customer Tier Distribution
    tier_stats = enriched_data.groupby('customer_tier').agg({
        'purchase_amount': ['count', 'mean', 'sum'],
        'customer_id': 'nunique'
    }).round(2)
    tier_stats.columns = ['Transaction_Count',
                          'Avg_Amount', 'Total_Revenue', 'Unique_Customers']

    fig_tier = px.bar(
        tier_stats,
        x=tier_stats.index,
        y='Avg_Amount',
        title="Average Transaction Value by Customer Tier",
        color='Avg_Amount',
        color_continuous_scale='viridis'
    )
    fig_tier.update_layout(
        height=400, xaxis_title="Customer Tier", yaxis_title="Average Amount (£)")

    # 3. Product Category Analysis
    category_stats = enriched_data.groupby('product_category').agg({
        'purchase_amount': ['count', 'mean', 'sum']
    }).round(2)
    category_stats.columns = [
        'Transaction_Count', 'Avg_Amount', 'Total_Revenue']

    fig_category = px.treemap(
        category_stats,
        path=[category_stats.index],
        values='Total_Revenue',
        color='Avg_Amount',
        color_continuous_scale='plasma',
        title="Product Category Revenue Distribution"
    )
    fig_category.update_layout(height=400)

    # 4. Transaction Type Analysis
    transaction_stats = enriched_data.groupby('transaction_type').agg({
        'purchase_amount': ['count', 'mean', 'sum']
    }).round(2)
    transaction_stats.columns = [
        'Transaction_Count', 'Avg_Amount', 'Total_Revenue']

    fig_transaction = px.pie(
        values=transaction_stats['Transaction_Count'],
        names=transaction_stats.index,
        title="Transaction Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_transaction.update_layout(height=400)

    # 5. Segment-Channel Heatmap
    segment_channel = pd.crosstab(
        enriched_data['segment'],
        enriched_data['channel'],
        values=enriched_data['purchase_amount'],
        aggfunc='mean'
    ).fillna(0)

    fig_segment_channel = px.imshow(
        segment_channel,
        text_auto=True,
        aspect="auto",
        title="Average Transaction Value: Segment vs Channel",
        color_continuous_scale='RdYlBu',
        labels=dict(x="Channel", y="Segment", color="Avg Amount (£)")
    )
    fig_segment_channel.update_layout(height=400)

    # 6. Segment-Tier Heatmap
    segment_tier = pd.crosstab(
        enriched_data['segment'],
        enriched_data['customer_tier'],
        values=enriched_data['purchase_amount'],
        aggfunc='mean'
    ).fillna(0)

    fig_segment_tier = px.imshow(
        segment_tier,
        text_auto=True,
        aspect="auto",
        title="Average Transaction Value: Segment vs Customer Tier",
        color_continuous_scale='YlOrRd',
        labels=dict(x="Customer Tier", y="Segment", color="Avg Amount (£)")
    )
    fig_segment_tier.update_layout(height=400)

    # Generate insights
    channel_insights = {
        'primary_channel': channel_stats['Transaction_Count'].idxmax(),
        'primary_percentage': (channel_stats['Transaction_Count'].max() / channel_stats['Transaction_Count'].sum()) * 100,
        'high_value_channel': channel_stats['Avg_Amount'].idxmax(),
        'high_value_amount': channel_stats['Avg_Amount'].max()
    }

    tier_insights = {
        'dominant_tier': tier_stats['Unique_Customers'].idxmax(),
        'dominant_percentage': (tier_stats['Unique_Customers'].max() / tier_stats['Unique_Customers'].sum()) * 100,
        'highest_value_tier': tier_stats['Avg_Amount'].idxmax(),
        'highest_value_amount': tier_stats['Avg_Amount'].max()
    }

    category_insights = {
        'top_category': category_stats['Transaction_Count'].idxmax(),
        'top_percentage': (category_stats['Transaction_Count'].max() / category_stats['Transaction_Count'].sum()) * 100,
        'highest_value_category': category_stats['Avg_Amount'].idxmax(),
        'highest_value_amount': category_stats['Avg_Amount'].max()
    }

    return {
        'channel_analysis': fig_channel,
        'tier_analysis': fig_tier,
        'category_analysis': fig_category,
        'transaction_analysis': fig_transaction,
        'segment_channel_heatmap': fig_segment_channel,
        'segment_tier_heatmap': fig_segment_tier,
        'channel_insights': channel_insights,
        'tier_insights': tier_insights,
        'category_insights': category_insights
    }


def create_temporal_analysis(original_df, rfm_df, cluster_labels, segment_names):
    """
    Create temporal analysis with time-series insights.

    Args:
        original_df: Original transaction data
        rfm_df: DataFrame with RFM metrics
        cluster_labels: Cluster labels
        segment_names: Segment names

    Returns:
        Dictionary with temporal analysis visualizations and insights
    """
    # Combine RFM data with original data
    rfm_with_segments = rfm_df.copy()
    rfm_with_segments['cluster'] = cluster_labels
    rfm_with_segments['segment'] = segment_names

    # Merge with original data
    customer_segments = rfm_with_segments[['customer_id', 'segment']]
    enriched_data = original_df.merge(
        customer_segments, on='customer_id', how='left')

    # Add time-based features
    enriched_data['month'] = enriched_data['purchase_date'].dt.to_period('M')
    enriched_data['quarter'] = enriched_data['purchase_date'].dt.quarter
    enriched_data['year'] = enriched_data['purchase_date'].dt.year

    # 1. Monthly Transaction Trends
    monthly_stats = enriched_data.groupby('month').agg({
        'purchase_amount': ['count', 'mean', 'sum'],
        'customer_id': 'nunique'
    }).round(2)
    monthly_stats.columns = ['Transaction_Count',
                             'Avg_Amount', 'Total_Revenue', 'Unique_Customers']
    monthly_stats = monthly_stats.reset_index()
    monthly_stats['month'] = monthly_stats['month'].astype(str)

    fig_monthly = px.line(
        monthly_stats,
        x='month',
        y=['Transaction_Count', 'Unique_Customers'],
        title="Monthly Transaction Trends",
        color_discrete_sequence=['#2E8B57', '#FF6B6B']
    )
    fig_monthly.update_layout(
        height=400, xaxis_title="Month", yaxis_title="Count")

    # 2. Seasonal Purchase Patterns
    seasonal_stats = enriched_data.groupby('quarter').agg({
        'purchase_amount': ['count', 'mean', 'sum']
    }).round(2)
    seasonal_stats.columns = [
        'Transaction_Count', 'Avg_Amount', 'Total_Revenue']
    seasonal_stats = seasonal_stats.reset_index()

    fig_seasonal = px.bar(
        seasonal_stats,
        x='quarter',
        y='Transaction_Count',
        title="Seasonal Purchase Patterns by Quarter",
        color='Avg_Amount',
        color_continuous_scale='plasma'
    )
    fig_seasonal.update_layout(
        height=400, xaxis_title="Quarter", yaxis_title="Transaction Count")

    # 3. Segment Evolution Over Time
    segment_monthly = enriched_data.groupby(
        ['month', 'segment']).size().reset_index(name='count')
    segment_monthly['month'] = segment_monthly['month'].astype(str)

    fig_segment_evolution = px.line(
        segment_monthly,
        x='month',
        y='count',
        color='segment',
        title="Segment Evolution Over Time",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_segment_evolution.update_layout(
        height=400, xaxis_title="Month", yaxis_title="Customer Count")

    # 4. Revenue Trends by Channel
    channel_monthly = enriched_data.groupby(['month', 'channel'])[
        'purchase_amount'].sum().reset_index()
    channel_monthly['month'] = channel_monthly['month'].astype(str)

    fig_channel_trends = px.line(
        channel_monthly,
        x='month',
        y='purchase_amount',
        color='channel',
        title="Revenue Trends by Channel",
        color_discrete_sequence=['#2E8B57', '#FF6B6B']
    )
    fig_channel_trends.update_layout(
        height=400, xaxis_title="Month", yaxis_title="Revenue (£)")

    # Generate temporal insights
    seasonal_insights = {
        'peak_month': monthly_stats.loc[monthly_stats['Transaction_Count'].idxmax(), 'month'],
        'peak_percentage': (monthly_stats['Transaction_Count'].max() / monthly_stats['Transaction_Count'].sum()) * 100,
        'lowest_month': monthly_stats.loc[monthly_stats['Transaction_Count'].idxmin(), 'month'],
        'lowest_percentage': (monthly_stats['Transaction_Count'].min() / monthly_stats['Transaction_Count'].sum()) * 100
    }

    # Calculate growth rates
    first_month = monthly_stats.iloc[0]
    last_month = monthly_stats.iloc[-1]

    transaction_growth = (
        (last_month['Transaction_Count'] - first_month['Transaction_Count']) / first_month['Transaction_Count']) * 100
    value_growth = (
        (last_month['Avg_Amount'] - first_month['Avg_Amount']) / first_month['Avg_Amount']) * 100

    growth_insights = {
        'transaction_growth': transaction_growth,
        'value_growth': value_growth
    }

    # Channel growth analysis
    channel_growth = channel_monthly.groupby('channel')['purchase_amount'].apply(
        lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) *
        100 if x.iloc[0] != 0 else 0
    )

    fastest_growing_channel = channel_growth.idxmax()
    growth_rate = channel_growth.max()

    channel_insights = {
        'fastest_growing_channel': fastest_growing_channel,
        'growth_rate': growth_rate
    }

    return {
        'monthly_trends': fig_monthly,
        'seasonal_patterns': fig_seasonal,
        'segment_evolution': fig_segment_evolution,
        'channel_trends': fig_channel_trends,
        'seasonal_insights': seasonal_insights,
        'growth_insights': growth_insights,
        'channel_insights': channel_insights
    }


def create_business_insights(rfm_df, cluster_labels, segment_names, original_df):
    """
    Generate business insights for each segment.

    Args:
        rfm_df: DataFrame with RFM metrics
        cluster_labels: Cluster labels
        segment_names: Segment names
        original_df: Original transaction data

    Returns:
        Dictionary with insights for each segment
    """
    # Combine RFM data with original data
    rfm_with_segments = rfm_df.copy()
    rfm_with_segments['cluster'] = cluster_labels
    rfm_with_segments['segment'] = segment_names

    # Merge with original data to get additional context
    customer_segments = rfm_with_segments[['customer_id', 'segment']]
    enriched_data = original_df.merge(
        customer_segments, on='customer_id', how='left')

    insights = {}

    for segment in rfm_with_segments['segment'].unique():
        segment_data = rfm_with_segments[rfm_with_segments['segment'] == segment]
        segment_transactions = enriched_data[enriched_data['segment'] == segment]

        # Basic RFM stats
        avg_recency = segment_data['recency'].mean()
        avg_frequency = segment_data['frequency'].mean()
        avg_monetary = segment_data['monetary'].mean()
        segment_size = len(segment_data)

        # Channel preference
        channel_dist = segment_transactions['channel'].value_counts(
            normalize=True)
        preferred_channel = channel_dist.index[0] if len(
            channel_dist) > 0 else "N/A"

        # Product category preference
        category_dist = segment_transactions['product_category'].value_counts(
            normalize=True)
        preferred_category = category_dist.index[0] if len(
            category_dist) > 0 else "N/A"

        # Customer tier distribution
        tier_dist = segment_transactions['customer_tier'].value_counts(
            normalize=True)
        dominant_tier = tier_dist.index[0] if len(tier_dist) > 0 else "N/A"

        # Generate recommendations
        recommendations = []
        if avg_recency < 30:
            recommendations.append(
                "Recent customers - seasonal collection previews")
        elif avg_recency > 90:
            recommendations.append(
                "At-risk customers - exclusive re-engagement events")

        if avg_frequency > 5:
            recommendations.append(
                "High-frequency buyers - VIP membership programs")
        elif avg_frequency < 2:
            recommendations.append(
                "Low-frequency buyers - personalized styling sessions")

        if avg_monetary > 500:
            recommendations.append(
                "High-value segment - VIP services and exclusive collections")
        elif avg_monetary < 100:
            recommendations.append(
                "Low-value segment - premium product introductions")

        # Calculate revenue metrics
        total_revenue = segment_transactions['purchase_amount'].sum()
        avg_transaction = segment_transactions['purchase_amount'].mean() if len(segment_transactions) > 0 else 0
        total_revenue_all = enriched_data['purchase_amount'].sum()
        revenue_percentage = (total_revenue / total_revenue_all * 100) if total_revenue_all > 0 else 0

        insights[segment] = {
            'size': segment_size,
            'avg_recency': avg_recency,
            'avg_frequency': avg_frequency,
            'avg_monetary': avg_monetary,
            'total_revenue': total_revenue,
            'revenue_percentage': revenue_percentage,
            'avg_transaction': avg_transaction,
            'preferred_channel': preferred_channel,
            'preferred_category': preferred_category,
            'dominant_tier': dominant_tier,
            'recommendations': recommendations
        }

    return insights


def create_non_loyal_insights(rfm_df, cluster_labels, segment_names, original_df):
    """
    Create insights for non-loyal customers.

    Args:
        rfm_df: DataFrame with RFM metrics
        cluster_labels: Cluster labels
        segment_names: Segment names
        original_df: Original transaction data

    Returns:
        Dictionary with insights for non-loyal customers
    """
    # Combine RFM data with original data
    rfm_with_segments = rfm_df.copy()
    rfm_with_segments['cluster'] = cluster_labels
    rfm_with_segments['segment'] = segment_names

    # Merge with original data
    customer_segments = rfm_with_segments[['customer_id', 'segment']]
    enriched_data = original_df.merge(
        customer_segments, on='customer_id', how='left')

    # Define loyal vs non-loyal segments
    loyal_segments = [seg for seg in rfm_with_segments['segment'].unique()
                      if "Loyal" in seg or "VIP" in seg or "Champion" in seg]
    non_loyal_segments = [seg for seg in rfm_with_segments['segment'].unique()
                          if seg not in loyal_segments]

    loyal_data = rfm_with_segments[rfm_with_segments['segment'].isin(
        loyal_segments)]
    non_loyal_data = rfm_with_segments[rfm_with_segments['segment'].isin(
        non_loyal_segments)]

    loyal_transactions = enriched_data[enriched_data['segment'].isin(
        loyal_segments)]
    non_loyal_transactions = enriched_data[enriched_data['segment'].isin(
        non_loyal_segments)]

    # Overview Metrics
    total_non_loyal_customers = len(non_loyal_data)
    percentage_of_total = (total_non_loyal_customers / len(rfm_df)) * 100 if len(rfm_df) > 0 else 0

    # Risk Analysis
    risk_breakdown = {
        'high_risk': len(non_loyal_data[non_loyal_data['recency'] > 180]), # Example: high risk if recency > 180 days
        'medium_risk': len(non_loyal_data[(non_loyal_data['recency'] <= 180) & (non_loyal_data['frequency'] < 2)]), # Example: medium risk if recency <= 180 and frequency < 2
        'low_risk': len(non_loyal_data[(non_loyal_data['recency'] <= 180) & (non_loyal_data['frequency'] >= 2)])
    }
    risk_breakdown['high_risk_percentage'] = (risk_breakdown['high_risk'] / total_non_loyal_customers) * 100 if total_non_loyal_customers > 0 else 0

    # Value Analysis
    value_breakdown = {
        'high_value': len(non_loyal_data[non_loyal_data['monetary'] > 1000]), # Example: high value if monetary > 1000
        'medium_value': len(non_loyal_data[(non_loyal_data['monetary'] <= 1000) & (non_loyal_data['monetary'] > 200)]), # Example: medium value if monetary <= 1000 and > 200
        'low_value': len(non_loyal_data[non_loyal_data['monetary'] <= 200])
    }
    value_breakdown['high_value_percentage'] = (value_breakdown['high_value'] / total_non_loyal_customers) * 100 if total_non_loyal_customers > 0 else 0

    # Engagement Analysis
    engagement_breakdown = {
        'high_engagement': len(non_loyal_data[non_loyal_data['frequency'] > 3]), # Example: high engagement if frequency > 3
        'medium_engagement': len(non_loyal_data[(non_loyal_data['frequency'] <= 3) & (non_loyal_data['frequency'] > 1)]), # Example: medium engagement if frequency <= 3 and > 1
        'low_engagement': len(non_loyal_data[non_loyal_data['frequency'] <= 1])
    }
    engagement_breakdown['low_engagement_percentage'] = (engagement_breakdown['low_engagement'] / total_non_loyal_customers) * 100 if total_non_loyal_customers > 0 else 0

    # Demographics and Patterns
    demographics = {
        'dominant_tier': non_loyal_transactions['customer_tier'].value_counts().idxmax() if len(non_loyal_transactions) > 0 else "N/A",
        'dominant_tier_percentage': (non_loyal_transactions['customer_tier'].value_counts().max() / len(non_loyal_transactions)) * 100 if len(non_loyal_transactions) > 0 else 0,
        'preferred_channel': non_loyal_transactions['channel'].value_counts().idxmax() if len(non_loyal_transactions) > 0 else "N/A",
        'preferred_category': non_loyal_transactions['product_category'].value_counts().idxmax() if len(non_loyal_transactions) > 0 else "N/A"
    }

    transaction_patterns = {
        'return_rate': (non_loyal_transactions['transaction_type'].value_counts().get('Return', 0) / len(non_loyal_transactions)) * 100 if len(non_loyal_transactions) > 0 else 0,
        'preferred_transaction_type': non_loyal_transactions['transaction_type'].value_counts().idxmax() if len(non_loyal_transactions) > 0 else "N/A",
        'discount_usage': (non_loyal_transactions['transaction_type'].value_counts().get('Discount', 0) / len(non_loyal_transactions)) * 100 if len(non_loyal_transactions) > 0 else 0
    }

    # Strategic Recommendations
    strategic_recommendations = []
    if total_non_loyal_customers > 0:
        if len(non_loyal_data[non_loyal_data['recency'] > 180]) > 0:
            strategic_recommendations.append("Re-engage high-risk non-loyal customers with exclusive events and personalized offers.")
        if len(non_loyal_data[(non_loyal_data['recency'] <= 180) & (non_loyal_data['frequency'] < 2)]) > 0:
            strategic_recommendations.append("Focus on medium-risk non-loyal customers through targeted re-engagement campaigns.")
        if len(non_loyal_data[(non_loyal_data['recency'] <= 180) & (non_loyal_data['frequency'] >= 2)]) > 0:
            strategic_recommendations.append("Leverage low-risk non-loyal customers for cross-selling opportunities.")
        if len(non_loyal_data[non_loyal_data['monetary'] > 1000]) > 0:
            strategic_recommendations.append("Target high-value non-loyal customers with premium offerings and exclusive access.")
        if len(non_loyal_data[(non_loyal_data['monetary'] <= 1000) & (non_loyal_data['monetary'] > 200)]) > 0:
            strategic_recommendations.append("Engage medium-value non-loyal customers with tailored promotions.")
        if len(non_loyal_data[non_loyal_data['monetary'] <= 200]) > 0:
            strategic_recommendations.append("Focus on low-value non-loyal customers with introductory offers and value-based messaging.")
        if len(non_loyal_data[non_loyal_data['frequency'] > 3]) > 0:
            strategic_recommendations.append("Leverage high-engagement non-loyal customers for cross-selling and retention.")
        if len(non_loyal_data[(non_loyal_data['frequency'] <= 3) & (non_loyal_data['frequency'] > 1)]) > 0:
            strategic_recommendations.append("Engage medium-engagement non-loyal customers with personalized content.")
        if len(non_loyal_data[non_loyal_data['frequency'] <= 1]) > 0:
            strategic_recommendations.append("Focus on low-engagement non-loyal customers with introductory offers and value-based messaging.")

    # Seasonal Insights
    # Create month column from purchase_date
    non_loyal_transactions_with_month = non_loyal_transactions.copy()
    non_loyal_transactions_with_month['month'] = non_loyal_transactions_with_month['purchase_date'].dt.month
    
    seasonal_insights = {
        'peak_month': non_loyal_transactions_with_month['month'].value_counts().idxmax() if len(non_loyal_transactions_with_month) > 0 else "N/A",
        'lowest_month': non_loyal_transactions_with_month['month'].value_counts().idxmin() if len(non_loyal_transactions_with_month) > 0 else "N/A"
    }

    return {
        'total_non_loyal_customers': total_non_loyal_customers,
        'percentage_of_total': percentage_of_total,
        'risk_breakdown': risk_breakdown,
        'value_breakdown': value_breakdown,
        'engagement_breakdown': engagement_breakdown,
        'demographics': demographics,
        'transaction_patterns': transaction_patterns,
        'strategic_recommendations': strategic_recommendations,
        'seasonal_insights': seasonal_insights
    }





def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header">👗 Luxury Fashion RFM Segmentation Dashboard</h1>',
                unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        try:
            df = load_data()
        except:
            # If the original file is busy, try the luxury version
            df = pd.read_csv('rfm_enriched_luxury.csv')
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    if df is None:
        st.error(
            "Failed to load data. Please ensure 'rfm_enriched.csv' is in the project directory.")
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Date range filter
    min_date = df['purchase_date'].min()
    max_date = df['purchase_date'].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    # Channel filter
    channels = ['All'] + list(df['channel'].unique())
    selected_channel = st.sidebar.selectbox("Channel", channels)

    # Customer tier filter
    tiers = ['All'] + list(df['customer_tier'].unique())
    selected_tier = st.sidebar.selectbox("Customer Tier", tiers)

    # Apply filters
    filtered_df = df.copy()

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['purchase_date'].dt.date >= start_date) &
            (filtered_df['purchase_date'].dt.date <= end_date)
        ]

    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]

    if selected_tier != 'All':
        filtered_df = filtered_df[filtered_df['customer_tier']
                                  == selected_tier]

    # Calculate RFM metrics
    with st.spinner("Calculating RFM metrics..."):
        rfm_df = calculate_rfm(filtered_df)

    # Display basic statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(rfm_df))

    with col2:
        st.metric("Total Transactions", len(filtered_df))

    with col3:
        st.metric("Avg Transaction Value",
                  f"£{filtered_df['purchase_amount'].mean():.2f}")

    with col4:
        st.metric(
            "Date Range", f"{filtered_df['purchase_date'].min().strftime('%Y-%m-%d')} to {filtered_df['purchase_date'].max().strftime('%Y-%m-%d')}")

    st.markdown("---")

    # Clustering section
    st.header("🎯 Customer Segmentation")

    # Find optimal number of clusters
    with st.spinner("Finding optimal number of clusters..."):
        rfm_scaled, scaler, rfm_normalized = normalize_rfm(rfm_df)
        optimal_k, inertias, silhouette_scores, K_range = find_optimal_clusters(
            rfm_scaled)

    # Display clustering metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Method")
        fig_elbow = px.line(x=list(K_range), y=inertias,
                            title="Elbow Method for Optimal Clusters")
        fig_elbow.add_vline(x=optimal_k, line_dash="dash",
                            line_color="red", annotation_text=f"Optimal: {optimal_k}")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        st.subheader("Silhouette Score")
        fig_silhouette = px.line(
            x=list(K_range), y=silhouette_scores, title="Silhouette Score")
        fig_silhouette.add_vline(x=optimal_k, line_dash="dash",
                                 line_color="red", annotation_text=f"Optimal: {optimal_k}")
        st.plotly_chart(fig_silhouette, use_container_width=True)

    st.info(f"📊 Optimal number of clusters: **{optimal_k}**")

    # Perform clustering
    with st.spinner("Performing clustering..."):
        kmeans_model, cluster_labels = perform_clustering(
            rfm_scaled, optimal_k)
        segment_names = assign_segment_names(cluster_labels, rfm_df)

    # Multi-Dimensional Customer Behavior Analysis
    st.header("🔍 Multi-Dimensional Customer Behavior Analysis")

    # Create comprehensive behavioral analysis
    with st.spinner("Analyzing customer behavior patterns..."):
        behavioral_insights = create_behavioral_analysis(
            filtered_df, rfm_df, cluster_labels, segment_names)

    # Display behavioral insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Channel Performance Analysis")
        st.plotly_chart(
            behavioral_insights['channel_analysis'], use_container_width=True)

        st.subheader("🏷️ Customer Tier Distribution")
        st.plotly_chart(
            behavioral_insights['tier_analysis'], use_container_width=True)

    with col2:
        st.subheader("🛍️ Product Category Preferences")
        st.plotly_chart(
            behavioral_insights['category_analysis'], use_container_width=True)

        st.subheader("💳 Transaction Type Patterns")
        st.plotly_chart(
            behavioral_insights['transaction_analysis'], use_container_width=True)

    # Advanced Cross-Segment Analysis
    st.subheader("🎯 Cross-Segment Behavioral Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            behavioral_insights['segment_channel_heatmap'], use_container_width=True)

    with col2:
        st.plotly_chart(
            behavioral_insights['segment_tier_heatmap'], use_container_width=True)

    # Key Behavioral Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Key Behavioral Insights")

    insights_text = f"""
    **Channel Strategy Insights:**
    • {behavioral_insights['channel_insights']['primary_channel']} is the dominant channel with {behavioral_insights['channel_insights']['primary_percentage']:.1f}% of transactions
    • {behavioral_insights['channel_insights']['high_value_channel']} generates the highest average transaction value (£{behavioral_insights['channel_insights']['high_value_amount']:.2f})
    
    **Customer Tier Analysis:**
    • {behavioral_insights['tier_insights']['dominant_tier']} customers represent {behavioral_insights['tier_insights']['dominant_percentage']:.1f}% of the customer base
    • {behavioral_insights['tier_insights']['highest_value_tier']} customers have the highest average spend (£{behavioral_insights['tier_insights']['highest_value_amount']:.2f})
    
    **Product Category Trends:**
    • {behavioral_insights['category_insights']['top_category']} is the most popular category ({behavioral_insights['category_insights']['top_percentage']:.1f}% of sales)
    • {behavioral_insights['category_insights']['highest_value_category']} generates the highest revenue per transaction (£{behavioral_insights['category_insights']['highest_value_amount']:.2f})
    """

    st.markdown(insights_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # Time-Series Analysis
    st.header("📈 Temporal Behavior Analysis")

    with st.spinner("Analyzing temporal patterns..."):
        temporal_insights = create_temporal_analysis(
            filtered_df, rfm_df, cluster_labels, segment_names)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🕒 Monthly Transaction Trends")
        st.plotly_chart(
            temporal_insights['monthly_trends'], use_container_width=True)

        st.subheader("📅 Seasonal Purchase Patterns")
        st.plotly_chart(
            temporal_insights['seasonal_patterns'], use_container_width=True)

    with col2:
        st.subheader("🎯 Segment Evolution Over Time")
        st.plotly_chart(
            temporal_insights['segment_evolution'], use_container_width=True)

        st.subheader("💹 Revenue Trends by Channel")
        st.plotly_chart(
            temporal_insights['channel_trends'], use_container_width=True)

    # Temporal Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("⏰ Temporal Insights")

    temporal_text = f"""
    **Seasonal Patterns:**
    • Peak shopping month: {temporal_insights['seasonal_insights']['peak_month']} with {temporal_insights['seasonal_insights']['peak_percentage']:.1f}% of annual transactions
    • Lowest activity month: {temporal_insights['seasonal_insights']['lowest_month']} with {temporal_insights['seasonal_insights']['lowest_percentage']:.1f}% of annual transactions
    
    **Growth Trends:**
    • Overall transaction growth: {temporal_insights['growth_insights']['transaction_growth']:.1f}% over the analysis period
    • Average transaction value growth: {temporal_insights['growth_insights']['value_growth']:.1f}% over the analysis period
    
    **Channel Performance:**
    • {temporal_insights['channel_insights']['fastest_growing_channel']} shows the fastest growth at {temporal_insights['channel_insights']['growth_rate']:.1f}% per month
    """

    st.markdown(temporal_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # Create visualizations
    with st.spinner("Creating visualizations..."):
        figures = create_rfm_visualizations(
            rfm_df, cluster_labels, segment_names)

    # Display visualizations
    st.header("📈 RFM Analysis Visualizations")

    # RFM Distribution
    st.subheader("RFM Metrics Distribution")
    st.plotly_chart(figures['rfm_distribution'], use_container_width=True)

    # Segment distribution and characteristics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segment Distribution")
        st.plotly_chart(figures['segment_distribution'],
                        use_container_width=True)

    with col2:
        st.subheader("Segment Characteristics")
        st.plotly_chart(figures['segment_heatmap'], use_container_width=True)

    # 3D Scatter plot
    st.subheader("3D RFM Scatter Plot")
    st.plotly_chart(figures['rfm_3d'], use_container_width=True)

    # Monetary vs Frequency
    st.subheader("Monetary vs Frequency Analysis")
    st.plotly_chart(figures['monetary_frequency'], use_container_width=True)

    # Business Insights
    st.header("💼 Business Insights & Recommendations")
    
    insights = create_business_insights(
        rfm_df, cluster_labels, segment_names, filtered_df)

    for segment, insight in insights.items():
        st.subheader(f"🎯 {segment}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Customer Count", insight['size'])
        
        with col2:
            st.metric("Avg Recency", f"{insight['avg_recency']:.0f} days")
        
        with col3:
            st.metric("Avg Frequency", f"{insight['avg_frequency']:.1f}")
        
        with col4:
            st.metric("Avg Monetary", f"£{insight['avg_monetary']:.0f}")

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Revenue", f"£{insight['total_revenue']:,.0f}")
        
        with col2:
            st.metric("Revenue %", f"{insight['revenue_percentage']:.1f}%")
        
        with col3:
            st.metric("Avg Transaction", f"£{insight['avg_transaction']:.0f}")
        
        with col4:
            st.metric("Dominant Tier", insight['dominant_tier'])

        st.subheader("🎯 Recommendations")
        for rec in insight['recommendations']:
            st.write(f"• {rec}")

    # LOYAL vs NON-LOYAL CUSTOMER SEGMENTATION DASHBOARD
    st.header("🔬 Loyal vs Non-Loyal Customer Segmentation Dashboard")
    st.markdown("""
    <div class="insight-box">
    <h4>🧪 Segmentation Overview</h4>
    <p>This section provides comprehensive analysis and insights for both loyal and non-loyal customer segments to understand behavioral differences and strategic opportunities.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create segmentation data
    rfm_with_segments = rfm_df.copy()
    rfm_with_segments['cluster'] = cluster_labels
    rfm_with_segments['segment'] = segment_names
    
    # Merge with original data
    customer_segments = rfm_with_segments[['customer_id', 'segment']]
    enriched_data = filtered_df.merge(
        customer_segments, on='customer_id', how='left')
    
    # Define loyal vs non-loyal segments
    loyal_segments = [seg for seg in rfm_with_segments['segment'].unique()
                      if "Loyal" in seg or "VIP" in seg or "Champion" in seg]
    non_loyal_segments = [seg for seg in rfm_with_segments['segment'].unique()
                          if seg not in loyal_segments]

    loyal_data = rfm_with_segments[rfm_with_segments['segment'].isin(
        loyal_segments)]
    non_loyal_data = rfm_with_segments[rfm_with_segments['segment'].isin(
        non_loyal_segments)]

    loyal_transactions = enriched_data[enriched_data['segment'].isin(
        loyal_segments)]
    non_loyal_transactions = enriched_data[enriched_data['segment'].isin(
        non_loyal_segments)]

    # Overview Comparison
    st.subheader("📊 Customer Segment Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👑 Loyal Customers")
        if len(loyal_data) > 0:
            st.metric("Total Customers", len(loyal_data))
            st.metric("Percentage",
                      f"{(len(loyal_data) / len(rfm_df)) * 100:.1f}%")
            st.metric("Avg Recency",
                      f"{loyal_data['recency'].mean():.0f} days")
            st.metric("Avg Frequency", f"{loyal_data['frequency'].mean():.1f}")
            st.metric("Avg Monetary", f"£{loyal_data['monetary'].mean():.0f}")
            st.metric("Total Revenue",
                      f"£{loyal_transactions['purchase_amount'].sum():,.0f}")
        else:
            st.info("No loyal customers found")
    
    with col2:
        st.markdown("### 🎯 Non-Loyal Customers")
        if len(non_loyal_data) > 0:
            st.metric("Total Customers", len(non_loyal_data))
            st.metric("Percentage",
                      f"{(len(non_loyal_data) / len(rfm_df)) * 100:.1f}%")
            st.metric("Avg Recency",
                      f"{non_loyal_data['recency'].mean():.0f} days")
            st.metric("Avg Frequency",
                      f"{non_loyal_data['frequency'].mean():.1f}")
            st.metric("Avg Monetary",
                      f"£{non_loyal_data['monetary'].mean():.0f}")
            st.metric(
                "Total Revenue", f"£{non_loyal_transactions['purchase_amount'].sum():,.0f}")
        else:
            st.info("No non-loyal customers found")

    # Behavioral Patterns Comparison
    if len(loyal_data) > 0 and len(non_loyal_data) > 0:
        st.subheader("🎭 Behavioral Patterns Comparison")
        
        # Channel Preference
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📺 Channel Preference - Loyal")
            if len(loyal_transactions) > 0:
                loyal_channel = loyal_transactions['channel'].value_counts()
                fig = px.pie(values=loyal_channel.values, names=loyal_channel.index,
                              title="Loyal Customer Channel Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📺 Channel Preference - Non-Loyal")
            if len(non_loyal_transactions) > 0:
                non_loyal_channel = non_loyal_transactions['channel'].value_counts(
                )
                fig = px.pie(values=non_loyal_channel.values, names=non_loyal_channel.index,
                              title="Non-Loyal Customer Channel Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Product Category Preference
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🛍️ Product Category - Loyal")
            if len(loyal_transactions) > 0:
                loyal_category = loyal_transactions['product_category'].value_counts(
                )
                fig = px.bar(x=loyal_category.index, y=loyal_category.values,
                              title="Loyal Customer Category Preferences")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🛍️ Product Category - Non-Loyal")
            if len(non_loyal_transactions) > 0:
                non_loyal_category = non_loyal_transactions['product_category'].value_counts(
                )
                fig = px.bar(x=non_loyal_category.index, y=non_loyal_category.values,
                              title="Non-Loyal Customer Category Preferences")
                st.plotly_chart(fig, use_container_width=True)
        
        # Transaction Type Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💳 Transaction Type - Loyal")
            if len(loyal_transactions) > 0:
                loyal_transaction = loyal_transactions['transaction_type'].value_counts(
                )
                fig = px.pie(values=loyal_transaction.values, names=loyal_transaction.index,
                              title="Loyal Customer Transaction Types")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💳 Transaction Type - Non-Loyal")
            if len(non_loyal_transactions) > 0:
                non_loyal_transaction = non_loyal_transactions['transaction_type'].value_counts(
                )
                fig = px.pie(values=non_loyal_transaction.values, names=non_loyal_transaction.index,
                              title="Non-Loyal Customer Transaction Types")
                st.plotly_chart(fig, use_container_width=True)

    # Strategic Insights Comparison
    st.subheader("🎯 Strategic Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👑 Loyal Customer Strategy")
        if len(loyal_data) > 0:
            st.success(f"""
            **Retention Focus:**
            • {len(loyal_data)} customers to retain
            • Average spend: £{loyal_data['monetary'].mean():.0f}
            • Purchase frequency: {loyal_data['frequency'].mean():.1f} times
            
            **Key Actions:**
            • VIP programs and exclusive access
            • Early access to new collections
            • Personalized luxury experiences
            • Referral programs
            """)
        else:
            st.info("No loyal customers to analyze")
    
    with col2:
        st.markdown("### 🎯 Non-Loyal Customer Strategy")
        if len(non_loyal_data) > 0:
            st.warning(f"""
            **Acquisition & Conversion Focus:**
            • {len(non_loyal_data)} customers to convert
            • Average spend: £{non_loyal_data['monetary'].mean():.0f}
            • Purchase frequency: {non_loyal_data['frequency'].mean():.1f} times
            
            **Key Actions:**
            • Re-engagement campaigns
            • Introductory offers
            • Cross-selling opportunities
            • Customer education programs
            """)
        else:
            st.info("No non-loyal customers to analyze")

    # Non-Loyal Customer Deep Insights
    st.header("🎯 Non-Loyal Customer Deep Insights")

    with st.spinner("Analyzing non-loyal customer patterns..."):
        non_loyal_insights = create_non_loyal_insights(
            rfm_df, cluster_labels, segment_names, filtered_df)

    if non_loyal_insights:
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Non-Loyal Customers",
                      non_loyal_insights['total_non_loyal_customers'])

        with col2:
            st.metric("Percentage of Total",
                      f"{non_loyal_insights['percentage_of_total']:.1f}%")

        with col3:
            st.metric("High-Risk Customers",
                      non_loyal_insights['risk_breakdown']['high_risk'])

        with col4:
            st.metric("High-Value Non-Loyal",
                      non_loyal_insights['value_breakdown']['high_value'])

        # Risk Analysis
        st.subheader("⚠️ Risk Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("High Risk",
                      f"{non_loyal_insights['risk_breakdown']['high_risk']} customers",
                      f"{non_loyal_insights['risk_breakdown']['high_risk_percentage']:.1f}%")

        with col2:
            st.metric("Medium Risk",
                      f"{non_loyal_insights['risk_breakdown']['medium_risk']} customers")

        with col3:
            st.metric("Low Risk",
                      f"{non_loyal_insights['risk_breakdown']['low_risk']} customers")

        # Value Analysis
        st.subheader("💰 Value Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("High Value",
                      f"{non_loyal_insights['value_breakdown']['high_value']} customers",
                      f"{non_loyal_insights['value_breakdown']['high_value_percentage']:.1f}%")

        with col2:
            st.metric("Medium Value",
                      f"{non_loyal_insights['value_breakdown']['medium_value']} customers")

        with col3:
            st.metric("Low Value",
                      f"{non_loyal_insights['value_breakdown']['low_value']} customers")

        # Engagement Analysis
        st.subheader("📊 Engagement Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("High Engagement",
                      f"{non_loyal_insights['engagement_breakdown']['high_engagement']} customers")

        with col2:
            st.metric("Medium Engagement",
                      f"{non_loyal_insights['engagement_breakdown']['medium_engagement']} customers")

        with col3:
            st.metric("Low Engagement",
                      f"{non_loyal_insights['engagement_breakdown']['low_engagement']} customers",
                      f"{non_loyal_insights['engagement_breakdown']['low_engagement_percentage']:.1f}%")

        # Demographics and Patterns
        st.subheader("👥 Demographics & Patterns")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Dominant Tier",
                      non_loyal_insights['demographics']['dominant_tier'],
                      f"{non_loyal_insights['demographics']['dominant_tier_percentage']:.1f}%")

        with col2:
            st.metric("Preferred Channel",
                      non_loyal_insights['demographics']['preferred_channel'])

        with col3:
            st.metric("Preferred Category",
                      non_loyal_insights['demographics']['preferred_category'])

        with col4:
            st.metric("Return Rate",
                      f"{non_loyal_insights['transaction_patterns']['return_rate']:.1f}%")

        # Strategic Recommendations
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🎯 Strategic Recommendations for Non-Loyal Customers")

        for i, rec in enumerate(non_loyal_insights['strategic_recommendations'], 1):
            st.write(f"**{i}.** {rec}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Additional Insights
        st.subheader("📈 Additional Insights")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Seasonal Patterns:**
            • Peak shopping month: {non_loyal_insights['seasonal_insights']['peak_month']}
            • Lowest activity month: {non_loyal_insights['seasonal_insights']['lowest_month']}
            
            **Transaction Patterns:**
            • Preferred transaction type: {non_loyal_insights['transaction_patterns']['preferred_transaction_type']}
            • Discount usage: {non_loyal_insights['transaction_patterns']['discount_usage']:.1f}%
            """)

        with col2:
            st.success(f"""
            **Opportunity Areas:**
            • {non_loyal_insights['value_breakdown']['high_value']} high-value customers to convert
            • {non_loyal_insights['risk_breakdown']['high_risk']} high-risk customers to re-engage
            • {non_loyal_insights['engagement_breakdown']['low_engagement']} low-engagement customers to activate
            """)
    else:
        st.info("No non-loyal customers found in the current dataset.")

    # Download section
    st.header("📥 Download Results")

    # Prepare data for download
    rfm_results = rfm_df.copy()
    rfm_results['cluster'] = cluster_labels
    rfm_results['segment'] = segment_names

    # CSV download
    csv_data = rfm_results.to_csv(index=False)
    st.download_button(
        label="📊 Download RFM Results (CSV)",
        data=csv_data,
        file_name="rfm_segmentation_results.csv",
        mime="text/csv"
    )

    # Summary statistics
    st.header("📊 Summary Statistics")

    summary_stats = rfm_results.groupby('segment').agg({
        'recency': ['mean', 'std'],
        'frequency': ['mean', 'std'],
        'monetary': ['mean', 'std']
    }).round(2)

    st.dataframe(summary_stats)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Luxury Fashion RFM Segmentation Dashboard | Built with Streamlit</p>
            <p>Empowering luxury retail with data-driven customer insights and personalized marketing strategies</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
