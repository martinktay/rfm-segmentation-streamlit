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
    page_title="RFM Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .segment-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
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
            silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
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
    cluster_centers = cluster_df.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    
    # Sort clusters by monetary value (descending)
    cluster_centers_sorted = cluster_centers.sort_values('monetary', ascending=False)
    
    # Define segment names based on RFM characteristics
    segment_names = []
    for i, (cluster_id, center) in enumerate(cluster_centers_sorted.iterrows()):
        recency, frequency, monetary = center['recency'], center['frequency'], center['monetary']
        
        if recency < 30 and frequency > 5 and monetary > 1000:
            name = "Champions"
        elif recency < 60 and frequency > 3 and monetary > 500:
            name = "Loyal Customers"
        elif recency > 90 and frequency > 2 and monetary > 300:
            name = "At Risk"
        elif recency < 30 and frequency < 3 and monetary < 200:
            name = "New Customers"
        elif recency > 90 and frequency < 2 and monetary < 100:
            name = "Dormant"
        else:
            name = f"Segment {i+1}"
        
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
    
    # 1. RFM Distribution
    fig_rfm_dist = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Recency Distribution', 'Frequency Distribution', 'Monetary Distribution'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
    )
    
    fig_rfm_dist.add_trace(go.Histogram(x=rfm_viz['recency'], name='Recency', nbinsx=30), row=1, col=1)
    fig_rfm_dist.add_trace(go.Histogram(x=rfm_viz['frequency'], name='Frequency', nbinsx=30), row=1, col=2)
    fig_rfm_dist.add_trace(go.Histogram(x=rfm_viz['monetary'], name='Monetary', nbinsx=30), row=1, col=3)
    
    fig_rfm_dist.update_layout(height=400, showlegend=False, title_text="RFM Metrics Distribution")
    
    # 2. Segment Distribution
    segment_counts = rfm_viz['segment'].value_counts()
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segment Distribution"
    )
    fig_segments.update_layout(height=400)
    
    # 3. RFM Scatter Plot (3D)
    fig_3d = px.scatter_3d(
        rfm_viz,
        x='recency',
        y='frequency',
        z='monetary',
        color='segment',
        title="3D RFM Scatter Plot by Segment",
        labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
    )
    fig_3d.update_layout(height=500)
    
    # 4. Segment Characteristics Heatmap
    segment_means = rfm_viz.groupby('segment')[['recency', 'frequency', 'monetary']].mean()
    
    fig_heatmap = px.imshow(
        segment_means.T,
        text_auto=True,
        aspect="auto",
        title="Segment Characteristics Heatmap",
        labels=dict(x="Segment", y="RFM Metric", color="Value")
    )
    fig_heatmap.update_layout(height=400)
    
    # 5. Monetary vs Frequency by Segment
    fig_scatter = px.scatter(
        rfm_viz,
        x='frequency',
        y='monetary',
        color='segment',
        size='recency',
        hover_data=['customer_id'],
        title="Monetary vs Frequency by Segment (Size = Recency)",
        labels={'frequency': 'Frequency', 'monetary': 'Monetary ($)', 'recency': 'Recency (days)'}
    )
    fig_scatter.update_layout(height=500)
    
    return {
        'rfm_distribution': fig_rfm_dist,
        'segment_distribution': fig_segments,
        'rfm_3d': fig_3d,
        'segment_heatmap': fig_heatmap,
        'monetary_frequency': fig_scatter
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
    enriched_data = original_df.merge(customer_segments, on='customer_id', how='left')
    
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
        channel_dist = segment_transactions['channel'].value_counts(normalize=True)
        preferred_channel = channel_dist.index[0] if len(channel_dist) > 0 else "N/A"
        
        # Product category preference
        category_dist = segment_transactions['product_category'].value_counts(normalize=True)
        preferred_category = category_dist.index[0] if len(category_dist) > 0 else "N/A"
        
        # Customer tier distribution
        tier_dist = segment_transactions['customer_tier'].value_counts(normalize=True)
        dominant_tier = tier_dist.index[0] if len(tier_dist) > 0 else "N/A"
        
        # Generate recommendations
        recommendations = []
        if avg_recency < 30:
            recommendations.append("Recent customers - focus on retention")
        elif avg_recency > 90:
            recommendations.append("At-risk customers - re-engagement campaigns needed")
        
        if avg_frequency > 5:
            recommendations.append("High-frequency buyers - loyalty programs")
        elif avg_frequency < 2:
            recommendations.append("Low-frequency buyers - increase engagement")
        
        if avg_monetary > 500:
            recommendations.append("High-value segment - premium services")
        elif avg_monetary < 100:
            recommendations.append("Low-value segment - upselling opportunities")
        
        insights[segment] = {
            'size': segment_size,
            'avg_recency': avg_recency,
            'avg_frequency': avg_frequency,
            'avg_monetary': avg_monetary,
            'preferred_channel': preferred_channel,
            'preferred_category': preferred_category,
            'dominant_tier': dominant_tier,
            'recommendations': recommendations
        }
    
    return insights

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 RFM Segmentation Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure 'rfm_enriched.csv' is in the project directory.")
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
        filtered_df = filtered_df[filtered_df['customer_tier'] == selected_tier]
    
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
        st.metric("Avg Transaction Value", f"${filtered_df['purchase_amount'].mean():.2f}")
    
    with col4:
        st.metric("Date Range", f"{filtered_df['purchase_date'].min().strftime('%Y-%m-%d')} to {filtered_df['purchase_date'].max().strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # Clustering section
    st.header("🎯 Customer Segmentation")
    
    # Find optimal number of clusters
    with st.spinner("Finding optimal number of clusters..."):
        rfm_scaled, scaler, rfm_normalized = normalize_rfm(rfm_df)
        optimal_k, inertias, silhouette_scores, K_range = find_optimal_clusters(rfm_scaled)
    
    # Display clustering metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Elbow Method")
        fig_elbow = px.line(x=list(K_range), y=inertias, title="Elbow Method for Optimal Clusters")
        fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal: {optimal_k}")
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        st.subheader("Silhouette Score")
        fig_silhouette = px.line(x=list(K_range), y=silhouette_scores, title="Silhouette Score")
        fig_silhouette.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal: {optimal_k}")
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    st.info(f"📊 Optimal number of clusters: **{optimal_k}**")
    
    # Perform clustering
    with st.spinner("Performing clustering..."):
        kmeans_model, cluster_labels = perform_clustering(rfm_scaled, optimal_k)
        segment_names = assign_segment_names(cluster_labels, rfm_df)
    
    # Create visualizations
    with st.spinner("Creating visualizations..."):
        figures = create_rfm_visualizations(rfm_df, cluster_labels, segment_names)
    
    # Display visualizations
    st.header("📈 RFM Analysis Visualizations")
    
    # RFM Distribution
    st.subheader("RFM Metrics Distribution")
    st.plotly_chart(figures['rfm_distribution'], use_container_width=True)
    
    # Segment distribution and characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segment Distribution")
        st.plotly_chart(figures['segment_distribution'], use_container_width=True)
    
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
    st.header("💡 Business Insights")
    
    with st.spinner("Generating business insights..."):
        insights = create_business_insights(rfm_df, cluster_labels, segment_names, filtered_df)
    
    # Display insights for each segment
    for segment, insight in insights.items():
        with st.expander(f"📋 {segment} - {insight['size']} customers"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Recency", f"{insight['avg_recency']:.1f} days")
                st.metric("Avg Frequency", f"{insight['avg_frequency']:.1f}")
            
            with col2:
                st.metric("Avg Monetary", f"${insight['avg_monetary']:.2f}")
                st.metric("Preferred Channel", insight['preferred_channel'])
            
            with col3:
                st.metric("Preferred Category", insight['preferred_category'])
                st.metric("Dominant Tier", insight['dominant_tier'])
            
            st.subheader("🎯 Recommendations")
            for rec in insight['recommendations']:
                st.write(f"• {rec}")
    
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
            <p>RFM Segmentation Dashboard | Built with Streamlit</p>
            <p>Helping marketers make data-driven decisions through customer segmentation</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 