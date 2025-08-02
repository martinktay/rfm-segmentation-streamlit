# RFM Segmentation Dashboard - Deployment Guide

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/martinktay/rfm-segmentation-streamlit.git
cd rfm-segmentation-streamlit
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Dashboard
```bash
streamlit run rfm_dashboard.py
```

### Step 4: Access the Dashboard
Open your web browser and navigate to:
- **Local URL:** http://localhost:8501
- **Network URL:** http://192.168.0.21:8501 (for access from other devices on your network)

## 🌐 Dashboard Features

### 📊 Main Dashboard Sections

1. **Overview Metrics**
   - Total Customers
   - Total Transactions
   - Average Transaction Value
   - Date Range

2. **Customer Segmentation**
   - Optimal cluster selection using Elbow Method and Silhouette Score
   - Interactive clustering analysis

3. **RFM Analysis Visualizations**
   - RFM Metrics Distribution (Histograms)
   - Customer Segment Distribution (Pie Chart)
   - 3D RFM Scatter Plot
   - Segment Characteristics Heatmap
   - Monetary vs Frequency Analysis

4. **Business Insights**
   - Detailed segment analysis
   - Customer behavior patterns
   - Marketing recommendations
   - Channel and product preferences

5. **Data Export**
   - Download RFM results as CSV
   - Summary statistics table

### 🔍 Interactive Filters

Use the sidebar to filter data by:
- **Date Range:** Select specific time periods
- **Channel:** Online or In-store transactions
- **Customer Tier:** Bronze, Silver, Gold, or Platinum

## 📈 Understanding the Results

### RFM Metrics
- **Recency:** Days since last purchase (lower is better)
- **Frequency:** Number of purchases (higher is better)
- **Monetary:** Total amount spent (higher is better)

### Customer Segments
The dashboard automatically identifies customer segments such as:
- **Champions:** High-value, recent, frequent buyers
- **Loyal Customers:** High-frequency, high-monetary customers
- **At Risk:** Previously good customers who haven't purchased recently
- **New Customers:** Recent but low-frequency buyers
- **Dormant:** Inactive customers

## 🛠️ Troubleshooting

### Common Issues

1. **Port 8501 already in use**
   ```bash
   # Kill existing process
   lsof -ti:8501 | xargs kill -9
   # Or use a different port
   streamlit run rfm_dashboard.py --server.port 8502
   ```

2. **Package installation errors**
   ```bash
   # Upgrade pip
   pip install --upgrade pip
   # Install packages individually
   pip install streamlit pandas numpy scikit-learn plotly
   ```

3. **Data loading issues**
   - Ensure `rfm_enriched.csv` is in the project root directory
   - Check file permissions
   - Verify CSV format is correct

### Performance Tips

1. **For large datasets:**
   - The dashboard uses caching for better performance
   - Consider reducing the date range for faster loading

2. **For better visualization:**
   - Use a modern browser (Chrome, Firefox, Safari)
   - Ensure sufficient screen resolution for 3D plots

## 🔧 Customization

### Modifying the Dashboard

1. **Add new visualizations:**
   - Edit `rfm_dashboard.py`
   - Add new Plotly figures in the `create_rfm_visualizations` function

2. **Change clustering parameters:**
   - Modify `max_clusters` in `find_optimal_clusters` function
   - Adjust segment naming logic in `assign_segment_names`

3. **Add new filters:**
   - Add filter widgets in the sidebar section
   - Update the filtering logic in the main function

### Data Format Requirements

The dashboard expects a CSV file with these columns:
- `customer_id`: Unique customer identifier
- `purchase_amount`: Transaction amount
- `purchase_date`: Date in YYYY-MM-DD format
- `channel`: Purchase channel
- `product_category`: Product category
- `transaction_type`: Transaction type
- `customer_tier`: Customer loyalty tier

## 📱 Mobile Access

The dashboard is responsive and works on mobile devices:
- Access via network URL from your phone
- Use touch gestures for 3D plot interaction
- Sidebar collapses automatically on small screens

## 🔒 Security Notes

- The dashboard runs locally by default
- For production deployment, consider:
  - Adding authentication
  - Using HTTPS
  - Implementing rate limiting
  - Setting up proper logging

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are correctly installed
4. Verify the data file format

## 🎯 Business Use Cases

This dashboard is designed for:
- **Marketing Teams:** Customer segmentation and targeting
- **Business Analysts:** Customer behavior analysis
- **Product Managers:** Understanding customer preferences
- **Sales Teams:** Identifying high-value prospects
- **Customer Success:** Retention and engagement strategies 