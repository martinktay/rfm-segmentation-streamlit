# RFM Segmentation Dashboard

An interactive customer segmentation dashboard built with Python and Streamlit that analyzes customer behavior using RFM (Recency, Frequency, Monetary) analysis and K-means clustering.

## Project Overview

This dashboard helps marketers and business analysts understand customer segments by:

- **Recency**: How recently a customer made a purchase
- **Frequency**: How often a customer makes purchases  
- **Monetary**: How much money a customer spends

The application uses K-means clustering to group customers into behavioral segments, providing actionable insights for targeted marketing campaigns.

## Features

- 📊 **RFM Analysis**: Calculate and visualize Recency, Frequency, and Monetary scores
- 🎯 **Customer Segmentation**: K-means clustering with optimal cluster selection
- 📈 **Interactive Visualizations**: Bar charts, heatmaps, and scatter plots
- 🔍 **Filtering & Analysis**: Filter by customer tier, channel, and product category
- 💡 **Business Insights**: Summary statistics and recommendations for each segment

## Sample Cluster Insights

Based on RFM analysis, customers are typically segmented into:

- **Segment 1 - Champions**: High recency, frequency, and monetary value
- **Segment 2 - Loyal Customers**: High frequency and monetary, moderate recency
- **Segment 3 - At Risk**: Low recency, moderate frequency and monetary
- **Segment 4 - New Customers**: High recency, low frequency and monetary
- **Segment 5 - Dormant**: Low recency, frequency, and monetary

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/martinktay/rfm-segmentation-streamlit.git
cd rfm-segmentation-streamlit
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. The dataset `rfm_enriched.csv` is already included in the project root directory.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run rfm_dashboard.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

The dashboard will automatically:
- Load the RFM dataset
- Calculate Recency, Frequency, and Monetary metrics
- Perform K-means clustering with optimal cluster selection
- Generate interactive visualizations
- Provide business insights and recommendations

### 🌐 Dashboard Access
- **Local URL:** http://localhost:8501
- **Network URL:** http://192.168.0.21:8501 (for access from other devices)

### 📱 Mobile Access
The dashboard is responsive and works on mobile devices. Access via network URL from your phone.

### 🛠️ Troubleshooting
- **Port 8501 already in use:** Use `streamlit run rfm_dashboard.py --server.port 8502`
- **Package installation errors:** Upgrade pip with `pip install --upgrade pip`
- **Data loading issues:** Ensure `rfm_enriched.csv` is in the project root directory

## Usage

Run the Streamlit dashboard:
```bash
streamlit run rfm_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Dataset Structure

The `rfm_enriched.csv` file contains **5,000 transactions** from **993 unique customers** with the following fields:
- `customer_id`: Unique customer identifier (1-1000)
- `purchase_amount`: Transaction amount (including returns as negative values)
- `purchase_date`: Date of purchase (spanning 2 years)
- `channel`: Purchase channel (Online/In-store)
- `product_category`: Product category (Electronics, Clothing, Home & Garden, Books, Sports, Beauty)
- `transaction_type`: Type of transaction (Full Price, Discount, Return)
- `customer_tier`: Customer loyalty tier (Bronze, Silver, Gold, Platinum)

**Dataset Statistics:**
- **Total Transactions:** 5,000
- **Unique Customers:** 993
- **Date Range:** 2 years of transaction history
- **Average Transaction Value:** $496.70
- **Average Customer Frequency:** 5.0 purchases
- **Average Customer Recency:** 72.5 days

## Business Value

This dashboard enables data-driven marketing decisions by:

- Identifying high-value customers for retention campaigns
- Targeting at-risk customers for re-engagement
- Understanding customer behavior patterns across channels
- Optimizing marketing spend based on segment characteristics
- Developing personalized marketing strategies

## Screenshots

The dashboard is currently running and accessible at:
- **Local:** http://localhost:8501
- **Network:** http://192.168.0.21:8501

*[Dashboard screenshots will be added here]*

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning (K-means clustering)
- **Plotly**: Interactive visualizations
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization

## Project Structure

```
rfm-segmentation-streamlit/
├── README.md
├── requirements.txt
├── rfm_dashboard.py
├── rfm_enriched.csv
└── .gitignore
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License. 