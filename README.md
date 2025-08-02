# 👗 Luxury Fashion RFM Segmentation Dashboard

An advanced customer segmentation dashboard designed specifically for luxury fashion retail, built with Python and Streamlit. This sophisticated analytics platform leverages RFM (Recency, Frequency, Monetary) analysis and machine learning clustering to deliver actionable insights for premium fashion brands seeking to optimize customer lifetime value and drive personalized marketing strategies.

## Project Overview

This luxury fashion retail analytics platform empowers premium brands to understand their high-value customer segments through sophisticated behavioral analysis:

- **Recency**: Days since last luxury purchase (critical for seasonal fashion cycles)
- **Frequency**: Purchase frequency patterns (indicates brand loyalty and engagement)
- **Monetary**: Total spend in GBP (reflects customer value and premium positioning)

The application employs advanced K-means clustering algorithms to identify distinct customer personas, enabling data-driven decisions for VIP services, exclusive collections, and personalized luxury experiences.

## Features

- 📊 **Advanced RFM Analysis**: Sophisticated calculation and visualization of luxury customer behavior patterns
- 🎯 **Premium Customer Segmentation**: Machine learning-powered clustering with optimal segment identification
- 📈 **Interactive Luxury Analytics**: High-quality visualizations including 3D scatter plots and heatmaps
- 🔍 **Granular Filtering**: Advanced filtering by customer tier, sales channel, and product categories
- 💡 **Strategic Business Insights**: Actionable recommendations for VIP services and premium marketing
- 💰 **GBP Currency Support**: Native British pound integration for UK luxury retail
- 📱 **Responsive Design**: Mobile-optimized interface for on-the-go luxury retail management

## Premium Customer Segments

Our advanced clustering algorithm identifies sophisticated luxury customer personas:

- **VIP Fashionistas**: Elite customers with recent, frequent, high-value purchases (£600+)
- **Loyal Luxury Buyers**: Consistent high-spending customers with strong brand affinity
- **At-Risk Premium Customers**: Previously valuable customers requiring re-engagement
- **New Luxury Prospects**: Recent customers with potential for premium product introductions
- **High-Value Fashion Enthusiasts**: High-spending customers (£500+) seeking exclusive experiences
- **Frequent Luxury Shoppers**: Regular buyers requiring VIP membership programs
- **Recent Fashion Buyers**: New customers with seasonal collection opportunities

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

## Luxury Fashion Dataset

The `rfm_enriched.csv` file contains **5,000 premium fashion transactions** from **993 luxury customers** with the following fields:

- `customer_id`: Unique customer identifier (1-1000)
- `purchase_amount`: Transaction amount in GBP (including returns as negative values)
- `purchase_date`: Date of purchase (spanning 2 years of seasonal collections)
- `channel`: Purchase channel (Online/In-store)
- `product_category`: Luxury product categories (Electronics, Clothing, Home & Garden, Books, Sports, Beauty)
- `transaction_type`: Type of transaction (Full Price, Discount, Return)
- `customer_tier`: Customer loyalty tier (Bronze, Silver, Gold, Platinum)

**Luxury Retail Statistics:**
- **Total Transactions:** 5,000 premium purchases
- **Unique Customers:** 993 luxury shoppers
- **Date Range:** 2 years of seasonal fashion cycles
- **Average Transaction Value:** £496.70 (premium positioning)
- **Average Customer Frequency:** 5.0 purchases (strong brand loyalty)
- **Average Customer Recency:** 72.5 days (seasonal engagement patterns)

## Strategic Business Value

This luxury fashion analytics platform delivers exceptional business value for premium retail brands:

- **VIP Customer Retention**: Identify and nurture high-value customers with exclusive services and early access to collections
- **Premium Re-engagement**: Target at-risk luxury customers with personalized re-engagement campaigns and exclusive events
- **Channel Optimization**: Understand customer behavior patterns across online and in-store channels for omnichannel excellence
- **Marketing ROI Optimization**: Allocate marketing spend based on segment characteristics and customer lifetime value
- **Personalized Luxury Experiences**: Develop sophisticated, data-driven marketing strategies for premium customer segments
- **Seasonal Collection Planning**: Leverage recency patterns to optimize seasonal fashion launches and inventory management

## Screenshots

The luxury fashion dashboard is currently running and accessible at:

- **Local:** http://localhost:8501
- **Network:** http://192.168.0.21:8501

_[Premium dashboard screenshots will be added here]_

**Key Visualizations to Capture:**
- Main dashboard header with luxury branding and GBP metrics
- Customer segmentation analysis with optimal cluster selection
- RFM distribution charts and premium segment pie chart
- 3D scatter plot showing luxury customer segments
- Business insights for VIP Fashionistas segment
- Download functionality and summary statistics

## Advanced Technologies

- **Python 3.8+**: Core programming language with advanced data science capabilities
- **Streamlit**: Modern web application framework for rapid deployment and interactive analytics
- **Pandas**: Sophisticated data manipulation and analysis for large-scale retail datasets
- **NumPy**: High-performance numerical computing for complex mathematical operations
- **Scikit-learn**: Advanced machine learning algorithms including K-means clustering with optimal parameter selection
- **Plotly**: Interactive, publication-quality visualizations with 3D plotting capabilities
- **Matplotlib & Seaborn**: Professional statistical data visualization and plotting libraries
- **GBP Currency Integration**: Native British pound support for UK luxury retail operations

## Project Structure

```
rfm-segmentation-streamlit/
├── README.md
├── requirements.txt
├── rfm_dashboard.py
├── rfm_enriched.csv
└── .gitignore
```

## Project Impact & Results

This luxury fashion RFM segmentation dashboard represents a sophisticated approach to customer analytics in the premium retail sector. By leveraging advanced machine learning algorithms and interactive visualizations, the platform delivers:

- **Enhanced Customer Understanding**: Deep insights into luxury customer behavior patterns and preferences
- **Improved Marketing ROI**: Data-driven allocation of marketing resources for maximum impact
- **Personalized Customer Experiences**: Tailored services and communications for different premium segments
- **Operational Efficiency**: Streamlined customer management processes through automated segmentation
- **Competitive Advantage**: Advanced analytics capabilities that differentiate premium brands in the luxury market

## Future Enhancements

Potential areas for expansion include:
- **Predictive Analytics**: Customer lifetime value forecasting and churn prediction
- **Real-time Integration**: Live data feeds from POS systems and e-commerce platforms
- **AI-Powered Recommendations**: Personalized product recommendations based on segment characteristics
- **Multi-brand Support**: Scalable architecture for managing multiple luxury brand portfolios
- **Advanced Visualizations**: Augmented reality dashboards and immersive analytics experiences

## Contributing

We welcome contributions from data scientists, retail analysts, and luxury industry professionals. Please submit issues and enhancement requests to help improve this platform for the luxury retail community.

## License

This project is licensed under the MIT License, enabling widespread adoption across the luxury retail industry.
