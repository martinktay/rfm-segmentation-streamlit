# 📊 RFM Segmentation Dashboard

An advanced customer segmentation dashboard built with Python and Streamlit. This sophisticated analytics platform leverages RFM (Recency, Frequency, Monetary) analysis and machine learning clustering to deliver actionable insights for businesses seeking to optimise customer lifetime value and drive personalised marketing strategies.

## Project Overview

This retail analytics platform empowers businesses to understand their high-value customer segments through sophisticated behavioural analysis:

- **Recency**: Days since last purchase (critical for customer engagement cycles)
- **Frequency**: Purchase frequency patterns (indicates brand loyalty and engagement)
- **Monetary**: Total spend in GBP (reflects customer value and positioning)

The application employs advanced K-means clustering algorithms to identify distinct customer personas, enabling data-driven decisions for VIP services, exclusive offerings, and personalised customer experiences.

## Features

- 📊 **Advanced RFM Analysis**: Sophisticated calculation and visualisation of customer behaviour patterns
- 🎯 **Premium Customer Segmentation**: Machine learning-powered clustering with optimal segment identification
- 📈 **Interactive Analytics**: High-quality visualisations including 3D scatter plots and heatmaps
- 🔍 **Granular Filtering**: Advanced filtering by customer tier, sales channel, and product categories
- 💡 **Strategic Business Insights**: Actionable recommendations for VIP services and marketing
- 💰 **GBP Currency Support**: Native British pound integration for UK retail
- 📱 **Responsive Design**: Mobile-optimised interface for on-the-go retail management

## Customer Segments

Our advanced clustering algorithm identifies sophisticated customer personas:

- **VIP Champions**: Elite customers with recent, frequent, high-value purchases (£600+)
- **Loyal Customers**: Consistent high-spending customers with strong brand affinity
- **At-Risk Customers**: Previously valuable customers requiring re-engagement
- **New Prospects**: Recent customers with potential for product introductions
- **High-Value Customers**: High-spending customers (£500+) seeking exclusive experiences
- **Frequent Buyers**: Regular buyers requiring VIP membership programmes
- **Recent Customers**: New customers with seasonal opportunities

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
- Generate interactive visualisations
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

## Dataset Overview

The `rfm_enriched.csv` file contains **5,000 retail transactions** from **993 customers** with the following fields:

- `customer_id`: Unique customer identifier (1-1000)
- `purchase_amount`: Transaction amount in GBP (including returns as negative values)
- `purchase_date`: Date of purchase (spanning approximately 1 year of data)
- `channel`: Purchase channel (Online/In-store)
- `product_category`: Product categories (Electronics, Books, Home, Apparel, Groceries)
- `transaction_type`: Type of transaction (Full Price, Discount, Return)
- `customer_tier`: Customer loyalty tier (Bronze, Silver, Gold, Platinum)

**Retail Statistics:**

- **Total Transactions:** 5,000 purchases
- **Unique Customers:** 993 shoppers
- **Date Range:** 2023-08-03 to 2024-07-31 (approximately 1 year of transaction data)
- **Average Transaction Value:** £98.64
- **Average Customer Frequency:** 5.0 purchases (strong brand loyalty)
- **Average Customer Recency:** 72.5 days (good engagement patterns)

## Strategic Business Value

This analytics platform delivers exceptional business value for retail brands:

- **VIP Customer Retention**: Identify and nurture high-value customers with exclusive services and early access to products
- **Customer Re-engagement**: Target at-risk customers with personalised re-engagement campaigns and exclusive events
- **Channel Optimisation**: Understand customer behaviour patterns across online and in-store channels for omnichannel excellence
- **Marketing ROI Optimisation**: Allocate marketing spend based on segment characteristics and customer lifetime value
- **Personalised Customer Experiences**: Develop sophisticated, data-driven marketing strategies for different customer segments
- **Inventory Planning**: Leverage recency patterns to optimise product launches and inventory management

## 📸 Dashboard Screenshots & Business Insights

### 1. **Main Dashboard Overview**

![Dashboard Overview](1.PNG)

**Key Metrics Displayed:**

- **Total Customers:** 993 (substantial customer base for analysis)
- **Total Transactions:** 5,000 (comprehensive dataset spanning multiple product categories)
- **Average Transaction Value:** £98.64 (healthy average spend across diverse product categories)
- **Date Range:** 2023-08-03 to 2024-07 (full year of retail data)

**Business Impact:** This overview demonstrates the platform's ability to process large-scale retail data and present key performance indicators in an executive-friendly format. The metrics reveal a well-established retail operation with significant transaction volume across various product categories including Electronics, Books, Groceries, Home, and Apparel.

### 2. **Customer Segmentation Analysis**

![Customer Segmentation](2.PNG)

**Machine Learning Optimisation:**

- **Elbow Method:** Demonstrates optimal cluster selection using Within-Cluster Sum of Squares (WCSS) analysis
- **Silhouette Score:** Shows cluster quality assessment with scores ranging from 0.26 to 0.37
- **Optimal Clusters:** Both methods indicate optimal segmentation for customer behaviour analysis

**Technical Excellence:** This section showcases advanced machine learning techniques for customer segmentation, demonstrating understanding of both statistical methods (Elbow Method) and quality metrics (Silhouette Score) for optimal cluster determination.

### 3. **Multi-Dimensional Customer Behaviour Analysis**

![Behavioural Analysis](3.PNG)

**Channel Performance Insights:**

- **In-store Dominance:** 50% of transactions occur in physical stores, generating £99.56 average transaction value
- **Online Channel:** Equally strong performance with £97.71 average transaction value
- **Omnichannel Strategy:** Both channels show similar customer engagement levels (900-950 unique customers each)

**Product Category Analysis:**

- **Electronics:** Highest revenue per transaction (£103.48) - premium positioning opportunity
- **Books:** Most popular category (21.9% of sales) - strong market demand
- **Balanced Portfolio:** Five distinct categories (Books, Electronics, Groceries, Home, Apparel) showing diverse customer preferences

**Strategic Value:** This analysis reveals the importance of maintaining strong omnichannel presence while identifying Electronics as the premium category for targeted marketing and Books as the volume driver.

### 4. **Customer Tier & Transaction Analysis**

![Tier Analysis](4.PNG)

**Customer Tier Insights:**

- **Bronze Tier:** Highest average spend (£100.30) despite being the entry-level tier
- **Silver Tier:** Also strong performance (£100.12 average spend)
- **Gold & Platinum:** Moderate performance, suggesting opportunity for tier optimisation

**Transaction Type Patterns:**

- **Full Price:** 60.8% of transactions (strong full-price acceptance)
- **Discount:** 30.1% of transactions (healthy promotional activity)
- **Returns:** 9.08% of transactions (acceptable return rate for retail)

**Business Intelligence:** The tier analysis reveals an interesting paradox where Bronze customers show the highest spend, indicating potential for tier restructuring or enhanced Bronze benefits to drive revenue.

### 5. **Cross-Segment Behavioural Patterns**

![Cross-Segment Analysis](5.PNG)

**Segment-Channel Heatmap:**

- **In-store Performance:** £99.56 average transaction for Loyal Customers
- **Online Performance:** £97.71 average transaction for Loyal Customers
- **Channel Preference:** Slight preference for in-store shopping among loyal customers

**Segment-Tier Heatmap:**

- **Bronze Loyal Customers:** £100.30 average spend (highest value segment)
- **Silver Loyal Customers:** £100.12 average spend (second highest)
- **Gold Loyal Customers:** £93.68 average spend (opportunity for engagement)

**Strategic Implications:** The heatmaps reveal that Bronze and Silver tier customers within the Loyal segment are the highest-value customers, suggesting a need to review tier benefits and potentially enhance Bronze/Silver offerings.

### 6. **Temporal Behaviour Analysis**

![Temporal Analysis](6.PNG)

**Key Behavioural Insights:**

- **Channel Strategy:** In-store dominates with 50% of transactions and highest average value (£99.56)
- **Customer Tier Analysis:** Bronze customers represent 32.3% of base with highest average spend (£100.30)
- **Product Category Trends:** Books lead in popularity (21.9% of sales), Electronics generate highest revenue per transaction (£103.48)

**Monthly Transaction Trends:**

- **Peak Periods:** October 2023, February 2024, May 2024 show highest transaction volumes
- **Seasonal Patterns:** Clear cyclical behaviour with peaks every 3-4 months
- **Customer Engagement:** Transaction count and unique customers follow similar patterns

**Business Intelligence:** The temporal analysis reveals strong seasonal patterns that can inform inventory planning, marketing campaigns, and staffing decisions.

### 7. **Seasonal & Revenue Trends**

![Seasonal Analysis](7.PNG)

**Seasonal Purchase Patterns:**

- **Quarterly Distribution:** All quarters show approximately 1,000 transactions each
- **Consistent Performance:** Minimal seasonal variation suggests stable business model
- **Average Transaction Values:** Range from £97-£100 across quarters

**Revenue Trends by Channel:**

- **In-store Revenue:** Peaks at £22.5k in January 2024, shows recovery patterns
- **Online Revenue:** More volatile, peaks at £25k in May 2024
- **Channel Growth:** Online shows fastest growth at 4.6% per month

**Strategic Value:** The analysis reveals that while in-store provides stability, online offers growth potential, supporting an omnichannel strategy with digital acceleration.

### 8. **Temporal Insights & RFM Distribution**

![Temporal Insights](8.PNG)

**Temporal Insights Summary:**

- **Peak Shopping Month:** May 2024 with 9.2% of annual transactions
- **Lowest Activity:** August 2023 with 7.6% of annual transactions
- **Growth Trends:** 6.3% transaction growth, -6.0% average value growth
- **Channel Performance:** Online fastest growing at 4.6% per month

**RFM Distribution Analysis:**

- **Recency Distribution:** Heavily skewed left (most customers made recent purchases)
- **Frequency Distribution:** Bell-shaped curve peaking at 5 purchases
- **Monetary Distribution:** Highly skewed left (most customers have lower spend)

**Business Intelligence:** The RFM distributions reveal a customer base with good recent engagement but opportunities for increasing average transaction values and purchase frequency.

### 9. **Customer Segment Distribution**

![Segment Distribution](9.PNG)

**Segment Characteristics:**

- **Loyal Customers:** 100% of analysed customer base (993 customers)
- **Recency:** 72.47 days average (good recent engagement)
- **Frequency:** 5.04 purchases average (strong brand loyalty)
- **Monetary:** £496.66 average spend (high-value customer base)

**Strategic Implications:** The analysis focuses on a highly engaged, loyal customer base with strong purchase frequency and high average spend, representing the core value segment for the business.

### 10. **3D RFM Scatter Plot**

![3D Scatter Plot](10.PNG)

**Three-Dimensional Analysis:**

- **Recency Range:** 0-350 days (comprehensive customer lifecycle coverage)
- **Frequency Range:** 0-12 purchases (diverse engagement levels)
- **Monetary Range:** 0-£2,000 (wide spending spectrum)
- **Data Point Distribution:** Dense clustering in low recency, high frequency areas

**Technical Excellence:** The 3D visualisation demonstrates advanced data visualisation capabilities and provides intuitive understanding of customer behaviour patterns across all three RFM dimensions.

### 11. **Monetary vs Frequency Analysis**

![Monetary Frequency Analysis](11.PNG)

**Customer Behaviour Patterns:**

- **Positive Correlation:** Higher frequency generally correlates with higher monetary value
- **Frequency Distribution:** Wide spread at lower frequencies (1-3), concentrated at higher frequencies (4-12)
- **High-Value Outliers:** Several customers with exceptional spend at various frequency levels
- **Recency Integration:** Bubble size represents recency, adding third dimension to analysis

**Business Intelligence:** The analysis reveals that increasing purchase frequency is a key driver of higher customer value, supporting strategies focused on customer engagement and repeat purchase programmes.

### 12. **Loyal Customer Strategy**

![Loyal Customer Strategy](12.PNG)

**Strategic Framework:**

- **Retention Focus:** 993 customers to retain with £497 average spend
- **Engagement Metrics:** 5.0 average purchase frequency, 72 days average recency
- **Key Actions:** VIP programmes, exclusive access, personalised experiences, referral programmes, premium service, exclusive events

**Business Value:** This strategic framework demonstrates the ability to translate data insights into actionable business strategies, showing understanding of customer retention and loyalty programme development.

### 13. **Business Insights & Recommendations**

![Business Insights](13.PNG)

**Loyal Customer Metrics:**

- **Customer Count:** 993 (substantial customer base)
- **Average Recency:** 72 days (good recent engagement)
- **Average Frequency:** 5.0 purchases (strong brand loyalty)
- **Average Monetary:** £497 (high-value segment)
- **Total Revenue:** £493,188 (significant revenue contribution)
- **Revenue Percentage:** 100% (core customer segment)
- **Average Transaction:** £99 (healthy transaction value)
- **Dominant Tier:** Bronze (interesting tier distribution)

**Recommendations:** High-frequency buyers - VIP membership programmes

**Strategic Value:** The insights demonstrate sophisticated understanding of customer value metrics and ability to generate specific, actionable recommendations for business growth.

### 14. **Business Insights Detail**

![Business Insights Detail](14.PNG)

**Comprehensive Customer Analysis:**

- **Revenue Contribution:** £493,188 total revenue from loyal customers
- **Tier Distribution:** Bronze tier dominance despite being entry-level
- **Transaction Value:** £99 average transaction across the segment
- **Strategic Focus:** VIP membership programmes for high-frequency buyers

**Business Intelligence:** The detailed analysis reveals the importance of the loyal customer segment and identifies specific opportunities for programme development and tier optimisation.

### 15. **Download & Summary Statistics**

![Download & Statistics](15.PNG)

**Data Export Capabilities:**

- **CSV Download:** Complete RFM results export functionality
- **Summary Statistics:** Comprehensive statistical analysis of customer segments
- **Statistical Measures:** Mean and standard deviation for all RFM metrics
- **Data Quality:** Professional-grade data export and reporting capabilities

**Technical Excellence:** The download functionality demonstrates understanding of business needs for data export and reporting, essential for enterprise analytics platforms.

### 16. **Summary Statistics Explanation**

**Statistical Measures Explained:**

**Mean (Average):** The central tendency of each RFM metric within each segment

- **Recency Mean:** Average days since last purchase (lower is better)
- **Frequency Mean:** Average number of purchases per customer (higher is better)
- **Monetary Mean:** Average total spend per customer in GBP (higher is better)

**Standard Deviation (Std):** Measures the variability/spread of values within each segment

- **Low Std:** Values are clustered close to the mean (consistent behaviour)
- **High Std:** Values are spread out from the mean (variable behaviour)

**Statistical Interpretation:**

**Recency Analysis:**

- **Mean:** Shows how recently customers in each segment made purchases
- **Standard Deviation:** Indicates consistency of purchase timing
- **Business Insight:** Lower recency with low std = consistently recent buyers

**Frequency Analysis:**

- **Mean:** Reveals purchase frequency patterns per segment
- **Standard Deviation:** Shows variability in purchase frequency
- **Business Insight:** Higher frequency with low std = loyal, consistent buyers

**Monetary Analysis:**

- **Mean:** Indicates average customer value per segment
- **Standard Deviation:** Shows spending consistency within segments
- **Business Insight:** Higher monetary with low std = high-value, predictable customers

**Strategic Implications:**

- **Low Std across all metrics:** Highly homogeneous segments, ideal for targeted marketing
- **High Std in any metric:** Diverse customer behaviour within segment, may need sub-segmentation
- **High Mean + Low Std:** Premium segments with consistent high-value behaviour

**Business Intelligence:** The comprehensive statistical analysis demonstrates advanced understanding of data science principles and ability to translate complex statistical measures into actionable business insights for marketing and customer management strategies.

## 🎯 Portfolio Impact

This dashboard showcases:

- **Advanced Analytics**: RFM analysis and machine learning clustering
- **Data Visualisation**: Interactive charts, 3D plots, and heatmaps
- **Business Intelligence**: Actionable insights and strategic recommendations
- **Technical Skills**: Python, Streamlit, Pandas, Scikit-learn, Plotly
- **Business Acumen**: Understanding of customer segmentation and marketing strategies

## Advanced Technologies

- **Python 3.8+**: Core programming language with advanced data science capabilities
- **Streamlit**: Modern web application framework for rapid deployment and interactive analytics
- **Pandas**: Sophisticated data manipulation and analysis for large-scale retail datasets
- **NumPy**: High-performance numerical computing for complex mathematical operations
- **Scikit-learn**: Advanced machine learning algorithms including K-means clustering with optimal parameter selection
- **Plotly**: Interactive, publication-quality visualisations with 3D plotting capabilities
- **Matplotlib & Seaborn**: Professional statistical data visualisation and plotting libraries
- **GBP Currency Integration**: Native British pound support for UK retail operations

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

This RFM segmentation dashboard represents a sophisticated approach to customer analytics in the retail sector. By leveraging advanced machine learning algorithms and interactive visualisations, the platform delivers:

- **Enhanced Customer Understanding**: Deep insights into customer behaviour patterns and preferences
- **Improved Marketing ROI**: Data-driven allocation of marketing resources for maximum impact
- **Personalised Customer Experiences**: Tailored services and communications for different customer segments
- **Operational Efficiency**: Streamlined customer management processes through automated segmentation
- **Competitive Advantage**: Advanced analytics capabilities that differentiate brands in the market

## Future Enhancements

Potential areas for expansion include:

- **Predictive Analytics**: Customer lifetime value forecasting and churn prediction
- **Real-time Integration**: Live data feeds from POS systems and e-commerce platforms
- **AI-Powered Recommendations**: Personalised product recommendations based on segment characteristics
- **Multi-brand Support**: Scalable architecture for managing multiple brand portfolios
- **Advanced Visualisations**: Augmented reality dashboards and immersive analytics experiences

## 📊 Detailed Business Analysis & Recommendations

### **Product Category Strategy**

The analysis reveals a diverse product portfolio with distinct customer behaviours:

**Electronics (£103.48 average transaction):**

- **Positioning:** Premium category with highest revenue per transaction
- **Strategy:** Focus on high-value customers, exclusive product launches, and premium service
- **Opportunity:** Develop VIP programmes specifically for electronics customers

**Books (21.9% of sales):**

- **Positioning:** Volume driver with broad customer appeal
- **Strategy:** Cross-selling opportunities, subscription services, and loyalty programmes
- **Opportunity:** Leverage high volume for customer acquisition and retention

**Groceries, Home, Apparel:**

- **Positioning:** Balanced portfolio supporting diverse customer needs
- **Strategy:** Seasonal campaigns, bundle offers, and category-specific promotions
- **Opportunity:** Develop category-specific loyalty programmes

### **Channel Optimisation Strategy**

**In-store Performance (£99.56 average transaction):**

- **Strengths:** Higher average transaction value, stable performance
- **Strategy:** Enhance in-store experience, staff training, and premium services
- **Opportunity:** Develop exclusive in-store events and VIP shopping experiences

**Online Performance (£97.71 average transaction, 4.6% monthly growth):**

- **Strengths:** Fastest growing channel, lower operational costs
- **Strategy:** Invest in digital marketing, personalised recommendations, and mobile optimisation
- **Opportunity:** Develop omnichannel integration and seamless customer journeys

### **Customer Tier Optimisation**

**Bronze Tier Paradox (£100.30 average spend):**

- **Insight:** Entry-level tier shows highest average spend
- **Strategy:** Review tier benefits and potentially enhance Bronze offerings
- **Recommendation:** Develop tier progression programmes to encourage upgrades

**Gold & Platinum Opportunities:**

- **Insight:** Lower than expected performance in premium tiers
- **Strategy:** Enhance tier benefits, exclusive services, and personalised experiences
- **Recommendation:** Develop targeted retention programmes for premium customers

## 🎯 Strategic Recommendations

### **Immediate Actions (0-3 months):**

1. **VIP Programme Development:**

   - Launch exclusive membership for high-frequency buyers (5+ purchases)
   - Offer early access to new products and exclusive events
   - Implement personalised styling sessions and premium customer service

2. **Channel Enhancement:**

   - Invest in online growth initiatives (4.6% monthly growth opportunity)
   - Enhance in-store experience to maintain £99.56 average transaction value
   - Develop omnichannel integration for seamless customer journeys

3. **Tier Optimisation:**
   - Review Bronze tier benefits given £100.30 average spend
   - Develop tier progression programmes to encourage upgrades
   - Enhance Gold and Platinum offerings to improve performance

### **Medium-term Strategy (3-12 months):**

1. **Product Category Focus:**

   - Develop Electronics VIP programme for £103.48 average transaction customers
   - Leverage Books category (21.9% of sales) for customer acquisition
   - Create category-specific loyalty programmes and cross-selling initiatives

2. **Customer Engagement:**

   - Implement personalised marketing campaigns based on RFM segments
   - Develop referral programmes and loyalty rewards
   - Launch exclusive events and product launches for high-value customers

3. **Data-Driven Marketing:**
   - Allocate marketing spend based on segment characteristics
   - Develop predictive analytics for customer lifetime value
   - Implement real-time customer behaviour tracking

### **Long-term Vision (12+ months):**

1. **Advanced Analytics:**

   - Implement predictive customer lifetime value modelling
   - Develop churn prediction and prevention programmes
   - Launch AI-powered personalised product recommendations

2. **Technology Integration:**

   - Real-time data feeds from POS systems and e-commerce platforms
   - Advanced visualisation capabilities including augmented reality
   - Multi-brand support for scalable growth

3. **Customer Experience Excellence:**
   - Seamless omnichannel customer journeys
   - Personalised customer experiences across all touchpoints
   - Advanced loyalty and retention programmes

## 📈 Conclusion

This RFM Segmentation Dashboard represents a sophisticated approach to customer analytics that delivers exceptional business value. The comprehensive analysis of 5,000 transactions across 993 customers reveals critical insights for retail strategy development:

### **Key Achievements:**

1. **Advanced Analytics Implementation:** Successfully deployed machine learning clustering algorithms with optimal parameter selection, demonstrating technical excellence in data science and business intelligence.

2. **Comprehensive Customer Understanding:** Deep insights into customer behaviour patterns across multiple dimensions including recency, frequency, monetary value, channels, tiers, and product categories.

3. **Actionable Business Intelligence:** Translation of complex data into strategic recommendations for VIP programmes, channel optimisation, and customer retention initiatives.

4. **Technical Excellence:** Professional-grade data visualisation, interactive analytics, and enterprise-ready export capabilities.

### **Business Impact:**

The dashboard reveals a highly engaged customer base with strong loyalty indicators (5.0 average purchase frequency, £497 average spend) and identifies specific opportunities for revenue growth through targeted strategies. The analysis supports data-driven decision making for marketing allocation, inventory planning, and customer experience development.

### **Strategic Value:**

This platform provides the foundation for sophisticated customer relationship management, enabling businesses to develop personalised marketing strategies, optimise channel performance, and maximise customer lifetime value. The insights support strategic initiatives across product development, marketing, and customer service functions.

### **Future Potential:**

The modular architecture and advanced analytics capabilities provide a strong foundation for future enhancements including predictive analytics, real-time integration, and AI-powered recommendations. The platform demonstrates the ability to scale and adapt to evolving business needs and technological advancements.

This project showcases the intersection of advanced data science, business intelligence, and strategic thinking, making it an excellent demonstration of capabilities for data science and analytics roles in retail and e-commerce organisations.
