# notebooks/retail_sales_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandasql import sqldf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import seaborn as sns

# --- settings
os.makedirs("outputs", exist_ok=True)
DATA_PATH = "data/sales.csv"  # if running from notebooks/; adjust if needed

# --- load
df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
df['month'] = df['order_date'].dt.to_period('M').astype(str)

# --- quick head
print("Data shape:", df.shape)
print(df.head())

# --- KPIs
total_revenue = df['sales'].sum()
n_orders = df['order_id'].nunique()
n_customers = df['customer_id'].nunique()
aov = total_revenue / n_orders
print(f"\nKPI - Revenue: ₹{total_revenue:,.2f} | Orders: {n_orders} | Customers: {n_customers} | AOV: ₹{aov:.2f}")

# --- monthly series
monthly = df.groupby('month').agg({'sales':'sum','order_id':'nunique'}).rename(columns={'order_id':'orders'})
monthly = monthly.reset_index()
monthly['month_dt'] = pd.to_datetime(monthly['month'])
monthly = monthly.sort_values('month_dt')

# plot monthly sales
plt.figure(figsize=(10,4))
plt.plot(monthly['month_dt'], monthly['sales'], marker='o')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("outputs/monthly_sales.png")
plt.close()

# --- top categories and products
top_categories = df.groupby('category')['sales'].sum().sort_values(ascending=False)
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)

print("\nTop categories by revenue:\n", top_categories)
print("\nTop 10 products by revenue:\n", top_products)

# category bar chart
plt.figure(figsize=(8,4))
top_categories.plot(kind='bar')
plt.title("Revenue by Category")
plt.tight_layout()
plt.savefig("outputs/category_revenue.png")
plt.close()

# top products bar chart
plt.figure(figsize=(10,4))
top_products.plot(kind='bar')
plt.title("Top 10 Products by Revenue")
plt.tight_layout()
plt.savefig("outputs/top_products.png")
plt.close()

# --- cohort-ish (first purchase month)
first_purchase = df.groupby('customer_id')['order_date'].min().reset_index()
first_purchase['cohort_month'] = first_purchase['order_date'].dt.to_period('M').astype(str)
cohort_counts = first_purchase.groupby('cohort_month').size().sort_index()
cohort_counts = cohort_counts.reset_index(name='new_customers')
cohort_counts['cohort_month_dt'] = pd.to_datetime(cohort_counts['cohort_month'])
plt.figure(figsize=(10,4))
plt.plot(cohort_counts['cohort_month_dt'], cohort_counts['new_customers'], marker='o')
plt.title("First Purchase (New Customers) by Month")
plt.tight_layout()
plt.savefig("outputs/cohort_new_customers.png")
plt.close()

# --- SQL style queries using pandasql
pysqldf = lambda q: sqldf(q, globals())

q1 = """
SELECT region, SUM(sales) as revenue, COUNT(DISTINCT order_id) as orders
FROM df
GROUP BY region
ORDER BY revenue DESC
"""
region_summary = pysqldf(q1)
print("\nRegion summary:\n", region_summary)

q2 = """
SELECT month, SUM(sales) as revenue, COUNT(DISTINCT order_id) as orders
FROM df
GROUP BY month
ORDER BY month
"""
monthly_sql = pysqldf(q2)

# month over month % change (pandas)
monthly['sales_prev'] = monthly['sales'].shift(1)
monthly['mom_pct'] = (monthly['sales'] - monthly['sales_prev']) / monthly['sales_prev'] * 100
monthly.fillna(0, inplace=True)
monthly.to_csv("outputs/monthly_summary.csv", index=False)

# --- simple forecasting baseline (linear regression on time index)
ts = monthly.copy().reset_index(drop=True)
ts['t'] = np.arange(len(ts))
X = ts[['t']]
y = ts['sales'].values
model = LinearRegression()
model.fit(X, y)
ts['pred'] = model.predict(X)
mae = mean_absolute_error(ts['sales'], ts['pred'])
print(f"\nForecast baseline (LinearRegression) MAE: {mae:.2f}")

# forecast next 3 months
future_t = np.arange(len(ts), len(ts)+3).reshape(-1,1)
future_pred = model.predict(future_t)
future_months = pd.date_range(start=ts['month_dt'].max() + pd.offsets.MonthBegin(1), periods=3, freq='MS')
forecast_df = pd.DataFrame({'month': future_months, 'predicted_sales': future_pred})
forecast_df.to_csv("outputs/forecast_next3.csv", index=False)
print("\nNext 3 months forecast:\n", forecast_df)

# --- save main summary to outputs
summary = {
    'total_revenue': total_revenue,
    'n_orders': n_orders,
    'n_customers': n_customers,
    'aov': aov,
    'forecast_mae': mae
}
pd.Series(summary).to_csv("outputs/summary_metrics.csv")

# --- Top insights (auto-generated)
top_cat_pct = top_categories / total_revenue * 100
dominant = top_cat_pct[top_cat_pct.cumsum() <= 60]  # categories covering first ~60% revenue
insights = []
insights.append(f"Top categories: {', '.join(top_categories.index.tolist()[:3])} account for {top_categories.values[:3].sum()/total_revenue*100:.1f}% revenue.")
peak_months = monthly.sort_values('sales', ascending=False).head(3)['month'].tolist()
insights.append(f"Peak months by revenue: {', '.join(peak_months)} — suggest marketing & inventory prep prior to these.")
insights.append(f"Simple forecast MAE: {mae:.2f}. Baseline shows trendable growth/decline; consider improved models (Prophet) later.")
with open("outputs/insights.txt", "w") as f:
    for i, it in enumerate(insights,1):
        f.write(f"{i}. {it}\n")
print("\nInsights saved to outputs/insights.txt")
