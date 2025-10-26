# data/generate_sales_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

n_customers = 2000
n_products = 80
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 9, 30)
days_range = (end_date - start_date).days

rows = []
for _ in range(20000):  # ~20k transactions
    order_date = start_date + timedelta(days=np.random.randint(0, days_range))
    customer_id = f"C{np.random.randint(1, n_customers+1):05d}"
    product_id = f"P{np.random.randint(1, n_products+1):03d}"
    category = np.random.choice(
        ["Electronics", "Home", "Clothing", "Sports", "Beauty", "Grocery"],
        p=[0.2, 0.18, 0.22, 0.12, 0.14, 0.14]
    )
    price = round(np.random.gamma(2.0, 30) + (10 if category == "Electronics" else 0), 2)
    quantity = np.random.choice([1, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05])
    region = np.random.choice(["North", "South", "East", "West"], p=[0.28, 0.25, 0.24, 0.23])
    discount = np.random.choice([0, 5, 10, 15], p=[0.7, 0.2, 0.08, 0.02])
    sales = round(price * quantity * (1 - discount / 100), 2)
    order_id = f"O{np.random.randint(100000, 999999)}"
    rows.append([order_id, order_date.strftime("%Y-%m-%d"), customer_id, product_id, category, price, quantity, discount, sales, region])

df = pd.DataFrame(rows, columns=[
    "order_id", "order_date", "customer_id", "product_id", "category",
    "price", "quantity", "discount_pct", "sales", "region"
])

df.to_csv("data/sales.csv", index=False)
print("✅ Dataset saved as data/sales.csv — shape:", df.shape)
