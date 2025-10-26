# app/streamlit_app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import subprocess

st.set_page_config(page_title="Retail Sales Insights", layout="wide")
st.title("Retail Sales Insights")

# Try to find the project root and data file in a few common places
def find_data_file():
    # Current file's parent directory
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "sales.csv",   # project_root/data/sales.csv when running from app/
        here / "data" / "sales.csv",          # app/data/sales.csv (less likely)
        here.parent.parent / "data" / "sales.csv",  # if nested deeper
        Path.cwd() / "data" / "sales.csv",    # if running from project root
        Path.cwd() / "app" / "data" / "sales.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

DATA_PATH = find_data_file()

# If data is missing, prompt to generate it automatically (calls the generator script)
def auto_generate_data():
    st.info("Generating synthetic data (this may take a few seconds)...")
    # Attempt to run the generator script from project root
    generator_paths = [
        Path(__file__).resolve().parent.parent / "data" / "generate_sales_data.py",
        Path(__file__).resolve().parent.parent.parent / "data" / "generate_sales_data.py",
        Path.cwd() / "data" / "generate_sales_data.py",
        Path.cwd() / "data" / "generate_sales_data.py",
    ]
    for gp in generator_paths:
        if gp.exists():
            try:
                # run using the same python interpreter that runs streamlit
                cmd = [os.sys.executable, str(gp)]
                subprocess.check_call(cmd)
                return find_data_file()
            except Exception as e:
                st.error(f"Auto-generation failed: {e}")
                return None
    st.error("Generator script not found. Please run `python data/generate_sales_data.py` from project root.")
    return None

if DATA_PATH is None:
    st.warning("data/sales.csv not found.")
    if st.button("Generate synthetic data now"):
        DATA_PATH = auto_generate_data()
    else:
        st.write("Either run `python data/generate_sales_data.py` from the project root, or click the button above to try auto-generation (requires the generator script to be present).")

if DATA_PATH is None:
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=["order_date"])

df = load_data(str(DATA_PATH))
df['month'] = df['order_date'].dt.to_period('M').astype(str)

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"₹{df['sales'].sum():,.0f}")
col2.metric("Orders", f"{df['order_id'].nunique()}")
col3.metric("Customers", f"{df['customer_id'].nunique()}")

st.markdown("### Monthly Sales")
monthly = df.groupby('month')['sales'].sum().reset_index()
monthly['month_dt'] = pd.to_datetime(monthly['month'])
monthly = monthly.sort_values('month_dt')
st.line_chart(data=monthly.set_index('month_dt')['sales'])

st.markdown("### Revenue by Category")
cat = df.groupby('category')['sales'].sum().reset_index().sort_values('sales', ascending=False)
st.bar_chart(cat.set_index('category')['sales'])

st.markdown("### Top 10 Products")
top_products = df.groupby('product_id')['sales'].sum().reset_index().sort_values('sales', ascending=False).head(10)
st.table(top_products)

st.markdown("### Region-wise revenue")
region = df.groupby('region')['sales'].sum().reset_index()
st.bar_chart(region.set_index('region')['sales'])

st.markdown("### Notes & Recommendations")
st.write("- Focus promotions & inventory on top categories (see chart).")
st.write("- Increase marketing budget 2–3 weeks before peak months.")
st.write("- Consider a more advanced forecasting model (Prophet) for the next sprint.")
