# Retail Sales Insights

**Short description:** End-to-end retail analytics prototype: synthetic transactions → EDA & SQL → KPI dashboard → baseline forecasting → business recommendations.

## Tech stack
Python (pandas, matplotlib), SQL (pandasql), scikit-learn, Streamlit

## Quick setup
```bash
git clone <your-repo-url>
cd RetailSalesProject
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# generate synthetic data
python data/generate_sales_data.py

# run analysis script (produces outputs/)
python notebooks/retail_sales_analysis.py

# run dashboard (optional)
streamlit run app/streamlit_app.py
