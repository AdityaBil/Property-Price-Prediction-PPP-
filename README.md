# Property-Price-Prediction

Predicts property prices using inflation forecasts and machine learning.

**What it does**

Predicts inflation rates based on GDP growth, unemployment, and money supply. Calculates future property prices for 5, 7, and 10 years. Shows price trends across different areas with interactive charts.

**Requirements**

Python 3.x, Streamlit, scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn

**Setup**

Install dependencies:
```bash
pip install -r requirements.txt
```

**Usage**

First, train the model:
```bash
python "Property Prediction Management.py"
```

Then run the app:
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Files**

- `app.py` - Streamlit app
- `Property Prediction Management.py` - Training script
- `pythonproj.xlsx` - Property data
- `.pkl` files - Saved model and predictions

**How it works**

The model uses a stacked ensemble with Lasso, XGBoost, Ridge, and Gradient Boosting. It combines inflation predictions with property data to forecast future prices.

Select a location and area, and the app calculates current and future prices based on predicted inflation and growth rates.

**Deployment**

Works on Streamlit Cloud, Heroku, or any Python hosting. Just include all `.pkl` files and the Excel file.
