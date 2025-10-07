# Stock Market Dashboard — v1.0

A Streamlit dashboard for visualizing stock performance, annual returns, return distributions, and simulating monthly investments using a Dollar-Cost Averaging (DCA) strategy.

---

## Features (v1.0)

- Interactive candlestick charts (daily, weekly, monthly)
- Annual returns with color-coded bars
- Return distribution histograms (daily, weekly, monthly)
- DCA simulation with configurable investment amount and day of month
- Automatic performance metrics (cumulative returns, annualized averages)

---

## Tech Stack

- Python 3.12+
- Streamlit
- yfinance
- matplotlib
- pandas

---

## How to Run

```bash
# Clone repository
git clone https://github.com/your-username/stock-dashboard.git
cd stock-dashboard

# (Optional) create environment
conda create -n stock_dash python=3.12
conda activate stock_dash

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
