
if 100 == 200:
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import numpy as np
    import pandas as pd
    import streamlit as st
    import yfinance as yf

    st.title("Market Dashboard")

    st.subheader(" Example of Ticker Inputs")

    # --- Example data ---
    example_data = {
        "Category": [
            "Brazilian Stock", "US Stock", "European Stock", "Gold ETF", "Bitcoin ETF"
        ],
        "Ticker": [
            "WEGE3.SA", "AAPL", "SAP.DE", "GLD", "BITO"
        ],
        "Exchange": [
            "B3", "NASDAQ", "XETRA", "NYSE", "NYSE"
        ]
    }

    example_df = pd.DataFrame(example_data)

    # --- Style dark ---
    st.write(
        example_df.style.set_properties(**{
            'background-color': '#0A0E1A',
            'color': 'white',
            'border-color': '#0A0E1A'
        }).set_table_styles([
            {'selector': 'thead', 'props': [
                ('background-color', '#0A0E1A'), ('color', 'white')]}
        ])
    )

    ticker = st.text_input("Ticker", "NVDA")

    full_data = yf.download(ticker, period="max")
    if isinstance(full_data.columns, pd.MultiIndex):
        full_data.columns = [col[0] for col in full_data.columns]

    max_years = max(1, (full_data.index[-1] - full_data.index[0]).days // 365)
    years = st.slider("Years to display", min_value=1,
                      max_value=max_years, value=min(3, max_years))
    start_date = full_data.index[-1] - pd.DateOffset(years=years)
    data = full_data.loc[start_date:].copy()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    freq_option = st.selectbox("Candlestick frequency", [
        "Daily", "Weekly", "Monthly"])

    if freq_option == "Weekly":
        data_resampled = data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    elif freq_option == "Monthly":
        data_resampled = data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    else:
        data_resampled = data

    # --- Dark style with white axes labels, no extra borders ---
    # --- Dark style with white axes labels, no extra borders, no grids ---
    mc = mpf.make_marketcolors(
        up='#00FF00', down='#FF3300', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mc,
        gridstyle='',        # no grid lines
        facecolor='#0A0E1A',
        figcolor='#0A0E1A',
        edgecolor='#0A0E1A',  # hide figure edges
        y_on_right=False,
        rc={'axes.labelcolor': 'white',
            'xtick.color': 'white', 'ytick.color': 'white'}
    )

    mpf_fig, mpf_ax = mpf.plot(
        data_resampled,
        type='candle',
        style=s,
        volume=True,
        returnfig=True,
        warn_too_much_data=10000,

    )
    st.pyplot(mpf_fig)

    # --- Select what to plot ---
    column_to_plot = st.selectbox("Select column to plot", [
        'Open', 'Close', 'Volume'])

    st.subheader(f"{column_to_plot} over time")

    # --- Date filter ---
    st.subheader("Optional: Filter by date range")

    min_date = data.index.min()
    max_date = data.index.max()

    date_range = st.date_input(
        "Select date range",
        value=[min_date.date(), max_date.date()],
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    # st.pyplot(fig_z)
    # --- Slice data according to selected date range ---
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_filter, end_filter = pd.to_datetime(
            date_range[0]), pd.to_datetime(date_range[1])
        data_filtered = data.loc[start_filter:end_filter].copy()
    else:
        data_filtered = data.copy()

    # --- Create main figure ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_filtered.index, data_filtered[column_to_plot],
            color='#3399FF' if column_to_plot != 'Volume' else '#AA33FF',
            linewidth=1.5)

    # --- Dark style for main figure ---
    ax.set_facecolor('#0A0E1A')
    fig.patch.set_facecolor('#0A0E1A')
    ax.tick_params(colors='white', which='both')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#0A0E1A')   # hide top spine
    ax.spines['right'].set_color('#0A0E1A')  # hide right spine
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel(column_to_plot, color='white')
    ax.grid(False)

    st.pyplot(fig)

    # --- Z-score parameters ---
    column_to_z = st.selectbox(
        "Column for Z-score", ['Open', 'Close', 'Volume'])
    st.subheader(f"Z-score of {column_to_z}")

    freq_z = st.selectbox("Frequency for Z-score",
                          ['Daily', 'Weekly', 'Monthly'])
    k = st.slider("Rolling window (k periods)",
                  min_value=5, max_value=100, value=20)

    # --- Prepare series for Z-score ---
    if freq_z == "Weekly":
        series = data_filtered[column_to_z].resample('W').last()
    elif freq_z == "Monthly":
        series = data_filtered[column_to_z].resample('M').last()
    else:
        series = data_filtered[column_to_z]

    # --- Compute rolling Z-score ---
    rolling_mean = series.rolling(k).mean()
    rolling_std = series.rolling(k).std()
    z_score = (series - rolling_mean) / rolling_std

    # --- Create Z-score figure ---
    fig_z, ax_z = plt.subplots(figsize=(12, 3))
    ax_z.plot(series.index, z_score, color='#3399FF', linewidth=1.5)

    # --- Horizontal reference lines for Z-score ---
    for level, color in zip([0, 1, -1, 2, -2, 3, -3],
                            ['white', '#AAAAFF', '#AAAAFF', '#77FF77', '#77FF77', '#FF7777', '#FF7777']):
        ax_z.axhline(level, color=color, linestyle='--', linewidth=0.8)

    # --- Dark style for Z-score ---
    ax_z.set_facecolor('#0A0E1A')
    fig_z.patch.set_facecolor('#0A0E1A')
    ax_z.tick_params(colors='white', which='both')
    ax_z.spines['bottom'].set_color('white')
    ax_z.spines['left'].set_color('white')
    ax_z.spines['top'].set_color('#0A0E1A')
    ax_z.spines['right'].set_color('#0A0E1A')
    ax_z.set_xlabel("Date", color='white')
    ax_z.set_ylabel(f"Z-score ({k} periods)", color='white')
    ax_z.grid(False)

    st.pyplot(fig_z)

    st.title("Annual Returns and Distribution")

    # --- Prices ---
    prices = data_filtered['Close'].astype(float)

    # --- Start and end of year prices ---
    start_year = prices.resample('Y').first()   # start-of-year price
    end_year = prices.resample('Y').last()      # end-of-year price

    # --- Annual returns based on start-of-year ---
    returns_yearly = ((end_year / start_year - 1) *
                      100).dropna()  # convert to %

    # --- Annual returns bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#77FF77' if r >= 0 else '#FF7777' for r in returns_yearly]
    returns_yearly.plot(kind='bar', color=colors, ax=ax)
    ax.set_facecolor('#0A0E1A')
    fig.patch.set_facecolor('#0A0E1A')
    ax.tick_params(colors='white')
    ax.set_xlabel("Year", color='white')
    ax.set_ylabel("Annual Return (%)", color='white')
    ax.set_title("Annual Returns (Start-of-Year Basis)", color='white')

    st.pyplot(fig)

    # ===================================================
    # --- Cumulative Return ---
    # ===================================================
    st.subheader("Cumulative Return Over Time")

    cumulative_return = (1 + prices.pct_change().fillna(0)).cumprod() - 1
    cumulative_return *= 100  # convert to %

    fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
    ax_cum.plot(cumulative_return.index, cumulative_return.values,
                color='#3399FF', linewidth=2)
    ax_cum.axhline(0, color='white', linestyle='--', linewidth=0.8)

    ax_cum.set_facecolor('#0A0E1A')
    fig_cum.patch.set_facecolor('#0A0E1A')
    ax_cum.tick_params(colors='white')
    ax_cum.set_xlabel("Date", color='white')
    ax_cum.set_ylabel("Cumulative Return (%)", color='white')
    ax_cum.set_title("Cumulative Performance", color='white')

    st.pyplot(fig_cum)

    # ===================================================
    # --- Histogram of returns ---
    # ===================================================
    st.subheader("Return Distribution")

    freq_ret = st.selectbox(
        "Histogram base frequency",
        ['Daily', 'Weekly', 'Monthly'],
        key="hist_freq"
    )

    # --- Base returns calculation ---
    if freq_ret == 'Weekly':
        base_returns = prices.resample('W').last().pct_change().dropna() * 100
    elif freq_ret == 'Monthly':
        base_returns = prices.resample('M').last().pct_change().dropna() * 100
    else:  # Daily
        base_returns = prices.pct_change().dropna() * 100

    # --- Histogram with color by sign ---
    fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
    bins = 30  # adjustable
    pos = base_returns[base_returns >= 0]
    neg = base_returns[base_returns < 0]

    ax_hist.hist(pos, bins=bins, color='#77FF77', alpha=0.7, label='Positive')
    ax_hist.hist(neg, bins=bins, color='#FF7777', alpha=0.7, label='Negative')

    ax_hist.set_facecolor('#0A0E1A')
    fig_hist.patch.set_facecolor('#0A0E1A')
    ax_hist.tick_params(colors='white')
    ax_hist.set_xlabel(f"{freq_ret} Returns (%)", color='white')
    ax_hist.set_ylabel("Frequency", color='white')
    ax_hist.set_title(f"Histogram of {freq_ret} Returns", color='white')
    ax_hist.legend(facecolor='#0A0E1A', edgecolor='white', labelcolor='white')

    st.pyplot(fig_hist)

    def simulate_dca(prices: pd.Series, monthly_investment: float, day_of_month: int = 5):
        """
        Simulates a DCA (Dollar-Cost Averaging) strategy with fixed monthly contributions.

        Args:
            prices (pd.Series): Price series (Close) with DateTimeIndex.
            monthly_investment (float): Amount invested each month.
            day_of_month (int): Day of the month to invest (default=5).

        Returns:
            pd.DataFrame: Capital curve with columns ['Price', 'Shares', 'Total Value', 'Invested', 'Return %'].
        """
        prices = prices.asfreq('D').ffill()  # daily freq to ensure coverage

        # Generate contribution dates
        start, end = prices.index.min(), prices.index.max()
        contrib_dates = pd.date_range(
            start, end, freq='MS') + pd.to_timedelta(day_of_month - 1, 'D')
        contrib_dates = contrib_dates[contrib_dates <= end]

        shares, invested = 0.0, 0.0
        history = []

        for d in contrib_dates:
            valid_dates = prices.index[prices.index >= d]
            if len(valid_dates) == 0:
                continue
            price_date = valid_dates[0]
            price = prices.loc[price_date]

            qty = monthly_investment / price
            shares += qty
            invested += monthly_investment

            total_value = shares * price
            ret_pct = (total_value / invested - 1) * 100
            history.append([price_date, price, shares,
                            invested, total_value, ret_pct])

        df = pd.DataFrame(history, columns=[
            'Date', 'Price', 'Shares', 'Invested', 'Total Value', 'Return %'])
        df.set_index('Date', inplace=True)
        return df

    st.title("Simulation")

    monthly_investment = st.number_input(
        "Monthly Investment ($)", value=500.0, min_value=50.0, step=50.0, key="dca_amount"
    )
    day_of_month = st.slider("Investment Day of Month",
                             1, 28, 5, key="dca_day")

    df_dca = simulate_dca(
        data_filtered['Close'], monthly_investment, day_of_month)

    # --- Line chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_dca.index, df_dca['Total Value'],
            label='Total Value', color='#66FF99', linewidth=2)
    ax.plot(df_dca.index, df_dca['Invested'], label='Invested',
            color='#3399FF', linewidth=2, linestyle='--')

    ax.set_facecolor('#0A0E1A')
    fig.patch.set_facecolor('#0A0E1A')
    ax.tick_params(colors='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Value ($)", color='white')
    ax.set_title("Portfolio Growth (DCA Strategy)", color='white')
    ax.legend(facecolor='#0A0E1A', edgecolor='white', labelcolor='white')

    st.pyplot(fig)

    # --- Metrics layout ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Portfolio Value",
                f"${df_dca['Total Value'].iloc[-1]:,.2f}")
    col2.metric("Total Invested", f"${df_dca['Invested'].iloc[-1]:,.2f}")
    col3.metric("Total Return", f"{df_dca['Return %'].iloc[-1]:.2f}%")


import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ===================================================
# --- Title & Description ---
# ===================================================
st.title("Market Dashboard")

st.subheader("Example of Ticker Inputs")
example_data = {
    "Category": [
        "Brazilian Stock", "US Stock", "European Stock", "Gold ETF", "Bitcoin ETF"
    ],
    "Ticker": ["WEGE3.SA", "AAPL", "SAP.DE", "GLD", "BITO"],
    "Exchange": ["B3", "NASDAQ", "XETRA", "NYSE", "NYSE"]
}
example_df = pd.DataFrame(example_data)

st.write(
    example_df.style.set_properties(**{
        'background-color': '#0A0E1A',
        'color': 'white',
        'border-color': '#0A0E1A'
    }).set_table_styles([
        {'selector': 'thead', 'props': [
            ('background-color', '#0A0E1A'), ('color', 'white')]}
    ])
)

# ===================================================
# --- Data Download & Selection ---
# ===================================================
ticker = st.text_input("Ticker", "NVDA")
full_data = yf.download(ticker, period="max")

if isinstance(full_data.columns, pd.MultiIndex):
    full_data.columns = [col[0] for col in full_data.columns]

max_years = max(1, (full_data.index[-1] - full_data.index[0]).days // 365)
years = st.slider("Years to display", min_value=1,
                  max_value=max_years, value=min(3, max_years))
start_date = full_data.index[-1] - pd.DateOffset(years=years)
data = full_data.loc[start_date:].copy()

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

freq_option = st.selectbox("Candlestick frequency", [
                           "Daily", "Weekly", "Monthly"])

# Resample if needed
if freq_option == "Weekly":
    data_resampled = data.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
elif freq_option == "Monthly":
    data_resampled = data.resample('M').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
else:
    data_resampled = data

# ===================================================
# --- Candlestick Chart (Dark Style) ---
# ===================================================
mc = mpf.make_marketcolors(
    up='#00FF00', down='#FF3300', edge='inherit', wick='inherit', volume='in')
s = mpf.make_mpf_style(
    base_mpf_style='charles',
    marketcolors=mc,
    gridstyle='',
    facecolor='#0A0E1A',
    figcolor='#0A0E1A',
    edgecolor='#0A0E1A',
    y_on_right=False,
    rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
)

mpf_fig, mpf_ax = mpf.plot(
    data_resampled,
    type='candle',
    style=s,
    volume=True,
    returnfig=True,
    warn_too_much_data=10000
)
st.pyplot(mpf_fig)

# ===================================================
# --- Line Chart Selection ---
# ===================================================
column_to_plot = st.selectbox("Select column to plot", [
                              'Open', 'Close', 'Volume'])
st.subheader(f"{column_to_plot} over time")

# Date range filter
min_date, max_date = data.index.min(), data.index.max()
date_range = st.date_input(
    "Select date range",
    value=[min_date.date(), max_date.date()],
    min_value=min_date.date(),
    max_value=max_date.date()
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_filter, end_filter = pd.to_datetime(
        date_range[0]), pd.to_datetime(date_range[1])
    data_filtered = data.loc[start_filter:end_filter].copy()
else:
    data_filtered = data.copy()

# Plot line chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    data_filtered.index, data_filtered[column_to_plot],
    color='#3399FF' if column_to_plot != 'Volume' else '#AA33FF', linewidth=1.5
)
ax.set_facecolor('#0A0E1A')
fig.patch.set_facecolor('#0A0E1A')
ax.tick_params(colors='white', which='both')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('#0A0E1A')
ax.spines['right'].set_color('#0A0E1A')
ax.set_xlabel("Date", color='white')
ax.set_ylabel(column_to_plot, color='white')
ax.grid(False)
st.pyplot(fig)

# ===================================================
# --- Z-score Analysis ---
# ===================================================
column_to_z = st.selectbox("Column for Z-score", ['Open', 'Close', 'Volume'])
st.subheader(f"Z-score of {column_to_z}")

freq_z = st.selectbox("Frequency for Z-score", ['Daily', 'Weekly', 'Monthly'])
k = st.slider("Rolling window (k periods)",
              min_value=5, max_value=100, value=20)

# Prepare series
if freq_z == "Weekly":
    series = data_filtered[column_to_z].resample('W').last()
elif freq_z == "Monthly":
    series = data_filtered[column_to_z].resample('M').last()
else:
    series = data_filtered[column_to_z]

rolling_mean = series.rolling(k).mean()
rolling_std = series.rolling(k).std()
z_score = (series - rolling_mean) / rolling_std

# Plot Z-score
fig_z, ax_z = plt.subplots(figsize=(12, 3))
ax_z.plot(series.index, z_score, color='#3399FF', linewidth=1.5)
for level, color in zip([0, 1, -1, 2, -2, 3, -3],
                        ['white', '#AAAAFF', '#AAAAFF', '#77FF77', '#77FF77', '#FF7777', '#FF7777']):
    ax_z.axhline(level, color=color, linestyle='--', linewidth=0.8)
ax_z.set_facecolor('#0A0E1A')
fig_z.patch.set_facecolor('#0A0E1A')
ax_z.tick_params(colors='white', which='both')
ax_z.spines['bottom'].set_color('white')
ax_z.spines['left'].set_color('white')
ax_z.spines['top'].set_color('#0A0E1A')
ax_z.spines['right'].set_color('#0A0E1A')
ax_z.set_xlabel("Date", color='white')
ax_z.set_ylabel(f"Z-score ({k} periods)", color='white')
ax_z.grid(False)
st.pyplot(fig_z)

# ===================================================
# --- Annual Returns & Distribution ---
# ===================================================
st.title("Annual Returns and Distribution")
prices = data_filtered['Close'].astype(float)
start_year = prices.resample('Y').first()
end_year = prices.resample('Y').last()
returns_yearly = ((end_year / start_year - 1) * 100).dropna()

# Annual returns bar chart
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#77FF77' if r >= 0 else '#FF7777' for r in returns_yearly]
returns_yearly.plot(kind='bar', color=colors, ax=ax)
ax.set_facecolor('#0A0E1A')
fig.patch.set_facecolor('#0A0E1A')
ax.tick_params(colors='white')
ax.set_xlabel("Year", color='white')
ax.set_ylabel("Annual Return (%)", color='white')
ax.set_title("Annual Returns (Start-of-Year Basis)", color='white')
st.pyplot(fig)

# ===================================================
# --- Cumulative Return ---
# ===================================================
st.subheader("Cumulative Return Over Time")
cumulative_return = (1 + prices.pct_change().fillna(0)).cumprod() - 1
cumulative_return *= 100

fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
ax_cum.plot(cumulative_return.index, cumulative_return.values,
            color='#3399FF', linewidth=2)
ax_cum.axhline(0, color='white', linestyle='--', linewidth=0.8)
ax_cum.set_facecolor('#0A0E1A')
fig_cum.patch.set_facecolor('#0A0E1A')
ax_cum.tick_params(colors='white')
ax_cum.set_xlabel("Date", color='white')
ax_cum.set_ylabel("Cumulative Return (%)", color='white')
ax_cum.set_title("Cumulative Performance", color='white')
st.pyplot(fig_cum)

# ===================================================
# --- Histogram of Returns ---
# ===================================================
st.subheader("Return Distribution")
freq_ret = st.selectbox("Histogram base frequency", [
                        'Daily', 'Weekly', 'Monthly'], key="hist_freq")

if freq_ret == 'Weekly':
    base_returns = prices.resample('W').last().pct_change().dropna() * 100
elif freq_ret == 'Monthly':
    base_returns = prices.resample('M').last().pct_change().dropna() * 100
else:
    base_returns = prices.pct_change().dropna() * 100

fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
pos = base_returns[base_returns >= 0]
neg = base_returns[base_returns < 0]
bins = 30
ax_hist.hist(pos, bins=bins, color='#77FF77', alpha=0.7, label='Positive')
ax_hist.hist(neg, bins=bins, color='#FF7777', alpha=0.7, label='Negative')
ax_hist.set_facecolor('#0A0E1A')
fig_hist.patch.set_facecolor('#0A0E1A')
ax_hist.tick_params(colors='white')
ax_hist.set_xlabel(f"{freq_ret} Returns (%)", color='white')
ax_hist.set_ylabel("Frequency", color='white')
ax_hist.set_title(f"Histogram of {freq_ret} Returns", color='white')
ax_hist.legend(facecolor='#0A0E1A', edgecolor='white', labelcolor='white')
st.pyplot(fig_hist)

# ===================================================
# --- DCA Simulation Function ---
# ===================================================


def simulate_dca(prices: pd.Series, monthly_investment: float, day_of_month: int = 5):
    prices = prices.asfreq('D').ffill()
    start, end = prices.index.min(), prices.index.max()
    contrib_dates = pd.date_range(
        start, end, freq='MS') + pd.to_timedelta(day_of_month - 1, 'D')
    contrib_dates = contrib_dates[contrib_dates <= end]

    shares, invested = 0.0, 0.0
    history = []

    for d in contrib_dates:
        valid_dates = prices.index[prices.index >= d]
        if len(valid_dates) == 0:
            continue
        price_date = valid_dates[0]
        price = prices.loc[price_date]
        qty = monthly_investment / price
        shares += qty
        invested += monthly_investment
        total_value = shares * price
        ret_pct = (total_value / invested - 1) * 100
        history.append([price_date, price, shares,
                       invested, total_value, ret_pct])

    df = pd.DataFrame(history, columns=[
                      'Date', 'Price', 'Shares', 'Invested', 'Total Value', 'Return %'])
    df.set_index('Date', inplace=True)
    return df


# ===================================================
# --- DCA Simulation Dashboard ---
# ===================================================
st.title("Simulation")
monthly_investment = st.number_input(
    "Monthly Investment ($)", value=500.0, min_value=50.0, step=50.0, key="dca_amount")
day_of_month = st.slider("Investment Day of Month", 1, 28, 5, key="dca_day")

df_dca = simulate_dca(data_filtered['Close'], monthly_investment, day_of_month)

# Line chart for DCA
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_dca.index, df_dca['Total Value'],
        label='Total Value', color='#66FF99', linewidth=2)
ax.plot(df_dca.index, df_dca['Invested'], label='Invested',
        color='#3399FF', linewidth=2, linestyle='--')
ax.set_facecolor('#0A0E1A')
fig.patch.set_facecolor('#0A0E1A')
ax.tick_params(colors='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Value ($)", color='white')
ax.set_title("Portfolio Growth (DCA Strategy)", color='white')
ax.legend(facecolor='#0A0E1A', edgecolor='white', labelcolor='white')
st.pyplot(fig)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Final Portfolio Value", f"${df_dca['Total Value'].iloc[-1]:,.2f}")
col2.metric("Total Invested", f"${df_dca['Invested'].iloc[-1]:,.2f}")
col3.metric("Total Return", f"{df_dca['Return %'].iloc[-1]:.2f}%")
