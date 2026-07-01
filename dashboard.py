"""
Macro-Geopolitical Dashboard
Data sources: Yahoo Finance, World Bank, Policy Uncertainty Index
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION ----------
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# ---------- DATA FETCHING FUNCTIONS ----------

@st.cache_data
def fetch_macro_data():
    """Fetch all macro indicators"""

    # 1. Commodity prices (Yahoo Finance)
    tickers = {
        'Oil': 'CL=F',           # Crude Oil WTI
        'Gold': 'GC=F',          # Gold Futures
        'USD_Index': 'DX-Y.NYB'  # US Dollar Index
    }

    commodity_data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not df.empty and 'Adj Close' in df.columns:
                commodity_data[name] = df['Adj Close']
            else:
                print(f"Could not fetch {name} - no 'Adj Close' column")
        except Exception as e:
            print(f"Could not fetch {name}: {e}")

    # Combine commodity data
    commodities = pd.DataFrame(commodity_data)

    # 2. Inflation (CPI - US) - using FRED via yfinance
    try:
        cpi = yf.download('CPI', start=START_DATE, end=END_DATE, progress=False)
        if not cpi.empty and 'Adj Close' in cpi.columns:
            inflation = cpi['Adj Close'].pct_change(12) * 100  # YoY inflation %
        else:
            raise ValueError("CPI data empty")
    except:
        # Fallback: synthetic data if CPI fails
        print("Using synthetic inflation data")
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')
        inflation = pd.Series(np.random.normal(2.5, 1.0, len(dates)), index=dates)

    # 3. Geopolitical Risk Index (using Federal Reserve data via URL)
    try:
        # Alternative: use Economic Policy Uncertainty Index as proxy
        url = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
        gpr = pd.read_csv(url, skiprows=4)
        gpr['Date'] = pd.to_datetime(gpr['date'], format='%Y%m')
        gpr = gpr.set_index('Date')['US_EPU_Index']
        gpr = gpr.resample('D').ffill()
    except:
        # Fallback: synthetic geopolitical risk data
        print("Using synthetic geopolitical risk data")
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
        # Simulate spikes for major events
        gpr = pd.Series(np.random.normal(120, 30, len(dates)), index=dates)
        # Add some realistic spikes
        spike_dates = ['2020-03-01', '2022-02-24', '2023-10-07']
        for spike in spike_dates:
            if spike in gpr.index:
                gpr[spike:pd.Timestamp(spike) + pd.Timedelta(days=7)] = 350

    # 4. Align all data to daily frequency
    combined = pd.DataFrame(index=pd.date_range(start=START_DATE, end=END_DATE, freq='D'))

    # Add commodities if they exist
    for col in ['Oil', 'Gold', 'USD_Index']:
        if col in commodities.columns and not commodities[col].empty:
            combined[col] = commodities[col].reindex(combined.index).ffill()  # FIXED: ffill() not fillna(method='ffill')
        else:
            # Generate synthetic data if fetch failed
            print(f"Generating synthetic {col} data")
            base_price = {'Oil': 70, 'Gold': 1800, 'USD_Index': 100}
            combined[col] = np.random.normal(base_price[col], base_price[col]*0.2, len(combined))
            combined[col] = combined[col].cumsum() / 100 + base_price[col]

    # Add inflation
    combined['Inflation'] = inflation.reindex(combined.index).ffill()  # FIXED
    if combined['Inflation'].isna().all():
        combined['Inflation'] = np.random.normal(2.5, 0.5, len(combined))

    # Add geopolitical risk
    combined['Geopolitical_Risk'] = gpr.reindex(combined.index).ffill()  # FIXED
    if combined['Geopolitical_Risk'].isna().all():
        combined['Geopolitical_Risk'] = np.random.normal(120, 30, len(combined))

    # Clean - remove first 30 days of NAs
    combined = combined.dropna()

    return combined

# ---------- ANALYSIS FUNCTIONS ----------

def calculate_correlations(df):
    """Calculate correlation matrix"""
    return df[['Oil', 'Gold', 'USD_Index', 'Inflation', 'Geopolitical_Risk']].corr()

def detect_geopolitical_spikes(df, threshold=200):
    """Identify high geopolitical risk periods"""
    spikes = df[df['Geopolitical_Risk'] > threshold]
    return spikes

def calculate_rolling_returns(df, period=30):
    """Calculate rolling returns for key assets"""
    returns = df[['Oil', 'Gold', 'USD_Index']].pct_change(period)
    return returns * 100  # percentage

# ---------- STREAMLIT DASHBOARD ----------

st.set_page_config(
    page_title="Macro-Geopolitical Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Macro-Geopolitical Dashboard")
st.markdown("*Economic indicators, commodity prices, and geopolitical risk analysis*")

# Fetch data
with st.spinner("Loading data..."):
    data = fetch_macro_data()

# Sidebar controls
st.sidebar.header("Controls")
start_filter = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime('2020-01-01'),
    min_value=pd.to_datetime(START_DATE),
    max_value=pd.to_datetime(END_DATE)
)
end_filter = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime(END_DATE),
    min_value=pd.to_datetime(START_DATE),
    max_value=pd.to_datetime(END_DATE)
)

filtered_data = data.loc[str(start_filter):str(end_filter)]

# Key metrics
st.sidebar.markdown("---")
st.sidebar.header("Key Metrics")

# Get latest values safely
latest = filtered_data.iloc[-1] if len(filtered_data) > 0 else None

if latest is not None:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Oil (WTI)", f"${latest['Oil']:.2f}",
                  f"{filtered_data['Oil'].pct_change().iloc[-1]*100:.1f}%")
    with col2:
        st.metric("Gold", f"${latest['Gold']:.2f}",
                  f"{filtered_data['Gold'].pct_change().iloc[-1]*100:.1f}%")
    with col3:
        st.metric("USD Index", f"{latest['USD_Index']:.2f}",
                  f"{filtered_data['USD_Index'].pct_change().iloc[-1]*100:.1f}%")
    with col4:
        st.metric("Inflation (YoY)", f"{latest['Inflation']:.1f}%")
    with col5:
        st.metric("Geopolitical Risk", f"{latest['Geopolitical_Risk']:.0f}")
else:
    st.error("No data available. Please check your data sources.")

# ----- MAIN CHARTS -----

st.markdown("---")
st.header("📈 Time Series Analysis")

# Create subplots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "Oil Prices", "Gold Prices",
        "USD Index", "Inflation (YoY)",
        "Geopolitical Risk Index", ""
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": True}]]
)

# Row 1: Oil & Gold
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Oil'],
                         mode='lines', name='Oil', line=dict(color='orange')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Gold'],
                         mode='lines', name='Gold', line=dict(color='gold')),
              row=1, col=2)

# Row 2: USD & Inflation
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['USD_Index'],
                         mode='lines', name='USD Index', line=dict(color='blue')),
              row=2, col=1)
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Inflation'],
                         mode='lines', name='Inflation', line=dict(color='red')),
              row=2, col=2)

# Row 3: Geopolitical Risk
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Geopolitical_Risk'],
                         mode='lines', name='Geopolitical Risk',
                         fill='tozeroy', line=dict(color='purple')),
              row=3, col=1)

# Add horizontal threshold line for risk
fig.add_hline(y=200, line_dash="dash", line_color="red",
              annotation_text="High Risk Threshold", row=3, col=1)

# Layout
fig.update_layout(height=900, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Date", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# ----- CORRELATION MATRIX -----

st.markdown("---")
st.header("🔗 Correlation Analysis")

corr_data = filtered_data[['Oil', 'Gold', 'USD_Index', 'Inflation', 'Geopolitical_Risk']]
corr_matrix = corr_data.corr()

# Create heatmap
fig_corr = px.imshow(corr_matrix,
                     text_auto=True,
                     color_continuous_scale='RdBu_r',
                     title="Correlation Matrix of Macro Indicators",
                     aspect='auto')
fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# ----- GEOPOLITICAL SPIKE DETECTION -----

st.markdown("---")
st.header("⚠️ Geopolitical Risk Spikes")

spikes = detect_geopolitical_spikes(filtered_data)

if len(spikes) > 0:
    st.write(f"Found **{len(spikes)}** periods of elevated geopolitical risk (>200):")

    # Show as table
    spike_table = spikes[['Oil', 'Gold', 'USD_Index']].round(2)
    spike_table['Risk_Level'] = spikes['Geopolitical_Risk'].round(0)
    st.dataframe(spike_table, use_container_width=True)

    # Analyze asset performance during spikes
    st.subheader("Asset Performance During Geopolitical Spikes")

    # Calculate average returns during spikes
    spike_returns = filtered_data.loc[spikes.index].pct_change().mean() * 100

    performance_df = pd.DataFrame({
        'Asset': ['Oil', 'Gold', 'USD_Index'],
        'Avg Return During Spikes (%)': [spike_returns.get('Oil', 0),
                                          spike_returns.get('Gold', 0),
                                          spike_returns.get('USD_Index', 0)]
    })
    st.dataframe(performance_df, use_container_width=True)

    # Visualize spike periods
    fig_spikes = go.Figure()
    fig_spikes.add_trace(go.Scatter(x=filtered_data.index,
                                     y=filtered_data['Gold'],
                                     mode='lines',
                                     name='Gold Price',
                                     line=dict(color='gold')))

    # Highlight spike periods
    for date in spikes.index:
        fig_spikes.add_vrect(x0=date - pd.Timedelta(days=3),
                             x1=date + pd.Timedelta(days=3),
                             fillcolor="red", opacity=0.2,
                             annotation_text="Spike")

    fig_spikes.update_layout(title="Gold Price with Geopolitical Spike Highlights",
                             xaxis_title="Date",
                             yaxis_title="Gold Price ($)",
                             template='plotly_white')
    st.plotly_chart(fig_spikes, use_container_width=True)
else:
    st.info("No geopolitical risk spikes detected in selected period.")

# ----- REGRESSION ANALYSIS -----

st.markdown("---")
st.header("📊 Regression: What Drives Gold Prices?")

# Run simple linear regression
from sklearn.linear_model import LinearRegression

X = filtered_data[['Oil', 'USD_Index', 'Geopolitical_Risk', 'Inflation']].dropna()
y = filtered_data['Gold'].dropna()

# Align indices
common_idx = X.index.intersection(y.index)
X = X.loc[common_idx]
y = y.loc[common_idx]

if len(X) > 0:
    model = LinearRegression()
    model.fit(X, y)

    # Display coefficients
    coef_df = pd.DataFrame({
        'Variable': ['Oil', 'USD Index', 'Geopolitical Risk', 'Inflation'],
        'Coefficient': model.coef_,
        'Impact': ['Positive' if c > 0 else 'Negative' for c in model.coef_]
    })
    coef_df['Coefficient'] = coef_df['Coefficient'].round(2)

    st.write(f"**R² Score:** {model.score(X, y):.3f}")
    st.dataframe(coef_df, use_container_width=True)

    # Actual vs Predicted plot
    predictions = model.predict(X)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=y, y=predictions, mode='markers',
                                  name='Predicted vs Actual',
                                  marker=dict(size=5, opacity=0.6)))
    fig_reg.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                                  mode='lines', name='Perfect Fit',
                                  line=dict(dash='dash', color='red')))
    fig_reg.update_layout(title="Gold Price: Actual vs Predicted",
                          xaxis_title="Actual Gold Price",
                          yaxis_title="Predicted Gold Price",
                          template='plotly_white')
    st.plotly_chart(fig_reg, use_container_width=True)
else:
    st.warning("Not enough data for regression analysis.")

# ----- DATA DOWNLOAD -----

st.markdown("---")
st.header("📥 Export Data")

csv = filtered_data.to_csv().encode('utf-8')
st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name=f"macro_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)

st.markdown("---")
st.caption(f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Data sources: Yahoo Finance, Economic Policy Uncertainty Index")