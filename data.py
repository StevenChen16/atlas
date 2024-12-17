import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def calculate_connors_rsi(data, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI"""
    # Component 1: Regular RSI on price changes
    price_rsi = pd.Series(ta.RSI(data['Close'].values, timeperiod=rsi_period), index=data.index)
    
    # Component 2: Streak RSI
    daily_returns = data['Close'].diff()
    streak = pd.Series(0.0, index=data.index)
    streak_count = 0.0
    
    for i in range(1, len(data)):
        if daily_returns.iloc[i] > 0:
            if streak_count < 0:
                streak_count = 1.0
            else:
                streak_count += 1.0
        elif daily_returns.iloc[i] < 0:
            if streak_count > 0:
                streak_count = -1.0
            else:
                streak_count -= 1.0
        else:
            streak_count = 0.0
        streak.iloc[i] = streak_count
    
    streak_values = streak.values.astype(np.float64)
    streak_rsi = pd.Series(ta.RSI(streak_values, timeperiod=streak_period), index=data.index)
    
    # Component 3: Percentage Rank (ROC)
    def percent_rank(series, period):
        return series.rolling(period).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / float(period) * 100,
            raw=True
        )
    
    pct_rank = percent_rank(data['Close'], rank_period)
    
    # Combine components with equal weighting
    crsi = (price_rsi + streak_rsi + pct_rank) / 3.0
    return crsi

def apply_kalman_filter(data, measurement_noise=0.1, process_noise=0.01):
    """Apply Kalman filter to price series"""
    prices = data['Close'].values
    state = np.array([prices[0], 0])
    P = np.array([[1, 0], [0, 1]])
    
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[process_noise/10, 0], [0, process_noise]])
    R = np.array([[measurement_noise]])
    
    filtered_prices = []
    trends = []
    
    for price in prices:
        # Predict
        state = np.dot(F, state)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        # Update
        y = price - np.dot(H, state)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        
        state = state + np.dot(K, y)
        P = P - np.dot(np.dot(K, H), P)
        
        filtered_prices.append(state[0])
        trends.append(state[1])
    
    return pd.Series(filtered_prices, index=data.index), pd.Series(trends, index=data.index)

def apply_fft_filter(data, cutoff_period):
    """Apply FFT filtering with specified cutoff period"""
    prices = data['Close'].values.astype(np.float64)
    n = len(prices)
    
    # Detrend the prices to reduce edge effects
    trend = np.linspace(prices[0], prices[-1], n)
    detrended = prices - trend
    
    # Perform FFT
    fft_result = fft(detrended)
    freqs = fftfreq(n, d=1)
    
    # Create low-pass filter
    filter_threshold = 1/cutoff_period
    filter_mask = np.abs(freqs) < filter_threshold
    fft_result_filtered = fft_result * filter_mask
    
    # Inverse FFT and add trend back
    filtered_detrended = np.real(ifft(fft_result_filtered))
    filtered_prices = filtered_detrended + trend
    
    return pd.Series(filtered_prices, index=data.index)

def download_and_prepare_data(symbol, start_date, end_date):
    # Download stock data
    stock = yf.download(symbol, start=start_date, end=end_date)

    if isinstance(stock.columns, pd.MultiIndex):
        stock = stock.xs(symbol, level=1, axis=1) if symbol in stock.columns.levels[1] else stock
    
    # Ensure we have data
    if stock.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    
    # Convert price columns to float64
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stock = stock.astype({col: np.float64 for col in price_columns})
    
    # Create a copy to avoid SettingWithCopyWarning
    stock = stock.copy()
    
    # Calculate basic technical indicators
    stock['MA5'] = ta.SMA(stock['Close'].values, timeperiod=5)
    stock['MA20'] = ta.SMA(stock['Close'].values, timeperiod=20)
    stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = ta.MACD(stock['Close'].values)
    stock['RSI'] = ta.RSI(stock['Close'].values)
    stock['Upper'], stock['Middle'], stock['Lower'] = ta.BBANDS(stock['Close'].values)
    stock['Volume_MA5'] = ta.SMA(stock['Volume'].values, timeperiod=5)
    
    # Add Connors RSI
    stock['CRSI'] = calculate_connors_rsi(stock)
    
    # Add Kalman Filter estimates
    stock['Kalman_Price'], stock['Kalman_Trend'] = apply_kalman_filter(stock)
    
    # Add FFT filtered prices
    stock['FFT_21'] = apply_fft_filter(stock, 21)
    stock['FFT_63'] = apply_fft_filter(stock, 63)
    
    # Forward fill any remaining NaN values from indicators
    stock = stock.fillna(method='ffill')
    
    # Backward fill any remaining NaN values at the beginning
    stock = stock.fillna(method='bfill')
    
    # Scale data
    scaler = StandardScaler()
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 
                     'MA5', 'MA20', 'MACD', 'RSI', 'Upper', 'Lower',
                     'Volume_MA5', 'Kalman_Price', 'FFT_21', 'FFT_63']
    
    # Ensure all columns exist before scaling
    missing_cols = [col for col in cols_to_scale if col not in stock.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    stock[cols_to_scale] = scaler.fit_transform(stock[cols_to_scale])
    
    # Don't scale percentage-based indicators
    stock['CRSI'] = stock['CRSI'].clip(0, 100)
    
    return stock

if __name__ == "__main__":
    # Test code with error handling
    try:
        data = download_and_prepare_data('AAPL', '2019-01-01', '2024-01-01')
        print("\nFirst few rows:")
        print(data.head())
        print("\nData shape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        
        # # Plot some of the new indicators
        # plt.figure(figsize=(15, 10))
        
        # # Plot 1: Price and filtered versions
        # plt.subplot(3, 1, 1)
        # plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
        # plt.plot(data.index, data['Kalman_Price'], label='Kalman Filtered')
        # plt.plot(data.index, data['FFT_21'], label='FFT 21-day filter')
        # plt.legend()
        # plt.title('Price Comparison')
        
        # # Plot 2: Connors RSI
        # plt.subplot(3, 1, 2)
        # plt.plot(data.index, data['CRSI'], label='Connors RSI')
        # plt.axhline(y=80, color='r', linestyle='--')
        # plt.axhline(y=20, color='g', linestyle='--')
        # plt.legend()
        # plt.title('Connors RSI')
        
        # # Plot 3: Kalman Trend
        # plt.subplot(3, 1, 3)
        # plt.plot(data.index, data['Kalman_Trend'], label='Kalman Trend')
        # plt.axhline(y=0, color='r', linestyle='--')
        # plt.legend()
        # plt.title('Kalman Trend')
        
        # plt.tight_layout()
        # plt.show()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")