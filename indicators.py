import numpy as np
import pandas as pd

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence/Divergence) values"""
    # Ensure prices is 1-dimensional
    if isinstance(prices, np.ndarray):
        prices = prices.flatten()
    prices = pd.Series(prices)
    
    # Calculate the fast and slow exponential moving averages
    # Initialize with SMA for better accuracy
    fast_sma = prices.rolling(window=fast, min_periods=1).mean()
    slow_sma = prices.rolling(window=slow, min_periods=1).mean()
    
    # Calculate EMAs using SMA as initial values
    fast_ema = prices.copy()
    slow_ema = prices.copy()
    
    # EMA multipliers
    fast_multiplier = 2 / (fast + 1)
    slow_multiplier = 2 / (slow + 1)
    
    # Calculate EMAs
    for i in range(len(prices)):
        if i == 0:
            fast_ema[i] = fast_sma[i]
            slow_ema[i] = slow_sma[i]
        else:
            fast_ema[i] = (prices[i] - fast_ema[i-1]) * fast_multiplier + fast_ema[i-1]
            slow_ema[i] = (prices[i] - slow_ema[i-1]) * slow_multiplier + slow_ema[i-1]
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line with initial SMA
    signal_sma = macd_line.rolling(window=signal, min_periods=1).mean()
    signal_line = macd_line.copy()
    signal_multiplier = 2 / (signal + 1)
    
    # Calculate signal EMA
    for i in range(len(macd_line)):
        if i == 0:
            signal_line[i] = signal_sma[i]
        else:
            signal_line[i] = (macd_line[i] - signal_line[i-1]) * signal_multiplier + signal_line[i-1]
    
    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_rvi(prices, periods=10):
    """Calculate RVI (Relative Vigor Index)"""
    prices = pd.Series(prices)
    opens = prices.shift(1)  # Previous close as open
    highs = prices.rolling(2).max()
    lows = prices.rolling(2).min()
    closes = prices
    
    # Calculate numerator (close - open)
    num = (closes - opens + 2 * (closes.shift(1) - opens.shift(1)) +
           2 * (closes.shift(2) - opens.shift(2)) + (closes.shift(3) - opens.shift(3))) / 6
    
    # Calculate denominator (high - low)
    den = (highs - lows + 2 * (highs.shift(1) - lows.shift(1)) +
           2 * (highs.shift(2) - lows.shift(2)) + (highs.shift(3) - lows.shift(3))) / 6
    
    # Calculate RVI
    rvi = num.rolling(window=periods, min_periods=periods).mean() / \
          den.rolling(window=periods, min_periods=periods).mean()
    
    return rvi

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands for given price data.
    
    Args:
        prices: Array of price data
        window: Moving average window size
        num_std: Number of standard deviations for bands
    
    Returns:
        Tuple of (upper_band, lower_band)
    """
    prices = np.array(prices).flatten()
    prices_series = pd.Series(prices)
    
    # Calculate rolling mean and standard deviation with min_periods=1
    rolling_mean = prices_series.rolling(window=window, min_periods=1).mean()
    rolling_std = prices_series.rolling(window=window, min_periods=1).std()
    
    # For the first window-1 periods, use expanding window statistics
    for i in range(window - 1):
        if i > 0:  # Skip the first point as it's already handled by min_periods=1
            rolling_mean[i] = prices_series[:i+1].mean()
            rolling_std[i] = prices_series[:i+1].std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_rsi(prices, periods=14):
    """Calculate RSI (Relative Strength Index).
    
    Args:
        prices: Array of price data
        periods: Lookback period for RSI calculation
    
    Returns:
        RSI values as pandas Series
    """
    prices = pd.Series(prices)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_indicators(price_data):
    """Prepare both MACD and RVI indicators for the given price data"""
    prices = np.array(price_data)
    
    # Calculate MACD
    macd_line, signal_line, macd_hist = calculate_macd(prices)
    
    # Calculate RVI
    rvi = calculate_rvi(prices)
    
    # Combine indicators
    indicators = pd.DataFrame({
        'price': prices,
        'macd': macd_line,
        'signal': signal_line,
        'macd_hist': macd_hist,
        'rvi': rvi
    })
    
    # Forward fill NaN values using the newer method
    indicators = indicators.ffill()
    
    # Convert to dictionary format for the environment
    return indicators.to_dict('records')