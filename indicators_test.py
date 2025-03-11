import numpy as np
import pandas as pd
from indicators import calculate_macd, calculate_rvi, calculate_bollinger_bands, calculate_rsi

# Create sample price data
prices = np.array([100, 102, 104, 103, 105, 107, 108, 107, 106, 105,
                   104, 103, 102, 101, 102, 103, 104, 105, 106, 107])

# Test MACD
print("\nTesting MACD calculation:")
macd_line, signal_line, macd_hist = calculate_macd(prices)
print(f"MACD Line (last 5 values): {macd_line[-5:].round(4)}")
print(f"Signal Line (last 5 values): {signal_line[-5:].round(4)}")
print(f"MACD Histogram (last 5 values): {macd_hist[-5:].round(4)}")

# Test RVI
print("\nTesting RVI calculation:")
rvi = calculate_rvi(prices)
print(f"RVI (last 5 values): {rvi[-5:].round(4)}")

# Test Bollinger Bands
print("\nTesting Bollinger Bands calculation:")
upper_band, lower_band = calculate_bollinger_bands(prices)
print(f"Upper Band (last 5 values): {upper_band[-5:].round(4)}")
print(f"Lower Band (last 5 values): {lower_band[-5:].round(4)}")

# Test RSI
print("\nTesting RSI calculation:")
rsi = calculate_rsi(prices)
print(f"RSI (last 5 values): {rsi[-5:].round(4)}")

# Test prepare_indicators
print("\nTesting prepare_indicators:")
from indicators import prepare_indicators
indicators_dict = prepare_indicators(prices)
print("\nSample of prepared indicators (last entry):")
print(indicators_dict[-1])