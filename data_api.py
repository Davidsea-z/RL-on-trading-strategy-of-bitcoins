import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Tuple, Union
import time
from urllib3.exceptions import MaxRetryError, SSLError
from requests.exceptions import RequestException

class BitcoinDataAPI:
    """API for fetching and processing Bitcoin price data using Yahoo Finance."""
    
    def __init__(self):
        self.symbol = 'BTC-USD'
    
    def fetch_historical_data(self, 
                            period: str = '1y', 
                            interval: str = '1d',
                            max_retries: int = 3,
                            initial_delay: float = 1.0) -> pd.DataFrame:
        """Fetch historical Bitcoin price data with retry mechanism.
        
        Args:
            period: Time period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            DataFrame with OHLCV data
        """
        attempt = 0
        delay = initial_delay
        
        while attempt < max_retries:
            try:
                print(f"Fetching {self.symbol} data for period: {period}, interval: {interval}...")
                print(f"Attempt {attempt + 1} of {max_retries}")
                
                data = yf.download(
                    tickers=self.symbol,
                    period=period,
                    interval=interval,
                    progress=False
                )
                
                if not data.empty:
                    print(f"Successfully retrieved {len(data)} data points")
                    return data
                
                print(f"Warning: No data retrieved for {self.symbol}. Retrying...")
                
            except (SSLError, MaxRetryError) as e:
                print(f"SSL/Connection error: {str(e)}")
            except RequestException as e:
                print(f"Network error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            
            attempt += 1
        
        print("Max retries exceeded. Please check your internet connection and try again.")
        return pd.DataFrame()
    
    def fetch_realtime_data(self, lookback_hours: int = 24) -> pd.DataFrame:
        """Fetch near real-time Bitcoin price data.
        
        Args:
            lookback_hours: Number of hours of historical data to include
            
        Returns:
            DataFrame with OHLCV data at 5-minute intervals
        """
        return self.fetch_historical_data(
            period=f"{lookback_hours}h",
            interval='5m'
        )
    
    def prepare_data_for_trading(self, 
                               data: Optional[pd.DataFrame] = None, 
                               period: str = '1y',
                               interval: str = '1d') -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare data for the trading environment.
        
        Args:
            data: Optional pre-fetched DataFrame. If None, will fetch new data
            period: Time period if fetching new data
            interval: Data interval if fetching new data
            
        Returns:
            Tuple of (price array, full DataFrame with indicators)
        """
        if data is None:
            data = self.fetch_historical_data(period, interval)
            
        if data.empty:
            raise ValueError("No data available for processing")
            
        # Extract closing prices as numpy array and ensure 1D
        prices = data['Close'].values.flatten()
        
        return prices, data
    
    def save_data(self, 
                  data: pd.DataFrame, 
                  filename: str = 'bitcoin_prices.csv') -> bool:
        """Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data.to_csv(filename)
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False

# Example usage
if __name__ == '__main__':
    api = BitcoinDataAPI()
    
    print("\nFetching Bitcoin Data...\n")
    
    # Fetch real-time data for the last 24 hours
    rt_data = api.fetch_realtime_data()
    
    if not rt_data.empty:
        print("Latest Bitcoin Data Points:")
        print("-" * 80)
        print("Timestamp\t\t\tPrice (USD)\t\tVolume\t\tHigh\t\tLow")
        print("-" * 80)
        
        # Display last 5 data points in a formatted list
        for idx, (timestamp, row) in enumerate(rt_data.tail().iterrows()):
            print(f"{timestamp}\t{float(row['Close']):,.2f}\t\t{int(row['Volume']):,.0f}\t{float(row['High']):,.2f}\t{float(row['Low']):,.2f}")
    else:
        print("No data available. Please check your internet connection.")
    
    print("\nTotal data points available:", len(rt_data))

    # Fetch and prepare historical data
    prices, data = api.prepare_data_for_trading()
    print(f"\nLoaded {len(prices)} historical price points")