import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from indicator_plots import plot_macd, plot_rsi, plot_rvi, plot_bollinger_bands
import os

def plot_price_candlesticks(df, output_dir):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    candlestick_data = df[['Open', 'High', 'Low', 'Close']]
    mpf.plot(candlestick_data, type='candle', style='charles',
             ax=ax, volume=False, show_nontrading=True)
    ax.set_title('Bitcoin Price', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price (USD)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitcoin_price_candlesticks.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_volume(df, output_dir):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.bar(df.index, df['Volume'], color='gray', alpha=0.5)
    ax.set_title('Volume', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Volume')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitcoin_volume.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_indicators(df, output_dir):
    prices = df['Close'].values
    
    # Bollinger Bands
    fig_bb = plt.figure(figsize=(12, 6))
    ax_bb = fig_bb.add_subplot(111)
    plot_bollinger_bands(ax_bb, df.index, prices)
    ax_bb.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_bb.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitcoin_bollinger_bands.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # MACD
    fig_macd = plt.figure(figsize=(12, 6))
    ax_macd = fig_macd.add_subplot(111)
    plot_macd(ax_macd, df.index, prices)
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_macd.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitcoin_macd.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # RSI
    fig_rsi = plt.figure(figsize=(12, 6))
    ax_rsi = fig_rsi.add_subplot(111)
    plot_rsi(ax_rsi, df.index, prices)
    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_rsi.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitcoin_rsi.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # RVI
    fig_rvi = plt.figure(figsize=(12, 6))
    ax_rvi = fig_rvi.add_subplot(111)
    plot_rvi(ax_rvi, df.index, prices)
    ax_rvi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_rvi.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitcoin_rvi.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_bitcoin_analysis(csv_file, output_dir='bitcoin_plots'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file, skipping metadata rows and using the first row as column names
    df = pd.read_csv(csv_file, skiprows=[1, 2])
    
    # Convert the Date column to datetime and set it as index
    df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
    
    # Select only the OHLCV columns and ensure correct order
    df = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    
    # Plot and save each component
    plot_price_candlesticks(df, output_dir)
    plot_volume(df, output_dir)
    plot_indicators(df, output_dir)
    
    print(f"All plots have been saved to the '{output_dir}' directory.")

if __name__ == '__main__':
    # Example usage
    csv_file = 'bitcoin_data_1y_1d.csv'
    plot_bitcoin_analysis(csv_file)