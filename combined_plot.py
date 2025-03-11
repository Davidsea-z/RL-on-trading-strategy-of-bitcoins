import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from indicators import calculate_macd, calculate_rvi, calculate_bollinger_bands, calculate_rsi

def plot_combined_indicators(prices, dates=None, volume=None):
    """Create a combined plot with all indicators in a 2x3 matrix layout.
    
    Args:
        prices: Array of price data
        dates: Optional array of dates for x-axis
        volume: Optional array of volume data
    """
    if dates is None:
        dates = np.arange(len(prices))
    
    # Create figure and grid specification
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    
    # Top row: Price, Volume, and Bollinger Bands
    # Price plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dates, prices, label='Price', color='blue', linewidth=1.5)
    ax1.set_title('Price', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Price (USD)')
    
    # Volume plot
    ax2 = fig.add_subplot(gs[0, 1])
    if volume is not None:
        ax2.bar(dates, volume, color='gray', alpha=0.5)
    else:
        # If no volume data, show a message
        ax2.text(0.5, 0.5, 'No Volume Data Available',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax2.transAxes)
    ax2.set_title('Volume', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Volume')
    
    # Bollinger Bands plot
    ax3 = fig.add_subplot(gs[0, 2])
    upper_band, lower_band = calculate_bollinger_bands(prices)
    ax3.plot(dates, prices, label='Price', color='blue', alpha=0.7)
    ax3.plot(dates, upper_band, 'r--', label='Upper BB', alpha=0.7)
    ax3.plot(dates, lower_band, 'r--', label='Lower BB', alpha=0.7)
    ax3.fill_between(dates, upper_band, lower_band, color='gray', alpha=0.1)
    ax3.set_title('Bollinger Bands', pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3.set_ylabel('Price (USD)')
    
    # Bottom row: MACD, RSI, and RVI
    # MACD plot
    ax4 = fig.add_subplot(gs[1, 0])
    macd_line, signal_line, macd_hist = calculate_macd(prices)
    ax4.plot(dates, macd_line, label='MACD', color='blue', linewidth=1.5)
    ax4.plot(dates, signal_line, label='Signal', color='orange', linewidth=1.5)
    ax4.bar(dates, macd_hist, color='gray', alpha=0.2)
    ax4.set_title('MACD', pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    ax4.set_ylabel('Value')
    
    # RSI plot
    ax5 = fig.add_subplot(gs[1, 1])
    rsi = calculate_rsi(prices)
    ax5.plot(dates, rsi, label='RSI', color='purple', linewidth=1.5)
    ax5.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax5.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    ax5.fill_between(dates, 70, 30, color='gray', alpha=0.1)
    ax5.set_title('RSI', pad=20)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper left')
    ax5.set_ylim(0, 100)
    ax5.set_ylabel('Value')
    
    # RVI plot
    ax6 = fig.add_subplot(gs[1, 2])
    rvi = calculate_rvi(prices)
    ax6.plot(dates, rvi, label='RVI', color='green', linewidth=1.5)
    ax6.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
    ax6.axhline(y=-0.7, color='r', linestyle='--', alpha=0.5)
    ax6.fill_between(dates, 0.7, -0.7, color='gray', alpha=0.1)
    ax6.set_title('RVI', pad=20)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper left')
    ax6.set_ylim(-1, 1)
    ax6.set_ylabel('Value')
    
    # Format x-axis dates if dates are datetime objects
    if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    n_points = 100
    prices = np.cumsum(np.random.randn(n_points)) + 100
    dates = pd.date_range(start='2024-01-01', periods=n_points)
    volume = np.random.randint(1000, 5000, size=n_points)
    
    # Create and show the combined plot
    fig = plot_combined_indicators(prices, dates, volume)
    plt.show()