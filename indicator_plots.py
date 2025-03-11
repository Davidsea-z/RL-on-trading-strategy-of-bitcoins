import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from indicators import calculate_macd, calculate_rvi, calculate_bollinger_bands, calculate_rsi

def plot_macd(ax, dates, prices, fast=12, slow=26, signal=9):
    """Plot MACD indicator on the given axis."""
    macd_line, signal_line, macd_hist = calculate_macd(prices, fast, slow, signal)
    
    ax.plot(dates, macd_line, label='MACD', color='blue', linewidth=1.5)
    ax.plot(dates, signal_line, label='Signal', color='orange', linewidth=1.5)
    ax.bar(dates, macd_hist, color='gray', alpha=0.2)
    ax.set_title('MACD', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_ylabel('Value')

def plot_rsi(ax, dates, prices, periods=14):
    """Plot RSI indicator on the given axis."""
    rsi = calculate_rsi(prices, periods)
    
    ax.plot(dates, rsi, label='RSI', color='purple', linewidth=1.5)
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(dates, 70, 30, color='gray', alpha=0.1)
    ax.set_title('RSI', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Value')

def plot_rvi(ax, dates, prices, periods=10):
    """Plot RVI indicator on the given axis."""
    rvi = calculate_rvi(prices, periods)
    
    ax.plot(dates, rvi, label='RVI', color='green', linewidth=1.5)
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.7, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(dates, 0.7, -0.7, color='gray', alpha=0.1)
    ax.set_title('RVI', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Value')

def plot_bollinger_bands(ax, dates, prices, window=20, num_std=2):
    """Plot Bollinger Bands on the given axis."""
    upper_band, lower_band = calculate_bollinger_bands(prices, window, num_std)
    
    ax.plot(dates, prices, label='Price', color='blue', alpha=0.7)
    ax.plot(dates, upper_band, 'r--', label='Upper BB', alpha=0.7)
    ax.plot(dates, lower_band, 'r--', label='Lower BB', alpha=0.7)
    ax.fill_between(dates, upper_band, lower_band, color='gray', alpha=0.1)
    ax.set_title('Bollinger Bands', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_ylabel('Price (USD)')

def plot_all_indicators(prices, dates=None):
    """Create a figure with all technical indicators."""
    if dates is None:
        dates = np.arange(len(prices))
    
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
    
    # Plot each indicator
    plot_bollinger_bands(fig.add_subplot(gs[0]), dates, prices)
    plot_macd(fig.add_subplot(gs[1]), dates, prices)
    plot_rsi(fig.add_subplot(gs[2]), dates, prices)
    plot_rvi(fig.add_subplot(gs[3]), dates, prices)
    
    # Format x-axis dates if dates are datetime objects
    if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
        for ax in fig.axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    return fig

def plot_single_indicator(prices, indicator_type, dates=None, **kwargs):
    """Create a single indicator plot.
    
    Args:
        prices: Array of price data
        indicator_type: String indicating which indicator to plot
                       ('macd', 'rsi', 'rvi', or 'bollinger')
        dates: Optional array of dates for x-axis
        **kwargs: Additional parameters for the specific indicator
    """
    if dates is None:
        dates = np.arange(len(prices))
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    if indicator_type.lower() == 'macd':
        plot_macd(ax, dates, prices, **kwargs)
    elif indicator_type.lower() == 'rsi':
        plot_rsi(ax, dates, prices, **kwargs)
    elif indicator_type.lower() == 'rvi':
        plot_rvi(ax, dates, prices, **kwargs)
    elif indicator_type.lower() == 'bollinger':
        plot_bollinger_bands(ax, dates, prices, **kwargs)
    else:
        raise ValueError(f"Unknown indicator type: {indicator_type}")
    
    if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
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
    
    # Plot all indicators
    fig_all = plot_all_indicators(prices, dates)
    plt.show()
    
    # Plot individual indicators
    indicators = ['macd', 'rsi', 'rvi', 'bollinger']
    for ind in indicators:
        fig = plot_single_indicator(prices, ind, dates)
        plt.show()