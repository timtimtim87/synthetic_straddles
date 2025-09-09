import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Black-Scholes functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    if K <= 0:
        return S
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call_price, 0)

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    if K <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put_price, 0)

def calculate_straddle_value(S, K, T, r, sigma):
    """Calculate straddle value (call + put)"""
    if K <= 0 or S <= 0:
        return 0
    
    call_value = black_scholes_call(S, K, T, r, sigma)
    put_value = black_scholes_put(S, K, T, r, sigma)
    return call_value + put_value

def generate_price_path_returns(initial_spot, strike_multiplier=1.25, 
                              price_changes=None, iv=0.25, risk_free_rate=0.04,
                              days_to_expiry=365):
    """
    Generate short straddle returns for different linear price paths
    """
    
    if price_changes is None:
        price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    # Calculate strike price
    strike_price = initial_spot * strike_multiplier
    
    print(f"SHORT STRADDLE RETURNS ANALYSIS")
    print("="*60)
    print(f"Initial Spot: ${initial_spot:.2f}")
    print(f"Strike Price: ${strike_price:.2f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"Implied Volatility: {iv*100:.0f}%")
    print(f"Analysis Period: {days_to_expiry} days")
    print("="*60)
    
    results = {}
    
    for pct_change in price_changes:
        # Create linear price path
        days = np.arange(days_to_expiry, -1, -1)  # 365 down to 0
        final_price = initial_spot * (1 + pct_change)
        spot_prices = np.linspace(initial_spot, final_price, len(days))
        
        # Calculate straddle values and returns
        straddle_values = []
        short_returns = []
        
        for day, spot in zip(days, spot_prices):
            time_to_expiry = max(day / 365.0, 1/365)
            straddle_val = calculate_straddle_value(spot, strike_price, time_to_expiry, risk_free_rate, iv)
            straddle_values.append(straddle_val)
        
        # Calculate short straddle returns
        initial_straddle = straddle_values[0]
        for straddle_val in straddle_values:
            pnl_percent = ((initial_straddle - straddle_val) / initial_straddle) * 100
            short_returns.append(pnl_percent)
        
        path_name = f"{pct_change*100:+.0f}%"
        results[path_name] = {
            'days_to_expiry': days,
            'spot_prices': spot_prices,
            'short_returns': np.array(short_returns),
            'final_return': short_returns[-1],
            'max_return': max(short_returns),
            'min_return': min(short_returns)
        }
        
        print(f"{path_name:>5} Path: Final Return {short_returns[-1]:+6.1f}% | Max {max(short_returns):+6.1f}% | Min {min(short_returns):+6.1f}%")
    
    return results, strike_price

def plot_short_straddle_returns(results, strike_price, initial_spot, save_plot=True):
    """
    Create individual return plots for each price path in one image
    """
    
    # Set up plot style
    plt.style.use('default')
    
    # Calculate grid dimensions
    n_paths = len(results)
    n_cols = 3
    n_rows = (n_paths + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle(f'Short Straddle Returns by Price Path\n(Strike: ${strike_price:.0f}, Initial Spot: ${initial_spot:.0f})', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easy indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # Color scheme
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results)))
    
    for i, (path_name, data) in enumerate(results.items()):
        ax = axes_flat[i]
        
        # Plot the return curve
        ax.plot(data['days_to_expiry'], data['short_returns'], 
               color=colors[i], linewidth=3, label=f'{path_name} Price Path')
        
        # Add horizontal reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='Breakeven')
        ax.axhline(y=data['final_return'], color=colors[i], linestyle='--', alpha=0.7, linewidth=2)
        
        # Fill profit/loss areas
        ax.fill_between(data['days_to_expiry'], data['short_returns'], 0,
                       where=(data['short_returns'] >= 0), alpha=0.3, color='green', label='Profit')
        ax.fill_between(data['days_to_expiry'], data['short_returns'], 0,
                       where=(data['short_returns'] < 0), alpha=0.3, color='red', label='Loss')
        
        # Formatting
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'{path_name} Price Movement\nFinal Return: {data["final_return"]:+.1f}%', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Countdown to expiry
        
        # Add key statistics text box
        stats_text = f'Max: {data["max_return"]:+.1f}%\nMin: {data["min_return"]:+.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Highlight final return point
        ax.scatter([0], [data['final_return']], color=colors[i], s=100, zorder=5, 
                  edgecolor='black', linewidth=2)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        filename = 'short_straddle_returns_by_path.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved: {filename}")
    
    plt.show()

def create_returns_summary(results):
    """Create a clean summary of returns by path"""
    
    print(f"\n" + "="*70)
    print("SHORT STRADDLE RETURNS SUMMARY")
    print("="*70)
    
    summary_data = []
    for path_name, data in results.items():
        summary_data.append({
            'Price_Path': path_name,
            'Final_Return_%': round(data['final_return'], 1),
            'Max_Return_%': round(data['max_return'], 1),
            'Min_Return_%': round(data['min_return'], 1),
            'Return_Range_%': round(data['max_return'] - data['min_return'], 1)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find best and worst scenarios
    best_path = max(results.items(), key=lambda x: x[1]['final_return'])
    worst_path = min(results.items(), key=lambda x: x[1]['final_return'])
    
    print(f"\n🎯 BEST SCENARIO: {best_path[0]} path with {best_path[1]['final_return']:+.1f}% final return")
    print(f"📉 WORST SCENARIO: {worst_path[0]} path with {worst_path[1]['final_return']:+.1f}% final return")
    
    return summary_df

def main():
    """Main function to run the analysis"""
    
    # Parameters
    initial_spot = 100.0
    strike_multiplier = 1.25  # 25% above spot
    price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # 0% to 50% increases
    iv = 0.25
    risk_free_rate = 0.04
    days_to_expiry = 365
    
    # Generate returns data
    results, strike_price = generate_price_path_returns(
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        price_changes=price_changes,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry
    )
    
    # Create the plots
    plot_short_straddle_returns(results, strike_price, initial_spot)
    
    # Show summary
    summary_df = create_returns_summary(results)
    
    # Save summary to CSV
    summary_df.to_csv('short_straddle_returns_summary.csv', index=False)
    print(f"\n💾 Summary saved to: short_straddle_returns_summary.csv")
    
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()