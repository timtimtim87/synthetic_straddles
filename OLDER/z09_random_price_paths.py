import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

def generate_geometric_brownian_motion_path(S0, target_return, days, annual_vol=0.25, dt=1/365, seed=None):
    """
    Generate a realistic random walk (Geometric Brownian Motion) that ends at a target price
    
    Parameters:
    - S0: Initial stock price
    - target_return: Target percentage change (e.g., 0.2 for 20% increase)
    - days: Number of days in the path
    - annual_vol: Annual volatility (default 25%)
    - dt: Time step (daily = 1/365)
    - seed: Random seed for reproducibility
    
    Returns:
    - Array of stock prices following GBM
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate required drift to hit target
    target_price = S0 * (1 + target_return)
    
    # We need to solve for mu such that E[S_T] = target_price
    # For GBM: E[S_T] = S0 * exp(mu * T)
    # So: mu = ln(target_price / S0) / T
    T = days * dt
    mu = np.log(target_price / S0) / T
    
    # Generate random walks
    n_steps = days
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    # Initialize price array
    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    
    # Generate GBM path
    for i in range(n_steps):
        prices[i + 1] = prices[i] * np.exp((mu - 0.5 * annual_vol**2) * dt + annual_vol * dW[i])
    
    # Adjust final price to exactly hit target (small correction)
    adjustment_factor = target_price / prices[-1]
    prices[-1] = target_price
    
    # Smoothly adjust the last few prices to make the ending more natural
    adjustment_window = min(10, len(prices) // 10)
    for i in range(adjustment_window):
        weight = (i + 1) / adjustment_window
        idx = -(adjustment_window - i)
        prices[idx] *= (1 + weight * (adjustment_factor - 1))
    
    return prices

def generate_realistic_price_paths(initial_spot, days_to_expiry, price_changes, annual_vol=0.25, n_simulations=1):
    """
    Generate realistic random walk price paths for different target returns
    """
    
    paths = {}
    
    for i, pct_change in enumerate(price_changes):
        # Use different seeds for different paths to ensure variety
        seed = 42 + i if n_simulations == 1 else None
        
        path_prices = generate_geometric_brownian_motion_path(
            S0=initial_spot,
            target_return=pct_change,
            days=days_to_expiry,
            annual_vol=annual_vol,
            seed=seed
        )
        
        # Create days array (365 down to 0)
        days = np.arange(days_to_expiry, -1, -1)
        
        paths[f"{pct_change*100:+.0f}%"] = {
            'days_to_expiry': days,
            'spot_prices': path_prices
        }
    
    return paths

def calculate_straddle_analysis_random_walk(initial_spot, strike_multiplier=1.25, 
                                          price_changes=None, iv=0.25, risk_free_rate=0.04,
                                          days_to_expiry=365, annual_vol=0.25):
    """
    Calculate straddle values for realistic random walk price paths
    """
    
    if price_changes is None:
        price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    # Calculate strike price
    strike_price = initial_spot * strike_multiplier
    
    print(f"ADVANCED STRADDLE ANALYSIS - RANDOM WALK PATHS")
    print("="*70)
    print(f"Initial Spot Price: ${initial_spot:.2f}")
    print(f"Strike Price: ${strike_price:.2f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"Initial Moneyness: {initial_spot/strike_price:.3f}")
    print(f"Implied Volatility: {iv*100:.0f}%")
    print(f"Stock Volatility: {annual_vol*100:.0f}% (for random walk)")
    print(f"Risk-free Rate: {risk_free_rate*100:.1f}%")
    print(f"Analysis Period: {days_to_expiry} days")
    print("="*70)
    
    # Generate realistic price paths
    price_paths = generate_realistic_price_paths(
        initial_spot, days_to_expiry, price_changes, annual_vol
    )
    
    results = {}
    
    for path_name, path_data in price_paths.items():
        days = path_data['days_to_expiry']
        spots = path_data['spot_prices']
        
        straddle_values = []
        moneyness_values = []
        call_values = []
        put_values = []
        short_returns = []
        
        for day, spot in zip(days, spots):
            # Convert days to years for Black-Scholes
            time_to_expiry = max(day / 365.0, 1/365)
            
            # Calculate option values
            call_val = black_scholes_call(spot, strike_price, time_to_expiry, risk_free_rate, iv)
            put_val = black_scholes_put(spot, strike_price, time_to_expiry, risk_free_rate, iv)
            straddle_val = call_val + put_val
            moneyness = spot / strike_price
            
            straddle_values.append(straddle_val)
            moneyness_values.append(moneyness)
            call_values.append(call_val)
            put_values.append(put_val)
        
        # Calculate short straddle returns
        initial_straddle = straddle_values[0]
        for straddle_val in straddle_values:
            pnl_percent = ((initial_straddle - straddle_val) / initial_straddle) * 100
            short_returns.append(pnl_percent)
        
        results[path_name] = {
            'days_to_expiry': days,
            'spot_prices': spots,
            'straddle_values': np.array(straddle_values),
            'moneyness': np.array(moneyness_values),
            'call_values': np.array(call_values),
            'put_values': np.array(put_values),
            'short_returns': np.array(short_returns),
            'final_spot': spots[-1],
            'final_straddle': straddle_values[-1],
            'initial_straddle': straddle_values[0],
            'total_decay': straddle_values[0] - straddle_values[-1],
            'final_return': short_returns[-1],
            'max_return': max(short_returns),
            'min_return': min(short_returns),
            'price_volatility': np.std(np.diff(np.log(spots))) * np.sqrt(365) * 100  # Realized vol
        }
        
        print(f"\n{path_name} Random Walk Path:")
        print(f"  Target/Final Spot: ${spots[-1]:.2f}")
        print(f"  Realized Volatility: {results[path_name]['price_volatility']:.1f}%")
        print(f"  Final Moneyness: {moneyness_values[-1]:.3f}")
        print(f"  Short Straddle Return: {short_returns[-1]:+.1f}%")
        print(f"  Max Return: {max(short_returns):+.1f}%")
        print(f"  Min Return: {min(short_returns):+.1f}%")
    
    return results, strike_price

def plot_comprehensive_straddle_analysis(results, strike_price, initial_spot, save_plots=True):
    """
    Create comprehensive plots of the random walk straddle analysis
    """
    
    # Set up the plot style
    plt.style.use('default')
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Create figure with subplots - 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Advanced Straddle Analysis - Random Walk Price Paths', fontsize=16, fontweight='bold')
    
    # Plot 1: Random walk price paths
    ax1 = axes[0, 0]
    for i, (path_name, data) in enumerate(results.items()):
        ax1.plot(data['days_to_expiry'], data['spot_prices'], 
                label=path_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.axhline(y=strike_price, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Strike Price')
    ax1.axhline(y=initial_spot, color='black', linestyle=':', alpha=0.7, linewidth=1, label='Initial Spot')
    ax1.set_xlabel('Days to Expiry')
    ax1.set_ylabel('Spot Price ($)')
    ax1.set_title('Random Walk Price Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Straddle values over time
    ax2 = axes[0, 1]
    for i, (path_name, data) in enumerate(results.items()):
        ax2.plot(data['days_to_expiry'], data['straddle_values'], 
                label=path_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Days to Expiry')
    ax2.set_ylabel('Straddle Value ($)')
    ax2.set_title('Straddle Value Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Short straddle returns over time
    ax3 = axes[0, 2]
    for i, (path_name, data) in enumerate(results.items()):
        ax3.plot(data['days_to_expiry'], data['short_returns'], 
                label=path_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='Breakeven')
    ax3.fill_between([0, 365], -100, 0, alpha=0.1, color='red', label='Loss Zone')
    ax3.fill_between([0, 365], 0, 200, alpha=0.1, color='green', label='Profit Zone')
    ax3.set_xlabel('Days to Expiry')
    ax3.set_ylabel('Short Straddle Return (%)')
    ax3.set_title('Short Straddle P&L Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Plot 4: Moneyness evolution
    ax4 = axes[1, 0]
    for i, (path_name, data) in enumerate(results.items()):
        ax4.plot(data['days_to_expiry'], data['moneyness'], 
                label=path_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM')
    ax4.fill_between([0, 365], 0.95, 1.05, alpha=0.2, color='red', label='Near ATM')
    ax4.set_xlabel('Days to Expiry')
    ax4.set_ylabel('Moneyness (S/K)')
    ax4.set_title('Moneyness Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    # Plot 5: Straddle value vs moneyness
    ax5 = axes[1, 1]
    for i, (path_name, data) in enumerate(results.items()):
        ax5.plot(data['moneyness'], data['straddle_values'], 
                label=path_name, color=colors[i], linewidth=2, marker='o', markersize=2, alpha=0.7)
    
    ax5.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM')
    ax5.set_xlabel('Moneyness (S/K)')
    ax5.set_ylabel('Straddle Value ($)')
    ax5.set_title('Straddle Value vs Moneyness')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final returns and volatility comparison
    ax6 = axes[1, 2]
    path_names = list(results.keys())
    final_returns = [results[path]['final_return'] for path in path_names]
    realized_vols = [results[path]['price_volatility'] for path in path_names]
    
    scatter = ax6.scatter(realized_vols, final_returns, c=range(len(path_names)), 
                         cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    
    for i, (name, ret, vol) in enumerate(zip(path_names, final_returns, realized_vols)):
        ax6.annotate(name, (vol, ret), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax6.set_xlabel('Realized Volatility (%)')
    ax6.set_ylabel('Final Return (%)')
    ax6.set_title('Return vs Realized Volatility')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'advanced_straddle_random_walk_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comprehensive plot saved: {filename}")
    
    plt.show()

def plot_individual_return_paths(results, strike_price, initial_spot, save_plots=True):
    """
    Create individual return plots for each random walk path
    """
    
    plt.style.use('default')
    
    # Calculate grid dimensions
    n_paths = len(results)
    n_cols = 3
    n_rows = (n_paths + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle(f'Short Straddle Returns - Random Walk Paths\n(Strike: ${strike_price:.0f}, Initial Spot: ${initial_spot:.0f})', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easy indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results)))
    
    for i, (path_name, data) in enumerate(results.items()):
        ax = axes_flat[i]
        
        # Plot return curve
        ax.plot(data['days_to_expiry'], data['short_returns'], 
               color=colors[i], linewidth=3, alpha=0.8)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axhline(y=data['final_return'], color=colors[i], linestyle='--', alpha=0.7, linewidth=2)
        
        # Fill areas
        ax.fill_between(data['days_to_expiry'], data['short_returns'], 0,
                       where=(data['short_returns'] >= 0), alpha=0.3, color='green')
        ax.fill_between(data['days_to_expiry'], data['short_returns'], 0,
                       where=(data['short_returns'] < 0), alpha=0.3, color='red')
        
        # Formatting
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'{path_name} Random Walk\nFinal Return: {data["final_return"]:+.1f}%', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Stats box
        stats_text = f'Max: {data["max_return"]:+.1f}%\nMin: {data["min_return"]:+.1f}%\nVol: {data["price_volatility"]:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Highlight final point
        ax.scatter([0], [data['final_return']], color=colors[i], s=100, zorder=5, 
                  edgecolor='black', linewidth=2)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'random_walk_straddle_returns_by_path.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Individual returns plot saved: {filename}")
    
    plt.show()

def main():
    """Main function to run the advanced analysis"""
    
    # Parameters
    initial_spot = 100.0
    strike_multiplier = 1.25  # 25% above spot
    price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    iv = 0.25  # Options IV
    annual_vol = 0.20  # Consistent stock volatility (20% for all paths)
    risk_free_rate = 0.04
    days_to_expiry = 365
    
    # Run the analysis
    results, strike_price = calculate_straddle_analysis_random_walk(
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        price_changes=price_changes,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry,
        annual_vol=annual_vol
    )
    
    # Create comprehensive plots
    plot_comprehensive_straddle_analysis(results, strike_price, initial_spot)
    
    # Create individual return plots
    plot_individual_return_paths(results, strike_price, initial_spot)
    
    # Save detailed data
    detailed_results = []
    for path_name, data in results.items():
        for i in range(len(data['days_to_expiry'])):
            detailed_results.append({
                'Price_Path': path_name,
                'Days_to_Expiry': data['days_to_expiry'][i],
                'Spot_Price': data['spot_prices'][i],
                'Straddle_Value': data['straddle_values'][i],
                'Short_Return_Percent': data['short_returns'][i],
                'Moneyness': data['moneyness'][i],
                'Call_Value': data['call_values'][i],
                'Put_Value': data['put_values'][i]
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('advanced_random_walk_straddle_data.csv', index=False)
    
    print(f"\n💾 Detailed data saved to: advanced_random_walk_straddle_data.csv")
    print(f"🎉 Advanced analysis complete!")

if __name__ == "__main__":
    main()