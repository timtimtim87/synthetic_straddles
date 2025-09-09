import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
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

def generate_linear_price_paths(initial_spot, days_to_expiry, price_changes):
    """
    Generate linear price paths from current spot to various percentage changes
    
    Parameters:
    - initial_spot: Starting spot price (365 DTE)
    - days_to_expiry: Number of days in the path (365)
    - price_changes: List of percentage changes (e.g., [0.1, 0.2, 0.3] for 10%, 20%, 30%)
    
    Returns:
    - Dictionary of price paths
    """
    
    paths = {}
    
    # Create daily steps
    days = np.arange(days_to_expiry, -1, -1)  # From 365 down to 0
    
    for pct_change in price_changes:
        # Calculate final price
        final_price = initial_spot * (1 + pct_change)
        
        # Create linear interpolation from initial to final
        price_path = np.linspace(initial_spot, final_price, len(days))
        
        paths[f"{pct_change*100:+.0f}%"] = {
            'days_to_expiry': days,
            'spot_prices': price_path
        }
    
    return paths

def calculate_theoretical_straddle_decay(initial_spot, strike_multiplier=1.25, 
                                       price_changes=None, iv=0.25, risk_free_rate=0.04,
                                       days_to_expiry=365):
    """
    Calculate theoretical straddle values for different linear price paths
    """
    
    if price_changes is None:
        price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # 0% to 50% increases
    
    # Calculate strike price (25% above initial spot)
    strike_price = initial_spot * strike_multiplier
    
    print(f"THEORETICAL STRADDLE DECAY ANALYSIS")
    print("="*60)
    print(f"Initial Spot Price: ${initial_spot:.2f}")
    print(f"Strike Price: ${strike_price:.2f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"Initial Moneyness: {initial_spot/strike_price:.3f}")
    print(f"Implied Volatility: {iv*100:.0f}%")
    print(f"Risk-free Rate: {risk_free_rate*100:.1f}%")
    print(f"Analysis Period: {days_to_expiry} days")
    print("="*60)
    
    # Generate price paths
    price_paths = generate_linear_price_paths(initial_spot, days_to_expiry, price_changes)
    
    results = {}
    
    for path_name, path_data in price_paths.items():
        days = path_data['days_to_expiry']
        spots = path_data['spot_prices']
        
        straddle_values = []
        moneyness_values = []
        call_values = []
        put_values = []
        
        for day, spot in zip(days, spots):
            # Convert days to years for Black-Scholes
            time_to_expiry = max(day / 365.0, 1/365)  # Minimum 1 day
            
            # Calculate option values
            call_val = black_scholes_call(spot, strike_price, time_to_expiry, risk_free_rate, iv)
            put_val = black_scholes_put(spot, strike_price, time_to_expiry, risk_free_rate, iv)
            straddle_val = call_val + put_val
            moneyness = spot / strike_price
            
            straddle_values.append(straddle_val)
            moneyness_values.append(moneyness)
            call_values.append(call_val)
            put_values.append(put_val)
        
        results[path_name] = {
            'days_to_expiry': days,
            'spot_prices': spots,
            'straddle_values': np.array(straddle_values),
            'moneyness': np.array(moneyness_values),
            'call_values': np.array(call_values),
            'put_values': np.array(put_values),
            'final_spot': spots[-1],
            'final_straddle': straddle_values[-1],
            'initial_straddle': straddle_values[0],
            'total_decay': straddle_values[0] - straddle_values[-1]
        }
        
        print(f"\n{path_name} Price Path:")
        print(f"  Final Spot: ${spots[-1]:.2f}")
        print(f"  Final Moneyness: {moneyness_values[-1]:.3f}")
        print(f"  Initial Straddle: ${straddle_values[0]:.2f}")
        print(f"  Final Straddle: ${straddle_values[-1]:.2f}")
        print(f"  Total Decay: ${straddle_values[0] - straddle_values[-1]:.2f}")
        print(f"  % Change: {((straddle_values[-1] - straddle_values[0])/straddle_values[0])*100:+.1f}%")
    
    return results, strike_price

def plot_straddle_decay_analysis(results, strike_price, initial_spot, save_plots=True):
    """
    Create comprehensive plots of the straddle decay analysis
    """
    
    # Set up the plot style
    plt.style.use('default')
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Theoretical Straddle Decay - Linear Price Paths', fontsize=16, fontweight='bold')
    
    # Plot 1: Spot price paths
    ax1 = axes[0, 0]
    for i, (path_name, data) in enumerate(results.items()):
        ax1.plot(data['days_to_expiry'], data['spot_prices'], 
                label=path_name, color=colors[i], linewidth=2)
    
    ax1.axhline(y=strike_price, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Strike Price')
    ax1.axhline(y=initial_spot, color='black', linestyle=':', alpha=0.7, linewidth=1, label='Initial Spot')
    ax1.set_xlabel('Days to Expiry')
    ax1.set_ylabel('Spot Price ($)')
    ax1.set_title('Linear Price Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Countdown to expiry
    
    # Plot 2: Straddle values over time
    ax2 = axes[0, 1]
    for i, (path_name, data) in enumerate(results.items()):
        ax2.plot(data['days_to_expiry'], data['straddle_values'], 
                label=path_name, color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Days to Expiry')
    ax2.set_ylabel('Straddle Value ($)')
    ax2.set_title('Straddle Value Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Moneyness over time
    ax3 = axes[1, 0]
    for i, (path_name, data) in enumerate(results.items()):
        ax3.plot(data['days_to_expiry'], data['moneyness'], 
                label=path_name, color=colors[i], linewidth=2)
    
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM')
    ax3.fill_between([0, 365], 0.95, 1.05, alpha=0.2, color='red', label='Near ATM')
    ax3.set_xlabel('Days to Expiry')
    ax3.set_ylabel('Moneyness (S/K)')
    ax3.set_title('Moneyness Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Plot 4: Straddle value vs moneyness
    ax4 = axes[1, 1]
    for i, (path_name, data) in enumerate(results.items()):
        ax4.plot(data['moneyness'], data['straddle_values'], 
                label=path_name, color=colors[i], linewidth=2, marker='o', markersize=3)
    
    ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM')
    ax4.set_xlabel('Moneyness (S/K)')
    ax4.set_ylabel('Straddle Value ($)')
    ax4.set_title('Straddle Value vs Moneyness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'theoretical_straddle_decay_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved: {filename}")
    
    plt.show()

def create_summary_table(results):
    """Create a summary table of key metrics for each path"""
    
    summary_data = []
    
    for path_name, data in results.items():
        summary_data.append({
            'Price_Path': path_name,
            'Final_Spot': data['final_spot'],
            'Final_Moneyness': data['moneyness'][-1],
            'Initial_Straddle': data['initial_straddle'],
            'Final_Straddle': data['final_straddle'],
            'Total_Decay_Dollar': data['total_decay'],
            'Total_Decay_Percent': ((data['final_straddle'] - data['initial_straddle'])/data['initial_straddle'])*100,
            'Max_Straddle': data['straddle_values'].max(),
            'Min_Straddle': data['straddle_values'].min()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.round(2).to_string(index=False))
    
    return summary_df

def main():
    """Main function to run the theoretical analysis"""
    
    # Parameters (you can modify these)
    initial_spot = 100.0  # Starting spot price
    strike_multiplier = 1.25  # Strike 25% above spot
    price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # Various % increases
    iv = 0.25  # 25% implied volatility
    risk_free_rate = 0.04  # 4% risk-free rate
    days_to_expiry = 365  # Full year analysis
    
    # Run the analysis
    results, strike_price = calculate_theoretical_straddle_decay(
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        price_changes=price_changes,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry
    )
    
    # Create plots
    plot_straddle_decay_analysis(results, strike_price, initial_spot)
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    # Save results to CSV
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
    detailed_df.to_csv('theoretical_straddle_decay_data.csv', index=False)
    summary_df.to_csv('theoretical_straddle_summary.csv', index=False)
    
    print(f"\n💾 Detailed data saved to: theoretical_straddle_decay_data.csv")
    print(f"📋 Summary saved to: theoretical_straddle_summary.csv")
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()