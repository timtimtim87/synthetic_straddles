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
            'spot_prices': price_path,
            'price_change': pct_change,
            'final_price': final_price
        }
    
    return paths

def calculate_comprehensive_straddle_analysis(initial_spot, strike_multiplier=1.25, 
                                            price_changes=None, iv=0.25, risk_free_rate=0.04,
                                            days_to_expiry=365):
    """
    Calculate comprehensive straddle analysis for linear price paths
    Combines theoretical decay analysis with short straddle return focus
    """
    
    if price_changes is None:
        price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # 0% to 50% increases
    
    # Calculate strike price (25% above initial spot)
    strike_price = initial_spot * strike_multiplier
    
    print(f"COMPREHENSIVE LINEAR PATH STRADDLE ANALYSIS")
    print("="*70)
    print(f"Initial Spot Price: ${initial_spot:.2f}")
    print(f"Strike Price: ${strike_price:.2f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"Initial Moneyness: {initial_spot/strike_price:.3f}")
    print(f"Implied Volatility: {iv*100:.0f}%")
    print(f"Risk-free Rate: {risk_free_rate*100:.1f}%")
    print(f"Analysis Period: {days_to_expiry} days")
    print(f"Price Targets: {[f'{pc*100:+.0f}%' for pc in price_changes]}")
    print("="*70)
    
    # Generate linear price paths
    price_paths = generate_linear_price_paths(initial_spot, days_to_expiry, price_changes)
    
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
            'price_change': path_data['price_change'],
            'final_spot': spots[-1],
            'final_straddle': straddle_values[-1],
            'initial_straddle': straddle_values[0],
            'total_decay': straddle_values[0] - straddle_values[-1],
            'final_return': short_returns[-1],
            'max_return': max(short_returns),
            'min_return': min(short_returns),
            'break_even_analysis': {
                'upper_breakeven': strike_price + initial_straddle,
                'lower_breakeven': strike_price - initial_straddle,
                'profit_zone_width': 2 * initial_straddle
            }
        }
        
        print(f"\n{path_name} Linear Path Analysis:")
        print(f"  Final Spot: ${spots[-1]:.2f}")
        print(f"  Final Moneyness: {moneyness_values[-1]:.3f}")
        print(f"  Initial Straddle: ${straddle_values[0]:.2f}")
        print(f"  Final Straddle: ${straddle_values[-1]:.2f}")
        print(f"  Total Decay: ${straddle_values[0] - straddle_values[-1]:.2f}")
        print(f"  Short Straddle Return: {short_returns[-1]:+.1f}%")
        print(f"  Max Return: {max(short_returns):+.1f}%")
        print(f"  Min Return: {min(short_returns):+.1f}%")
        print(f"  Upper Breakeven: ${results[path_name]['break_even_analysis']['upper_breakeven']:.2f}")
        print(f"  Lower Breakeven: ${results[path_name]['break_even_analysis']['lower_breakeven']:.2f}")
    
    return results, strike_price

def calculate_straddle_sensitivity_analysis(results, initial_spot, strike_price):
    """
    Calculate sensitivity metrics for straddle positions
    """
    
    print(f"\n" + "="*70)
    print("STRADDLE SENSITIVITY ANALYSIS")
    print("="*70)
    
    sensitivity_data = []
    
    for path_name, data in results.items():
        price_change = data['price_change']
        final_return = data['final_return']
        max_return = data['max_return']
        min_return = data['min_return']
        
        # Calculate sensitivity metrics
        return_per_price_move = final_return / (price_change * 100) if price_change != 0 else 0
        max_drawdown = max_return - min_return
        
        # Time decay analysis
        mid_point = len(data['short_returns']) // 2
        early_return = data['short_returns'][mid_point]  # At 6 months
        time_decay_benefit = early_return - final_return
        
        sensitivity_data.append({
            'Path': path_name,
            'Price_Change_Pct': price_change * 100,
            'Final_Return_Pct': final_return,
            'Max_Return_Pct': max_return,
            'Min_Return_Pct': min_return,
            'Max_Drawdown_Pct': max_drawdown,
            'Return_Per_Price_Move': return_per_price_move,
            'Early_Exit_Return_6mo': early_return,
            'Time_Decay_Benefit': time_decay_benefit
        })
        
        print(f"{path_name:>6}: Return/Move = {return_per_price_move:+.2f} | "
              f"Max DD = {max_drawdown:.1f}% | "
              f"6mo Exit = {early_return:+.1f}%")
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    # Find optimal scenarios
    best_return = sensitivity_df.loc[sensitivity_df['Final_Return_Pct'].idxmax()]
    worst_return = sensitivity_df.loc[sensitivity_df['Final_Return_Pct'].idxmin()]
    best_risk_adj = sensitivity_df.loc[(sensitivity_df['Final_Return_Pct'] / sensitivity_df['Max_Drawdown_Pct']).idxmax()]
    
    print(f"\nKEY INSIGHTS:")
    print(f"  Best Return: {best_return['Path']} with {best_return['Final_Return_Pct']:+.1f}%")
    print(f"  Worst Return: {worst_return['Path']} with {worst_return['Final_Return_Pct']:+.1f}%")
    print(f"  Best Risk-Adjusted: {best_risk_adj['Path']} (Return/DD ratio)")
    
    return sensitivity_df

def analyze_moneyness_impact(results):
    """
    Analyze how moneyness changes affect straddle behavior
    """
    
    print(f"\n" + "="*50)
    print("MONEYNESS IMPACT ANALYSIS")
    print("="*50)
    
    moneyness_analysis = {}
    
    for path_name, data in results.items():
        final_moneyness = data['moneyness'][-1]
        final_return = data['final_return']
        
        # Categorize by final moneyness
        if final_moneyness < 0.95:
            category = "Deep OTM Put"
        elif final_moneyness < 1.05:
            category = "Near ATM"
        elif final_moneyness < 1.15:
            category = "Slightly OTM Call"
        else:
            category = "OTM Call"
        
        if category not in moneyness_analysis:
            moneyness_analysis[category] = []
        
        moneyness_analysis[category].append({
            'path': path_name,
            'final_moneyness': final_moneyness,
            'final_return': final_return
        })
    
    for category, paths in moneyness_analysis.items():
        avg_return = np.mean([p['final_return'] for p in paths])
        avg_moneyness = np.mean([p['final_moneyness'] for p in paths])
        
        print(f"{category:>15}: Avg Return = {avg_return:+.1f}%, Avg Moneyness = {avg_moneyness:.3f}")
    
    return moneyness_analysis

# END OF PART 1
# ============================================================================
# This is the halfway point. The second half will contain:
# - Comprehensive plotting functions
# - Individual return path visualizations
# - Multi-panel overview analysis
# - Break-even analysis plots
# - Data export functionality
# - main() function
# ============================================================================


# PART 2 - CONTINUING FROM PART 1
# ============================================================================

def plot_comprehensive_straddle_analysis(results, strike_price, initial_spot, save_plots=True):
    """
    Create comprehensive plots combining theoretical decay and short straddle analysis
    """
    
    # Set up the plot style
    plt.style.use('default')
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Create figure with subplots - 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Linear Path Straddle Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Linear price paths
    ax1 = axes[0, 0]
    for i, (path_name, data) in enumerate(results.items()):
        ax1.plot(data['days_to_expiry'], data['spot_prices'], 
                label=path_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.axhline(y=strike_price, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Strike Price')
    ax1.axhline(y=initial_spot, color='black', linestyle=':', alpha=0.7, linewidth=1, label='Initial Spot')
    ax1.set_xlabel('Days to Expiry')
    ax1.set_ylabel('Spot Price ($)')
    ax1.set_title('Linear Price Paths')
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
    ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='20% TP')
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
    
    # Plot 5: Break-even analysis
    ax5 = axes[1, 1]
    path_names = list(results.keys())
    final_spots = [results[path]['final_spot'] for path in path_names]
    final_returns = [results[path]['final_return'] for path in path_names]
    
    # Get break-even points from first path (they're all the same)
    first_path = list(results.values())[0]
    upper_be = first_path['break_even_analysis']['upper_breakeven']
    lower_be = first_path['break_even_analysis']['lower_breakeven']
    
    bars = ax5.bar(range(len(path_names)), final_returns, 
                   color=['green' if r > 0 else 'red' for r in final_returns], alpha=0.7)
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.axvline(x=len(path_names)/2, color='blue', linestyle=':', alpha=0.5, label='Strike Reference')
    ax5.set_xlabel('Price Path')
    ax5.set_ylabel('Final Return (%)')
    ax5.set_title('Final Returns by Price Path')
    ax5.set_xticks(range(len(path_names)))
    ax5.set_xticklabels(path_names, rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ret in zip(bars, final_returns):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{ret:+.1f}%', ha='center', va='bottom' if ret > 0 else 'top', fontweight='bold')
    
    # Plot 6: Risk-Return scatter
    ax6 = axes[1, 2]
    max_returns = [results[path]['max_return'] for path in path_names]
    min_returns = [results[path]['min_return'] for path in path_names]
    drawdowns = [max_ret - min_ret for max_ret, min_ret in zip(max_returns, min_returns)]
    
    scatter = ax6.scatter(drawdowns, final_returns, c=range(len(path_names)), 
                         cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    
    for i, (name, ret, dd) in enumerate(zip(path_names, final_returns, drawdowns)):
        ax6.annotate(name, (dd, ret), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.set_xlabel('Max Drawdown (%)')
    ax6.set_ylabel('Final Return (%)')
    ax6.set_title('Risk vs Return Analysis')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'comprehensive_linear_path_straddle_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comprehensive plot saved: {filename}")
    
    plt.show()

def plot_individual_return_paths(results, strike_price, initial_spot, save_plots=True):
    """
    Create individual return plots for each linear path (enhanced version of z08_price_paths_2.py)
    """
    
    plt.style.use('default')
    
    # Calculate grid dimensions
    n_paths = len(results)
    n_cols = 3
    n_rows = (n_paths + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle(f'Short Straddle Returns - Linear Price Paths\n(Strike: ${strike_price:.0f}, Initial Spot: ${initial_spot:.0f})', 
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
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=1, label='20% TP')
        
        # Fill areas
        ax.fill_between(data['days_to_expiry'], data['short_returns'], 0,
                       where=(data['short_returns'] >= 0), alpha=0.3, color='green')
        ax.fill_between(data['days_to_expiry'], data['short_returns'], 0,
                       where=(data['short_returns'] < 0), alpha=0.3, color='red')
        
        # Formatting
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'{path_name} Linear Path\nFinal Return: {data["final_return"]:+.1f}%', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Set consistent y-limits
        ax.set_ylim(-40, 80)
        
        # Stats box
        stats_text = f'Max: {data["max_return"]:+.1f}%\nMin: {data["min_return"]:+.1f}%\nDecay: ${data["total_decay"]:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Highlight final point
        final_color = '#8B0000' if data['final_return'] < 0 else '#006400'
        ax.scatter([0], [data['final_return']], color=final_color, s=100, zorder=5, 
                  edgecolor='black', linewidth=2)
        
        # Add break-even reference
        be_analysis = data['break_even_analysis']
        if data['final_spot'] < be_analysis['lower_breakeven'] or data['final_spot'] > be_analysis['upper_breakeven']:
            be_status = "✓ Outside BE"
            be_color = 'green'
        else:
            be_status = "✗ Inside BE"
            be_color = 'red'
        
        ax.text(0.98, 0.02, be_status, transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=be_color, alpha=0.3),
                horizontalalignment='right', verticalalignment='bottom', fontsize=9)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'linear_path_straddle_returns_individual.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Individual returns plot saved: {filename}")
    
    plt.show()

def plot_sensitivity_analysis(sensitivity_df, save_plots=True):
    """
    Create sensitivity analysis plots
    """
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Straddle Sensitivity Analysis - Linear Paths', fontsize=16, fontweight='bold')
    
    # Plot 1: Return vs Price Move
    ax1 = axes[0, 0]
    ax1.scatter(sensitivity_df['Price_Change_Pct'], sensitivity_df['Final_Return_Pct'], 
               s=100, alpha=0.7, color='blue', edgecolors='black')
    
    for i, row in sensitivity_df.iterrows():
        ax1.annotate(row['Path'], (row['Price_Change_Pct'], row['Final_Return_Pct']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Price Change (%)')
    ax1.set_ylabel('Final Straddle Return (%)')
    ax1.set_title('Return vs Underlying Price Change')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk-Adjusted Returns
    ax2 = axes[0, 1]
    risk_adj_returns = sensitivity_df['Final_Return_Pct'] / sensitivity_df['Max_Drawdown_Pct']
    bars = ax2.bar(range(len(sensitivity_df)), risk_adj_returns, 
                   color=['green' if r > 0 else 'red' for r in risk_adj_returns], alpha=0.7)
    
    ax2.set_xlabel('Price Path')
    ax2.set_ylabel('Risk-Adjusted Return (Return/Max DD)')
    ax2.set_title('Risk-Adjusted Performance')
    ax2.set_xticks(range(len(sensitivity_df)))
    ax2.set_xticklabels(sensitivity_df['Path'], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, risk_adj_returns):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # Plot 3: Early Exit vs Hold to Expiry
    ax3 = axes[1, 0]
    width = 0.35
    x = np.arange(len(sensitivity_df))
    
    bars1 = ax3.bar(x - width/2, sensitivity_df['Early_Exit_Return_6mo'], width, 
                   label='6-Month Exit', alpha=0.7, color='orange')
    bars2 = ax3.bar(x + width/2, sensitivity_df['Final_Return_Pct'], width,
                   label='Hold to Expiry', alpha=0.7, color='blue')
    
    ax3.set_xlabel('Price Path')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Early Exit vs Hold to Expiry')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sensitivity_df['Path'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time Decay Benefit
    ax4 = axes[1, 1]
    time_decay_benefit = sensitivity_df['Time_Decay_Benefit']
    bars = ax4.bar(range(len(sensitivity_df)), time_decay_benefit,
                   color=['green' if t > 0 else 'red' for t in time_decay_benefit], alpha=0.7)
    
    ax4.set_xlabel('Price Path')
    ax4.set_ylabel('Time Decay Benefit (%)')
    ax4.set_title('Benefit of Holding vs Early Exit')
    ax4.set_xticks(range(len(sensitivity_df)))
    ax4.set_xticklabels(sensitivity_df['Path'], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, time_decay_benefit):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'straddle_sensitivity_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Sensitivity analysis plot saved: {filename}")
    
    plt.show()

def create_summary_table(results, sensitivity_df):
    """Create a comprehensive summary table of all results"""
    
    print(f"\n" + "="*90)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*90)
    
    summary_data = []
    for path_name, data in results.items():
        # Get corresponding sensitivity data
        sens_row = sensitivity_df[sensitivity_df['Path'] == path_name].iloc[0]
        
        summary_data.append({
            'Price_Path': path_name,
            'Price_Change_%': data['price_change'] * 100,
            'Final_Spot_$': data['final_spot'],
            'Final_Moneyness': data['moneyness'][-1],
            'Initial_Straddle_$': data['initial_straddle'],
            'Final_Straddle_$': data['final_straddle'],
            'Total_Decay_$': data['total_decay'],
            'Final_Return_%': data['final_return'],
            'Max_Return_%': data['max_return'],
            'Min_Return_%': data['min_return'],
            'Max_Drawdown_%': sens_row['Max_Drawdown_Pct'],
            'Risk_Adj_Return': sens_row['Final_Return_Pct'] / sens_row['Max_Drawdown_Pct'],
            'Upper_Breakeven_$': data['break_even_analysis']['upper_breakeven'],
            'Lower_Breakeven_$': data['break_even_analysis']['lower_breakeven'],
            'Early_Exit_6mo_%': sens_row['Early_Exit_Return_6mo']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display formatted table
    print(summary_df.round(2).to_string(index=False))
    
    # Find best and worst scenarios
    best_path = max(results.items(), key=lambda x: x[1]['final_return'])
    worst_path = min(results.items(), key=lambda x: x[1]['final_return'])
    best_risk_adj = summary_df.loc[summary_df['Risk_Adj_Return'].idxmax()]
    
    print(f"\n🎯 BEST SCENARIO: {best_path[0]} with {best_path[1]['final_return']:+.1f}% final return")
    print(f"📉 WORST SCENARIO: {worst_path[0]} with {worst_path[1]['final_return']:+.1f}% final return")
    print(f"⚖️  BEST RISK-ADJUSTED: {best_risk_adj['Price_Path']} with {best_risk_adj['Risk_Adj_Return']:.2f} ratio")
    
    return summary_df

def save_detailed_results(results, sensitivity_df, summary_df):
    """Save all results to CSV files"""
    
    print(f"\n💾 Saving detailed results...")
    
    # Save detailed path data
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
    detailed_df.to_csv('linear_path_straddle_detailed_data.csv', index=False)
    
    # Save sensitivity analysis
    sensitivity_df.to_csv('linear_path_sensitivity_analysis.csv', index=False)
    
    # Save summary table
    summary_df.to_csv('linear_path_straddle_summary.csv', index=False)
    
    print(f"  📄 linear_path_straddle_detailed_data.csv ({len(detailed_df)} records)")
    print(f"  📄 linear_path_sensitivity_analysis.csv ({len(sensitivity_df)} paths)")
    print(f"  📄 linear_path_straddle_summary.csv ({len(summary_df)} paths)")

def main():
    """
    Main function to run the comprehensive linear path analysis
    """
    
    # Parameters (you can modify these)
    initial_spot = 100.0  # Starting spot price
    strike_multiplier = 1.25  # Strike 25% above spot
    price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # Various % increases
    iv = 0.25  # 25% implied volatility
    risk_free_rate = 0.04  # 4% risk-free rate
    days_to_expiry = 365  # Full year analysis
    
    print(f"COMPREHENSIVE LINEAR PATH STRADDLE ANALYSIS")
    print("="*80)
    print(f"Combining theoretical decay analysis with practical short straddle returns")
    print(f"Enhanced with sensitivity analysis and break-even calculations")
    print("="*80)
    
    # Step 1: Run comprehensive straddle analysis
    results, strike_price = calculate_comprehensive_straddle_analysis(
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        price_changes=price_changes,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry
    )
    
    # Step 2: Calculate sensitivity analysis
    sensitivity_df = calculate_straddle_sensitivity_analysis(results, initial_spot, strike_price)
    
    # Step 3: Analyze moneyness impact
    moneyness_analysis = analyze_moneyness_impact(results)
    
    # Step 4: Create comprehensive plots
    plot_comprehensive_straddle_analysis(results, strike_price, initial_spot)
    
    # Step 5: Create individual return plots
    plot_individual_return_paths(results, strike_price, initial_spot)
    
    # Step 6: Create sensitivity analysis plots
    plot_sensitivity_analysis(sensitivity_df)
    
    # Step 7: Generate summary table
    summary_df = create_summary_table(results, sensitivity_df)
    
    # Step 8: Save all results
    save_detailed_results(results, sensitivity_df, summary_df)
    
    print(f"\n🎉 Comprehensive linear path analysis complete!")
    print(f"📊 Analyzed {len(price_changes)} different price paths")
    print(f"📈 Strike price: ${strike_price:.0f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"⚖️  Includes sensitivity analysis and risk-adjusted metrics")
    print(f"📁 Check output files for detailed data")

if __name__ == "__main__":
    main()