import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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

def generate_geometric_brownian_motion_path(S0, target_return, days, annual_vol=0.20, dt=1/365, seed=None):
    """
    Generate a realistic random walk (Geometric Brownian Motion) that ends at a target price
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate required drift to hit target exactly
    target_price = S0 * (1 + target_return)
    T = days * dt
    
    required_log_return = np.log(target_price / S0)
    mu = (required_log_return + 0.5 * annual_vol**2 * T) / T
    
    # Generate random increments
    n_steps = days
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    # Initialize arrays
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)
    
    # Generate log price path
    for i in range(n_steps):
        log_prices[i + 1] = log_prices[i] + (mu - 0.5 * annual_vol**2) * dt + annual_vol * dW[i]
    
    # Convert back to prices
    prices = np.exp(log_prices)
    
    # Fine-tune to hit exact target
    prices[-1] = target_price
    
    return prices

def run_multiple_simulations_analysis(initial_spot, strike_multiplier=1.25, 
                                    price_changes=None, n_simulations=10,
                                    iv=0.25, risk_free_rate=0.04, 
                                    days_to_expiry=365, annual_vol=0.20):
    """
    Run multiple random walk simulations for each price target
    """
    
    if price_changes is None:
        price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    # Calculate strike price
    strike_price = initial_spot * strike_multiplier
    
    print(f"ENHANCED RANDOM WALK STRADDLE ANALYSIS")
    print("="*80)
    print(f"Initial Spot: ${initial_spot:.2f}")
    print(f"Strike Price: ${strike_price:.2f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"Simulations per target: {n_simulations}")
    print(f"Stock Volatility: {annual_vol*100:.0f}%")
    print(f"Options IV: {iv*100:.0f}%")
    print("="*80)
    
    all_results = []
    
    for target_return in price_changes:
        target_name = f"{target_return*100:+.0f}%"
        print(f"\nSimulating {target_name} target with {n_simulations} random walks...")
        
        target_results = []
        
        for sim in range(n_simulations):
            # Generate unique seed for each simulation
            seed = 1000 + int(target_return * 1000) + sim
            
            # Generate price path
            price_path = generate_geometric_brownian_motion_path(
                S0=initial_spot,
                target_return=target_return,
                days=days_to_expiry,
                annual_vol=annual_vol,
                seed=seed
            )
            
            # Calculate straddle values and returns
            days_array = np.arange(days_to_expiry, -1, -1)
            straddle_values = []
            short_returns = []
            moneyness_values = []
            
            for day, spot in zip(days_array, price_path):
                time_to_expiry = max(day / 365.0, 1/365)
                
                straddle_val = calculate_straddle_value(
                    spot, strike_price, time_to_expiry, risk_free_rate, iv
                )
                moneyness = spot / strike_price
                
                straddle_values.append(straddle_val)
                moneyness_values.append(moneyness)
            
            # Calculate short straddle returns
            initial_straddle = straddle_values[0]
            for straddle_val in straddle_values:
                pnl_percent = ((initial_straddle - straddle_val) / initial_straddle) * 100
                short_returns.append(pnl_percent)
            
            # Store this simulation's results
            sim_result = {
                'target_return': target_return,
                'target_name': target_name,
                'simulation': sim,
                'days_to_expiry': days_array,
                'spot_prices': price_path,
                'straddle_values': np.array(straddle_values),
                'short_returns': np.array(short_returns),
                'moneyness': np.array(moneyness_values),
                'final_return': short_returns[-1],
                'max_return': max(short_returns),
                'min_return': min(short_returns),
                'path_volatility': np.std(np.diff(np.log(price_path))) * np.sqrt(365) * 100
            }
            
            target_results.append(sim_result)
        
        all_results.extend(target_results)
        
        # Summary for this target
        final_returns = [r['final_return'] for r in target_results]
        print(f"  Final returns range: {min(final_returns):+.1f}% to {max(final_returns):+.1f}%")
        print(f"  Average final return: {np.mean(final_returns):+.1f}%")
        print(f"  Standard deviation: {np.std(final_returns):.1f}%")
    
    return all_results, strike_price

def calculate_straddle_volatility_by_dte(all_results):
    """
    Calculate straddle price volatility by days to expiry
    """
    
    print(f"\nCalculating straddle volatility by DTE...")
    
    # Collect all straddle values by DTE across all simulations
    dte_volatility_data = {}
    
    for result in all_results:
        days_to_expiry = result['days_to_expiry']
        straddle_values = result['straddle_values']
        
        for dte, straddle_val in zip(days_to_expiry, straddle_values):
            if dte not in dte_volatility_data:
                dte_volatility_data[dte] = []
            dte_volatility_data[dte].append(straddle_val)
    
    # Calculate volatility for each DTE
    dte_summary = []
    for dte in sorted(dte_volatility_data.keys(), reverse=True):  # 365 down to 0
        values = dte_volatility_data[dte]
        if len(values) > 1:
            mean_val = np.mean(values)
            std_val = np.std(values)
            volatility_pct = (std_val / mean_val) * 100 if mean_val > 0 else 0
            
            dte_summary.append({
                'DTE': dte,
                'Mean_Straddle_Value': mean_val,
                'Std_Straddle_Value': std_val,
                'Volatility_Percent': volatility_pct,
                'Sample_Size': len(values)
            })
    
    volatility_df = pd.DataFrame(dte_summary)
    
    print(f"Volatility analysis complete for {len(volatility_df)} DTE levels")
    return volatility_df

def plot_individual_target_analysis(target_results, target_return, strike_price, initial_spot):
    """
    Create individual plot for each price target showing all 10 simulations
    """
    
    plt.style.use('default')
    
    # Create figure with subplots for each simulation
    n_sims = len(target_results)
    n_cols = 5  # 5 columns
    n_rows = 2  # 2 rows (for 10 simulations)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    fig.suptitle(f'Short Straddle Returns - {target_return*100:+.0f}% Price Target\n(Strike: ${strike_price:.0f}, Initial Spot: ${initial_spot:.0f})', 
                 fontsize=16, fontweight='bold')
    
    # Colors for each simulation
    colors = plt.cm.Set3(np.linspace(0, 1, n_sims))
    
    for i, result in enumerate(target_results):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot the return curve
        ax.plot(result['days_to_expiry'], result['short_returns'], 
               linewidth=2, color=colors[i], alpha=0.8)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=1, label='20% TP')
        
        # Fill profit/loss areas
        ax.fill_between(result['days_to_expiry'], result['short_returns'], 0,
                       where=(result['short_returns'] >= 0), alpha=0.3, color='green')
        ax.fill_between(result['days_to_expiry'], result['short_returns'], 0,
                       where=(result['short_returns'] < 0), alpha=0.3, color='red')
        
        # Formatting
        final_return = result['final_return']
        path_vol = result['path_volatility']
        
        ax.set_title(f'Sim {result["simulation"]+1}\nFinal: {final_return:+.1f}% | Vol: {path_vol:.1f}%', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Days to Expiry', fontsize=9)
        ax.set_ylabel('Return (%)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Set consistent y-limits for comparison
        ax.set_ylim(-50, 50)
        
        # Highlight final point
        final_color = '#8B0000' if final_return < 0 else '#006400'
        ax.scatter([result['days_to_expiry'][-1]], [final_return], 
                  color=final_color, s=50, zorder=5, edgecolor='white', linewidth=1.5)
        
        # Add statistics text box
        stats_text = f'Max: {result["max_return"]:+.1f}%\nMin: {result["min_return"]:+.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    
    # Save individual plot
    filename = f'random_walk_target_{target_return*100:+.0f}pct.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {filename}")
    
    plt.show()

def plot_overview_analysis(all_results, volatility_df, strike_price, initial_spot):
    """
    Create overview plots showing all targets together and volatility analysis
    """
    
    plt.style.use('default')
    
    # Get unique targets
    targets = sorted(list(set([r['target_return'] for r in all_results])))
    colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))
    target_colors = {target: color for target, color in zip(targets, colors)}
    
    # Create main overview figure
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: All return curves overlaid
    ax1 = plt.subplot(2, 3, 1)
    
    for result in all_results:
        target = result['target_return']
        color = target_colors[target]
        
        ax1.plot(result['days_to_expiry'], result['short_returns'], 
                color=color, alpha=0.3, linewidth=1)
    
    # Add representative lines for legend
    for target in targets:
        representative = [r for r in all_results if r['target_return'] == target][0]
        ax1.plot(representative['days_to_expiry'], representative['short_returns'], 
                color=target_colors[target], linewidth=2, 
                label=f"{target*100:+.0f}% target")
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax1.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='20% TP')
    ax1.set_xlabel('Days to Expiry')
    ax1.set_ylabel('Short Straddle Return (%)')
    ax1.set_title('All Short Straddle Returns (60 Simulations)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Return distribution by target
    ax2 = plt.subplot(2, 3, 2)
    
    final_returns_by_target = {}
    for target in targets:
        target_results = [r for r in all_results if r['target_return'] == target]
        final_returns = [r['final_return'] for r in target_results]
        final_returns_by_target[f"{target*100:+.0f}%"] = final_returns
    
    # Create violin plot for better distribution view
    positions = range(1, len(final_returns_by_target) + 1)
    violin_parts = ax2.violinplot(final_returns_by_target.values(), positions=positions, showmeans=True)
    
    # Color the violins
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(final_returns_by_target.keys())
    ax2.set_xlabel('Target Price Movement')
    ax2.set_ylabel('Final Return (%)')
    ax2.set_title('Return Distribution by Target')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Straddle volatility by DTE
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.plot(volatility_df['DTE'], volatility_df['Volatility_Percent'], 
             linewidth=3, color='purple', marker='o', markersize=3)
    ax3.set_xlabel('Days to Expiry')
    ax3.set_ylabel('Straddle Price Volatility (%)')
    ax3.set_title('Straddle Price Volatility by DTE')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Highlight high volatility region
    high_vol_threshold = volatility_df['Volatility_Percent'].quantile(0.75)
    high_vol_data = volatility_df[volatility_df['Volatility_Percent'] >= high_vol_threshold]
    if not high_vol_data.empty:
        ax3.scatter(high_vol_data['DTE'], high_vol_data['Volatility_Percent'], 
                   color='red', s=50, alpha=0.7, label=f'High Vol (>{high_vol_threshold:.1f}%)')
        ax3.legend()
    
    # Plot 4: Price paths by target
    ax4 = plt.subplot(2, 3, 4)
    
    for result in all_results:
        target = result['target_return']
        color = target_colors[target]
        
        ax4.plot(result['days_to_expiry'], result['spot_prices'], 
                color=color, alpha=0.3, linewidth=1)
    
    # Add representative lines for legend
    for target in targets:
        representative = [r for r in all_results if r['target_return'] == target][0]
        ax4.plot(representative['days_to_expiry'], representative['spot_prices'], 
                color=target_colors[target], linewidth=2, 
                label=f"{target*100:+.0f}% target")
    
    ax4.axhline(y=strike_price, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Strike')
    ax4.set_xlabel('Days to Expiry')
    ax4.set_ylabel('Spot Price ($)')
    ax4.set_title('Random Walk Price Paths')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    # Plot 5: Volatility zones analysis
    ax5 = plt.subplot(2, 3, 5)
    
    # Create volatility zones
    volatility_df['Zone'] = pd.cut(volatility_df['DTE'], 
                                  bins=[0, 30, 90, 180, 365], 
                                  labels=['0-30 DTE', '30-90 DTE', '90-180 DTE', '180-365 DTE'])
    
    zone_stats = volatility_df.groupby('Zone')['Volatility_Percent'].agg(['mean', 'std']).reset_index()
    
    bars = ax5.bar(zone_stats['Zone'], zone_stats['mean'], 
                   yerr=zone_stats['std'], capsize=5, alpha=0.7, 
                   color=['red', 'orange', 'yellow', 'green'])
    ax5.set_xlabel('DTE Zones')
    ax5.set_ylabel('Average Volatility (%)')
    ax5.set_title('Straddle Volatility by Time Zones')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean_val in zip(bars, zone_stats['mean']):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate key volatility statistics
    near_expiry_vol = volatility_df[volatility_df['DTE'] <= 30]['Volatility_Percent'].mean()
    far_expiry_vol = volatility_df[volatility_df['DTE'] >= 300]['Volatility_Percent'].mean()
    max_vol = volatility_df['Volatility_Percent'].max()
    max_vol_dte = volatility_df.loc[volatility_df['Volatility_Percent'].idxmax(), 'DTE']
    
    # Overall return statistics
    all_final_returns = [r['final_return'] for r in all_results]
    profitable_count = sum(1 for r in all_final_returns if r > 0)
    
    stats_text = f"""
ENHANCED ANALYSIS SUMMARY
{'='*30}

SIMULATIONS:
  Total: {len(all_results)}
  Per Target: {len(all_results) // len(targets)}
  Targets: {len(targets)}

OVERALL RETURNS:
  Profitable: {profitable_count}/{len(all_results)} ({profitable_count/len(all_results)*100:.0f}%)
  Average: {np.mean(all_final_returns):+.1f}%
  Best: {max(all_final_returns):+.1f}%
  Worst: {min(all_final_returns):+.1f}%

VOLATILITY ANALYSIS:
  Near Expiry (<30 DTE): {near_expiry_vol:.1f}%
  Far Expiry (>300 DTE): {far_expiry_vol:.1f}%
  Maximum: {max_vol:.1f}% at {max_vol_dte:.0f} DTE
  
VOLATILITY PATTERN:
  {'High near expiry' if near_expiry_vol > far_expiry_vol else 'High far from expiry'}
  Ratio: {near_expiry_vol/far_expiry_vol:.1f}x
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save overview plot
    filename = 'random_walk_overview_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Overview analysis saved: {filename}")
    
    plt.show()

def plot_combined_target_analysis(all_results, strike_price, initial_spot):
    """
    Create a single image showing all 6 price targets with their random walk returns
    """
    
    plt.style.use('default')
    
    # Get unique targets and organize results
    targets = sorted(list(set([r['target_return'] for r in all_results])))
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Short Straddle Returns - Random Walk Paths\n(Strike: {strike_price:.0f}, Initial Spot: {initial_spot:.0f})', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme for each target
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Process each target
    for i, target in enumerate(targets):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Get all simulations for this target
        target_results = [r for r in all_results if r['target_return'] == target]
        
        # Calculate average path and statistics
        all_returns = []
        all_days = None
        
        for result in target_results:
            all_returns.append(result['short_returns'])
            if all_days is None:
                all_days = result['days_to_expiry']
        
        # Convert to numpy arrays for easier manipulation
        all_returns = np.array(all_returns)
        
        # Calculate statistics across all simulations
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        max_returns = np.max(all_returns, axis=0)
        min_returns = np.min(all_returns, axis=0)
        
        # Plot all individual paths with transparency
        for j, result in enumerate(target_results):
            ax.plot(result['days_to_expiry'], result['short_returns'], 
                   color=colors[i], alpha=0.3, linewidth=1)
        
        # Plot mean path with thicker line
        ax.plot(all_days, mean_returns, 
               color=colors[i], linewidth=3, alpha=0.9, label='Average')
        
        # Fill between min and max for envelope
        ax.fill_between(all_days, min_returns, max_returns, 
                       color=colors[i], alpha=0.2, label='Range')
        
        # Fill profit/loss areas based on mean
        ax.fill_between(all_days, mean_returns, 0,
                       where=(mean_returns >= 0), alpha=0.4, color='green', 
                       interpolate=True)
        ax.fill_between(all_days, mean_returns, 0,
                       where=(mean_returns < 0), alpha=0.4, color='red', 
                       interpolate=True)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=1)
        
        # Calculate final statistics
        final_returns = [result['final_return'] for result in target_results]
        avg_final = np.mean(final_returns)
        max_final = max(final_returns)
        min_final = min(final_returns)
        avg_volatility = np.mean([result['path_volatility'] for result in target_results])
        
        # Formatting
        ax.set_title(f'{target*100:+.0f}% Random Walk\nFinal Return: {avg_final:+.1f}%', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Days to Expiry', fontsize=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Set consistent y-limits for comparison
        ax.set_ylim(-50, 100)
        
        # Add statistics text box
        stats_text = f'Max: {max_final:+.1f}%\nMin: {min_final:+.1f}%\nVol: {avg_volatility:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Highlight final average point
        final_color = '#8B0000' if avg_final < 0 else '#006400'
        ax.scatter([all_days[-1]], [avg_final], 
                  color=final_color, s=80, zorder=5, edgecolor='white', linewidth=2)
    
    plt.tight_layout()
    
    # Save the combined plot
    filename = 'short_straddle_random_walk_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Combined analysis saved: {filename}")
    
    plt.show()

def plot_enhanced_analysis(all_results, volatility_df, strike_price, initial_spot):
    """
    Create enhanced plots including the main combined analysis and overview
    """
    
    print(f"\n📊 Creating combined target analysis...")
    
    # Create the main combined plot (like your reference image)
    plot_combined_target_analysis(all_results, strike_price, initial_spot)
    
    print(f"\n📊 Creating detailed overview analysis...")
    
    # Create detailed overview plot
    plot_overview_analysis(all_results, volatility_df, strike_price, initial_spot)

def create_detailed_summary_report(all_results, volatility_df):
    """
    Create detailed summary report and save data
    """
    
    print(f"\n" + "="*80)
    print("DETAILED SUMMARY REPORT")
    print("="*80)
    
    # Group results by target
    targets = sorted(list(set([r['target_return'] for r in all_results])))
    
    print(f"\nSIMULATION OVERVIEW:")
    print(f"  Total simulations: {len(all_results)}")
    print(f"  Targets tested: {len(targets)}")
    print(f"  Simulations per target: {len(all_results) // len(targets)}")
    
    print(f"\nRESULTS BY TARGET:")
    for target in targets:
        target_results = [r for r in all_results if r['target_return'] == target]
        final_returns = [r['final_return'] for r in target_results]
        
        print(f"\n  {target*100:+.0f}% Price Target:")
        print(f"    Simulations: {len(target_results)}")
        print(f"    Final returns: {min(final_returns):+.1f}% to {max(final_returns):+.1f}%")
        print(f"    Average: {np.mean(final_returns):+.1f}% ± {np.std(final_returns):.1f}%")
        print(f"    Profitable simulations: {sum(1 for r in final_returns if r > 0)}/{len(final_returns)}")
    
    # Save detailed data
    detailed_data = []
    for result in all_results:
        for i, (dte, spot, straddle, ret) in enumerate(zip(
            result['days_to_expiry'], result['spot_prices'], 
            result['straddle_values'], result['short_returns'])):
            
            detailed_data.append({
                'Target_Return': result['target_return'],
                'Target_Name': result['target_name'],
                'Simulation': result['simulation'],
                'Day_Index': i,
                'Days_to_Expiry': dte,
                'Spot_Price': spot,
                'Straddle_Value': straddle,
                'Short_Return_Pct': ret,
                'Moneyness': result['moneyness'][i]
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('enhanced_random_walk_detailed_data.csv', index=False)
    
    # Save volatility analysis
    volatility_df.to_csv('straddle_volatility_by_dte.csv', index=False)
    
    # Save summary statistics
    summary_stats = []
    for target in targets:
        target_results = [r for r in all_results if r['target_return'] == target]
        final_returns = [r['final_return'] for r in target_results]
        
        summary_stats.append({
            'Target_Return': target,
            'Target_Name': f"{target*100:+.0f}%",
            'Simulations': len(target_results),
            'Mean_Final_Return': np.mean(final_returns),
            'Std_Final_Return': np.std(final_returns),
            'Min_Final_Return': min(final_returns),
            'Max_Final_Return': max(final_returns),
            'Profitable_Count': sum(1 for r in final_returns if r > 0),
            'Profitable_Rate': sum(1 for r in final_returns if r > 0) / len(final_returns) * 100
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('enhanced_random_walk_summary.csv', index=False)
    
    print(f"\n💾 Data files saved:")
    print(f"  enhanced_random_walk_detailed_data.csv ({len(detailed_df)} records)")
    print(f"  straddle_volatility_by_dte.csv ({len(volatility_df)} DTE levels)")
    print(f"  enhanced_random_walk_summary.csv ({len(summary_df)} targets)")

def main():
    """
    Main function to run the enhanced analysis
    """
    
    # Parameters
    initial_spot = 100.0
    strike_multiplier = 1.25  # 25% above spot
    price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    n_simulations = 10  # 10 simulations per target
    iv = 0.25
    annual_vol = 0.20  # Consistent 20% volatility
    risk_free_rate = 0.04
    days_to_expiry = 365
    
    print(f"ENHANCED RANDOM WALK STRADDLE ANALYSIS")
    print(f"Running {n_simulations} simulations for each price target...")
    
    # Run multiple simulations
    all_results, strike_price = run_multiple_simulations_analysis(
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        price_changes=price_changes,
        n_simulations=n_simulations,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry,
        annual_vol=annual_vol
    )
    
    # Calculate straddle volatility by DTE
    volatility_df = calculate_straddle_volatility_by_dte(all_results)
    
    # Create enhanced plots
    plot_enhanced_analysis(all_results, volatility_df, strike_price, initial_spot)
    
    # Generate detailed report
    create_detailed_summary_report(all_results, volatility_df)
    
    print(f"\n🎉 Enhanced analysis complete!")
    print(f"Total simulations run: {len(all_results)}")
    print(f"Straddle volatility analyzed across {len(volatility_df)} DTE levels")

if __name__ == "__main__":
    main()