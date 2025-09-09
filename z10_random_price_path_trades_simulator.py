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

def simulate_threshold_analysis(initial_spot, strike_multiplier=1.25, price_changes=None, 
                              threshold_return=20.0, n_simulations=10, iv=0.25, 
                              risk_free_rate=0.04, days_to_expiry=365, annual_vol=0.20):
    """
    Simulate multiple random walks for each price target and track when 20% return threshold is hit
    """
    
    if price_changes is None:
        price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    # Calculate strike price
    strike_price = initial_spot * strike_multiplier
    
    print(f"SHORT STRADDLE 20% THRESHOLD ANALYSIS")
    print("="*80)
    print(f"Initial Spot Price: ${initial_spot:.2f}")
    print(f"Strike Price: ${strike_price:.2f} ({strike_multiplier*100-100:+.0f}% above spot)")
    print(f"Threshold Return: {threshold_return}%")
    print(f"Simulations per target: {n_simulations}")
    print(f"Stock Volatility: {annual_vol*100:.0f}%")
    print(f"Options IV: {iv*100:.0f}%")
    print("="*80)
    
    all_results = []
    
    for pct_change in price_changes:
        target_name = f"{pct_change*100:+.0f}%"
        print(f"\nSimulating {target_name} price target...")
        
        target_results = []
        
        for sim in range(n_simulations):
            # Generate unique path with different seed
            seed = 1000 + int(pct_change * 1000) + sim
            
            # Generate price path
            price_path = generate_geometric_brownian_motion_path(
                S0=initial_spot,
                target_return=pct_change,
                days=days_to_expiry,
                annual_vol=annual_vol,
                seed=seed
            )
            
            # Calculate straddle values and returns for this path
            days_array = np.arange(days_to_expiry, -1, -1)
            straddle_values = []
            short_returns = []
            
            for day, spot in zip(days_array, price_path):
                time_to_expiry = max(day / 365.0, 1/365)
                straddle_val = calculate_straddle_value(spot, strike_price, time_to_expiry, risk_free_rate, iv)
                straddle_values.append(straddle_val)
            
            # Calculate short straddle returns
            initial_straddle = straddle_values[0]
            for straddle_val in straddle_values:
                pnl_percent = ((initial_straddle - straddle_val) / initial_straddle) * 100
                short_returns.append(pnl_percent)
            
            # Find if and when threshold was hit
            threshold_hit = False
            days_to_threshold = None
            threshold_day_index = None
            
            for i, return_pct in enumerate(short_returns):
                if return_pct >= threshold_return:
                    threshold_hit = True
                    threshold_day_index = i
                    days_to_threshold = days_to_expiry - days_array[i]  # Days since start
                    break
            
            # Store results for this simulation
            sim_result = {
                'Target_Change': target_name,
                'Simulation': sim + 1,
                'Threshold_Hit': threshold_hit,
                'Days_to_Threshold': days_to_threshold,
                'Final_Return': short_returns[-1],
                'Max_Return': max(short_returns),
                'Final_Spot': price_path[-1],
                'Path_Volatility': np.std(np.diff(np.log(price_path))) * np.sqrt(365) * 100
            }
            
            target_results.append(sim_result)
            all_results.append(sim_result)
        
        # Summarize results for this target
        hits = sum(1 for r in target_results if r['Threshold_Hit'])
        if hits > 0:
            avg_days = np.mean([r['Days_to_Threshold'] for r in target_results if r['Threshold_Hit']])
            min_days = min([r['Days_to_Threshold'] for r in target_results if r['Threshold_Hit']])
            max_days = max([r['Days_to_Threshold'] for r in target_results if r['Threshold_Hit']])
            
            print(f"  Hits: {hits}/{n_simulations} ({hits/n_simulations*100:.0f}%)")
            print(f"  Avg days to 20%: {avg_days:.0f}")
            print(f"  Range: {min_days:.0f} - {max_days:.0f} days")
        else:
            print(f"  Hits: 0/{n_simulations} (0%) - No paths reached 20% threshold")
    
    return pd.DataFrame(all_results), strike_price

def analyze_threshold_results(results_df):
    """Analyze and summarize threshold hitting results"""
    
    print(f"\n" + "="*80)
    print("THRESHOLD ANALYSIS SUMMARY")
    print("="*80)
    
    # Summary by target
    summary_by_target = results_df.groupby('Target_Change').agg({
        'Threshold_Hit': ['sum', 'count', 'mean'],
        'Days_to_Threshold': ['mean', 'std', 'min', 'max'],
        'Final_Return': ['mean', 'std'],
        'Max_Return': ['mean', 'std']
    }).round(1)
    
    # Flatten column names
    summary_by_target.columns = [
        'Hits', 'Total_Sims', 'Hit_Rate',
        'Avg_Days', 'Std_Days', 'Min_Days', 'Max_Days',
        'Avg_Final_Return', 'Std_Final_Return',
        'Avg_Max_Return', 'Std_Max_Return'
    ]
    
    print("\nSUMMARY BY TARGET:")
    print("-" * 80)
    for target, row in summary_by_target.iterrows():
        print(f"{target:>5} Target:")
        print(f"  Hit Rate: {row['Hits']:.0f}/{row['Total_Sims']:.0f} ({row['Hit_Rate']*100:.0f}%)")
        if row['Hits'] > 0:
            print(f"  Days to 20%: {row['Avg_Days']:.0f} ± {row['Std_Days']:.0f} (range: {row['Min_Days']:.0f}-{row['Max_Days']:.0f})")
        print(f"  Final Return: {row['Avg_Final_Return']:+.1f}% ± {row['Std_Final_Return']:.1f}%")
        print(f"  Max Return: {row['Avg_Max_Return']:+.1f}% ± {row['Std_Max_Return']:.1f}%")
        print()
    
    return summary_by_target

def plot_threshold_analysis(results_df, threshold_return=20.0):
    """Create comprehensive plots of threshold analysis"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Short Straddle {threshold_return}% Threshold Analysis\n(10 Simulations per Target)', 
                 fontsize=16, fontweight='bold')
    
    targets = sorted(results_df['Target_Change'].unique(), key=lambda x: float(x.replace('%', '').replace('+', '')))
    colors = plt.cm.Set3(np.linspace(0, 1, len(targets)))
    
    # Plot 1: Hit rates by target
    ax1 = axes[0, 0]
    hit_rates = results_df.groupby('Target_Change')['Threshold_Hit'].mean() * 100
    hit_rates = hit_rates.reindex(targets)
    
    bars = ax1.bar(targets, hit_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Target Price Change')
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title(f'{threshold_return}% Threshold Hit Rates')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, hit_rates):
        if not np.isnan(rate):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Days to threshold (for successful hits)
    ax2 = axes[0, 1]
    hits_only = results_df[results_df['Threshold_Hit'] == True]
    
    if not hits_only.empty:
        box_data = [hits_only[hits_only['Target_Change'] == target]['Days_to_Threshold'].dropna().values 
                   for target in targets]
        box_labels = [f"{target}\n(n={len(data)})" for target, data in zip(targets, box_data)]
        
        # Only plot if we have data
        valid_data = [(label, data) for label, data in zip(box_labels, box_data) if len(data) > 0]
        if valid_data:
            valid_labels, valid_box_data = zip(*valid_data)
            ax2.boxplot(valid_box_data, labels=valid_labels)
        
        ax2.set_xlabel('Target Price Change')
        ax2.set_ylabel('Days to Reach 20%')
        ax2.set_title('Time to Reach Threshold (Successful Hits Only)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No threshold hits\nto display', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Time to Reach Threshold')
    
    # Plot 3: Final returns distribution
    ax3 = axes[0, 2]
    for i, target in enumerate(targets):
        target_data = results_df[results_df['Target_Change'] == target]['Final_Return']
        ax3.hist(target_data, alpha=0.6, label=target, color=colors[i], bins=8)
    
    ax3.axvline(x=threshold_return, color='red', linestyle='--', linewidth=2, label=f'{threshold_return}% Threshold')
    ax3.set_xlabel('Final Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Final Return Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hit rate vs target change scatter
    ax4 = axes[1, 0]
    target_nums = [float(t.replace('%', '').replace('+', '')) for t in targets]
    hit_rates_list = [hit_rates[t] for t in targets]
    
    ax4.scatter(target_nums, hit_rates_list, s=100, c=colors, alpha=0.7, edgecolors='black')
    for x, y, label in zip(target_nums, hit_rates_list, targets):
        ax4.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('Target Price Change (%)')
    ax4.set_ylabel('Hit Rate (%)')
    ax4.set_title('Hit Rate vs Target Price Movement')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Individual simulation results (hit/miss pattern)
    ax5 = axes[1, 1]
    
    # Create a matrix showing hits (1) and misses (0)
    pivot_data = results_df.pivot(index='Simulation', columns='Target_Change', values='Threshold_Hit')
    pivot_data = pivot_data.reindex(columns=targets)
    
    im = ax5.imshow(pivot_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_xlabel('Simulation Number')
    ax5.set_ylabel('Target Price Change')
    ax5.set_title('Hit/Miss Pattern by Simulation')
    ax5.set_yticks(range(len(targets)))
    ax5.set_yticklabels(targets)
    ax5.set_xticks(range(len(pivot_data)))
    ax5.set_xticklabels(range(1, len(pivot_data) + 1))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Threshold Hit (1=Yes, 0=No)')
    
    # Plot 6: Average days to threshold by target
    ax6 = axes[1, 2]
    avg_days = hits_only.groupby('Target_Change')['Days_to_Threshold'].mean()
    avg_days = avg_days.reindex(targets)
    
    valid_targets = [t for t in targets if not np.isnan(avg_days[t])]
    valid_days = [avg_days[t] for t in valid_targets]
    valid_colors = [colors[i] for i, t in enumerate(targets) if t in valid_targets]
    
    if valid_targets:
        bars = ax6.bar(valid_targets, valid_days, color=valid_colors, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Target Price Change')
        ax6.set_ylabel('Average Days to 20%')
        ax6.set_title('Average Time to Reach Threshold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, days in zip(bars, valid_days):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{days:.0f}d', ha='center', va='bottom', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No threshold hits\nto display', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=14)
        ax6.set_title('Average Time to Reach Threshold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'straddle_{threshold_return}pct_threshold_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Analysis plots saved: {filename}")
    
    plt.show()

def main():
    """Main function to run the threshold analysis"""
    
    # Parameters
    initial_spot = 100.0
    strike_multiplier = 1.25
    price_changes = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    threshold_return = 20.0  # 20% return threshold
    n_simulations = 10  # 10 simulations per target
    iv = 0.25
    annual_vol = 0.20
    risk_free_rate = 0.04
    days_to_expiry = 365
    
    # Run simulations
    results_df, strike_price = simulate_threshold_analysis(
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        price_changes=price_changes,
        threshold_return=threshold_return,
        n_simulations=n_simulations,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry,
        annual_vol=annual_vol
    )
    
    # Analyze results
    summary_df = analyze_threshold_results(results_df)
    
    # Create plots
    plot_threshold_analysis(results_df, threshold_return)
    
    # Save detailed results
    results_df.to_csv('straddle_threshold_simulation_results.csv', index=False)
    summary_df.to_csv('straddle_threshold_summary.csv', index=True)
    
    print(f"\n💾 Detailed results saved to: straddle_threshold_simulation_results.csv")
    print(f"📋 Summary saved to: straddle_threshold_summary.csv")
    print(f"🎉 Analysis complete!")

if __name__ == "__main__":
    main()