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

def generate_realistic_gbm_path(S0, annual_drift, days=365, annual_vol=0.20, dt=1/365, seed=None):
    """
    Generate pure Geometric Brownian Motion path with realistic market behavior
    No artificial endpoint constraints - let the market decide where it goes!
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Pure GBM parameters
    n_steps = days
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    # Initialize price array
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)
    
    # Generate realistic GBM path
    for i in range(n_steps):
        log_prices[i + 1] = log_prices[i] + (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * dW[i]
    
    # Convert back to prices
    prices = np.exp(log_prices)
    
    return prices

def select_market_scenario(seed=None):
    """
    Select realistic market scenario with probability weighting
    Returns drift rate for that scenario
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Market scenario probabilities
    scenarios = {
        'bull': {'prob': 0.40, 'drift_range': (0.08, 0.25)},      # 40% chance, 8-25% annual
        'moderate': {'prob': 0.30, 'drift_range': (0.02, 0.08)},  # 30% chance, 2-8% annual  
        'sideways': {'prob': 0.20, 'drift_range': (-0.02, 0.02)}, # 20% chance, -2 to +2% annual
        'bear': {'prob': 0.10, 'drift_range': (-0.15, -0.02)}     # 10% chance, -15 to -2% annual
    }
    
    # Select scenario based on probabilities
    rand_val = np.random.random()
    cumulative_prob = 0
    
    for scenario, params in scenarios.items():
        cumulative_prob += params['prob']
        if rand_val <= cumulative_prob:
            # Select random drift within the scenario range
            drift = np.random.uniform(params['drift_range'][0], params['drift_range'][1])
            return scenario, drift
    
    # Fallback (shouldn't reach here)
    return 'moderate', 0.05

def generate_many_realistic_paths(initial_spot, n_simulations=500, days_to_expiry=365, annual_vol=0.20):
    """
    Generate many realistic random walks and return their details
    """
    
    print(f"Generating {n_simulations} realistic random walks...")
    print("Market scenarios: 40% Bull | 30% Moderate | 20% Sideways | 10% Bear")
    
    all_simulations = []
    
    for sim in range(n_simulations):
        # Select market scenario
        scenario_seed = 10000 + sim
        scenario, drift = select_market_scenario(seed=scenario_seed)
        
        # Generate realistic price path
        path_seed = 20000 + sim
        price_path = generate_realistic_gbm_path(
            S0=initial_spot,
            annual_drift=drift,
            days=days_to_expiry,
            annual_vol=annual_vol,
            seed=path_seed
        )
        
        # Calculate final return
        final_return = (price_path[-1] - initial_spot) / initial_spot
        final_return_pct = final_return * 100
        
        # Store simulation details
        simulation_data = {
            'simulation_id': sim,
            'scenario': scenario,
            'annual_drift': drift,
            'price_path': price_path,
            'final_price': price_path[-1],
            'final_return': final_return,
            'final_return_pct': final_return_pct,
            'path_volatility': np.std(np.diff(np.log(price_path))) * np.sqrt(365) * 100
        }
        
        all_simulations.append(simulation_data)
        
        if (sim + 1) % 100 == 0:
            print(f"  Generated {sim + 1}/{n_simulations} simulations...")
    
    print(f"✅ Generated {len(all_simulations)} realistic random walks")
    
    return all_simulations

def group_simulations_by_outcome(all_simulations, initial_spot, target_groups=None, sims_per_group=10):
    """
    Group simulations by their final price outcome (post-hoc grouping)
    """
    
    if target_groups is None:
        # Define outcome buckets (% change from initial spot)
        target_groups = [
            (-10, 0),    # -10% to 0%
            (0, 10),     # 0% to +10%
            (10, 20),    # +10% to +20%
            (20, 30),    # +20% to +30%
            (30, 40),    # +30% to +40%
            (40, 50),    # +40% to +50%
        ]
    
    print(f"\nGrouping simulations by final price outcome...")
    print(f"Target: {sims_per_group} simulations per group")
    
    grouped_results = {}
    
    # Sort simulations by final return for easier debugging
    sorted_sims = sorted(all_simulations, key=lambda x: x['final_return_pct'])
    
    print(f"\nOutcome distribution of {len(sorted_sims)} simulations:")
    print(f"  Worst: {sorted_sims[0]['final_return_pct']:+.1f}%")
    print(f"  Best: {sorted_sims[-1]['final_return_pct']:+.1f}%")
    print(f"  Median: {sorted_sims[len(sorted_sims)//2]['final_return_pct']:+.1f}%")
    
    for group_min, group_max in target_groups:
        group_name = f"{group_min:+.0f}% to {group_max:+.0f}%"
        
        # Find simulations that fall in this range
        matching_sims = [
            sim for sim in all_simulations 
            if group_min <= sim['final_return_pct'] < group_max
        ]
        
        print(f"\n{group_name}: Found {len(matching_sims)} matching simulations")
        
        if len(matching_sims) >= sims_per_group:
            # Randomly select the required number
            np.random.seed(42)  # For reproducible selection
            selected_sims = np.random.choice(matching_sims, size=sims_per_group, replace=False)
            grouped_results[group_name] = list(selected_sims)
            
            # Show scenario breakdown
            scenarios = [sim['scenario'] for sim in selected_sims]
            scenario_counts = pd.Series(scenarios).value_counts()
            print(f"  ✅ Selected {sims_per_group} simulations")
            print(f"  Scenario breakdown: {dict(scenario_counts)}")
            
        else:
            print(f"  ⚠️  Not enough simulations ({len(matching_sims)} < {sims_per_group})")
            if len(matching_sims) > 0:
                grouped_results[group_name] = matching_sims
                print(f"  📝 Using all {len(matching_sims)} available simulations")
    
    return grouped_results

def calculate_straddle_analysis_for_groups(grouped_results, initial_spot, strike_multiplier=1.25, 
                                         iv=0.25, risk_free_rate=0.04, days_to_expiry=365):
    """
    Calculate straddle values and returns for grouped simulations
    """
    
    strike_price = initial_spot * strike_multiplier
    
    print(f"\nCalculating straddle analysis...")
    print(f"Strike price: ${strike_price:.0f} ({strike_multiplier*100-100:+.0f}% above ${initial_spot:.0f})")
    
    enhanced_results = {}
    
    for group_name, simulations in grouped_results.items():
        print(f"\nProcessing {group_name} group ({len(simulations)} simulations)...")
        
        group_results = []
        
        for sim in simulations:
            price_path = sim['price_path']
            days_array = np.arange(days_to_expiry, -1, -1)
            
            # Calculate straddle values and returns
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
            
            # Enhanced simulation data
            enhanced_sim = {
                **sim,  # Include all original data
                'days_to_expiry': days_array,
                'straddle_values': np.array(straddle_values),
                'short_returns': np.array(short_returns),
                'moneyness': np.array(moneyness_values),
                'final_straddle_return': short_returns[-1],
                'max_straddle_return': max(short_returns),
                'min_straddle_return': min(short_returns),
                'strike_price': strike_price
            }
            
            group_results.append(enhanced_sim)
        
        enhanced_results[group_name] = group_results
        
        # Group statistics
        final_returns = [sim['final_straddle_return'] for sim in group_results]
        print(f"  Straddle returns: {min(final_returns):+.1f}% to {max(final_returns):+.1f}%")
        print(f"  Average straddle return: {np.mean(final_returns):+.1f}%")
    
    return enhanced_results

def calculate_proper_straddle_volatility_by_dte(enhanced_results, method='daily_returns'):
    """
    Calculate proper straddle volatility - how much straddle prices change day-to-day
    
    Methods:
    - 'daily_returns': Standard financial volatility (daily % changes)
    - 'rolling_range': Rolling 5-day high-low range as % of price
    - 'rolling_std': 5-day rolling standard deviation of daily changes
    """
    
    print(f"\nCalculating proper straddle volatility using {method} method...")
    
    all_volatility_data = []
    
    for group_name, simulations in enhanced_results.items():
        for sim in simulations:
            days_to_expiry = sim['days_to_expiry']
            straddle_values = sim['straddle_values']
            
            if method == 'daily_returns':
                # Calculate daily % changes in straddle values
                daily_returns = np.diff(straddle_values) / straddle_values[:-1] * 100
                # Use absolute value of daily returns as volatility measure
                volatilities = np.abs(daily_returns)
                dte_array = days_to_expiry[1:]  # Match length after diff
                
            elif method == 'rolling_range':
                # 5-day rolling high-low range as % of current price
                window = 5
                volatilities = []
                dte_array = []
                
                for i in range(window, len(straddle_values)):
                    window_values = straddle_values[i-window:i]
                    range_vol = (np.max(window_values) - np.min(window_values)) / straddle_values[i] * 100
                    volatilities.append(range_vol)
                    dte_array.append(days_to_expiry[i])
                    
                volatilities = np.array(volatilities)
                dte_array = np.array(dte_array)
                
            elif method == 'rolling_std':
                # 5-day rolling standard deviation of daily % changes
                window = 5
                daily_returns = np.diff(straddle_values) / straddle_values[:-1] * 100
                volatilities = []
                dte_array = []
                
                for i in range(window, len(daily_returns)):
                    window_returns = daily_returns[i-window:i]
                    vol = np.std(window_returns)
                    volatilities.append(vol)
                    dte_array.append(days_to_expiry[i+1])  # Adjust for diff offset
                    
                volatilities = np.array(volatilities)
                dte_array = np.array(dte_array)
            
            # Store each day's volatility measurement
            for dte, vol in zip(dte_array, volatilities):
                all_volatility_data.append({
                    'DTE': dte,
                    'Volatility': vol,
                    'Group': group_name,
                    'Simulation': sim['simulation_id']
                })
    
    # Convert to DataFrame and aggregate by DTE
    vol_df = pd.DataFrame(all_volatility_data)
    
    if len(vol_df) == 0:
        print("Warning: No volatility data calculated")
        return pd.DataFrame(), pd.DataFrame()
    
    dte_summary = vol_df.groupby('DTE').agg({
        'Volatility': ['mean', 'std', 'count']
    }).round(2)
    
    dte_summary.columns = ['Mean_Volatility', 'Std_Volatility', 'Sample_Size']
    dte_summary = dte_summary.reset_index()
    
    print(f"Proper volatility analysis complete for {len(dte_summary)} DTE levels")
    
    return dte_summary, vol_df

# END OF PART 1
# ============================================================================
# This is the halfway point. The second half will contain:
# - compare_volatility_methods()
# - All plotting functions
# - save_detailed_results()
# - main() function
# ============================================================================



# PART 2 - CONTINUING FROM PART 1
# ============================================================================

def compare_volatility_methods(enhanced_results):
    """
    Compare all three volatility calculation methods
    """
    
    methods = {
        'daily_returns': 'Daily % Change (Absolute)',
        'rolling_range': '5-Day Rolling Range %',
        'rolling_std': '5-Day Rolling Std Dev'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Straddle Volatility Analysis - Method Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    
    for i, (method, label) in enumerate(methods.items()):
        ax = axes[i//2, i%2] if i < 3 else None
        
        if ax is not None:
            dte_summary, vol_df = calculate_proper_straddle_volatility_by_dte(enhanced_results, method=method)
            
            if len(dte_summary) > 0:
                # Plot mean volatility by DTE
                ax.plot(dte_summary['DTE'], dte_summary['Mean_Volatility'], 
                       linewidth=2, color=colors[i], marker='o', markersize=3)
                
                # Add error bands
                ax.fill_between(dte_summary['DTE'], 
                               dte_summary['Mean_Volatility'] - dte_summary['Std_Volatility'],
                               dte_summary['Mean_Volatility'] + dte_summary['Std_Volatility'],
                               alpha=0.3, color=colors[i])
                
                ax.set_xlabel('Days to Expiry')
                ax.set_ylabel('Volatility (%)')
                ax.set_title(f'{label}')
                ax.grid(True, alpha=0.3)
                ax.invert_xaxis()
                
                # Show key statistics
                max_vol = dte_summary['Mean_Volatility'].max()
                max_vol_dte = dte_summary.loc[dte_summary['Mean_Volatility'].idxmax(), 'DTE']
                
                ax.text(0.02, 0.98, f'Max: {max_vol:.1f}% at {max_vol_dte:.0f} DTE', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       verticalalignment='top')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    # Fourth plot: comparison overlay
    ax4 = axes[1, 1]
    
    for i, (method, label) in enumerate(methods.items()):
        dte_summary, _ = calculate_proper_straddle_volatility_by_dte(enhanced_results, method=method)
        if len(dte_summary) > 0:
            ax4.plot(dte_summary['DTE'], dte_summary['Mean_Volatility'], 
                    linewidth=2, color=colors[i], label=label, alpha=0.8)
    
    ax4.set_xlabel('Days to Expiry')
    ax4.set_ylabel('Volatility (%)')
    ax4.set_title('All Methods Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('straddle_volatility_methods_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Volatility comparison saved: straddle_volatility_methods_comparison.png")
    plt.show()

def plot_individual_group_analysis(simulations, group_name, initial_spot, strike_price):
    """
    Create individual plot for each outcome group showing all simulations as separate subplots
    """
    
    plt.style.use('default')
    
    # Determine grid size based on number of simulations
    n_sims = len(simulations)
    if n_sims <= 10:
        n_cols = 5
        n_rows = 2
    else:
        n_cols = 5
        n_rows = (n_sims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle(f'Short Straddle Returns - {group_name} Final Price Outcome\n(Strike: ${strike_price:.0f}, Initial Spot: ${initial_spot:.0f})', 
                 fontsize=16, fontweight='bold')
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for each simulation
    colors = plt.cm.Set3(np.linspace(0, 1, n_sims))
    
    for i, sim in enumerate(simulations):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot the return curve
        ax.plot(sim['days_to_expiry'], sim['short_returns'], 
               linewidth=2, color=colors[i], alpha=0.8)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=1)
        
        # Fill profit/loss areas
        ax.fill_between(sim['days_to_expiry'], sim['short_returns'], 0,
                       where=(sim['short_returns'] >= 0), alpha=0.3, color='green')
        ax.fill_between(sim['days_to_expiry'], sim['short_returns'], 0,
                       where=(sim['short_returns'] < 0), alpha=0.3, color='red')
        
        # Formatting
        final_straddle_return = sim['final_straddle_return']
        final_price_return = sim['final_return_pct']
        scenario = sim['scenario']
        
        ax.set_title(f'Sim {i+1} ({scenario.title()})\nStraddle: {final_straddle_return:+.1f}% | Price: {final_price_return:+.1f}%', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Days to Expiry', fontsize=9)
        ax.set_ylabel('Straddle Return (%)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Set consistent y-limits for comparison
        ax.set_ylim(-60, 60)
        
        # Highlight final point
        final_color = '#8B0000' if final_straddle_return < 0 else '#006400'
        ax.scatter([sim['days_to_expiry'][-1]], [final_straddle_return], 
                  color=final_color, s=50, zorder=5, edgecolor='white', linewidth=1.5)
        
        # Add statistics text box
        stats_text = f'Max: {sim["max_straddle_return"]:+.1f}%\nMin: {sim["min_straddle_return"]:+.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
    
    # Hide unused subplots
    for i in range(n_sims, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save individual plot
    filename = f'realistic_random_walk_{group_name.replace(" ", "_").replace("%", "pct").replace("+", "plus").replace("-", "minus")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {filename}")
    
    plt.show()

def plot_overview_analysis(enhanced_results, volatility_df, initial_spot, strike_price):
    """
    Create comprehensive overview analysis
    """
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # Get all groups and their data
    group_names = list(enhanced_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(group_names)))
    group_colors = {name: color for name, color in zip(group_names, colors)}
    
    # Plot 1: Average return curves by outcome group
    ax1 = plt.subplot(2, 3, 1)
    
    for group_name, simulations in enhanced_results.items():
        # Calculate average return curve for this group
        all_returns = np.array([sim['short_returns'] for sim in simulations])
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        days = simulations[0]['days_to_expiry']
        
        # Plot mean with error bands
        ax1.plot(days, mean_returns, color=group_colors[group_name], 
                linewidth=2, label=group_name)
        ax1.fill_between(days, mean_returns - std_returns, mean_returns + std_returns,
                        color=group_colors[group_name], alpha=0.2)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax1.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='20% TP')
    ax1.set_xlabel('Days to Expiry')
    ax1.set_ylabel('Average Straddle Return (%)')
    ax1.set_title('Average Straddle Returns by Price Outcome')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Final straddle return distributions
    ax2 = plt.subplot(2, 3, 2)
    
    final_returns_by_group = {}
    for group_name, simulations in enhanced_results.items():
        final_returns = [sim['final_straddle_return'] for sim in simulations]
        final_returns_by_group[group_name] = final_returns
    
    # Create violin plot
    positions = range(1, len(final_returns_by_group) + 1)
    violin_parts = ax2.violinplot(final_returns_by_group.values(), positions=positions, showmeans=True)
    
    # Color the violins
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([name.replace(' to ', '\nto\n') for name in final_returns_by_group.keys()], fontsize=8)
    ax2.set_xlabel('Final Price Outcome Group')
    ax2.set_ylabel('Final Straddle Return (%)')
    ax2.set_title('Straddle Return Distributions by Price Outcome')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Straddle volatility by DTE
    ax3 = plt.subplot(2, 3, 3)
    
    if len(volatility_df) > 0:
        ax3.plot(volatility_df['DTE'], volatility_df['Mean_Volatility'], 
                 linewidth=3, color='purple', marker='o', markersize=3)
        ax3.set_xlabel('Days to Expiry')
        ax3.set_ylabel('Straddle Price Volatility (%)')
        ax3.set_title('Straddle Price Volatility by DTE')
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
        
        # Highlight high volatility region
        high_vol_threshold = volatility_df['Mean_Volatility'].quantile(0.75)
        high_vol_data = volatility_df[volatility_df['Mean_Volatility'] >= high_vol_threshold]
        if not high_vol_data.empty:
            ax3.scatter(high_vol_data['DTE'], high_vol_data['Mean_Volatility'], 
                       color='red', s=50, alpha=0.7, label=f'High Vol (>{high_vol_threshold:.1f}%)')
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No volatility data', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Market scenario distribution
    ax4 = plt.subplot(2, 3, 4)
    
    all_scenarios = []
    for simulations in enhanced_results.values():
        all_scenarios.extend([sim['scenario'] for sim in simulations])
    
    scenario_counts = pd.Series(all_scenarios).value_counts()
    scenario_colors_pie = ['#2E8B57', '#FF8C00', '#87CEEB', '#DC143C']
    
    wedges, texts, autotexts = ax4.pie(scenario_counts.values, labels=scenario_counts.index,
                                      autopct='%1.0f%%', colors=scenario_colors_pie[:len(scenario_counts)], 
                                      startangle=90)
    ax4.set_title('Market Scenario Distribution\n(Across All Selected Paths)')
    
    # Plot 5: Price outcome vs straddle outcome correlation
    ax5 = plt.subplot(2, 3, 5)
    
    price_returns = []
    straddle_returns = []
    scenario_labels = []
    
    for simulations in enhanced_results.values():
        for sim in simulations:
            price_returns.append(sim['final_return_pct'])
            straddle_returns.append(sim['final_straddle_return'])
            scenario_labels.append(sim['scenario'])
    
    # Color by scenario
    scenario_color_map = {'bull': '#2E8B57', 'moderate': '#FF8C00', 'sideways': '#87CEEB', 'bear': '#DC143C'}
    point_colors = [scenario_color_map.get(scenario, 'gray') for scenario in scenario_labels]
    
    ax5.scatter(price_returns, straddle_returns, c=point_colors, alpha=0.7, s=50)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Final Price Return (%)')
    ax5.set_ylabel('Final Straddle Return (%)')
    ax5.set_title('Price vs Straddle Return Correlation')
    ax5.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(price_returns) > 1:
        correlation = np.corrcoef(price_returns, straddle_returns)[0, 1]
        ax5.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax5.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate overall statistics
    total_simulations = sum(len(sims) for sims in enhanced_results.values())
    all_straddle_returns = [sim['final_straddle_return'] for sims in enhanced_results.values() for sim in sims]
    profitable_straddles = sum(1 for r in all_straddle_returns if r > 0)
    
    # Volatility statistics
    if len(volatility_df) > 0:
        near_expiry_vol = volatility_df[volatility_df['DTE'] <= 30]['Mean_Volatility'].mean()
        far_expiry_vol = volatility_df[volatility_df['DTE'] >= 300]['Mean_Volatility'].mean()
        max_vol = volatility_df['Mean_Volatility'].max()
        max_vol_dte = volatility_df.loc[volatility_df['Mean_Volatility'].idxmax(), 'DTE']
    else:
        near_expiry_vol = far_expiry_vol = max_vol = max_vol_dte = 0
    
    # Correlation
    correlation = np.corrcoef(price_returns, straddle_returns)[0, 1] if len(price_returns) > 1 else 0
    
    stats_text = f"""
REALISTIC RANDOM WALK ANALYSIS
{'='*34}

SIMULATION OVERVIEW:
  Total Paths: {total_simulations}
  Outcome Groups: {len(enhanced_results)}
  Natural Market Behavior: ✓

STRADDLE PERFORMANCE:
  Profitable: {profitable_straddles}/{total_simulations} ({profitable_straddles/total_simulations*100:.0f}%)
  Average Return: {np.mean(all_straddle_returns):+.1f}%
  Best: {max(all_straddle_returns):+.1f}%
  Worst: {min(all_straddle_returns):+.1f}%

PRICE vs STRADDLE:
  Correlation: {correlation:.3f}
  (Negative = good for short straddles)

VOLATILITY ANALYSIS:
  Near Expiry (<30 DTE): {near_expiry_vol:.1f}%
  Far Expiry (>300 DTE): {far_expiry_vol:.1f}%
  Peak: {max_vol:.1f}% at {max_vol_dte:.0f} DTE

REALISM IMPROVEMENT:
  ✓ Natural price movements
  ✓ No artificial endpoints
  ✓ Market scenario weighting
  ✓ Post-hoc outcome grouping
  ✓ Proper volatility calculation
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save overview plot
    filename = 'realistic_random_walk_overview.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Overview analysis saved: {filename}")
    
    plt.show()

def plot_grouped_analysis(enhanced_results, volatility_df, initial_spot, strike_price):
    """
    Create plots showing the grouped analysis results
    """
    
    print(f"\n📊 Creating grouped analysis plots...")
    
    # Plot individual target analysis for each group
    for group_name, simulations in enhanced_results.items():
        plot_individual_group_analysis(simulations, group_name, initial_spot, strike_price)
    
    # Create overview plot
    plot_overview_analysis(enhanced_results, volatility_df, initial_spot, strike_price)

def save_detailed_results(enhanced_results, volatility_df):
    """
    Save detailed results to CSV files
    """
    
    print(f"\n💾 Saving detailed results...")
    
    # Save detailed simulation data
    detailed_data = []
    for group_name, simulations in enhanced_results.items():
        for sim in simulations:
            for i, (dte, spot, straddle, straddle_ret) in enumerate(zip(
                sim['days_to_expiry'], sim['price_path'], 
                sim['straddle_values'], sim['short_returns'])):
                
                detailed_data.append({
                    'Group': group_name,
                    'Simulation_ID': sim['simulation_id'],
                    'Scenario': sim['scenario'],
                    'Annual_Drift': sim['annual_drift'],
                    'Day_Index': i,
                    'Days_to_Expiry': dte,
                    'Spot_Price': spot,
                    'Straddle_Value': straddle,
                    'Short_Return_Pct': straddle_ret,
                    'Moneyness': sim['moneyness'][i],
                    'Final_Price_Return_Pct': sim['final_return_pct'],
                    'Final_Straddle_Return_Pct': sim['final_straddle_return']
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('realistic_random_walk_detailed_data.csv', index=False)
    
    # Save volatility analysis
    volatility_df.to_csv('realistic_straddle_volatility_by_dte.csv', index=False)
    
    # Save summary by group
    summary_data = []
    for group_name, simulations in enhanced_results.items():
        straddle_returns = [sim['final_straddle_return'] for sim in simulations]
        price_returns = [sim['final_return_pct'] for sim in simulations]
        scenarios = [sim['scenario'] for sim in simulations]
        
        summary_data.append({
            'Group': group_name,
            'Simulations': len(simulations),
            'Avg_Price_Return': np.mean(price_returns),
            'Avg_Straddle_Return': np.mean(straddle_returns),
            'Std_Straddle_Return': np.std(straddle_returns),
            'Min_Straddle_Return': min(straddle_returns),
            'Max_Straddle_Return': max(straddle_returns),
            'Profitable_Count': sum(1 for r in straddle_returns if r > 0),
            'Profitable_Rate': sum(1 for r in straddle_returns if r > 0) / len(straddle_returns) * 100,
            'Bull_Count': scenarios.count('bull'),
            'Moderate_Count': scenarios.count('moderate'),
            'Sideways_Count': scenarios.count('sideways'),
            'Bear_Count': scenarios.count('bear')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('realistic_random_walk_summary.csv', index=False)
    
    print(f"  📄 realistic_random_walk_detailed_data.csv ({len(detailed_df)} records)")
    print(f"  📄 realistic_straddle_volatility_by_dte.csv ({len(volatility_df)} DTE levels)")
    print(f"  📄 realistic_random_walk_summary.csv ({len(summary_df)} groups)")

def main():
    """
    Main function to run the realistic random walk analysis with proper volatility calculation
    """
    
    # Parameters
    initial_spot = 100.0
    strike_multiplier = 1.25  # Strike 25% above spot
    n_simulations = 500  # Generate many simulations to get good distribution
    iv = 0.25
    annual_vol = 0.20  # Consistent 20% volatility
    risk_free_rate = 0.04
    days_to_expiry = 365
    sims_per_group = 10  # Target 10 simulations per outcome group
    
    print(f"REALISTIC RANDOM WALK STRADDLE ANALYSIS")
    print("="*80)
    print(f"Strategy: Post-hoc grouping by natural market outcomes")
    print(f"Initial simulations: {n_simulations}")
    print(f"Target per group: {sims_per_group}")
    print(f"Volatility: {annual_vol*100:.0f}% (consistent across all paths)")
    print(f"Strike: {strike_multiplier*100-100:+.0f}% above spot (${initial_spot * strike_multiplier:.0f})")
    print(f"✅ IMPROVED: Proper straddle volatility calculation (day-to-day changes)")
    print("="*80)
    
    # Step 1: Generate many realistic random walks
    all_simulations = generate_many_realistic_paths(
        initial_spot=initial_spot,
        n_simulations=n_simulations,
        days_to_expiry=days_to_expiry,
        annual_vol=annual_vol
    )
    
    # Step 2: Group simulations by their natural outcomes
    grouped_results = group_simulations_by_outcome(
        all_simulations=all_simulations,
        initial_spot=initial_spot,
        sims_per_group=sims_per_group
    )
    
    if not grouped_results:
        print("❌ No groups could be formed with sufficient simulations")
        print("Try increasing n_simulations or decreasing sims_per_group")
        return
    
    # Step 3: Calculate straddle analysis for grouped results
    enhanced_results = calculate_straddle_analysis_for_groups(
        grouped_results=grouped_results,
        initial_spot=initial_spot,
        strike_multiplier=strike_multiplier,
        iv=iv,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry
    )
    
    # Step 4: Calculate proper straddle volatility analysis
    print(f"\n📊 Running improved volatility analysis...")
    
    # Compare different volatility calculation methods
    compare_volatility_methods(enhanced_results)
    
    # Use daily returns method as the primary measure
    volatility_df, vol_detail_df = calculate_proper_straddle_volatility_by_dte(
        enhanced_results, method='daily_returns'
    )
    
    # Step 5: Create all plots
    plot_grouped_analysis(enhanced_results, volatility_df, initial_spot, initial_spot * strike_multiplier)
    
    # Step 6: Save detailed results
    save_detailed_results(enhanced_results, volatility_df)
    
    # Final summary
    total_selected = sum(len(sims) for sims in enhanced_results.values())
    all_final_returns = [sim['final_straddle_return'] for sims in enhanced_results.values() for sim in sims]
    
    print(f"\n🎉 Realistic random walk analysis complete!")
    print(f"📊 Generated {n_simulations} natural random walks")
    print(f"🎯 Selected {total_selected} paths across {len(enhanced_results)} outcome groups")
    print(f"📈 Average straddle return: {np.mean(all_final_returns):+.1f}%")
    print(f"🎲 Natural market behavior with realistic volatility patterns")
    print(f"✅ IMPROVED: Proper volatility measures day-to-day straddle price changes")
    print(f"📁 Check output files for detailed analysis")
    
    if len(volatility_df) > 0:
        print(f"\n📊 VOLATILITY INSIGHTS:")
        print(f"  Average straddle volatility: {volatility_df['Mean_Volatility'].mean():.1f}%")
        print(f"  Maximum volatility: {volatility_df['Mean_Volatility'].max():.1f}% at {volatility_df.loc[volatility_df['Mean_Volatility'].idxmax(), 'DTE']:.0f} DTE")
        print(f"  Minimum volatility: {volatility_df['Mean_Volatility'].min():.1f}% at {volatility_df.loc[volatility_df['Mean_Volatility'].idxmin(), 'DTE']:.0f} DTE")

if __name__ == "__main__":
    main()
    