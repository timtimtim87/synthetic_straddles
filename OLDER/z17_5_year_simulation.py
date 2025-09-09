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

def select_trade_scenario(trade_num, cumulative_return_deficit=0.0, seed=None):
    """
    Select trading scenario with probability weighting and long-term balance adjustment
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Base probabilities: 70% bull, 20% sideways, 10% bear
    base_bull_prob = 0.70
    base_sideways_prob = 0.20
    base_bear_prob = 0.10
    
    # Adjust probabilities based on cumulative performance
    if cumulative_return_deficit > 0.05:  # More than 5% behind target
        bull_prob = min(0.80, base_bull_prob + 0.10)
        bear_prob = max(0.05, base_bear_prob - 0.05)
        sideways_prob = 1.0 - bull_prob - bear_prob
    elif cumulative_return_deficit < -0.05:  # More than 5% ahead of target
        bull_prob = max(0.60, base_bull_prob - 0.10)
        bear_prob = min(0.15, base_bear_prob + 0.05)
        sideways_prob = 1.0 - bull_prob - bear_prob
    else:
        bull_prob = base_bull_prob
        sideways_prob = base_sideways_prob
        bear_prob = base_bear_prob
    
    # Select scenario
    random_val = np.random.random()
    
    if random_val < bull_prob:
        scenario = 'bull'
        drift = np.random.uniform(0.12, 0.25)
    elif random_val < (bull_prob + sideways_prob):
        scenario = 'sideways'
        drift = np.random.uniform(0.02, 0.12)
    else:
        scenario = 'bear'
        drift = np.random.uniform(-0.10, 0.05)
    
    return scenario, drift

def generate_realistic_gbm_path(S0, annual_drift, days=365, annual_vol=0.20, dt=1/365, seed=None):
    """
    Generate pure Geometric Brownian Motion path with specified drift
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate GBM path
    n_steps = days
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)
    
    # Pure GBM: dS/S = mu*dt + sigma*dW
    for i in range(n_steps):
        log_prices[i + 1] = log_prices[i] + (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * dW[i]
    
    prices = np.exp(log_prices)
    
    return prices

def simulate_portfolio_trade_group(group_num, entry_spots, portfolio_value, 
                                 strike_multiplier=1.25, tp_threshold=20.0, days_to_expiry=365,
                                 iv=0.25, risk_free_rate=0.04, annual_vol=0.20, n_assets=5):
    """
    Simulate a group of 5 simultaneous short straddle trades with group-level TP
    """
    
    print(f"  Simulating Group {group_num} with {n_assets} assets...")
    
    # Equal allocation across assets
    capital_per_asset = portfolio_value / n_assets
    
    # Generate scenario and path for each asset
    assets_data = []
    for asset_num in range(n_assets):
        # Generate scenario for this asset
        scenario_seed = 5000 + group_num * 100 + asset_num
        return_deficit = 0.0  # Could be enhanced to track per-asset
        scenario, drift = select_trade_scenario(group_num * n_assets + asset_num, return_deficit, seed=scenario_seed)
        
        # Generate price path
        path_seed = 10000 + group_num * 100 + asset_num
        price_path = generate_realistic_gbm_path(
            S0=entry_spots[asset_num],
            annual_drift=drift,
            days=days_to_expiry,
            annual_vol=annual_vol,
            seed=path_seed
        )
        
        # Calculate strike and entry straddle value
        strike_price = entry_spots[asset_num] * strike_multiplier
        entry_straddle_value = calculate_straddle_value(
            entry_spots[asset_num], strike_price, 1.0, risk_free_rate, iv
        )
        
        # Position size for this asset
        position_size = capital_per_asset / entry_straddle_value
        
        assets_data.append({
            'asset_num': asset_num,
            'entry_spot': entry_spots[asset_num],
            'scenario': scenario,
            'drift': drift,
            'price_path': price_path,
            'strike_price': strike_price,
            'entry_straddle_value': entry_straddle_value,
            'position_size': position_size
        })
        
        print(f"    Asset {asset_num+1}: {scenario.title()} scenario, Entry ${entry_spots[asset_num]:.2f}, Strike ${strike_price:.2f}")
    
    # Simulate daily progression
    days_array = np.arange(days_to_expiry, -1, -1)
    daily_data = []
    
    exit_triggered = False
    exit_day = None
    exit_reason = "Expiry"
    
    for i, day in enumerate(days_array):
        time_to_expiry = max(day / 365.0, 1/365)
        
        # Calculate current values for all assets
        total_portfolio_value = 0
        total_straddle_value = 0
        total_pnl = 0
        asset_day_data = []
        
        for asset_data in assets_data:
            spot_price = asset_data['price_path'][i]
            
            # Calculate current straddle value
            current_straddle_value = calculate_straddle_value(
                spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv
            )
            
            # Short straddle P&L for this asset
            pnl_per_contract = asset_data['entry_straddle_value'] - current_straddle_value
            asset_pnl = pnl_per_contract * asset_data['position_size']
            asset_portfolio_value = capital_per_asset + asset_pnl
            
            # Asset return calculation
            asset_return_pct = (pnl_per_contract / asset_data['entry_straddle_value']) * 100
            
            # Store asset data
            asset_day_data.append({
                'asset_num': asset_data['asset_num'],
                'spot_price': spot_price,
                'straddle_value': current_straddle_value,
                'asset_return_pct': asset_return_pct,
                'asset_pnl': asset_pnl,
                'asset_portfolio_value': asset_portfolio_value
            })
            
            # Accumulate totals
            total_straddle_value += current_straddle_value * asset_data['position_size']
            total_pnl += asset_pnl
            total_portfolio_value += asset_portfolio_value
        
        # Calculate group-level metrics
        total_entry_value = sum(asset['entry_straddle_value'] * asset['position_size'] for asset in assets_data)
        group_return_pct = (total_pnl / (portfolio_value - total_pnl + total_pnl)) * 100  # Return on initial capital
        
        # Check for group-level 20% TP
        if group_return_pct >= tp_threshold and not exit_triggered:
            exit_triggered = True
            exit_day = i
            exit_reason = "20% TP Hit"
        
        # Store daily group data
        daily_data.append({
            'Group_Num': group_num,
            'Day': i,
            'Days_to_Expiry': day,
            'Total_Portfolio_Value': total_portfolio_value,
            'Group_Return_Pct': group_return_pct,
            'Total_PnL': total_pnl,
            'Exit_Triggered': exit_triggered,
            'Assets_Data': asset_day_data.copy()  # Store individual asset data
        })
        
        # Exit if TP hit
        if exit_triggered and exit_day == i:
            break
    
    # Calculate final results for each asset
    final_day_data = daily_data[-1]
    asset_results = []
    
    for i, asset_data in enumerate(assets_data):
        final_asset_data = final_day_data['Assets_Data'][i]
        actual_underlying_return = (asset_data['price_path'][-1] - asset_data['entry_spot']) / asset_data['entry_spot']
        
        asset_results.append({
            'Asset_Num': asset_data['asset_num'],
            'Scenario': asset_data['scenario'],
            'Annual_Drift': asset_data['drift'],
            'Entry_Spot': asset_data['entry_spot'],
            'Final_Spot': final_asset_data['spot_price'],
            'Strike_Price': asset_data['strike_price'],
            'Actual_Underlying_Return': actual_underlying_return,
            'Asset_Return_Pct': final_asset_data['asset_return_pct'],
            'Asset_PnL': final_asset_data['asset_pnl'],
            'Position_Size': asset_data['position_size']
        })
    
    # Group-level results
    group_result = {
        'Group_Num': group_num,
        'Entry_Portfolio': portfolio_value,
        'Exit_Portfolio': final_day_data['Total_Portfolio_Value'],
        'Group_PnL': final_day_data['Total_PnL'],
        'Group_Return_Pct': final_day_data['Group_Return_Pct'],
        'Portfolio_Return_Pct': ((final_day_data['Total_Portfolio_Value'] - portfolio_value) / portfolio_value) * 100,
        'Days_Held': len(daily_data),
        'Exit_Reason': exit_reason,
        'TP_Hit': exit_triggered,
        'N_Assets': n_assets
    }
    
    return group_result, daily_data, asset_results

def run_portfolio_trading_simulation(n_groups=10, initial_capital=100000, target_annual_return=0.15, n_assets=5):
    """
    Run portfolio trading simulation with groups of simultaneous straddles
    """
    
    print(f"PORTFOLIO SHORT STRADDLE TRADING SIMULATION")
    print("="*80)
    print(f"Number of groups: {n_groups}")
    print(f"Assets per group: {n_assets}")
    print(f"Total individual positions: {n_groups * n_assets}")
    print(f"Initial capital: ${initial_capital:,.0f}")
    print(f"Capital per asset: ${initial_capital/n_assets:,.0f}")
    print(f"Target annual return: {target_annual_return*100:+.0f}%")
    print(f"Group TP threshold: 20% (applied to group, not individual assets)")
    print("="*80)
    
    # Initialize tracking
    portfolio_value = initial_capital
    # Starting spots for 5 hypothetical assets
    entry_spots = [100.0, 95.0, 105.0, 98.0, 102.0]  # Slightly different starting points
    cumulative_underlying_returns = [0.0] * n_assets  # Track per-asset performance
    
    all_group_results = []
    all_daily_data = []
    all_asset_results = []
    
    # Run each group
    for group_num in range(1, n_groups + 1):
        print(f"\nGroup {group_num}: Portfolio ${portfolio_value:,.0f}")
        print(f"  Entry spots: {[f'${spot:.2f}' for spot in entry_spots]}")
        
        # Simulate the group
        group_result, daily_data, asset_results = simulate_portfolio_trade_group(
            group_num=group_num,
            entry_spots=entry_spots,
            portfolio_value=portfolio_value
        )
        
        # Update portfolio value
        portfolio_value = group_result['Exit_Portfolio']
        
        # Update entry spots for next group (use final spots from this group)
        final_day = daily_data[-1]
        entry_spots = [asset_data['spot_price'] for asset_data in final_day['Assets_Data']]
        
        # Update cumulative returns
        for i, asset_result in enumerate(asset_results):
            cumulative_underlying_returns[i] += asset_result['Actual_Underlying_Return']
        
        print(f"  Exit: {group_result['Exit_Reason']}")
        print(f"  Days Held: {group_result['Days_Held']}")
        print(f"  Group Return: {group_result['Group_Return_Pct']:+.1f}%")
        print(f"  New Portfolio: ${portfolio_value:,.0f}")
        print(f"  Final spots: {[f'${spot:.2f}' for spot in entry_spots]}")
        
        # Store results
        all_group_results.append(group_result)
        
        # Add group number to daily data for tracking
        for day_data in daily_data:
            all_daily_data.append(day_data)
        
        # Add group number to asset results
        for asset_result in asset_results:
            asset_result['Group_Num'] = group_num
            all_asset_results.append(asset_result)
    
    # Convert to DataFrames
    groups_df = pd.DataFrame(all_group_results)
    assets_df = pd.DataFrame(all_asset_results)
    
    return groups_df, all_daily_data, assets_df

def analyze_portfolio_simulation_results(groups_df, daily_data, assets_df, initial_capital):
    """
    Enhanced analysis for portfolio simulation results
    """
    
    print(f"\n" + "="*80)
    print("PORTFOLIO SIMULATION RESULTS ANALYSIS")
    print("="*80)
    
    final_portfolio = groups_df['Exit_Portfolio'].iloc[-1]
    total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
    n_assets = groups_df['N_Assets'].iloc[0]
    
    # Overall performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Initial Capital: ${initial_capital:,.0f}")
    print(f"  Final Portfolio: ${final_portfolio:,.0f}")
    print(f"  Total Return: {total_return:+.1f}%")
    print(f"  Total Groups: {len(groups_df)}")
    print(f"  Assets per Group: {n_assets}")
    print(f"  Total Positions: {len(assets_df)}")
    
    # Group statistics
    winning_groups = (groups_df['Group_Return_Pct'] > 0).sum()
    tp_hits = groups_df['TP_Hit'].sum()
    
    print(f"\nGROUP STATISTICS:")
    print(f"  Winning Groups: {winning_groups}/{len(groups_df)} ({winning_groups/len(groups_df)*100:.1f}%)")
    print(f"  TP Hits: {tp_hits}/{len(groups_df)} ({tp_hits/len(groups_df)*100:.1f}%)")
    print(f"  Average Group Return: {groups_df['Group_Return_Pct'].mean():+.1f}%")
    print(f"  Best Group: {groups_df['Group_Return_Pct'].max():+.1f}%")
    print(f"  Worst Group: {groups_df['Group_Return_Pct'].min():+.1f}%")
    print(f"  Average Days Held: {groups_df['Days_Held'].mean():.0f}")
    
    # Asset-level analysis
    print(f"\nASSET-LEVEL ANALYSIS:")
    asset_scenarios = assets_df.groupby('Scenario').agg({
        'Asset_Return_Pct': ['count', 'mean', 'std'],
        'Actual_Underlying_Return': 'mean'
    }).round(2)
    
    for scenario in ['bull', 'sideways', 'bear']:
        if scenario in asset_scenarios.index:
            stats = asset_scenarios.loc[scenario]
            count = int(stats[('Asset_Return_Pct', 'count')])
            mean_return = stats[('Asset_Return_Pct', 'mean')]
            underlying_return = stats[('Actual_Underlying_Return', 'mean')] * 100
            
            print(f"  {scenario.title()}: {count} positions ({count/len(assets_df)*100:.0f}%)")
            print(f"    Avg Straddle Return: {mean_return:+.1f}%")
            print(f"    Avg Underlying Move: {underlying_return:+.1f}%")
    
    # Portfolio metrics
    portfolio_returns = groups_df['Portfolio_Return_Pct']
    
    # Calculate max drawdown at group level
    peak_portfolio = groups_df['Exit_Portfolio'].expanding().max()
    drawdown = (groups_df['Exit_Portfolio'] - peak_portfolio) / peak_portfolio * 100
    max_drawdown = drawdown.min()
    
    print(f"\nPORTFOLIO METRICS:")
    print(f"  Max Drawdown: {max_drawdown:.1f}%")
    print(f"  Return Volatility: {portfolio_returns.std():.1f}%")
    if portfolio_returns.std() > 0:
        print(f"  Sharpe Ratio: {portfolio_returns.mean()/portfolio_returns.std():.2f}")
    
    # Diversification analysis
    print(f"\nDIVERSIFICATION ANALYSIS:")
    
    # Calculate intra-group correlations
    group_correlations = []
    for group_num in groups_df['Group_Num']:
        group_assets = assets_df[assets_df['Group_Num'] == group_num]
        if len(group_assets) >= 2:
            returns = group_assets['Asset_Return_Pct'].values
            if len(set(returns)) > 1:  # Check for variation
                # Calculate pairwise correlations (simplified)
                asset_returns_matrix = group_assets.pivot(index='Group_Num', columns='Asset_Num', values='Asset_Return_Pct')
                if not asset_returns_matrix.empty:
                    corr_matrix = asset_returns_matrix.corr()
                    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    if not np.isnan(avg_correlation):
                        group_correlations.append(avg_correlation)
    
    if group_correlations:
        avg_correlation = np.mean(group_correlations)
        print(f"  Average intra-group correlation: {avg_correlation:.3f}")
    
    # Asset performance spread within groups
    group_spreads = []
    for group_num in groups_df['Group_Num']:
        group_assets = assets_df[assets_df['Group_Num'] == group_num]
        if len(group_assets) > 1:
            spread = group_assets['Asset_Return_Pct'].max() - group_assets['Asset_Return_Pct'].min()
            group_spreads.append(spread)
    
    if group_spreads:
        avg_spread = np.mean(group_spreads)
        print(f"  Average intra-group return spread: {avg_spread:.1f}%")

def plot_portfolio_simulation_results(groups_df, daily_data, assets_df, initial_capital):
    """
    Create comprehensive plots for portfolio simulation results
    """
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: Portfolio equity curve
    ax1 = plt.subplot(4, 2, 1)
    portfolio_values = [initial_capital] + groups_df['Exit_Portfolio'].tolist()
    group_numbers = list(range(len(portfolio_values)))
    
    ax1.plot(group_numbers, portfolio_values, linewidth=3, color='blue', marker='o', markersize=6)
    ax1.fill_between(group_numbers, portfolio_values, alpha=0.3, color='blue')
    ax1.set_xlabel('Group Number')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Equity Curve - Group Trading')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Group returns distribution
    ax2 = plt.subplot(4, 2, 2)
    ax2.hist(groups_df['Group_Return_Pct'], bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax2.axvline(x=20, color='blue', linestyle='--', linewidth=2, label='20% TP')
    ax2.set_xlabel('Group Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Group Returns Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Asset returns by scenario
    ax3 = plt.subplot(4, 2, 3)
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = assets_df[assets_df['Scenario'] == scenario]['Asset_Return_Pct']
        if not scenario_data.empty:
            ax3.hist(scenario_data, alpha=0.6, label=f'{scenario.title()} ({len(scenario_data)} assets)', 
                    color=scenario_colors.get(scenario, 'gray'), bins=15)
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Breakeven')
    ax3.set_xlabel('Individual Asset Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Individual Asset Returns by Scenario')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Days held distribution
    ax4 = plt.subplot(4, 2, 4)
    tp_groups = groups_df[groups_df['TP_Hit'] == True]['Days_Held']
    expiry_groups = groups_df[groups_df['TP_Hit'] == False]['Days_Held']
    
    if len(tp_groups) > 0 and len(expiry_groups) > 0:
        ax4.hist([tp_groups, expiry_groups], bins=15, alpha=0.7, 
                 label=[f'TP Hits ({len(tp_groups)})', f'Held to Expiry ({len(expiry_groups)})'], 
                 color=['green', 'orange'])
    elif len(tp_groups) > 0:
        ax4.hist(tp_groups, bins=15, alpha=0.7, label=f'TP Hits ({len(tp_groups)})', color='green')
    elif len(expiry_groups) > 0:
        ax4.hist(expiry_groups, bins=15, alpha=0.7, label=f'Held to Expiry ({len(expiry_groups)})', color='orange')
    
    ax4.set_xlabel('Days Held')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Group Holding Period Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Intra-group diversification
    ax5 = plt.subplot(4, 2, 5)
    
    # Calculate return spread within each group
    group_spreads = []
    group_nums = []
    for group_num in groups_df['Group_Num']:
        group_assets = assets_df[assets_df['Group_Num'] == group_num]
        if len(group_assets) > 1:
            spread = group_assets['Asset_Return_Pct'].max() - group_assets['Asset_Return_Pct'].min()
            group_spreads.append(spread)
            group_nums.append(group_num)
    
    if group_spreads:
        colors = ['green' if groups_df[groups_df['Group_Num'] == gn]['TP_Hit'].iloc[0] else 'red' 
                 for gn in group_nums]
        ax5.bar(group_nums, group_spreads, color=colors, alpha=0.7)
        ax5.set_xlabel('Group Number')
        ax5.set_ylabel('Return Spread (%)')
        ax5.set_title('Intra-Group Return Spread (Green=TP Hit)')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Asset correlation heatmap (sample)
    ax6 = plt.subplot(4, 2, 6)
    
    # Create a correlation matrix for first few groups
    sample_groups = groups_df['Group_Num'].head(min(5, len(groups_df)))
    
    if len(sample_groups) > 0:
        corr_data = []
        for group_num in sample_groups:
            group_assets = assets_df[assets_df['Group_Num'] == group_num]
            returns_by_asset = {}
            for _, asset in group_assets.iterrows():
                returns_by_asset[f"Asset_{asset['Asset_Num']}"] = asset['Asset_Return_Pct']
            corr_data.append(returns_by_asset)
        
        if corr_data and len(corr_data) > 1:
            corr_df = pd.DataFrame(corr_data)
            correlation_matrix = corr_df.corr()
            
            im = ax6.imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            ax6.set_xticks(range(len(correlation_matrix.columns)))
            ax6.set_yticks(range(len(correlation_matrix.columns)))
            ax6.set_xticklabels(correlation_matrix.columns, rotation=45)
            ax6.set_yticklabels(correlation_matrix.columns)
            ax6.set_title('Asset Correlation Matrix (Sample Groups)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor correlation matrix', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    
    # Plot 7: Win rate by asset position
    ax7 = plt.subplot(4, 2, 7)
    
    # Calculate win rate by asset number
    asset_win_rates = assets_df.groupby('Asset_Num')['Asset_Return_Pct'].apply(lambda x: (x > 0).mean() * 100)
    
    bars = ax7.bar(asset_win_rates.index, asset_win_rates.values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(asset_win_rates)],
                   alpha=0.7)
    ax7.set_xlabel('Asset Number')
    ax7.set_ylabel('Win Rate (%)')
    ax7.set_title('Win Rate by Asset Position')
    ax7.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, asset_win_rates.values):
        ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 8: Summary statistics
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Calculate key statistics
    total_return = (groups_df['Exit_Portfolio'].iloc[-1] / initial_capital - 1) * 100
    avg_group_return = groups_df['Group_Return_Pct'].mean()
    group_volatility = groups_df['Group_Return_Pct'].std()
    max_drawdown = ((groups_df['Exit_Portfolio'].expanding().max() - groups_df['Exit_Portfolio']) / groups_df['Exit_Portfolio'].expanding().max() * 100).max()
    
    # Asset level stats
    avg_asset_return = assets_df['Asset_Return_Pct'].mean()
    asset_volatility = assets_df['Asset_Return_Pct'].std()
    
    # Diversification benefit
    theoretical_single_vol = asset_volatility
    actual_group_vol = group_volatility
    diversification_benefit = (theoretical_single_vol - actual_group_vol) / theoretical_single_vol * 100
    
    stats_text = f"""
PORTFOLIO SIMULATION SUMMARY
{'='*35}

PERFORMANCE:
Total Return: {total_return:+.1f}%
Avg Group Return: {avg_group_return:+.1f}%
Group Volatility: {group_volatility:.1f}%
Max Drawdown: {max_drawdown:.1f}%

DIVERSIFICATION:
Avg Asset Return: {avg_asset_return:+.1f}%
Asset Volatility: {asset_volatility:.1f}%
Diversification Benefit: {diversification_benefit:.0f}%

TRADE STATS:
Groups: {len(groups_df)}
Total Positions: {len(assets_df)}
Group Win Rate: {(groups_df['Group_Return_Pct'] > 0).mean()*100:.0f}%
TP Hit Rate: {groups_df['TP_Hit'].mean()*100:.0f}%

RISK REDUCTION:
Single Asset Risk: {theoretical_single_vol:.1f}%
Portfolio Risk: {actual_group_vol:.1f}%
Risk Reduction: {diversification_benefit:.0f}%
"""
    
    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    filename = 'portfolio_trading_simulation_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Portfolio simulation plots saved: {filename}")
    
    plt.show()

def create_continuous_portfolio_timeline(daily_data, groups_df):
    """
    Create continuous daily timeline for portfolio analysis
    """
    
    print("Creating continuous portfolio timeline...")
    
    continuous_data = []
    global_day = 0
    
    # Process each group's daily data
    for group_num in sorted(groups_df['Group_Num'].unique()):
        group_daily = [d for d in daily_data if d['Group_Num'] == group_num]
        group_daily = sorted(group_daily, key=lambda x: x['Day'])
        
        for day_data in group_daily:
            continuous_data.append({
                'Global_Day': global_day,
                'Group_Num': group_num,
                'Group_Day': day_data['Day'],
                'Days_to_Expiry': day_data['Days_to_Expiry'],
                'Total_Portfolio_Value': day_data['Total_Portfolio_Value'],
                'Group_Return_Pct': day_data['Group_Return_Pct'],
                'Total_PnL': day_data['Total_PnL'],
                'Exit_Triggered': day_data['Exit_Triggered']
            })
            global_day += 1
    
    continuous_df = pd.DataFrame(continuous_data)
    
    # Calculate drawdown and daily returns
    continuous_df['Peak_Portfolio'] = continuous_df['Total_Portfolio_Value'].expanding().max()
    continuous_df['Drawdown'] = (continuous_df['Total_Portfolio_Value'] - continuous_df['Peak_Portfolio']) / continuous_df['Peak_Portfolio'] * 100
    continuous_df['Daily_Return'] = continuous_df['Total_Portfolio_Value'].pct_change() * 100
    continuous_df['Daily_Return'] = continuous_df['Daily_Return'].fillna(0)
    
    print(f"Created continuous timeline with {len(continuous_df)} days")
    
    return continuous_df

def compare_with_single_asset_simulation(groups_df, assets_df):
    """
    Create comparison analysis showing benefits of portfolio approach vs single asset
    """
    
    print(f"\n" + "="*80)
    print("PORTFOLIO vs SINGLE ASSET COMPARISON")
    print("="*80)
    
    # Portfolio metrics
    portfolio_returns = groups_df['Group_Return_Pct']
    portfolio_volatility = portfolio_returns.std()
    portfolio_mean = portfolio_returns.mean()
    portfolio_sharpe = portfolio_mean / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Single asset metrics (average of all individual positions)
    asset_returns = assets_df['Asset_Return_Pct']
    asset_volatility = asset_returns.std()
    asset_mean = asset_returns.mean()
    asset_sharpe = asset_mean / asset_volatility if asset_volatility > 0 else 0
    
    # Risk reduction calculation
    risk_reduction = (asset_volatility - portfolio_volatility) / asset_volatility * 100
    
    print(f"\nRETURN COMPARISON:")
    print(f"  Portfolio Average: {portfolio_mean:+.1f}%")
    print(f"  Single Asset Average: {asset_mean:+.1f}%")
    print(f"  Difference: {portfolio_mean - asset_mean:+.1f}%")
    
    print(f"\nRISK COMPARISON:")
    print(f"  Portfolio Volatility: {portfolio_volatility:.1f}%")
    print(f"  Single Asset Volatility: {asset_volatility:.1f}%")
    print(f"  Risk Reduction: {risk_reduction:.1f}%")
    
    print(f"\nRISK-ADJUSTED RETURNS:")
    print(f"  Portfolio Sharpe: {portfolio_sharpe:.2f}")
    print(f"  Single Asset Sharpe: {asset_sharpe:.2f}")
    print(f"  Sharpe Improvement: {portfolio_sharpe - asset_sharpe:+.2f}")
    
    # Drawdown comparison (simplified)
    portfolio_drawdowns = []
    current_peak = groups_df['Exit_Portfolio'].iloc[0]
    for portfolio_value in groups_df['Exit_Portfolio']:
        if portfolio_value > current_peak:
            current_peak = portfolio_value
        drawdown = (portfolio_value - current_peak) / current_peak * 100
        portfolio_drawdowns.append(drawdown)
    
    max_portfolio_dd = min(portfolio_drawdowns)
    
    print(f"\nDRAWDOWN COMPARISON:")
    print(f"  Portfolio Max Drawdown: {max_portfolio_dd:.1f}%")
    print(f"  (Single asset drawdowns would typically be higher)")
    
    return {
        'portfolio_volatility': portfolio_volatility,
        'asset_volatility': asset_volatility,
        'risk_reduction': risk_reduction,
        'portfolio_sharpe': portfolio_sharpe,
        'asset_sharpe': asset_sharpe
    }

def main():
    """
    Main function to run the portfolio simulation
    """
    
    # Run portfolio simulation
    groups_df, daily_data, assets_df = run_portfolio_trading_simulation(
        n_groups=10,  # 10 groups of trades
        initial_capital=100000,
        target_annual_return=0.15,
        n_assets=5  # 5 assets per group
    )
    
    # Analyze results
    analyze_portfolio_simulation_results(groups_df, daily_data, assets_df, 100000)
    
    # Create comparison analysis
    comparison_metrics = compare_with_single_asset_simulation(groups_df, assets_df)
    
    # Create plots
    plot_portfolio_simulation_results(groups_df, daily_data, assets_df, 100000)
    
    # Create continuous timeline for detailed analysis
    continuous_df = create_continuous_portfolio_timeline(daily_data, groups_df)
    
    # Save results to CSV
    groups_df.to_csv('portfolio_simulation_groups.csv', index=False)
    assets_df.to_csv('portfolio_simulation_assets.csv', index=False)
    continuous_df.to_csv('portfolio_continuous_timeline.csv', index=False)
    
    # Create detailed daily data CSV
    detailed_daily_data = []
    for day_data in daily_data:
        base_data = {
            'Group_Num': day_data['Group_Num'],
            'Day': day_data['Day'],
            'Days_to_Expiry': day_data['Days_to_Expiry'],
            'Total_Portfolio_Value': day_data['Total_Portfolio_Value'],
            'Group_Return_Pct': day_data['Group_Return_Pct'],
            'Total_PnL': day_data['Total_PnL'],
            'Exit_Triggered': day_data['Exit_Triggered']
        }
        
        # Add individual asset data
        for asset_data in day_data['Assets_Data']:
            asset_row = base_data.copy()
            asset_row.update({
                'Asset_Num': asset_data['asset_num'],
                'Asset_Spot_Price': asset_data['spot_price'],
                'Asset_Straddle_Value': asset_data['straddle_value'],
                'Asset_Return_Pct': asset_data['asset_return_pct'],
                'Asset_PnL': asset_data['asset_pnl'],
                'Asset_Portfolio_Value': asset_data['asset_portfolio_value']
            })
            detailed_daily_data.append(asset_row)
    
    detailed_daily_df = pd.DataFrame(detailed_daily_data)
    detailed_daily_df.to_csv('portfolio_simulation_daily_detailed.csv', index=False)
    
    print(f"\n💾 Portfolio simulation results saved:")
    print(f"  - portfolio_simulation_groups.csv ({len(groups_df)} groups)")
    print(f"  - portfolio_simulation_assets.csv ({len(assets_df)} individual assets)")
    print(f"  - portfolio_continuous_timeline.csv ({len(continuous_df)} daily records)")
    print(f"  - portfolio_simulation_daily_detailed.csv ({len(detailed_daily_df)} detailed daily records)")
    
    print(f"\n🎉 Portfolio simulation complete!")
    print(f"📈 Portfolio grew from $100,000 to ${groups_df['Exit_Portfolio'].iloc[-1]:,.0f}")
    print(f"📊 Risk reduction achieved: {comparison_metrics['risk_reduction']:.1f}%")
    print(f"📉 Volatility reduced from {comparison_metrics['asset_volatility']:.1f}% to {comparison_metrics['portfolio_volatility']:.1f}%")

if __name__ == "__main__":
    main()