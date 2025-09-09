import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1/3: CORE FUNCTIONS AND BLACK-SCHOLES CALCULATIONS
# ============================================================================

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

def simulate_comprehensive_trade_group(group_num, entry_spots, portfolio_value, 
                                     strike_multiplier=1.25, tp_threshold=20.0, days_to_expiry=365,
                                     iv=0.25, risk_free_rate=0.04, annual_vol=0.20, n_assets=5):
    """
    Enhanced simulation that tracks detailed daily data for both individual assets and group
    """
    
    print(f"  📊 Simulating Group {group_num} with detailed tracking...")
    
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
    
    # Simulate daily progression with comprehensive tracking
    days_array = np.arange(days_to_expiry, -1, -1)
    comprehensive_daily_data = []
    
    exit_triggered = False
    exit_day = None
    exit_reason = "Expiry"
    
    for i, day in enumerate(days_array):
        time_to_expiry = max(day / 365.0, 1/365)
        
        # Calculate current values for all assets
        total_portfolio_value = 0
        total_straddle_value = 0
        total_pnl = 0
        
        # Individual asset calculations
        asset_day_details = []
        
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
            
            # Asset return calculations
            asset_return_pct = (pnl_per_contract / asset_data['entry_straddle_value']) * 100
            underlying_return_pct = ((spot_price - asset_data['entry_spot']) / asset_data['entry_spot']) * 100
            moneyness = spot_price / asset_data['strike_price']
            
            # Greeks calculations (simplified)
            call_value = black_scholes_call(spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv)
            put_value = black_scholes_put(spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv)
            
            # Store detailed asset data
            asset_details = {
                'asset_num': asset_data['asset_num'],
                'scenario': asset_data['scenario'],
                'drift': asset_data['drift'],
                'spot_price': spot_price,
                'strike_price': asset_data['strike_price'],
                'straddle_value': current_straddle_value,
                'call_value': call_value,
                'put_value': put_value,
                'asset_return_pct': asset_return_pct,
                'underlying_return_pct': underlying_return_pct,
                'asset_pnl': asset_pnl,
                'asset_portfolio_value': asset_portfolio_value,
                'moneyness': moneyness,
                'position_size': asset_data['position_size'],
                'entry_straddle_value': asset_data['entry_straddle_value'],
                'entry_spot': asset_data['entry_spot']
            }
            
            asset_day_details.append(asset_details)
            
            # Accumulate totals
            total_straddle_value += current_straddle_value * asset_data['position_size']
            total_pnl += asset_pnl
            total_portfolio_value += asset_portfolio_value
        
        # Calculate group-level metrics
        total_entry_value = sum(asset['entry_straddle_value'] * asset['position_size'] for asset in assets_data)
        group_return_pct = (total_pnl / portfolio_value) * 100  # Return on initial capital
        portfolio_return_pct = ((total_portfolio_value - portfolio_value) / portfolio_value) * 100
        
        # Check for group-level 20% TP
        if group_return_pct >= tp_threshold and not exit_triggered:
            exit_triggered = True
            exit_day = i
            exit_reason = "20% TP Hit"
        
        # Store comprehensive daily data
        daily_data_entry = {
            'Group_Num': group_num,
            'Day_Index': i,
            'Calendar_Day': i,  # Could be enhanced with actual dates
            'Days_to_Expiry': day,
            'Total_Portfolio_Value': total_portfolio_value,
            'Group_Return_Pct': group_return_pct,
            'Portfolio_Return_Pct': portfolio_return_pct,
            'Total_PnL': total_pnl,
            'Total_Straddle_Value': total_straddle_value,
            'Exit_Triggered': exit_triggered,
            'Exit_Reason': exit_reason if exit_triggered else None,
            'Assets_Data': asset_day_details.copy()  # Store individual asset data
        }
        
        comprehensive_daily_data.append(daily_data_entry)
        
        # Exit if TP hit
        if exit_triggered and exit_day == i:
            break
    
    # Calculate final results for each asset
    final_day_data = comprehensive_daily_data[-1]
    asset_results = []
    
    for i, asset_data in enumerate(assets_data):
        final_asset_data = final_day_data['Assets_Data'][i]
        
        # Calculate path volatility (realized volatility)
        if len(asset_data['price_path']) > 1:
            log_returns = np.diff(np.log(asset_data['price_path'][:len(comprehensive_daily_data)]))
            path_volatility = np.std(log_returns) * np.sqrt(365) * 100
        else:
            path_volatility = 0
        
        asset_results.append({
            'Asset_Num': asset_data['asset_num'],
            'Group_Num': group_num,
            'Scenario': asset_data['scenario'],
            'Annual_Drift': asset_data['drift'],
            'Entry_Spot': asset_data['entry_spot'],
            'Final_Spot': final_asset_data['spot_price'],
            'Strike_Price': asset_data['strike_price'],
            'Entry_Straddle_Value': asset_data['entry_straddle_value'],
            'Final_Straddle_Value': final_asset_data['straddle_value'],
            'Position_Size': asset_data['position_size'],
            'Final_Asset_Return_Pct': final_asset_data['asset_return_pct'],
            'Final_Underlying_Return_Pct': final_asset_data['underlying_return_pct'],
            'Asset_PnL': final_asset_data['asset_pnl'],
            'Path_Volatility': path_volatility,
            'Final_Moneyness': final_asset_data['moneyness']
        })
    
    # Group-level results
    group_result = {
        'Group_Num': group_num,
        'Entry_Portfolio': portfolio_value,
        'Exit_Portfolio': final_day_data['Total_Portfolio_Value'],
        'Group_PnL': final_day_data['Total_PnL'],
        'Group_Return_Pct': final_day_data['Group_Return_Pct'],
        'Portfolio_Return_Pct': final_day_data['Portfolio_Return_Pct'],
        'Days_Held': len(comprehensive_daily_data),
        'Exit_Reason': exit_reason,
        'TP_Hit': exit_triggered,
        'N_Assets': n_assets,
        'Start_Date': datetime.now(),  # Could be enhanced with actual start dates
        'End_Date': datetime.now() + timedelta(days=len(comprehensive_daily_data))
    }
    
    return group_result, comprehensive_daily_data, asset_results



# END OF PART 1
# ============================================================================


# PART 2 - CONTINUING FROM PART 1
# ============================================================================


# ============================================================================
# PART 2/3: MAIN SIMULATION RUNNER AND PERFORMANCE ANALYSIS
# ============================================================================

def run_comprehensive_portfolio_simulation(n_groups=10, initial_capital=100000, 
                                         target_annual_return=0.15, n_assets=5,
                                         tp_threshold=20.0, strike_multiplier=1.25,
                                         iv=0.25, risk_free_rate=0.04, annual_vol=0.20):
    """
    Main simulation runner with comprehensive tracking and analysis
    """
    
    print(f"🚀 COMPREHENSIVE PORTFOLIO TRADING SIMULATION")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"   Groups: {n_groups}")
    print(f"   Assets per group: {n_assets}")
    print(f"   Initial capital: ${initial_capital:,.0f}")
    print(f"   Target annual return: {target_annual_return*100:.1f}%")
    print(f"   TP threshold: {tp_threshold:.1f}%")
    print(f"   Strike multiplier: {strike_multiplier:.2f}")
    print(f"   IV: {iv*100:.0f}%")
    print(f"   Risk-free rate: {risk_free_rate*100:.1f}%")
    print(f"   Annual volatility: {annual_vol*100:.0f}%")
    print("="*80)
    
    # Initialize tracking
    portfolio_value = initial_capital
    entry_spots = [100.0, 95.0, 105.0, 98.0, 102.0]  # Starting spots for 5 assets
    
    # Storage for all results
    all_group_results = []
    all_comprehensive_daily_data = []
    all_asset_results = []
    
    # Performance tracking
    simulation_start_time = datetime.now()
    
    # Run each group
    for group_num in range(1, n_groups + 1):
        print(f"\n🔥 Group {group_num}/{n_groups}: Portfolio ${portfolio_value:,.0f}")
        print(f"   Entry spots: {[f'${spot:.2f}' for spot in entry_spots]}")
        
        # Simulate the group with comprehensive tracking
        group_result, daily_data, asset_results = simulate_comprehensive_trade_group(
            group_num=group_num,
            entry_spots=entry_spots,
            portfolio_value=portfolio_value,
            strike_multiplier=strike_multiplier,
            tp_threshold=tp_threshold,
            iv=iv,
            risk_free_rate=risk_free_rate,
            annual_vol=annual_vol,
            n_assets=n_assets
        )
        
        # Update portfolio value for next group
        portfolio_value = group_result['Exit_Portfolio']
        
        # Update entry spots for next group (use final spots from this group)
        final_day = daily_data[-1]
        entry_spots = [asset_data['spot_price'] for asset_data in final_day['Assets_Data']]
        
        # Store results
        all_group_results.append(group_result)
        all_comprehensive_daily_data.extend(daily_data)
        all_asset_results.extend(asset_results)
        
        # Progress update
        print(f"   ✅ Exit: {group_result['Exit_Reason']}")
        print(f"   📅 Days held: {group_result['Days_Held']}")
        print(f"   📈 Group return: {group_result['Group_Return_Pct']:+.1f}%")
        print(f"   💰 New portfolio: ${portfolio_value:,.0f}")
        print(f"   📊 Final spots: {[f'${spot:.2f}' for spot in entry_spots]}")
    
    # Convert to DataFrames for analysis
    groups_df = pd.DataFrame(all_group_results)
    assets_df = pd.DataFrame(all_asset_results)
    
    # Create comprehensive daily DataFrame
    daily_records = []
    for day_data in all_comprehensive_daily_data:
        base_record = {
            'Group_Num': day_data['Group_Num'],
            'Day_Index': day_data['Day_Index'],
            'Calendar_Day': day_data['Calendar_Day'],
            'Days_to_Expiry': day_data['Days_to_Expiry'],
            'Total_Portfolio_Value': day_data['Total_Portfolio_Value'],
            'Group_Return_Pct': day_data['Group_Return_Pct'],
            'Portfolio_Return_Pct': day_data['Portfolio_Return_Pct'],
            'Total_PnL': day_data['Total_PnL'],
            'Total_Straddle_Value': day_data['Total_Straddle_Value'],
            'Exit_Triggered': day_data['Exit_Triggered'],
            'Exit_Reason': day_data['Exit_Reason']
        }
        
        # Add individual asset data
        for asset_data in day_data['Assets_Data']:
            asset_record = base_record.copy()
            asset_record.update({
                'Asset_Num': asset_data['asset_num'],
                'Asset_Scenario': asset_data['scenario'],
                'Asset_Drift': asset_data['drift'],
                'Asset_Spot_Price': asset_data['spot_price'],
                'Asset_Strike_Price': asset_data['strike_price'],
                'Asset_Straddle_Value': asset_data['straddle_value'],
                'Asset_Call_Value': asset_data['call_value'],
                'Asset_Put_Value': asset_data['put_value'],
                'Asset_Return_Pct': asset_data['asset_return_pct'],
                'Asset_Underlying_Return_Pct': asset_data['underlying_return_pct'],
                'Asset_PnL': asset_data['asset_pnl'],
                'Asset_Portfolio_Value': asset_data['asset_portfolio_value'],
                'Asset_Moneyness': asset_data['moneyness'],
                'Asset_Position_Size': asset_data['position_size'],
                'Asset_Entry_Straddle_Value': asset_data['entry_straddle_value'],
                'Asset_Entry_Spot': asset_data['entry_spot']
            })
            daily_records.append(asset_record)
    
    comprehensive_daily_df = pd.DataFrame(daily_records)
    
    # Calculate simulation metrics
    simulation_end_time = datetime.now()
    simulation_duration = simulation_end_time - simulation_start_time
    
    print(f"\n🎉 Simulation Complete!")
    print(f"⏱️  Duration: {simulation_duration}")
    print(f"📊 Generated {len(groups_df)} groups, {len(assets_df)} individual assets")
    print(f"📅 Total daily records: {len(comprehensive_daily_df)}")
    
    return groups_df, assets_df, comprehensive_daily_df, all_comprehensive_daily_data

def calculate_comprehensive_performance_metrics(groups_df, assets_df, comprehensive_daily_df, initial_capital):
    """
    Calculate comprehensive performance metrics including annualized returns, Sharpe ratios, etc.
    """
    
    print(f"\n📈 CALCULATING COMPREHENSIVE PERFORMANCE METRICS")
    print("="*60)
    
    # Basic performance metrics
    final_portfolio = groups_df['Exit_Portfolio'].iloc[-1]
    total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
    
    # Calculate simulation period
    total_days = comprehensive_daily_df.groupby('Group_Num')['Day_Index'].max().sum()
    simulation_years = total_days / 365.25
    
    # Annualized return
    if simulation_years > 0:
        annualized_return = ((final_portfolio / initial_capital) ** (1/simulation_years) - 1) * 100
    else:
        annualized_return = 0
    
    # Group-level metrics
    group_returns = groups_df['Group_Return_Pct']
    portfolio_returns = groups_df['Portfolio_Return_Pct']
    
    # Win rates
    group_win_rate = (group_returns > 0).mean() * 100
    tp_hit_rate = groups_df['TP_Hit'].mean() * 100
    
    # Risk metrics
    group_volatility = group_returns.std()
    portfolio_volatility = portfolio_returns.std()
    
    if group_volatility > 0:
        group_sharpe = group_returns.mean() / group_volatility
    else:
        group_sharpe = 0
    
    # Maximum drawdown calculation
    portfolio_values = [initial_capital] + groups_df['Exit_Portfolio'].tolist()
    peak_values = pd.Series(portfolio_values).expanding().max()
    drawdowns = (pd.Series(portfolio_values) - peak_values) / peak_values * 100
    max_drawdown = drawdowns.min()
    
    # Asset-level metrics
    asset_returns = assets_df['Final_Asset_Return_Pct']
    asset_win_rate = (asset_returns > 0).mean() * 100
    asset_avg_return = asset_returns.mean()
    asset_volatility = asset_returns.std()
    
    # Scenario analysis
    scenario_performance = assets_df.groupby('Scenario').agg({
        'Final_Asset_Return_Pct': ['count', 'mean', 'std'],
        'Final_Underlying_Return_Pct': 'mean',
        'Path_Volatility': 'mean'
    }).round(2)
    
    # Holding period analysis
    avg_holding_period = groups_df['Days_Held'].mean()
    median_holding_period = groups_df['Days_Held'].median()
    
    # Diversification metrics
    theoretical_single_vol = asset_volatility
    actual_group_vol = group_volatility
    if theoretical_single_vol > 0:
        diversification_benefit = (theoretical_single_vol - actual_group_vol) / theoretical_single_vol * 100
    else:
        diversification_benefit = 0
    
    # Performance consistency
    positive_groups = (group_returns > 0).sum()
    negative_groups = (group_returns <= 0).sum()
    
    # Create performance summary dictionary
    performance_metrics = {
        # Overall Performance
        'initial_capital': initial_capital,
        'final_portfolio': final_portfolio,
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'simulation_years': simulation_years,
        'total_days': total_days,
        
        # Group-level Performance
        'avg_group_return_pct': group_returns.mean(),
        'median_group_return_pct': group_returns.median(),
        'group_volatility_pct': group_volatility,
        'group_sharpe_ratio': group_sharpe,
        'group_win_rate_pct': group_win_rate,
        'tp_hit_rate_pct': tp_hit_rate,
        'positive_groups': positive_groups,
        'negative_groups': negative_groups,
        
        # Risk Metrics
        'max_drawdown_pct': max_drawdown,
        'portfolio_volatility_pct': portfolio_volatility,
        
        # Asset-level Metrics
        'asset_avg_return_pct': asset_avg_return,
        'asset_volatility_pct': asset_volatility,
        'asset_win_rate_pct': asset_win_rate,
        
        # Holding Period
        'avg_holding_period_days': avg_holding_period,
        'median_holding_period_days': median_holding_period,
        
        # Diversification
        'diversification_benefit_pct': diversification_benefit,
        'theoretical_single_vol_pct': theoretical_single_vol,
        'actual_group_vol_pct': actual_group_vol,
        
        # Scenario Performance
        'scenario_performance': scenario_performance
    }
    
    # Print summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   💰 Total Return: {total_return:+.1f}% (${initial_capital:,.0f} → ${final_portfolio:,.0f})")
    print(f"   📈 Annualized Return: {annualized_return:+.1f}%")
    print(f"   📅 Simulation Period: {simulation_years:.1f} years ({total_days:.0f} days)")
    print(f"   🎯 Group Win Rate: {group_win_rate:.1f}%")
    print(f"   🚀 TP Hit Rate: {tp_hit_rate:.1f}%")
    print(f"   📉 Max Drawdown: {max_drawdown:.1f}%")
    print(f"   📊 Sharpe Ratio: {group_sharpe:.2f}")
    print(f"   🔄 Avg Holding Period: {avg_holding_period:.0f} days")
    print(f"   🛡️  Diversification Benefit: {diversification_benefit:.1f}%")
    
    print(f"\n📈 ASSET-LEVEL PERFORMANCE:")
    print(f"   📊 Asset Win Rate: {asset_win_rate:.1f}%")
    print(f"   💵 Avg Asset Return: {asset_avg_return:+.1f}%")
    print(f"   📈 Asset Volatility: {asset_volatility:.1f}%")
    
    print(f"\n🎭 SCENARIO BREAKDOWN:")
    for scenario in ['bull', 'sideways', 'bear']:
        if scenario in scenario_performance.index:
            count = int(scenario_performance.loc[scenario, ('Final_Asset_Return_Pct', 'count')])
            mean_return = scenario_performance.loc[scenario, ('Final_Asset_Return_Pct', 'mean')]
            print(f"   {scenario.title()}: {count} assets, avg return {mean_return:+.1f}%")
    
    return performance_metrics

def create_continuous_portfolio_timeline(comprehensive_daily_df):
    """
    Create continuous timeline showing portfolio progression across all groups
    """
    
    print(f"\n📅 Creating continuous portfolio timeline...")
    
    # Sort by group and day
    timeline_df = comprehensive_daily_df.copy()
    timeline_df = timeline_df.sort_values(['Group_Num', 'Day_Index'])
    
    # Create global day counter
    global_day = 0
    timeline_records = []
    
    for group_num in sorted(timeline_df['Group_Num'].unique()):
        group_data = timeline_df[timeline_df['Group_Num'] == group_num]
        
        # Get unique days for this group (group-level data)
        group_days = group_data.groupby(['Group_Num', 'Day_Index']).first().reset_index()
        
        for _, day_record in group_days.iterrows():
            timeline_records.append({
                'Global_Day': global_day,
                'Group_Num': day_record['Group_Num'],
                'Group_Day': day_record['Day_Index'],
                'Days_to_Expiry': day_record['Days_to_Expiry'],
                'Total_Portfolio_Value': day_record['Total_Portfolio_Value'],
                'Group_Return_Pct': day_record['Group_Return_Pct'],
                'Portfolio_Return_Pct': day_record['Portfolio_Return_Pct'],
                'Total_PnL': day_record['Total_PnL'],
                'Exit_Triggered': day_record['Exit_Triggered']
            })
            global_day += 1
    
    continuous_timeline_df = pd.DataFrame(timeline_records)
    
    # Calculate additional metrics
    if len(continuous_timeline_df) > 0:
        continuous_timeline_df['Peak_Portfolio'] = continuous_timeline_df['Total_Portfolio_Value'].expanding().max()
        continuous_timeline_df['Drawdown_Pct'] = ((continuous_timeline_df['Total_Portfolio_Value'] - 
                                                  continuous_timeline_df['Peak_Portfolio']) / 
                                                  continuous_timeline_df['Peak_Portfolio'] * 100)
        continuous_timeline_df['Daily_Return_Pct'] = continuous_timeline_df['Total_Portfolio_Value'].pct_change() * 100
        continuous_timeline_df['Daily_Return_Pct'] = continuous_timeline_df['Daily_Return_Pct'].fillna(0)
    
    print(f"   ✅ Created timeline with {len(continuous_timeline_df)} daily records")
    
    return continuous_timeline_df

def save_comprehensive_results(groups_df, assets_df, comprehensive_daily_df, 
                             continuous_timeline_df, performance_metrics):
    """
    Save all comprehensive results to CSV files with timestamps
    """
    
    print(f"\n💾 Saving comprehensive results...")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main dataframes
    groups_filename = f'comprehensive_portfolio_groups_{timestamp}.csv'
    assets_filename = f'comprehensive_portfolio_assets_{timestamp}.csv'
    daily_filename = f'comprehensive_portfolio_daily_{timestamp}.csv'
    timeline_filename = f'comprehensive_portfolio_timeline_{timestamp}.csv'
    
    groups_df.to_csv(groups_filename, index=False)
    assets_df.to_csv(assets_filename, index=False)
    comprehensive_daily_df.to_csv(daily_filename, index=False)
    continuous_timeline_df.to_csv(timeline_filename, index=False)
    
    # Save performance metrics as JSON-like CSV
    metrics_filename = f'comprehensive_portfolio_metrics_{timestamp}.csv'
    metrics_data = []
    
    for key, value in performance_metrics.items():
        if key != 'scenario_performance':  # Handle this separately
            metrics_data.append({'Metric': key, 'Value': value})
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_filename, index=False)
    
    # Save scenario performance separately
    scenario_filename = f'comprehensive_portfolio_scenarios_{timestamp}.csv'
    if 'scenario_performance' in performance_metrics:
        performance_metrics['scenario_performance'].to_csv(scenario_filename)
    
    print(f"   📄 Groups: {groups_filename} ({len(groups_df)} records)")
    print(f"   📄 Assets: {assets_filename} ({len(assets_df)} records)")
    print(f"   📄 Daily: {daily_filename} ({len(comprehensive_daily_df)} records)")
    print(f"   📄 Timeline: {timeline_filename} ({len(continuous_timeline_df)} records)")
    print(f"   📄 Metrics: {metrics_filename}")
    print(f"   📄 Scenarios: {scenario_filename}")
    
    return {
        'groups_file': groups_filename,
        'assets_file': assets_filename,
        'daily_file': daily_filename,
        'timeline_file': timeline_filename,
        'metrics_file': metrics_filename,
        'scenarios_file': scenario_filename
    }


# END OF PART 2
# ============================================================================


# ============================================================================
# PART 3/3: COMPREHENSIVE PLOTTING AND VISUALIZATION FUNCTIONS
# ============================================================================

def plot_individual_group_progression(group_num, comprehensive_daily_df, groups_df, assets_df, save_plots=True):
    """
    Plot individual group progression showing all 5 assets + group total over time
    """
    
    # Get data for this specific group
    group_data = comprehensive_daily_df[comprehensive_daily_df['Group_Num'] == group_num].copy()
    group_info = groups_df[groups_df['Group_Num'] == group_num].iloc[0]
    group_assets = assets_df[assets_df['Group_Num'] == group_num]
    
    if group_data.empty:
        print(f"No data found for Group {group_num}")
        return
    
    # Sort by day index
    group_data = group_data.sort_values(['Day_Index', 'Asset_Num'])
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Colors and markers for assets
    asset_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    asset_markers = ['o', 's', '^', 'D', 'v']
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    # Plot 1: Individual asset returns over time
    ax1 = axes[0, 0]
    
    # Get group-level data (one record per day)
    group_level_data = group_data.groupby('Day_Index').first().reset_index()
    days_from_start = group_level_data['Day_Index']
    
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            asset_info = group_assets[group_assets['Asset_Num'] == asset_num]
            scenario = asset_info.iloc[0]['Scenario'] if not asset_info.empty else 'unknown'
            
            ax1.plot(asset_data['Day_Index'], asset_data['Asset_Return_Pct'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.8,
                    marker=asset_markers[asset_num], markersize=3, markevery=10,
                    label=f'Asset {asset_num+1} ({scenario[:4]})')
    
    # Add group average line
    ax1.plot(group_level_data['Day_Index'], group_level_data['Group_Return_Pct'], 
            color='black', linewidth=3, alpha=0.9, label='Group Avg')
    
    # Reference lines and styling
    ax1.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=2, label='20% TP')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
    ax1.fill_between(days_from_start, group_level_data['Group_Return_Pct'], 0,
                    where=(group_level_data['Group_Return_Pct'] >= 0), 
                    alpha=0.2, color='green', interpolate=True)
    ax1.fill_between(days_from_start, group_level_data['Group_Return_Pct'], 0,
                    where=(group_level_data['Group_Return_Pct'] < 0), 
                    alpha=0.2, color='red', interpolate=True)
    
    ax1.set_xlabel('Days from Start')
    ax1.set_ylabel('Return (%)')
    ax1.set_title(f'Group {group_num} - Asset Returns Over Time')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-50, 30)
    
    # Plot 2: Portfolio value progression
    ax2 = axes[0, 1]
    ax2.plot(group_level_data['Day_Index'], group_level_data['Total_Portfolio_Value'], 
            linewidth=3, color='blue', alpha=0.8)
    ax2.fill_between(group_level_data['Day_Index'], group_level_data['Total_Portfolio_Value'], 
                    group_info['Entry_Portfolio'], alpha=0.3, color='blue')
    
    ax2.axhline(y=group_info['Entry_Portfolio'], color='black', linestyle=':', alpha=0.7, 
               linewidth=2, label='Entry Value')
    ax2.set_xlabel('Days from Start')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Portfolio Value Progression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 3: Underlying price movements
    ax3 = axes[0, 2]
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            # Calculate percentage change from entry
            entry_spot = asset_data['Asset_Entry_Spot'].iloc[0]
            price_change_pct = ((asset_data['Asset_Spot_Price'] - entry_spot) / entry_spot) * 100
            
            ax3.plot(asset_data['Day_Index'], price_change_pct, 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.7,
                    label=f'Asset {asset_num+1}')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Days from Start')
    ax3.set_ylabel('Underlying Price Change (%)')
    ax3.set_title('Underlying Asset Movements')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Moneyness evolution
    ax4 = axes[1, 0]
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            ax4.plot(asset_data['Day_Index'], asset_data['Asset_Moneyness'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.7,
                    label=f'Asset {asset_num+1}')
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM')
    ax4.fill_between(days_from_start, 0.95, 1.05, alpha=0.2, color='red', label='Near ATM')
    ax4.set_xlabel('Days from Start')
    ax4.set_ylabel('Moneyness (S/K)')
    ax4.set_title('Moneyness Evolution')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Straddle values over time
    ax5 = axes[1, 1]
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            ax5.plot(asset_data['Day_Index'], asset_data['Asset_Straddle_Value'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.7,
                    label=f'Asset {asset_num+1}')
    
    ax5.set_xlabel('Days from Start')
    ax5.set_ylabel('Straddle Value ($)')
    ax5.set_title('Straddle Values Over Time')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics and info
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    final_return = group_info['Group_Return_Pct']
    days_held = group_info['Days_Held']
    exit_reason = group_info['Exit_Reason']
    tp_hit = group_info['TP_Hit']
    
    # Asset summaries
    asset_summary_text = ""
    for _, asset in group_assets.iterrows():
        asset_summary_text += f"Asset {asset['Asset_Num']+1}: {asset['Scenario']} ({asset['Final_Asset_Return_Pct']:+.1f}%)\n"
    
    summary_text = f"""
GROUP {group_num} SUMMARY
{'='*20}

PERFORMANCE:
Final Return: {final_return:+.1f}%
Days Held: {days_held}
Exit Reason: {exit_reason}
TP Hit: {'Yes' if tp_hit else 'No'}

PORTFOLIO:
Entry: ${group_info['Entry_Portfolio']:,.0f}
Exit: ${group_info['Exit_Portfolio']:,.0f}
P&L: ${group_info['Group_PnL']:,.0f}

ASSETS:
{asset_summary_text.strip()}

MARKET CONDITIONS:
Bull: {sum(1 for _, a in group_assets.iterrows() if a['Scenario'] == 'bull')}
Sideways: {sum(1 for _, a in group_assets.iterrows() if a['Scenario'] == 'sideways')}
Bear: {sum(1 for _, a in group_assets.iterrows() if a['Scenario'] == 'bear')}
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Group {group_num} Comprehensive Analysis\n'
                f'Return: {final_return:+.1f}% | Days: {days_held} | Exit: {exit_reason}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        filename = f'group_{group_num}_comprehensive_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
    
    plt.show()

def plot_all_group_progressions(comprehensive_daily_df, groups_df, assets_df, max_groups=None):
    """
    Plot progression charts for all groups (or specified maximum)
    """
    
    unique_groups = sorted(comprehensive_daily_df['Group_Num'].unique())
    
    if max_groups:
        unique_groups = unique_groups[:max_groups]
    
    print(f"\n📊 Creating individual progression plots for {len(unique_groups)} groups...")
    
    for group_num in unique_groups:
        print(f"   📈 Plotting Group {group_num}...")
        plot_individual_group_progression(group_num, comprehensive_daily_df, groups_df, assets_df)

def plot_portfolio_performance_dashboard(groups_df, assets_df, continuous_timeline_df, performance_metrics, save_plots=True):
    """
    Create comprehensive portfolio performance dashboard
    """
    
    print(f"\n📊 Creating portfolio performance dashboard...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Portfolio equity curve
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(continuous_timeline_df['Global_Day'], continuous_timeline_df['Total_Portfolio_Value'], 
             linewidth=3, color='blue', alpha=0.8)
    ax1.fill_between(continuous_timeline_df['Global_Day'], continuous_timeline_df['Total_Portfolio_Value'], 
                     performance_metrics['initial_capital'], alpha=0.3, color='blue')
    
    ax1.set_xlabel('Days Since Start')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Equity Curve')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Drawdown chart
    ax2 = plt.subplot(3, 4, 2)
    ax2.fill_between(continuous_timeline_df['Global_Day'], continuous_timeline_df['Drawdown_Pct'], 0, 
                     color='red', alpha=0.6)
    ax2.plot(continuous_timeline_df['Global_Day'], continuous_timeline_df['Drawdown_Pct'], 
             linewidth=2, color='darkred')
    
    max_dd = continuous_timeline_df['Drawdown_Pct'].min()
    max_dd_day = continuous_timeline_df.loc[continuous_timeline_df['Drawdown_Pct'].idxmin(), 'Global_Day']
    ax2.annotate(f'Max: {max_dd:.1f}%', xy=(max_dd_day, max_dd), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('Days Since Start')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Portfolio Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Group returns distribution
    ax3 = plt.subplot(3, 4, 3)
    group_returns = groups_df['Group_Return_Pct']
    tp_groups = groups_df[groups_df['TP_Hit'] == True]['Group_Return_Pct']
    exp_groups = groups_df[groups_df['TP_Hit'] == False]['Group_Return_Pct']
    
    if len(tp_groups) > 0:
        ax3.hist(tp_groups, alpha=0.7, label=f'TP Hits ({len(tp_groups)})', 
                color='green', bins=15)
    if len(exp_groups) > 0:
        ax3.hist(exp_groups, alpha=0.7, label=f'Expiry ({len(exp_groups)})', 
                color='orange', bins=15)
    
    ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(20, color='green', linestyle='--', alpha=0.7, label='20% TP')
    ax3.set_xlabel('Group Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Group Returns Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Asset performance by scenario
    ax4 = plt.subplot(3, 4, 4)
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = assets_df[assets_df['Scenario'] == scenario]['Final_Asset_Return_Pct']
        if not scenario_data.empty:
            ax4.hist(scenario_data, alpha=0.6, label=f'{scenario.title()} ({len(scenario_data)})', 
                    color=scenario_colors.get(scenario, 'gray'), bins=15)
    
    ax4.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Asset Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Asset Returns by Scenario')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Rolling performance
    ax5 = plt.subplot(3, 4, 5)
    window = 30
    if len(continuous_timeline_df) >= window:
        rolling_returns = continuous_timeline_df['Daily_Return_Pct'].rolling(window=window).mean()
        rolling_vol = continuous_timeline_df['Daily_Return_Pct'].rolling(window=window).std()
        
        ax5.plot(continuous_timeline_df['Global_Day'], rolling_returns, 
                linewidth=2, color='green', label=f'{window}d Avg Return')
        ax5.plot(continuous_timeline_df['Global_Day'], rolling_vol, 
                linewidth=2, color='red', label=f'{window}d Volatility')
        
        ax5.set_xlabel('Days Since Start')
        ax5.set_ylabel('Return/Volatility (%)')
        ax5.set_title('Rolling Performance Metrics')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Holding periods
    ax6 = plt.subplot(3, 4, 6)
    holding_periods = groups_df['Days_Held']
    tp_periods = groups_df[groups_df['TP_Hit'] == True]['Days_Held']
    exp_periods = groups_df[groups_df['TP_Hit'] == False]['Days_Held']
    
    if len(tp_periods) > 0 and len(exp_periods) > 0:
        ax6.hist([tp_periods, exp_periods], bins=15, alpha=0.7, 
                 label=[f'TP Hits', f'Expiry'], color=['green', 'orange'])
    elif len(tp_periods) > 0:
        ax6.hist(tp_periods, bins=15, alpha=0.7, label=f'TP Hits', color='green')
    elif len(exp_periods) > 0:
        ax6.hist(exp_periods, bins=15, alpha=0.7, label=f'Expiry', color='orange')
    
    ax6.set_xlabel('Days Held')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Holding Period Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Win rate by group
    ax7 = plt.subplot(3, 4, 7)
    group_nums = groups_df['Group_Num']
    colors = ['green' if ret > 0 else 'red' for ret in groups_df['Group_Return_Pct']]
    
    bars = ax7.bar(group_nums, groups_df['Group_Return_Pct'], color=colors, alpha=0.7)
    ax7.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax7.axhline(20, color='green', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Group Number')
    ax7.set_ylabel('Group Return (%)')
    ax7.set_title('Performance by Group')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Asset correlation analysis
    ax8 = plt.subplot(3, 4, 8)
    
    # Calculate correlation matrix for asset returns within groups
    correlation_data = []
    for group_num in groups_df['Group_Num'].unique():
        group_assets = assets_df[assets_df['Group_Num'] == group_num]
        if len(group_assets) == 5:  # Full group
            returns_dict = {}
            for _, asset in group_assets.iterrows():
                returns_dict[f'Asset_{asset["Asset_Num"]}'] = asset['Final_Asset_Return_Pct']
            correlation_data.append(returns_dict)
    
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        im = ax8.imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax8.set_xticks(range(len(correlation_matrix.columns)))
        ax8.set_yticks(range(len(correlation_matrix.columns)))
        ax8.set_xticklabels([f'A{i}' for i in range(5)], rotation=45)
        ax8.set_yticklabels([f'A{i}' for i in range(5)])
        ax8.set_title('Asset Return Correlations')
        
        # Add colorbar
        plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
    
    # Plot 9: Performance metrics summary
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    metrics_text = f"""
PERFORMANCE SUMMARY
{'='*20}

RETURNS:
Total: {performance_metrics['total_return_pct']:+.1f}%
Annualized: {performance_metrics['annualized_return_pct']:+.1f}%
Period: {performance_metrics['simulation_years']:.1f} years

RISK METRICS:
Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%
Volatility: {performance_metrics['group_volatility_pct']:.1f}%
Sharpe Ratio: {performance_metrics['group_sharpe_ratio']:.2f}

SUCCESS RATES:
Group Win Rate: {performance_metrics['group_win_rate_pct']:.1f}%
TP Hit Rate: {performance_metrics['tp_hit_rate_pct']:.1f}%
Asset Win Rate: {performance_metrics['asset_win_rate_pct']:.1f}%

DIVERSIFICATION:
Risk Reduction: {performance_metrics['diversification_benefit_pct']:.1f}%
Avg Hold Period: {performance_metrics['avg_holding_period_days']:.0f} days
"""
    
    ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Plot 10: Scenario performance comparison
    ax10 = plt.subplot(3, 4, 10)
    
    scenario_stats = assets_df.groupby('Scenario')['Final_Asset_Return_Pct'].agg(['mean', 'std', 'count'])
    scenarios = scenario_stats.index
    means = scenario_stats['mean']
    stds = scenario_stats['std']
    
    bars = ax10.bar(scenarios, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=[scenario_colors.get(s, 'gray') for s in scenarios])
    ax10.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax10.set_ylabel('Average Return (%)')
    ax10.set_title('Performance by Market Scenario')
    ax10.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, scenario_stats['count']):
        ax10.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Plot 11: Portfolio value vs benchmark (if available)
    ax11 = plt.subplot(3, 4, 11)
    
    # Simple buy-and-hold comparison (simplified)
    initial_value = performance_metrics['initial_capital']
    portfolio_values = continuous_timeline_df['Total_Portfolio_Value']
    days = continuous_timeline_df['Global_Day']
    
    # Simulate 10% annual return benchmark
    benchmark_annual_return = 0.10
    benchmark_values = [initial_value * (1 + benchmark_annual_return) ** (day/365.25) for day in days]
    
    ax11.plot(days, portfolio_values, linewidth=2, color='blue', label='Portfolio')
    ax11.plot(days, benchmark_values, linewidth=2, color='gray', linestyle='--', label='10% Benchmark')
    
    ax11.set_xlabel('Days Since Start')
    ax11.set_ylabel('Value ($)')
    ax11.set_title('Portfolio vs Benchmark')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 12: Final summary table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    final_summary_text = f"""
FINAL RESULTS
{'='*15}

PORTFOLIO:
Start: ${performance_metrics['initial_capital']:,.0f}
End: ${performance_metrics['final_portfolio']:,.0f}
Gain: ${performance_metrics['final_portfolio'] - performance_metrics['initial_capital']:,.0f}

GROUPS:
Total: {len(groups_df)}
Winners: {performance_metrics['positive_groups']}
Losers: {performance_metrics['negative_groups']}

ASSETS:
Total: {len(assets_df)}
Avg Return: {performance_metrics['asset_avg_return_pct']:+.1f}%

EFFICIENCY:
Risk Reduction: {performance_metrics['diversification_benefit_pct']:.1f}%
Return/Risk: {performance_metrics['group_sharpe_ratio']:.2f}
"""
    
    ax12.text(0.1, 0.9, final_summary_text, transform=ax12.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle(f'Comprehensive Portfolio Performance Dashboard\n'
                f'Total Return: {performance_metrics["total_return_pct"]:+.1f}% | '
                f'Annualized: {performance_metrics["annualized_return_pct"]:+.1f}% | '
                f'Sharpe: {performance_metrics["group_sharpe_ratio"]:.2f}', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'comprehensive_portfolio_dashboard.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
    
    plt.show()

def main():
    """
    Main function to run the comprehensive portfolio simulation and analysis
    """
    
    print("🚀 COMPREHENSIVE PORTFOLIO TRADING SIMULATOR")
    print("="*80)
    print("Integrated simulation and visualization with detailed tracking")
    print("="*80)
    
    # Simulation parameters
    n_groups = 12  # Run 12 groups for comprehensive analysis
    initial_capital = 100000
    target_annual_return = 0.15
    n_assets = 5
    tp_threshold = 20.0
    
    # Step 1: Run comprehensive simulation
    print(f"\n🎯 Running comprehensive simulation...")
    groups_df, assets_df, comprehensive_daily_df, all_daily_data = run_comprehensive_portfolio_simulation(
        n_groups=n_groups,
        initial_capital=initial_capital,
        target_annual_return=target_annual_return,
        n_assets=n_assets,
        tp_threshold=tp_threshold
    )
    
    # Step 2: Calculate comprehensive performance metrics
    print(f"\n📊 Calculating performance metrics...")
    performance_metrics = calculate_comprehensive_performance_metrics(
        groups_df, assets_df, comprehensive_daily_df, initial_capital
    )
    
    # Step 3: Create continuous timeline
    print(f"\n📅 Creating continuous timeline...")
    continuous_timeline_df = create_continuous_portfolio_timeline(comprehensive_daily_df)
    
    # Step 4: Save all results
    print(f"\n💾 Saving comprehensive results...")
    saved_files = save_comprehensive_results(
        groups_df, assets_df, comprehensive_daily_df, 
        continuous_timeline_df, performance_metrics
    )
    
    # Step 5: Create comprehensive visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    
    # Portfolio dashboard
    plot_portfolio_performance_dashboard(
        groups_df, assets_df, continuous_timeline_df, performance_metrics
    )
    
    # Individual group progressions (first 6 groups to avoid overwhelming output)
    print(f"\n📈 Creating individual group progression plots...")
    plot_all_group_progressions(comprehensive_daily_df, groups_df, assets_df, max_groups=6)
    
    # Final summary
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📈 Portfolio Performance: ${initial_capital:,.0f} → ${performance_metrics['final_portfolio']:,.0f}")
    print(f"📊 Total Return: {performance_metrics['total_return_pct']:+.1f}%")
    print(f"📅 Annualized Return: {performance_metrics['annualized_return_pct']:+.1f}%")
    print(f"🎯 Win Rate: {performance_metrics['group_win_rate_pct']:.1f}%")
    print(f"🚀 TP Hit Rate: {performance_metrics['tp_hit_rate_pct']:.1f}%")
    print(f"📉 Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%")
    print(f"📊 Sharpe Ratio: {performance_metrics['group_sharpe_ratio']:.2f}")
    print(f"🛡️  Risk Reduction: {performance_metrics['diversification_benefit_pct']:.1f}%")
    print("="*80)
    print(f"📁 All data saved with timestamps for future analysis")
    print(f"📊 All plots generated and saved")
    print(f"✅ Ready for detailed review and further analysis!")

if __name__ == "__main__":
    main()