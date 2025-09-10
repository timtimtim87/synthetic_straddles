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

def check_individual_exit_conditions(spot_price, strike_price, entry_spot, exit_buffer=0.02):
    """
    Check if individual asset should be exited based on strike breach
    Since strike is 25% above entry, we exit if spot crosses the strike level
    
    Parameters:
    - spot_price: Current spot price
    - strike_price: Strike price of the straddle (25% above entry)
    - entry_spot: Entry spot price
    - exit_buffer: Buffer around strike (2% default)
    
    Returns:
    - should_exit: Boolean indicating if position should be closed
    - exit_reason: String describing the exit reason
    """
    
    # Since strike is ABOVE entry spot, we only care about upward movement toward strike
    # Add buffer around strike price
    exit_threshold = strike_price * (1 - exit_buffer)  # Exit when approaching strike from below
    
    # Check if spot price has moved up enough to cross our threshold
    if spot_price >= exit_threshold:
        return True, f"Strike Approach: ${spot_price:.2f} >= ${exit_threshold:.2f} (Strike: ${strike_price:.2f})"
    
    return False, None

def simulate_comprehensive_trade_group_with_individual_exits(group_num, entry_spots, portfolio_value, 
                                                           strike_multiplier=1.25, tp_threshold=20.0, 
                                                           days_to_expiry=365, iv=0.25, risk_free_rate=0.04, 
                                                           annual_vol=0.20, n_assets=5, strike_exit_buffer=0.02):
    """
    Enhanced simulation with individual asset exit conditions for strike breaches
    """
    
    print(f"  📊 Simulating Group {group_num} with individual exit conditions...")
    print(f"      Strike exit buffer: ±{strike_exit_buffer*100:.1f}%")
    
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
            'position_size': position_size,
            'is_active': True,  # Track if position is still active
            'exit_day': None,   # Track when position was closed
            'exit_reason': None,  # Track why position was closed
            'exit_value': None    # Track exit straddle value
        })
        
        print(f"    Asset {asset_num+1}: {scenario.title()} scenario, Entry ${entry_spots[asset_num]:.2f}, Strike ${strike_price:.2f}")
    
    # Simulate daily progression with comprehensive tracking and individual exits
    days_array = np.arange(days_to_expiry, -1, -1)
    comprehensive_daily_data = []
    
    group_exit_triggered = False
    group_exit_day = None
    group_exit_reason = "Expiry"
    
    for i, day in enumerate(days_array):
        time_to_expiry = max(day / 365.0, 1/365)
        
        # Calculate current values for all assets (including closed positions)
        total_portfolio_value = 0
        total_straddle_value = 0
        total_pnl = 0
        active_assets_count = 0
        
        # Individual asset calculations
        asset_day_details = []
        
        for asset_data in assets_data:
            spot_price = asset_data['price_path'][i]
            
            # Check if this asset should be individually exited (only if still active)
            if asset_data['is_active']:
                should_exit, exit_reason = check_individual_exit_conditions(
                    spot_price, asset_data['strike_price'], asset_data['entry_spot'], strike_exit_buffer
                )
                
                if should_exit:
                    # Close this individual position
                    asset_data['is_active'] = False
                    asset_data['exit_day'] = i
                    asset_data['exit_reason'] = exit_reason
                    
                    # Calculate exit straddle value
                    exit_straddle_value = calculate_straddle_value(
                        spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv
                    )
                    asset_data['exit_value'] = exit_straddle_value
                    
                    print(f"      🚪 Asset {asset_data['asset_num']+1} EXITED on day {i}: {exit_reason}")
            
            # Calculate current values (whether active or closed)
            if asset_data['is_active']:
                # Still active - calculate current straddle value
                current_straddle_value = calculate_straddle_value(
                    spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv
                )
                active_assets_count += 1
            else:
                # Position closed - use exit value
                current_straddle_value = asset_data['exit_value']
            
            # Short straddle P&L for this asset
            pnl_per_contract = asset_data['entry_straddle_value'] - current_straddle_value
            asset_pnl = pnl_per_contract * asset_data['position_size']
            asset_portfolio_value = capital_per_asset + asset_pnl
            
            # Asset return calculations
            asset_return_pct = (pnl_per_contract / asset_data['entry_straddle_value']) * 100
            underlying_return_pct = ((spot_price - asset_data['entry_spot']) / asset_data['entry_spot']) * 100
            moneyness = spot_price / asset_data['strike_price']
            
            # Greeks calculations (simplified)
            if asset_data['is_active']:
                call_value = black_scholes_call(spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv)
                put_value = black_scholes_put(spot_price, asset_data['strike_price'], time_to_expiry, risk_free_rate, iv)
            else:
                # For closed positions, calculate what the values would be (for analysis)
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
                'entry_spot': asset_data['entry_spot'],
                'is_active': asset_data['is_active'],
                'exit_day': asset_data['exit_day'],
                'exit_reason': asset_data['exit_reason'],
                'exit_value': asset_data['exit_value']
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
        
        # Check for group-level 20% TP (only if we have active positions or this is the first check)
        if group_return_pct >= tp_threshold and not group_exit_triggered:
            group_exit_triggered = True
            group_exit_day = i
            group_exit_reason = "20% TP Hit"
            
            # Close any remaining active positions
            for asset_data in assets_data:
                if asset_data['is_active']:
                    asset_data['is_active'] = False
                    asset_data['exit_day'] = i
                    asset_data['exit_reason'] = "Group TP Hit"
                    exit_straddle_value = calculate_straddle_value(
                        asset_data['price_path'][i], asset_data['strike_price'], 
                        time_to_expiry, risk_free_rate, iv
                    )
                    asset_data['exit_value'] = exit_straddle_value
            
            print(f"      🎯 GROUP {group_num} TP HIT on day {i}: {group_return_pct:.1f}%")
        
        # Check if all positions are closed
        if active_assets_count == 0 and not group_exit_triggered:
            group_exit_triggered = True
            group_exit_day = i
            group_exit_reason = "All Positions Closed"
            print(f"      🏁 GROUP {group_num} ALL POSITIONS CLOSED on day {i}")
        
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
            'Active_Assets_Count': active_assets_count,
            'Exit_Triggered': group_exit_triggered,
            'Exit_Reason': group_exit_reason if group_exit_triggered else None,
            'Assets_Data': asset_day_details.copy()  # Store individual asset data
        }
        
        comprehensive_daily_data.append(daily_data_entry)
        
        # Exit if group TP hit or all positions closed
        if group_exit_triggered:
            break
    
    # Calculate final results for each asset
    final_day_data = comprehensive_daily_data[-1]
    asset_results = []
    
    for i, asset_data in enumerate(assets_data):
        final_asset_data = final_day_data['Assets_Data'][i]
        
        # Calculate path volatility (realized volatility) up to exit point
        exit_day_index = asset_data['exit_day'] if asset_data['exit_day'] is not None else len(comprehensive_daily_data) - 1
        price_segment = asset_data['price_path'][:exit_day_index + 1]
        
        if len(price_segment) > 1:
            log_returns = np.diff(np.log(price_segment))
            path_volatility = np.std(log_returns) * np.sqrt(365) * 100
        else:
            path_volatility = 0
        
        # Determine final values
        final_spot = asset_data['price_path'][exit_day_index] if exit_day_index < len(asset_data['price_path']) else asset_data['price_path'][-1]
        days_held = (asset_data['exit_day'] + 1) if asset_data['exit_day'] is not None else len(comprehensive_daily_data)
        
        asset_results.append({
            'Asset_Num': asset_data['asset_num'],
            'Group_Num': group_num,
            'Scenario': asset_data['scenario'],
            'Annual_Drift': asset_data['drift'],
            'Entry_Spot': asset_data['entry_spot'],
            'Final_Spot': final_spot,
            'Strike_Price': asset_data['strike_price'],
            'Entry_Straddle_Value': asset_data['entry_straddle_value'],
            'Final_Straddle_Value': final_asset_data['straddle_value'],
            'Exit_Straddle_Value': asset_data['exit_value'],
            'Position_Size': asset_data['position_size'],
            'Final_Asset_Return_Pct': final_asset_data['asset_return_pct'],
            'Final_Underlying_Return_Pct': final_asset_data['underlying_return_pct'],
            'Asset_PnL': final_asset_data['asset_pnl'],
            'Path_Volatility': path_volatility,
            'Final_Moneyness': final_asset_data['moneyness'],
            'Days_Held': days_held,
            'Exit_Day': asset_data['exit_day'],
            'Exit_Reason': asset_data['exit_reason'],
            'Was_Individually_Closed': asset_data['exit_day'] is not None and asset_data['exit_reason'] != "Group TP Hit"
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
        'Exit_Reason': group_exit_reason,
        'TP_Hit': group_exit_reason == "20% TP Hit",
        'All_Positions_Closed': group_exit_reason == "All Positions Closed",
        'Individual_Exits_Count': sum(1 for asset in assets_data if asset['exit_reason'] and asset['exit_reason'] not in ["Group TP Hit", "Expiry"]),
        'N_Assets': n_assets,
        'Start_Date': datetime.now(),  # Could be enhanced with actual start dates
        'End_Date': datetime.now() + timedelta(days=len(comprehensive_daily_data))
    }
    
    # Summary of individual exits
    individual_exits = [asset for asset in assets_data if asset['exit_reason'] and asset['exit_reason'] not in ["Group TP Hit", "Expiry"]]
    if individual_exits:
        print(f"      📋 Individual exits summary:")
        for asset in individual_exits:
            print(f"        Asset {asset['asset_num']+1}: {asset['exit_reason']} on day {asset['exit_day']}")
    
    return group_result, comprehensive_daily_data, asset_results


# ============================================================================



# ============================================================================
# PART 2/3: MAIN SIMULATION RUNNER AND PERFORMANCE ANALYSIS
# ============================================================================

def run_comprehensive_portfolio_simulation_with_exits(n_groups=10, initial_capital=100000, 
                                                     target_annual_return=0.15, n_assets=5,
                                                     tp_threshold=20.0, strike_multiplier=1.25,
                                                     iv=0.25, risk_free_rate=0.04, annual_vol=0.20,
                                                     strike_exit_buffer=0.02):
    """
    Main simulation runner with individual exit conditions
    """
    
    print(f"🚀 COMPREHENSIVE PORTFOLIO TRADING SIMULATION WITH INDIVIDUAL EXITS")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"   Groups: {n_groups}")
    print(f"   Assets per group: {n_assets}")
    print(f"   Initial capital: ${initial_capital:,.0f}")
    print(f"   Target annual return: {target_annual_return*100:.1f}%")
    print(f"   TP threshold: {tp_threshold:.1f}%")
    print(f"   Strike multiplier: {strike_multiplier:.2f}")
    print(f"   Strike exit buffer: ±{strike_exit_buffer*100:.1f}%")
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
        
        # Simulate the group with individual exit conditions
        group_result, daily_data, asset_results = simulate_comprehensive_trade_group_with_individual_exits(
            group_num=group_num,
            entry_spots=entry_spots,
            portfolio_value=portfolio_value,
            strike_multiplier=strike_multiplier,
            tp_threshold=tp_threshold,
            iv=iv,
            risk_free_rate=risk_free_rate,
            annual_vol=annual_vol,
            n_assets=n_assets,
            strike_exit_buffer=strike_exit_buffer
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
        
        # Progress update with enhanced metrics
        individual_exits = group_result['Individual_Exits_Count']
        print(f"   ✅ Exit: {group_result['Exit_Reason']}")
        print(f"   📅 Days held: {group_result['Days_Held']}")
        print(f"   📈 Group return: {group_result['Group_Return_Pct']:+.1f}%")
        print(f"   🚪 Individual exits: {individual_exits}/{n_assets}")
        print(f"   💰 New portfolio: ${portfolio_value:,.0f}")
        print(f"   📊 Final spots: {[f'${spot:.2f}' for spot in entry_spots]}")
    
    # Convert to DataFrames for analysis
    groups_df = pd.DataFrame(all_group_results)
    assets_df = pd.DataFrame(all_asset_results)
    
    # Create comprehensive daily DataFrame with enhanced exit tracking
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
            'Active_Assets_Count': day_data['Active_Assets_Count'],
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
                'Asset_Entry_Spot': asset_data['entry_spot'],
                'Asset_Is_Active': asset_data['is_active'],
                'Asset_Exit_Day': asset_data['exit_day'],
                'Asset_Exit_Reason': asset_data['exit_reason'],
                'Asset_Exit_Value': asset_data['exit_value']
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

def calculate_enhanced_performance_metrics_with_exits(groups_df, assets_df, comprehensive_daily_df, initial_capital):
    """
    Calculate enhanced performance metrics including individual exit analysis
    """
    
    print(f"\n📈 CALCULATING ENHANCED PERFORMANCE METRICS WITH EXIT ANALYSIS")
    print("="*70)
    
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
    
    # NEW: Individual exit analysis
    total_individual_exits = groups_df['Individual_Exits_Count'].sum()
    total_possible_exits = len(groups_df) * 5  # 5 assets per group
    individual_exit_rate = (total_individual_exits / total_possible_exits) * 100
    
    # Groups with individual exits
    groups_with_exits = (groups_df['Individual_Exits_Count'] > 0).sum()
    groups_with_exits_rate = (groups_with_exits / len(groups_df)) * 100
    
    # All positions closed rate
    all_closed_rate = groups_df['All_Positions_Closed'].mean() * 100
    
    # Asset-level exit analysis
    individually_closed_assets = assets_df['Was_Individually_Closed'].sum()
    individual_close_rate_assets = (individually_closed_assets / len(assets_df)) * 100
    
    # Performance comparison: individually closed vs held to group exit
    individual_closed = assets_df[assets_df['Was_Individually_Closed'] == True]
    group_closed = assets_df[assets_df['Was_Individually_Closed'] == False]
    
    if len(individual_closed) > 0 and len(group_closed) > 0:
        individual_avg_return = individual_closed['Final_Asset_Return_Pct'].mean()
        group_avg_return = group_closed['Final_Asset_Return_Pct'].mean()
        individual_avg_days = individual_closed['Days_Held'].mean()
        group_avg_days = group_closed['Days_Held'].mean()
    else:
        individual_avg_return = 0
        group_avg_return = assets_df['Final_Asset_Return_Pct'].mean()
        individual_avg_days = 0
        group_avg_days = assets_df['Days_Held'].mean()
    
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
    
    # Exit reason analysis
    exit_reasons = assets_df['Exit_Reason'].value_counts()
    
    # Scenario analysis with exit breakdown
    scenario_exit_analysis = assets_df.groupby(['Scenario', 'Was_Individually_Closed']).agg({
        'Final_Asset_Return_Pct': ['count', 'mean', 'std'],
        'Days_Held': 'mean'
    }).round(2)
    
    # Holding period analysis
    avg_holding_period = groups_df['Days_Held'].mean()
    median_holding_period = groups_df['Days_Held'].median()
    
    # Create enhanced performance summary dictionary
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
        'all_closed_rate_pct': all_closed_rate,
        
        # NEW: Individual Exit Metrics
        'total_individual_exits': total_individual_exits,
        'individual_exit_rate_pct': individual_exit_rate,
        'groups_with_exits': groups_with_exits,
        'groups_with_exits_rate_pct': groups_with_exits_rate,
        'individual_close_rate_assets_pct': individual_close_rate_assets,
        
        # Performance Comparison
        'individual_avg_return_pct': individual_avg_return,
        'group_avg_return_pct': group_avg_return,
        'individual_avg_days': individual_avg_days,
        'group_avg_days': group_avg_days,
        'exit_benefit_pct': individual_avg_return - group_avg_return,
        
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
        
        # Exit Analysis
        'exit_reasons': exit_reasons,
        'scenario_exit_analysis': scenario_exit_analysis
    }
    
    # Print enhanced summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   💰 Total Return: {total_return:+.1f}% (${initial_capital:,.0f} → ${final_portfolio:,.0f})")
    print(f"   📈 Annualized Return: {annualized_return:+.1f}%")
    print(f"   📅 Simulation Period: {simulation_years:.1f} years ({total_days:.0f} days)")
    print(f"   🎯 Group Win Rate: {group_win_rate:.1f}%")
    print(f"   🚀 TP Hit Rate: {tp_hit_rate:.1f}%")
    print(f"   🏁 All Closed Rate: {all_closed_rate:.1f}%")
    print(f"   📉 Max Drawdown: {max_drawdown:.1f}%")
    print(f"   📊 Sharpe Ratio: {group_sharpe:.2f}")
    
    print(f"\n🚪 INDIVIDUAL EXIT ANALYSIS:")
    print(f"   📊 Individual Exit Rate: {individual_exit_rate:.1f}% ({total_individual_exits}/{total_possible_exits} assets)")
    print(f"   🏢 Groups with Exits: {groups_with_exits}/{len(groups_df)} ({groups_with_exits_rate:.1f}%)")
    print(f"   📈 Individual Exit Avg Return: {individual_avg_return:+.1f}%")
    print(f"   📈 Group Exit Avg Return: {group_avg_return:+.1f}%")
    print(f"   ✨ Exit Benefit: {individual_avg_return - group_avg_return:+.1f}%")
    print(f"   📅 Individual Avg Days: {individual_avg_days:.0f}")
    print(f"   📅 Group Avg Days: {group_avg_days:.0f}")
    
    print(f"\n📋 EXIT REASONS:")
    for reason, count in exit_reasons.items():
        if pd.notna(reason):
            print(f"   {reason}: {count} assets")
    
    return performance_metrics

def create_enhanced_continuous_timeline(comprehensive_daily_df):
    """
    Create enhanced continuous timeline with individual exit tracking
    """
    
    print(f"\n📅 Creating enhanced continuous timeline with exit tracking...")
    
    # Sort by group and day
    timeline_df = comprehensive_daily_df.copy()
    timeline_df = timeline_df.sort_values(['Group_Num', 'Day_Index'])
    
    # Create global day counter
    global_day = 0
    timeline_records = []
    
    for group_num in sorted(timeline_df['Group_Num'].unique()):
        group_data = timeline_df[timeline_df['Group_Num'] == group_num]
        
        # Get unique days for this group (group-level data)
        group_days = group_data.groupby(['Group_Num', 'Day_Index']).agg({
            'Total_Portfolio_Value': 'first',
            'Group_Return_Pct': 'first',
            'Portfolio_Return_Pct': 'first',
            'Total_PnL': 'first',
            'Active_Assets_Count': 'first',
            'Exit_Triggered': 'first',
            'Days_to_Expiry': 'first'
        }).reset_index()
        
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
                'Active_Assets_Count': day_record['Active_Assets_Count'],
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
        continuous_timeline_df['Active_Assets_Pct'] = (continuous_timeline_df['Active_Assets_Count'] / 5) * 100
    
    print(f"   ✅ Created enhanced timeline with {len(continuous_timeline_df)} daily records")
    
    return continuous_timeline_df

def save_enhanced_results_with_exits(groups_df, assets_df, comprehensive_daily_df, 
                                   continuous_timeline_df, performance_metrics):
    """
    Save enhanced results including individual exit analysis
    """
    
    print(f"\n💾 Saving enhanced results with exit analysis...")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main dataframes with "_individual_exits" suffix
    groups_filename = f'portfolio_groups_individual_exits_{timestamp}.csv'
    assets_filename = f'portfolio_assets_individual_exits_{timestamp}.csv'
    daily_filename = f'portfolio_daily_individual_exits_{timestamp}.csv'
    timeline_filename = f'portfolio_timeline_individual_exits_{timestamp}.csv'
    
    groups_df.to_csv(groups_filename, index=False)
    assets_df.to_csv(assets_filename, index=False)
    comprehensive_daily_df.to_csv(daily_filename, index=False)
    continuous_timeline_df.to_csv(timeline_filename, index=False)
    
    # Save enhanced performance metrics
    metrics_filename = f'portfolio_metrics_individual_exits_{timestamp}.csv'
    metrics_data = []
    
    for key, value in performance_metrics.items():
        if key not in ['exit_reasons', 'scenario_exit_analysis']:  # Handle these separately
            metrics_data.append({'Metric': key, 'Value': value})
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_filename, index=False)
    
    # Save exit reasons analysis
    exit_reasons_filename = f'portfolio_exit_reasons_{timestamp}.csv'
    if 'exit_reasons' in performance_metrics:
        exit_reasons_df = performance_metrics['exit_reasons'].reset_index()
        exit_reasons_df.columns = ['Exit_Reason', 'Count']
        exit_reasons_df.to_csv(exit_reasons_filename, index=False)
    
    # Save scenario exit analysis
    scenario_exit_filename = f'portfolio_scenario_exits_{timestamp}.csv'
    if 'scenario_exit_analysis' in performance_metrics:
        performance_metrics['scenario_exit_analysis'].to_csv(scenario_exit_filename)
    
    print(f"   📄 Groups: {groups_filename} ({len(groups_df)} records)")
    print(f"   📄 Assets: {assets_filename} ({len(assets_df)} records)")
    print(f"   📄 Daily: {daily_filename} ({len(comprehensive_daily_df)} records)")
    print(f"   📄 Timeline: {timeline_filename} ({len(continuous_timeline_df)} records)")
    print(f"   📄 Metrics: {metrics_filename}")
    print(f"   📄 Exit Reasons: {exit_reasons_filename}")
    print(f"   📄 Scenario Exits: {scenario_exit_filename}")
    
    return {
        'groups_file': groups_filename,
        'assets_file': assets_filename,
        'daily_file': daily_filename,
        'timeline_file': timeline_filename,
        'metrics_file': metrics_filename,
        'exit_reasons_file': exit_reasons_filename,
        'scenario_exits_file': scenario_exit_filename
    }



# ============================================================================
# PART 3/3: ENHANCED PLOTTING AND VISUALIZATION WITH INDIVIDUAL EXIT ANALYSIS
# ============================================================================

def plot_enhanced_individual_group_progression(group_num, comprehensive_daily_df, groups_df, assets_df, save_plots=True):
    """
    Enhanced individual group plot showing individual exits and active asset tracking
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
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    # Colors and markers for assets
    asset_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    asset_markers = ['o', 's', '^', 'D', 'v']
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    # Get group-level data (one record per day)
    group_level_data = group_data.groupby('Day_Index').first().reset_index()
    days_from_start = group_level_data['Day_Index']
    
    # Plot 1: Individual asset returns with exit markers
    ax1 = axes[0, 0]
    
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            asset_info = group_assets[group_assets['Asset_Num'] == asset_num]
            scenario = asset_info.iloc[0]['Scenario'] if not asset_info.empty else 'unknown'
            
            # Plot the return curve
            ax1.plot(asset_data['Day_Index'], asset_data['Asset_Return_Pct'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.8,
                    marker=asset_markers[asset_num], markersize=2, markevery=15,
                    label=f'Asset {asset_num+1} ({scenario[:4]})')
            
            # Mark individual exits
            exit_data = asset_data[asset_data['Asset_Is_Active'] == False]
            if not exit_data.empty:
                first_exit = exit_data.iloc[0]
                ax1.scatter([first_exit['Day_Index']], [first_exit['Asset_Return_Pct']], 
                           color='red', s=100, marker='X', zorder=10, 
                           edgecolor='black', linewidth=2)
                ax1.annotate(f'EXIT\n{first_exit["Asset_Exit_Reason"][:10]}...', 
                           xy=(first_exit['Day_Index'], first_exit['Asset_Return_Pct']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
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
    ax1.set_title(f'Group {group_num} - Asset Returns with Individual Exits')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-60, 40)
    
    # Plot 2: Active assets count over time
    ax2 = axes[0, 1]
    ax2.plot(group_level_data['Day_Index'], group_level_data['Active_Assets_Count'], 
            linewidth=3, color='purple', marker='o', markersize=4)
    ax2.fill_between(group_level_data['Day_Index'], group_level_data['Active_Assets_Count'], 0,
                    alpha=0.3, color='purple')
    
    ax2.set_xlabel('Days from Start')
    ax2.set_ylabel('Active Assets Count')
    ax2.set_title('Active Assets Over Time')
    ax2.set_ylim(0, 5.5)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for asset exits
    for asset_info in group_assets.iterrows():
        asset = asset_info[1]
        if asset['Exit_Day'] is not None and asset['Was_Individually_Closed']:
            ax2.annotate(f'A{asset["Asset_Num"]+1}', 
                        xy=(asset['Exit_Day'], 5 - asset['Asset_Num'] * 0.1),
                        xytext=(0, -10), textcoords='offset points',
                        fontsize=8, ha='center', color='red', fontweight='bold')
    
    # Plot 3: Portfolio value progression
    ax3 = axes[0, 2]
    ax3.plot(group_level_data['Day_Index'], group_level_data['Total_Portfolio_Value'], 
            linewidth=3, color='blue', alpha=0.8)
    ax3.fill_between(group_level_data['Day_Index'], group_level_data['Total_Portfolio_Value'], 
                    group_info['Entry_Portfolio'], alpha=0.3, color='blue')
    
    ax3.axhline(y=group_info['Entry_Portfolio'], color='black', linestyle=':', alpha=0.7, 
               linewidth=2, label='Entry Value')
    ax3.set_xlabel('Days from Start')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_title('Portfolio Value Progression')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 4: Underlying price movements with strike levels
    ax4 = axes[1, 0]
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            ax4.plot(asset_data['Day_Index'], asset_data['Asset_Spot_Price'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.7,
                    label=f'Asset {asset_num+1}')
            
            # Add strike price line
            strike_price = asset_data['Asset_Strike_Price'].iloc[0]
            ax4.axhline(y=strike_price, color=asset_colors[asset_num], 
                       linestyle='--', alpha=0.5, linewidth=1)
            
            # Mark exit point if individually closed
            asset_info = group_assets[group_assets['Asset_Num'] == asset_num].iloc[0]
            if asset_info['Was_Individually_Closed'] and asset_info['Exit_Day'] is not None:
                exit_day = asset_info['Exit_Day']
                # Fixed: Proper variable scope and null checking
                if exit_day is not None and exit_day < len(asset_data):
                    exit_price = asset_data.iloc[int(exit_day)]['Asset_Spot_Price']
                    ax4.scatter([exit_day], [exit_price], color='red', s=60, marker='X', zorder=10)
    
    ax4.set_xlabel('Days from Start')
    ax4.set_ylabel('Spot Price ($)')
    ax4.set_title('Underlying Movements vs Strike Prices')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Moneyness evolution with breach zones
    ax5 = axes[1, 1]
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            ax5.plot(asset_data['Day_Index'], asset_data['Asset_Moneyness'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.7,
                    label=f'Asset {asset_num+1}')
    
    # Add exit zones (assuming 2% buffer)
    ax5.axhline(y=1.02, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Upper Exit (102%)')
    ax5.axhline(y=0.98, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Lower Exit (98%)')
    ax5.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='ATM')
    ax5.fill_between(days_from_start, 0.98, 1.02, alpha=0.2, color='yellow', label='Danger Zone')
    
    ax5.set_xlabel('Days from Start')
    ax5.set_ylabel('Moneyness (S/K)')
    ax5.set_title('Moneyness with Exit Zones')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Straddle values over time
    ax6 = axes[1, 2]
    for asset_num in range(5):
        asset_data = group_data[group_data['Asset_Num'] == asset_num].copy()
        if not asset_data.empty:
            ax6.plot(asset_data['Day_Index'], asset_data['Asset_Straddle_Value'], 
                    color=asset_colors[asset_num], linewidth=2, alpha=0.7,
                    label=f'Asset {asset_num+1}')
            
            # Mark individual exit - Fixed: Proper variable scope
            asset_info = group_assets[group_assets['Asset_Num'] == asset_num].iloc[0]
            if asset_info['Was_Individually_Closed'] and asset_info['Exit_Day'] is not None:
                exit_day = asset_info['Exit_Day']
                if exit_day is not None and exit_day < len(asset_data):
                    exit_value = asset_data.iloc[int(exit_day)]['Asset_Straddle_Value']
                    ax6.scatter([exit_day], [exit_value], color='red', s=60, marker='X', zorder=10)
    
    ax6.set_xlabel('Days from Start')
    ax6.set_ylabel('Straddle Value ($)')
    ax6.set_title('Straddle Values Over Time')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Exit analysis summary
    ax7 = axes[2, 0]
    ax7.axis('off')
    
    # Calculate exit statistics for this group
    individual_exits = group_assets[group_assets['Was_Individually_Closed'] == True]
    group_exits = group_assets[group_assets['Was_Individually_Closed'] == False]
    
    exit_summary_text = f"""
EXIT ANALYSIS - GROUP {group_num}
{'='*25}

INDIVIDUAL EXITS: {len(individual_exits)}/5
"""
    
    for _, asset in individual_exits.iterrows():
        exit_summary_text += f"Asset {asset['Asset_Num']+1}: Day {asset['Exit_Day']}\n"
        exit_summary_text += f"  Reason: {asset['Exit_Reason']}\n"
        exit_summary_text += f"  Return: {asset['Final_Asset_Return_Pct']:+.1f}%\n"
    
    if len(group_exits) > 0:
        exit_summary_text += f"\nGROUP EXITS: {len(group_exits)}/5\n"
        exit_summary_text += f"Avg Return: {group_exits['Final_Asset_Return_Pct'].mean():+.1f}%\n"
    
    exit_summary_text += f"\nCOMPARISON:\n"
    if len(individual_exits) > 0 and len(group_exits) > 0:
        ind_avg = individual_exits['Final_Asset_Return_Pct'].mean()
        grp_avg = group_exits['Final_Asset_Return_Pct'].mean()
        exit_summary_text += f"Individual: {ind_avg:+.1f}%\n"
        exit_summary_text += f"Group: {grp_avg:+.1f}%\n"
        exit_summary_text += f"Benefit: {ind_avg - grp_avg:+.1f}%\n"
    
    ax7.text(0.1, 0.9, exit_summary_text, transform=ax7.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
    
    # Plot 8: Performance comparison
    ax8 = axes[2, 1]
    
    if len(individual_exits) > 0 and len(group_exits) > 0:
        categories = ['Individual Exits', 'Group Exits']
        returns = [individual_exits['Final_Asset_Return_Pct'].mean(), 
                  group_exits['Final_Asset_Return_Pct'].mean()]
        colors_bar = ['red', 'blue']
        
        bars = ax8.bar(categories, returns, color=colors_bar, alpha=0.7)
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax8.set_ylabel('Average Return (%)')
        ax8.set_title('Individual vs Group Exit Performance')
        ax8.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ret in zip(bars, returns):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{ret:+.1f}%', ha='center', va='bottom' if ret > 0 else 'top', 
                    fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Performance Comparison')
    
    # Plot 9: Summary statistics and info
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Create enhanced summary text
    final_return = group_info['Group_Return_Pct']
    days_held = group_info['Days_Held']
    exit_reason = group_info['Exit_Reason']
    tp_hit = group_info['TP_Hit']
    individual_exits_count = group_info['Individual_Exits_Count']
    
    summary_text = f"""
GROUP {group_num} ENHANCED SUMMARY
{'='*28}

PERFORMANCE:
Final Return: {final_return:+.1f}%
Days Held: {days_held}
Exit Reason: {exit_reason}
TP Hit: {'Yes' if tp_hit else 'No'}

PORTFOLIO:
Entry: ${group_info['Entry_Portfolio']:,.0f}
Exit: ${group_info['Exit_Portfolio']:,.0f}
P&L: ${group_info['Group_PnL']:,.0f}

RISK MANAGEMENT:
Individual Exits: {individual_exits_count}/5
All Closed: {'Yes' if group_info['All_Positions_Closed'] else 'No'}

SCENARIOS:
Bull: {sum(1 for _, a in group_assets.iterrows() if a['Scenario'] == 'bull')}
Sideways: {sum(1 for _, a in group_assets.iterrows() if a['Scenario'] == 'sideways')}
Bear: {sum(1 for _, a in group_assets.iterrows() if a['Scenario'] == 'bear')}

EFFECTIVENESS:
Risk Mitigation: {'Active' if individual_exits_count > 0 else 'None'}
"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Group {group_num} Enhanced Analysis with Individual Exit Tracking\n'
                f'Return: {final_return:+.1f}% | Days: {days_held} | Individual Exits: {individual_exits_count}/5 | Exit: {exit_reason}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        filename = f'group_{group_num}_enhanced_individual_exits_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
    
    plt.show()

def plot_enhanced_portfolio_dashboard_with_exits(groups_df, assets_df, continuous_timeline_df, performance_metrics, save_plots=True):
    """
    Create enhanced portfolio dashboard with individual exit analysis
    """
    
    print(f"\n📊 Creating enhanced portfolio dashboard with individual exit analysis...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(28, 20))
    
    # Plot 1: Portfolio equity curve
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(continuous_timeline_df['Global_Day'], continuous_timeline_df['Total_Portfolio_Value'], 
             linewidth=3, color='blue', alpha=0.8)
    ax1.fill_between(continuous_timeline_df['Global_Day'], continuous_timeline_df['Total_Portfolio_Value'], 
                     performance_metrics['initial_capital'], alpha=0.3, color='blue')
    
    ax1.set_xlabel('Days Since Start')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Equity Curve')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Active assets over time
    ax2 = plt.subplot(4, 4, 2)
    ax2.plot(continuous_timeline_df['Global_Day'], continuous_timeline_df['Active_Assets_Pct'], 
             linewidth=2, color='purple', alpha=0.8)
    ax2.fill_between(continuous_timeline_df['Global_Day'], continuous_timeline_df['Active_Assets_Pct'], 0,
                     alpha=0.3, color='purple')
    
    ax2.set_xlabel('Days Since Start')
    ax2.set_ylabel('Active Assets (%)')
    ax2.set_title('Active Assets Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Plot 3: Individual vs Group exit performance
    ax3 = plt.subplot(4, 4, 3)
    individual_closed = assets_df[assets_df['Was_Individually_Closed'] == True]
    group_closed = assets_df[assets_df['Was_Individually_Closed'] == False]
    
    if len(individual_closed) > 0 and len(group_closed) > 0:
        categories = ['Individual\nExits', 'Group\nExits']
        returns = [individual_closed['Final_Asset_Return_Pct'].mean(), 
                  group_closed['Final_Asset_Return_Pct'].mean()]
        counts = [len(individual_closed), len(group_closed)]
        colors_bar = ['red', 'blue']
        
        bars = ax3.bar(categories, returns, color=colors_bar, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Average Return (%)')
        ax3.set_title('Exit Type Performance Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add count and value labels
        for bar, ret, count in zip(bars, returns, counts):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{ret:+.1f}%\n(n={count})', ha='center', 
                    va='bottom' if ret > 0 else 'top', fontweight='bold', fontsize=9)
    
    # Plot 4: Enhanced performance metrics summary
    ax4 = plt.subplot(4, 4, 4)
    ax4.axis('off')
    
    enhanced_metrics_text = f"""
ENHANCED PERFORMANCE SUMMARY
{'='*25}

RETURNS:
Total: {performance_metrics['total_return_pct']:+.1f}%
Annualized: {performance_metrics['annualized_return_pct']:+.1f}%

RISK MANAGEMENT:
Individual Exit Rate: {performance_metrics['individual_exit_rate_pct']:.1f}%
Groups with Exits: {performance_metrics['groups_with_exits_rate_pct']:.1f}%
Exit Benefit: {performance_metrics['exit_benefit_pct']:+.1f}%

SUCCESS RATES:
Group Win Rate: {performance_metrics['group_win_rate_pct']:.1f}%
TP Hit Rate: {performance_metrics['tp_hit_rate_pct']:.1f}%

RISK METRICS:
Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%
Sharpe Ratio: {performance_metrics['group_sharpe_ratio']:.2f}
"""
    
    ax4.text(0.1, 0.9, enhanced_metrics_text, transform=ax4.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Continue with remaining plots (5-16) as before, but simplified for space
    # ... Additional plots can be added here following the same pattern
    
    plt.suptitle(f'Enhanced Portfolio Dashboard with Individual Exit Analysis\n'
                f'Total Return: {performance_metrics["total_return_pct"]:+.1f}% | '
                f'Individual Exit Rate: {performance_metrics["individual_exit_rate_pct"]:.1f}% | '
                f'Exit Benefit: {performance_metrics["exit_benefit_pct"]:+.1f}% | '
                f'Sharpe: {performance_metrics["group_sharpe_ratio"]:.2f}', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'enhanced_portfolio_dashboard_individual_exits.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
    
    plt.show()

def plot_all_enhanced_group_progressions(comprehensive_daily_df, groups_df, assets_df, max_groups=None):
    """
    Plot enhanced progression charts for all groups with individual exit tracking
    """
    
    unique_groups = sorted(comprehensive_daily_df['Group_Num'].unique())
    
    if max_groups:
        unique_groups = unique_groups[:max_groups]
    
    print(f"\n📊 Creating enhanced individual progression plots for {len(unique_groups)} groups...")
    
    for group_num in unique_groups:
        print(f"   📈 Plotting Enhanced Group {group_num}...")
        plot_enhanced_individual_group_progression(group_num, comprehensive_daily_df, groups_df, assets_df)

def main():
    """
    Main function to run the enhanced portfolio simulation with individual exit conditions
    """
    
    print("🚀 ENHANCED PORTFOLIO TRADING SIMULATOR WITH INDIVIDUAL EXITS")
    print("="*80)
    print("Advanced simulation with individual asset exit conditions for risk management")
    print("Assets automatically exit when underlying breaches strike price levels")
    print("="*80)
    
    # Enhanced simulation parameters
    n_groups = 12  # Run 12 groups for comprehensive analysis
    initial_capital = 100000
    target_annual_return = 0.15
    n_assets = 5
    tp_threshold = 20.0
    strike_exit_buffer = 0.02  # 2% buffer around strike for exits
    
    # Step 1: Run enhanced simulation with individual exits
    print(f"\n🎯 Running enhanced simulation with individual exit conditions...")
    groups_df, assets_df, comprehensive_daily_df, all_daily_data = run_comprehensive_portfolio_simulation_with_exits(
        n_groups=n_groups,
        initial_capital=initial_capital,
        target_annual_return=target_annual_return,
        n_assets=n_assets,
        tp_threshold=tp_threshold,
        strike_exit_buffer=strike_exit_buffer
    )
    
    # Step 2: Calculate enhanced performance metrics
    print(f"\n📊 Calculating enhanced performance metrics with exit analysis...")
    performance_metrics = calculate_enhanced_performance_metrics_with_exits(
        groups_df, assets_df, comprehensive_daily_df, initial_capital
    )
    
    # Step 3: Create enhanced continuous timeline
    print(f"\n📅 Creating enhanced continuous timeline...")
    continuous_timeline_df = create_enhanced_continuous_timeline(comprehensive_daily_df)
    
    # Step 4: Save enhanced results
    print(f"\n💾 Saving enhanced results with individual exit analysis...")
    saved_files = save_enhanced_results_with_exits(
        groups_df, assets_df, comprehensive_daily_df, 
        continuous_timeline_df, performance_metrics
    )
    
    # Step 5: Create enhanced visualizations
    print(f"\n📊 Creating enhanced visualizations with individual exit analysis...")
    
    # Enhanced portfolio dashboard
    plot_enhanced_portfolio_dashboard_with_exits(
        groups_df, assets_df, continuous_timeline_df, performance_metrics
    )
    
    # Enhanced individual group progressions (first 6 groups)
    print(f"\n📈 Creating enhanced individual group progression plots...")
    plot_all_enhanced_group_progressions(comprehensive_daily_df, groups_df, assets_df, max_groups=6)
    
    # Final enhanced summary
    print(f"\n🎉 ENHANCED ANALYSIS WITH INDIVIDUAL EXITS COMPLETE!")
    print("="*80)
    print(f"📈 Portfolio Performance: ${initial_capital:,.0f} → ${performance_metrics['final_portfolio']:,.0f}")
    print(f"📊 Total Return: {performance_metrics['total_return_pct']:+.1f}%")
    print(f"📅 Annualized Return: {performance_metrics['annualized_return_pct']:+.1f}%")
    print(f"🎯 Group Win Rate: {performance_metrics['group_win_rate_pct']:.1f}%")
    print(f"🚀 TP Hit Rate: {performance_metrics['tp_hit_rate_pct']:.1f}%")
    print(f"🚪 Individual Exit Rate: {performance_metrics['individual_exit_rate_pct']:.1f}%")
    print(f"✨ Exit Benefit: {performance_metrics['exit_benefit_pct']:+.1f}%")
    print(f"📉 Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%")
    print(f"📊 Sharpe Ratio: {performance_metrics['group_sharpe_ratio']:.2f}")
    print("="*80)
    print(f"🛡️  RISK MANAGEMENT EFFECTIVENESS:")
    print(f"   Individual Exits: {performance_metrics['total_individual_exits']} assets")
    print(f"   Groups with Exits: {performance_metrics['groups_with_exits']}/{len(groups_df)}")
    print(f"   Average Days Saved: {performance_metrics['group_avg_days'] - performance_metrics['individual_avg_days']:.0f}")
    print(f"   Risk Mitigation: {'EFFECTIVE' if performance_metrics['exit_benefit_pct'] > 0 else 'NEEDS IMPROVEMENT'}")
    print("="*80)
    print(f"📁 All enhanced data saved with individual exit analysis")
    print(f"📊 All enhanced plots generated with exit tracking")
    print(f"✅ Ready for detailed review of individual exit strategy effectiveness!")

if __name__ == "__main__":
    main()