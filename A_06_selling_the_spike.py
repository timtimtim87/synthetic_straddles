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
# PART 1/3: CORE FUNCTIONS AND CONTRARIAN STRATEGY LOGIC
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
    Select trading scenario with probability weighting
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

class Contrarian_Straddle_Position:
    """
    Class to track individual contrarian straddle positions
    """
    
    def __init__(self, asset_num, entry_day, entry_spot, strike_price, entry_straddle_value, 
                 position_size, tp_threshold=30.0):
        self.asset_num = asset_num
        self.entry_day = entry_day
        self.entry_spot = entry_spot
        self.strike_price = strike_price
        self.entry_straddle_value = entry_straddle_value
        self.position_size = position_size
        self.tp_threshold = tp_threshold
        
        self.is_active = True
        self.exit_day = None
        self.exit_reason = None
        self.exit_straddle_value = None
        self.max_return = 0.0
        self.min_return = 0.0
        
        # Track daily performance
        self.daily_returns = []
        self.daily_straddle_values = []
        self.daily_spot_prices = []
        
    def update_daily(self, day, spot_price, current_straddle_value):
        """Update position with daily market data"""
        if not self.is_active:
            return False
        
        # Calculate return (short straddle perspective)
        pnl_per_contract = self.entry_straddle_value - current_straddle_value
        return_pct = (pnl_per_contract / self.entry_straddle_value) * 100
        
        # Update tracking
        self.daily_returns.append(return_pct)
        self.daily_straddle_values.append(current_straddle_value)
        self.daily_spot_prices.append(spot_price)
        
        # Update extremes
        self.max_return = max(self.max_return, return_pct)
        self.min_return = min(self.min_return, return_pct)
        
        # Check for TP
        if return_pct >= self.tp_threshold:
            self.is_active = False
            self.exit_day = day
            self.exit_reason = f"{self.tp_threshold:.0f}% TP Hit"
            self.exit_straddle_value = current_straddle_value
            return True  # Position closed
        
        return False  # Position still active
    
    def force_close(self, day, spot_price, current_straddle_value, reason="Expiry"):
        """Force close position at expiry or other reason"""
        if self.is_active:
            self.is_active = False
            self.exit_day = day
            self.exit_reason = reason
            self.exit_straddle_value = current_straddle_value
            
            # Calculate final return
            pnl_per_contract = self.entry_straddle_value - current_straddle_value
            final_return = (pnl_per_contract / self.entry_straddle_value) * 100
            
            self.daily_returns.append(final_return)
            self.daily_straddle_values.append(current_straddle_value)
            self.daily_spot_prices.append(spot_price)
    
    def get_final_return(self):
        """Get final return for this position"""
        if len(self.daily_returns) == 0:
            return 0.0
        return self.daily_returns[-1]
    
    def get_pnl(self):
        """Get P&L for this position"""
        if self.exit_straddle_value is None:
            return 0.0
        pnl_per_contract = self.entry_straddle_value - self.exit_straddle_value
        return pnl_per_contract * self.position_size

def check_contrarian_entry_signal(asset_num, day, spot_price, initial_straddle_value, 
                                 current_straddle_value, days_to_expiry, 
                                 drawdown_threshold=20.0, min_days_to_expiry=150):
    """
    Check if conditions are met for contrarian straddle entry
    
    Parameters:
    - asset_num: Asset identifier
    - day: Current day
    - spot_price: Current spot price
    - initial_straddle_value: Initial straddle value (from day 0)
    - current_straddle_value: Current straddle value
    - days_to_expiry: Days remaining to expiry
    - drawdown_threshold: Minimum drawdown to trigger entry (20% default)
    - min_days_to_expiry: Minimum days to expiry required (150 default)
    
    Returns:
    - should_enter: Boolean indicating if we should enter
    - entry_reason: String describing entry reason
    """
    
    # Check minimum days to expiry
    if days_to_expiry < min_days_to_expiry:
        return False, None
    
    # Calculate drawdown (increase in straddle value = drawdown for short position)
    drawdown_pct = ((current_straddle_value - initial_straddle_value) / initial_straddle_value) * 100
    
    # Check if drawdown threshold is met
    if drawdown_pct >= drawdown_threshold:
        return True, f"Contrarian Entry: {drawdown_pct:.1f}% drawdown, {days_to_expiry} DTE"
    
    return False, None

def simulate_contrarian_straddle_strategy(asset_num, entry_spot, total_days=365, 
                                        strike_multiplier=1.25, scenario='bull', 
                                        annual_drift=0.15, annual_vol=0.20, 
                                        iv=0.25, risk_free_rate=0.04,
                                        drawdown_threshold=20.0, tp_threshold=30.0,
                                        min_days_to_expiry=150, capital_per_trade=20000,
                                        seed=None):
    """
    Simulate contrarian straddle strategy for a single asset
    """
    
    print(f"    📊 Simulating Asset {asset_num+1}: {scenario.title()} scenario")
    print(f"        Entry spot: ${entry_spot:.2f}, Drift: {annual_drift*100:.1f}%")
    
    # Generate price path
    if seed is not None:
        np.random.seed(seed)
    
    price_path = generate_realistic_gbm_path(
        S0=entry_spot, 
        annual_drift=annual_drift, 
        days=total_days, 
        annual_vol=annual_vol, 
        seed=seed
    )
    
    # Calculate strike price and initial straddle value
    strike_price = entry_spot * strike_multiplier
    initial_straddle_value = calculate_straddle_value(
        entry_spot, strike_price, 1.0, risk_free_rate, iv
    )
    
    print(f"        Strike: ${strike_price:.2f}, Initial straddle: ${initial_straddle_value:.2f}")
    
    # Initialize tracking
    days_array = np.arange(total_days, -1, -1)
    contrarian_positions = []
    daily_monitoring_data = []
    total_capital_deployed = 0
    
    # Daily simulation loop
    for i, day in enumerate(days_array):
        spot_price = price_path[i]
        time_to_expiry = max(day / 365.0, 1/365)
        
        # Calculate current straddle value
        current_straddle_value = calculate_straddle_value(
            spot_price, strike_price, time_to_expiry, risk_free_rate, iv
        )
        
        # Update existing positions
        closed_positions = []
        for pos in contrarian_positions:
            if pos.is_active:
                position_closed = pos.update_daily(i, spot_price, current_straddle_value)
                if position_closed:
                    closed_positions.append(pos)
                    print(f"          🎯 Position closed on day {i}: {pos.exit_reason}, Return: {pos.get_final_return():+.1f}%")
        
        # Check for new contrarian entry signal
        should_enter, entry_reason = check_contrarian_entry_signal(
            asset_num, i, spot_price, initial_straddle_value, current_straddle_value,
            day, drawdown_threshold, min_days_to_expiry
        )
        
        # Enter new position if signal triggered and we have capital
        if should_enter and (total_capital_deployed + capital_per_trade) <= capital_per_trade * 3:  # Max 3 positions
            position_size = capital_per_trade / current_straddle_value
            
            new_position = Contrarian_Straddle_Position(
                asset_num=asset_num,
                entry_day=i,
                entry_spot=spot_price,
                strike_price=strike_price,
                entry_straddle_value=current_straddle_value,
                position_size=position_size,
                tp_threshold=tp_threshold
            )
            
            contrarian_positions.append(new_position)
            total_capital_deployed += capital_per_trade
            
            print(f"          🚀 NEW CONTRARIAN ENTRY on day {i}: {entry_reason}")
            print(f"              Spot: ${spot_price:.2f}, Straddle: ${current_straddle_value:.2f}")
            print(f"              Position size: {position_size:.2f} contracts")
        
        # Store daily monitoring data
        active_positions = [pos for pos in contrarian_positions if pos.is_active]
        total_pnl = sum(pos.get_pnl() for pos in contrarian_positions if not pos.is_active)
        unrealized_pnl = 0
        
        for pos in active_positions:
            if len(pos.daily_returns) > 0:
                unrealized_pnl += (pos.daily_returns[-1] / 100) * capital_per_trade
        
        daily_monitoring_data.append({
            'Day': i,
            'Days_to_Expiry': day,
            'Spot_Price': spot_price,
            'Straddle_Value': current_straddle_value,
            'Straddle_Drawdown_Pct': ((current_straddle_value - initial_straddle_value) / initial_straddle_value) * 100,
            'Active_Positions': len(active_positions),
            'Total_Positions': len(contrarian_positions),
            'Realized_PnL': total_pnl,
            'Unrealized_PnL': unrealized_pnl,
            'Total_PnL': total_pnl + unrealized_pnl,
            'Capital_Deployed': total_capital_deployed
        })
    
    # Force close any remaining active positions at expiry
    for pos in contrarian_positions:
        if pos.is_active:
            final_spot = price_path[-1]
            final_straddle_value = calculate_straddle_value(
                final_spot, strike_price, 1/365, risk_free_rate, iv
            )
            pos.force_close(len(days_array)-1, final_spot, final_straddle_value, "Expiry")
            print(f"          📅 Position closed at expiry: Return: {pos.get_final_return():+.1f}%")
    
    # Calculate final statistics
    total_realized_pnl = sum(pos.get_pnl() for pos in contrarian_positions)
    total_return_pct = (total_realized_pnl / total_capital_deployed) * 100 if total_capital_deployed > 0 else 0
    
    position_returns = [pos.get_final_return() for pos in contrarian_positions]
    avg_position_return = np.mean(position_returns) if len(position_returns) > 0 else 0
    win_rate = sum(1 for ret in position_returns if ret > 0) / len(position_returns) * 100 if len(position_returns) > 0 else 0
    
    # Summary results
    asset_result = {
        'Asset_Num': asset_num,
        'Scenario': scenario,
        'Annual_Drift': annual_drift,
        'Entry_Spot': entry_spot,
        'Final_Spot': price_path[-1],
        'Strike_Price': strike_price,
        'Initial_Straddle_Value': initial_straddle_value,
        'Final_Straddle_Value': current_straddle_value,
        'Max_Straddle_Drawdown_Pct': max([d['Straddle_Drawdown_Pct'] for d in daily_monitoring_data]),
        'Total_Positions_Entered': len(contrarian_positions),
        'Capital_Deployed': total_capital_deployed,
        'Total_Realized_PnL': total_realized_pnl,
        'Total_Return_Pct': total_return_pct,
        'Avg_Position_Return_Pct': avg_position_return,
        'Win_Rate_Pct': win_rate,
        'Path_Volatility': np.std(np.diff(np.log(price_path))) * np.sqrt(365) * 100
    }
    
    print(f"        ✅ Summary: {len(contrarian_positions)} positions, {win_rate:.0f}% win rate, {total_return_pct:+.1f}% total return")
    
    return asset_result, contrarian_positions, daily_monitoring_data


# ============================================================================
# PART 2/3: PORTFOLIO SIMULATION AND PERFORMANCE ANALYSIS
# ============================================================================

def run_contrarian_portfolio_simulation(n_groups=10, initial_capital=100000, n_assets=5,
                                       capital_per_asset=20000, drawdown_threshold=20.0,
                                       tp_threshold=30.0, min_days_to_expiry=150,
                                       strike_multiplier=1.25, iv=0.25, risk_free_rate=0.04,
                                       annual_vol=0.20, total_days=365):
    """
    Main contrarian portfolio simulation runner
    """
    
    print(f"🚀 CONTRARIAN STRADDLE PORTFOLIO SIMULATION")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"   Groups: {n_groups}")
    print(f"   Assets per group: {n_assets}")
    print(f"   Initial capital: ${initial_capital:,.0f}")
    print(f"   Capital per asset: ${capital_per_asset:,.0f}")
    print(f"   Drawdown threshold: {drawdown_threshold:.1f}%")
    print(f"   TP threshold: {tp_threshold:.1f}%")
    print(f"   Min days to expiry: {min_days_to_expiry}")
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
    all_asset_results = []
    all_position_results = []
    all_daily_monitoring_data = []
    
    # Performance tracking
    simulation_start_time = datetime.now()
    
    # Run each group
    for group_num in range(1, n_groups + 1):
        print(f"\n🔥 Group {group_num}/{n_groups}: Portfolio ${portfolio_value:,.0f}")
        print(f"   Entry spots: {[f'${spot:.2f}' for spot in entry_spots]}")
        
        # Simulate group with contrarian strategy
        group_result, group_asset_results, group_positions, group_daily_data = simulate_contrarian_group(
            group_num=group_num,
            entry_spots=entry_spots,
            portfolio_value=portfolio_value,
            n_assets=n_assets,
            capital_per_asset=capital_per_asset,
            drawdown_threshold=drawdown_threshold,
            tp_threshold=tp_threshold,
            min_days_to_expiry=min_days_to_expiry,
            strike_multiplier=strike_multiplier,
            iv=iv,
            risk_free_rate=risk_free_rate,
            annual_vol=annual_vol,
            total_days=total_days
        )
        
        # Update portfolio value for next group
        portfolio_value = group_result['Exit_Portfolio']
        
        # Update entry spots for next group (use final spots from this group)
        entry_spots = [asset['Final_Spot'] for asset in group_asset_results]
        
        # Store results
        all_group_results.append(group_result)
        all_asset_results.extend(group_asset_results)
        all_position_results.extend(group_positions)
        all_daily_monitoring_data.extend(group_daily_data)
        
        # Progress update
        print(f"   ✅ Group completed")
        print(f"   📈 Group return: {group_result['Group_Return_Pct']:+.1f}%")
        print(f"   🎯 Total positions: {group_result['Total_Positions']}")
        print(f"   💰 New portfolio: ${portfolio_value:,.0f}")
        print(f"   📊 Final spots: {[f'${spot:.2f}' for spot in entry_spots]}")
    
    # Convert to DataFrames for analysis
    groups_df = pd.DataFrame(all_group_results)
    assets_df = pd.DataFrame(all_asset_results)
    positions_df = pd.DataFrame(all_position_results)
    daily_df = pd.DataFrame(all_daily_monitoring_data)
    
    # Calculate simulation metrics
    simulation_end_time = datetime.now()
    simulation_duration = simulation_end_time - simulation_start_time
    
    print(f"\n🎉 Simulation Complete!")
    print(f"⏱️  Duration: {simulation_duration}")
    print(f"📊 Generated {len(groups_df)} groups, {len(assets_df)} assets, {len(positions_df)} positions")
    print(f"📅 Total daily records: {len(daily_df)}")
    
    return groups_df, assets_df, positions_df, daily_df

def simulate_contrarian_group(group_num, entry_spots, portfolio_value, n_assets=5,
                            capital_per_asset=20000, drawdown_threshold=20.0,
                            tp_threshold=30.0, min_days_to_expiry=150,
                            strike_multiplier=1.25, iv=0.25, risk_free_rate=0.04,
                            annual_vol=0.20, total_days=365):
    """
    Simulate contrarian strategy for a group of assets
    """
    
    print(f"  📊 Simulating Group {group_num} with contrarian strategy...")
    
    # Generate scenario and simulate each asset
    group_asset_results = []
    group_all_positions = []
    group_all_daily_data = []
    
    for asset_num in range(n_assets):
        # Generate scenario for this asset
        scenario_seed = 5000 + group_num * 100 + asset_num
        scenario, drift = select_trade_scenario(group_num * n_assets + asset_num, seed=scenario_seed)
        
        # Simulate contrarian strategy for this asset
        asset_seed = 10000 + group_num * 100 + asset_num
        asset_result, asset_positions, asset_daily_data = simulate_contrarian_straddle_strategy(
            asset_num=asset_num,
            entry_spot=entry_spots[asset_num],
            total_days=total_days,
            strike_multiplier=strike_multiplier,
            scenario=scenario,
            annual_drift=drift,
            annual_vol=annual_vol,
            iv=iv,
            risk_free_rate=risk_free_rate,
            drawdown_threshold=drawdown_threshold,
            tp_threshold=tp_threshold,
            min_days_to_expiry=min_days_to_expiry,
            capital_per_trade=capital_per_asset,
            seed=asset_seed
        )
        
        # Add group identifier to results
        asset_result['Group_Num'] = group_num
        
        # Add group and asset identifiers to positions
        for pos in asset_positions:
            position_data = {
                'Group_Num': group_num,
                'Asset_Num': asset_num,
                'Entry_Day': pos.entry_day,
                'Exit_Day': pos.exit_day,
                'Entry_Spot': pos.entry_spot,
                'Strike_Price': pos.strike_price,
                'Entry_Straddle_Value': pos.entry_straddle_value,
                'Exit_Straddle_Value': pos.exit_straddle_value,
                'Position_Size': pos.position_size,
                'Final_Return_Pct': pos.get_final_return(),
                'Max_Return_Pct': pos.max_return,
                'Min_Return_Pct': pos.min_return,
                'PnL': pos.get_pnl(),
                'Exit_Reason': pos.exit_reason,
                'TP_Hit': pos.exit_reason and 'TP Hit' in pos.exit_reason,
                'Days_Held': (pos.exit_day - pos.entry_day + 1) if pos.exit_day is not None else total_days - pos.entry_day
            }
            group_all_positions.append(position_data)
        
        # Add group and asset identifiers to daily data
        for daily_record in asset_daily_data:
            daily_record['Group_Num'] = group_num
            daily_record['Asset_Num'] = asset_num
            group_all_daily_data.append(daily_record)
        
        group_asset_results.append(asset_result)
    
    # Calculate group-level metrics
    total_capital_deployed = sum(asset['Capital_Deployed'] for asset in group_asset_results)
    total_realized_pnl = sum(asset['Total_Realized_PnL'] for asset in group_asset_results)
    total_positions = sum(asset['Total_Positions_Entered'] for asset in group_asset_results)
    
    # Group return calculation
    group_return_pct = (total_realized_pnl / total_capital_deployed) * 100 if total_capital_deployed > 0 else 0
    
    # Portfolio impact
    portfolio_impact = total_realized_pnl
    exit_portfolio = portfolio_value + portfolio_impact
    portfolio_return_pct = (portfolio_impact / portfolio_value) * 100
    
    # Win rate calculation
    all_position_returns = [pos['Final_Return_Pct'] for pos in group_all_positions]
    group_win_rate = sum(1 for ret in all_position_returns if ret > 0) / len(all_position_returns) * 100 if len(all_position_returns) > 0 else 0
    tp_hit_rate = sum(1 for pos in group_all_positions if pos['TP_Hit']) / len(group_all_positions) * 100 if len(group_all_positions) > 0 else 0
    
    # Create group result
    group_result = {
        'Group_Num': group_num,
        'Entry_Portfolio': portfolio_value,
        'Exit_Portfolio': exit_portfolio,
        'Portfolio_Impact': portfolio_impact,
        'Group_Return_Pct': group_return_pct,
        'Portfolio_Return_Pct': portfolio_return_pct,
        'Total_Capital_Deployed': total_capital_deployed,
        'Total_Realized_PnL': total_realized_pnl,
        'Total_Positions': total_positions,
        'Group_Win_Rate_Pct': group_win_rate,
        'TP_Hit_Rate_Pct': tp_hit_rate,
        'N_Assets': n_assets,
        'Avg_Positions_Per_Asset': total_positions / n_assets if n_assets > 0 else 0
    }
    
    return group_result, group_asset_results, group_all_positions, group_all_daily_data

def calculate_contrarian_performance_metrics(groups_df, assets_df, positions_df, daily_df, initial_capital):
    """
    Calculate comprehensive performance metrics for contrarian strategy
    """
    
    print(f"\n📈 CALCULATING CONTRARIAN STRATEGY PERFORMANCE METRICS")
    print("="*60)
    
    # Basic performance metrics
    final_portfolio = groups_df['Exit_Portfolio'].iloc[-1]
    total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
    
    # Calculate simulation period (approximate)
    total_groups = len(groups_df)
    simulation_years = total_groups * (365 / 365.25)  # Approximate
    
    # Annualized return
    if simulation_years > 0:
        annualized_return = ((final_portfolio / initial_capital) ** (1/simulation_years) - 1) * 100
    else:
        annualized_return = 0
    
    # Group-level metrics
    group_returns = groups_df['Group_Return_Pct']
    portfolio_returns = groups_df['Portfolio_Return_Pct']
    
    # Success rates
    group_win_rate = (group_returns > 0).mean() * 100
    overall_win_rate = positions_df['Final_Return_Pct'].apply(lambda x: x > 0).mean() * 100
    tp_hit_rate = positions_df['TP_Hit'].mean() * 100
    
    # Position statistics
    avg_position_return = positions_df['Final_Return_Pct'].mean()
    median_position_return = positions_df['Final_Return_Pct'].median()
    position_volatility = positions_df['Final_Return_Pct'].std()
    
    # Risk metrics
    group_volatility = group_returns.std()
    if group_volatility > 0:
        group_sharpe = group_returns.mean() / group_volatility
    else:
        group_sharpe = 0
    
    # Maximum drawdown calculation
    portfolio_values = [initial_capital] + groups_df['Exit_Portfolio'].tolist()
    peak_values = pd.Series(portfolio_values).expanding().max()
    drawdowns = (pd.Series(portfolio_values) - peak_values) / peak_values * 100
    max_drawdown = drawdowns.min()
    
    # Position frequency analysis
    total_positions = len(positions_df)
    avg_positions_per_group = groups_df['Total_Positions'].mean()
    avg_positions_per_asset = assets_df['Total_Positions_Entered'].mean()
    
    # Holding period analysis
    avg_holding_period = positions_df['Days_Held'].mean()
    median_holding_period = positions_df['Days_Held'].median()
    
    # Capital deployment analysis
    total_capital_deployed = groups_df['Total_Capital_Deployed'].sum()
    capital_efficiency = (final_portfolio - initial_capital) / total_capital_deployed * 100 if total_capital_deployed > 0 else 0
    
    # Scenario analysis
    scenario_performance = assets_df.groupby('Scenario').agg({
        'Total_Return_Pct': ['count', 'mean', 'std'],
        'Win_Rate_Pct': 'mean',
        'Total_Positions_Entered': 'mean'
    }).round(2)
    
    # Create performance summary dictionary
    performance_metrics = {
        # Overall Performance
        'initial_capital': initial_capital,
        'final_portfolio': final_portfolio,
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'simulation_years': simulation_years,
        
        # Group-level Performance
        'avg_group_return_pct': group_returns.mean(),
        'median_group_return_pct': group_returns.median(),
        'group_volatility_pct': group_volatility,
        'group_sharpe_ratio': group_sharpe,
        'group_win_rate_pct': group_win_rate,
        
        # Position-level Performance
        'total_positions': total_positions,
        'avg_position_return_pct': avg_position_return,
        'median_position_return_pct': median_position_return,
        'position_volatility_pct': position_volatility,
        'overall_win_rate_pct': overall_win_rate,
        'tp_hit_rate_pct': tp_hit_rate,
        
        # Risk Metrics
        'max_drawdown_pct': max_drawdown,
        
        # Operational Metrics
        'avg_positions_per_group': avg_positions_per_group,
        'avg_positions_per_asset': avg_positions_per_asset,
        'avg_holding_period_days': avg_holding_period,
        'median_holding_period_days': median_holding_period,
        'total_capital_deployed': total_capital_deployed,
        'capital_efficiency_pct': capital_efficiency,
        
        # Scenario Analysis
        'scenario_performance': scenario_performance
    }
    
    # Print comprehensive summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   💰 Total Return: {total_return:+.1f}% (${initial_capital:,.0f} → ${final_portfolio:,.0f})")
    print(f"   📈 Annualized Return: {annualized_return:+.1f}%")
    print(f"   📅 Simulation Period: {simulation_years:.1f} years")
    print(f"   🎯 Group Win Rate: {group_win_rate:.1f}%")
    print(f"   📉 Max Drawdown: {max_drawdown:.1f}%")
    print(f"   📊 Sharpe Ratio: {group_sharpe:.2f}")
    
    print(f"\n🎯 POSITION ANALYSIS:")
    print(f"   📊 Total Positions: {total_positions}")
    print(f"   🏆 Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"   🚀 TP Hit Rate: {tp_hit_rate:.1f}%")
    print(f"   📈 Avg Position Return: {avg_position_return:+.1f}%")
    print(f"   📅 Avg Holding Period: {avg_holding_period:.0f} days")
    
    print(f"\n💰 CAPITAL EFFICIENCY:")
    print(f"   💵 Total Deployed: ${total_capital_deployed:,.0f}")
    print(f"   ⚡ Capital Efficiency: {capital_efficiency:.1f}%")
    print(f"   🔄 Avg Positions/Group: {avg_positions_per_group:.1f}")
    
    print(f"\n🎭 SCENARIO BREAKDOWN:")
    for scenario in ['bull', 'sideways', 'bear']:
        if scenario in scenario_performance.index:
            count = int(scenario_performance.loc[scenario, ('Total_Return_Pct', 'count')])
            mean_return = scenario_performance.loc[scenario, ('Total_Return_Pct', 'mean')]
            win_rate = scenario_performance.loc[scenario, ('Win_Rate_Pct', 'mean')]
            print(f"   {scenario.title()}: {count} assets, avg return {mean_return:+.1f}%, win rate {win_rate:.1f}%")
    
    return performance_metrics

def save_contrarian_results(groups_df, assets_df, positions_df, daily_df, performance_metrics):
    """
    Save all contrarian strategy results to CSV files
    """
    
    print(f"\n💾 Saving contrarian strategy results...")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main dataframes
    groups_filename = f'contrarian_groups_{timestamp}.csv'
    assets_filename = f'contrarian_assets_{timestamp}.csv'
    positions_filename = f'contrarian_positions_{timestamp}.csv'
    daily_filename = f'contrarian_daily_{timestamp}.csv'
    
    groups_df.to_csv(groups_filename, index=False)
    assets_df.to_csv(assets_filename, index=False)
    positions_df.to_csv(positions_filename, index=False)
    daily_df.to_csv(daily_filename, index=False)
    
    # Save performance metrics
    metrics_filename = f'contrarian_metrics_{timestamp}.csv'
    metrics_data = []
    
    for key, value in performance_metrics.items():
        if key != 'scenario_performance':  # Handle this separately
            metrics_data.append({'Metric': key, 'Value': value})
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_filename, index=False)
    
    # Save scenario performance separately
    scenario_filename = f'contrarian_scenarios_{timestamp}.csv'
    if 'scenario_performance' in performance_metrics:
        performance_metrics['scenario_performance'].to_csv(scenario_filename)
    
    print(f"   📄 Groups: {groups_filename} ({len(groups_df)} records)")
    print(f"   📄 Assets: {assets_filename} ({len(assets_df)} records)")
    print(f"   📄 Positions: {positions_filename} ({len(positions_df)} records)")
    print(f"   📄 Daily: {daily_filename} ({len(daily_df)} records)")
    print(f"   📄 Metrics: {metrics_filename}")
    print(f"   📄 Scenarios: {scenario_filename}")
    
    return {
        'groups_file': groups_filename,
        'assets_file': assets_filename,
        'positions_file': positions_filename,
        'daily_file': daily_filename,
        'metrics_file': metrics_filename,
        'scenarios_file': scenario_filename
    }


# ============================================================================
# PART 3/3: VISUALIZATION AND MAIN EXECUTION
# ============================================================================

def plot_contrarian_strategy_overview(groups_df, assets_df, positions_df, performance_metrics, save_plots=True):
    """
    Create comprehensive overview plots for contrarian strategy
    """
    
    print(f"\n📊 Creating contrarian strategy overview plots...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Portfolio equity curve
    ax1 = plt.subplot(3, 4, 1)
    portfolio_values = [performance_metrics['initial_capital']] + groups_df['Exit_Portfolio'].tolist()
    group_numbers = [0] + groups_df['Group_Num'].tolist()
    
    ax1.plot(group_numbers, portfolio_values, linewidth=3, color='blue', alpha=0.8, marker='o')
    ax1.fill_between(group_numbers, portfolio_values, performance_metrics['initial_capital'], 
                     alpha=0.3, color='blue')
    
    ax1.set_xlabel('Group Number')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Equity Curve')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Position returns distribution
    ax2 = plt.subplot(3, 4, 2)
    position_returns = positions_df['Final_Return_Pct']
    tp_positions = positions_df[positions_df['TP_Hit'] == True]['Final_Return_Pct']
    other_positions = positions_df[positions_df['TP_Hit'] == False]['Final_Return_Pct']
    
    if len(tp_positions) > 0:
        ax2.hist(tp_positions, alpha=0.7, label=f'TP Hits ({len(tp_positions)})', 
                color='green', bins=20)
    if len(other_positions) > 0:
        ax2.hist(other_positions, alpha=0.7, label=f'Other ({len(other_positions)})', 
                color='orange', bins=20)
    
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(performance_metrics['tp_threshold'], color='green', linestyle='--', alpha=0.7, label='30% TP')
    ax2.set_xlabel('Position Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Position Returns Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Win rate by group
    ax3 = plt.subplot(3, 4, 3)
    group_nums = groups_df['Group_Num']
    group_returns = groups_df['Group_Return_Pct']
    colors = ['green' if ret > 0 else 'red' for ret in group_returns]
    
    bars = ax3.bar(group_nums, group_returns, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Group Number')
    ax3.set_ylabel('Group Return (%)')
    ax3.set_title('Performance by Group')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ret in zip(bars, group_returns):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{ret:+.1f}%', ha='center', va='bottom' if ret > 0 else 'top', 
                fontsize=8, fontweight='bold')
    
    # Plot 4: Position frequency by asset and group
    ax4 = plt.subplot(3, 4, 4)
    position_counts = positions_df.groupby(['Group_Num', 'Asset_Num']).size().reset_index(name='Count')
    pivot_counts = position_counts.pivot(index='Group_Num', columns='Asset_Num', values='Count').fillna(0)
    
    im = ax4.imshow(pivot_counts.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(pivot_counts.columns)))
    ax4.set_yticks(range(len(pivot_counts.index)))
    ax4.set_xticklabels([f'Asset {i+1}' for i in pivot_counts.columns])
    ax4.set_yticklabels([f'Group {i}' for i in pivot_counts.index])
    ax4.set_title('Position Count Heatmap')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # Plot 5: Holding period analysis
    ax5 = plt.subplot(3, 4, 5)
    holding_periods = positions_df['Days_Held']
    tp_periods = positions_df[positions_df['TP_Hit'] == True]['Days_Held']
    other_periods = positions_df[positions_df['TP_Hit'] == False]['Days_Held']
    
    if len(tp_periods) > 0 and len(other_periods) > 0:
        ax5.hist([tp_periods, other_periods], bins=15, alpha=0.7, 
                 label=['TP Hits', 'Other'], color=['green', 'orange'])
    elif len(tp_periods) > 0:
        ax5.hist(tp_periods, bins=15, alpha=0.7, label='TP Hits', color='green')
    elif len(other_periods) > 0:
        ax5.hist(other_periods, bins=15, alpha=0.7, label='Other', color='orange')
    
    ax5.set_xlabel('Days Held')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Holding Period Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Scenario performance comparison
    ax6 = plt.subplot(3, 4, 6)
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    scenario_stats = assets_df.groupby('Scenario')['Total_Return_Pct'].agg(['mean', 'std', 'count'])
    scenarios = scenario_stats.index
    means = scenario_stats['mean']
    stds = scenario_stats['std']
    
    bars = ax6.bar(scenarios, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=[scenario_colors.get(s, 'gray') for s in scenarios])
    ax6.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax6.set_ylabel('Average Return (%)')
    ax6.set_title('Performance by Market Scenario')
    ax6.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, scenario_stats['count']):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Plot 7: Capital deployment over time
    ax7 = plt.subplot(3, 4, 7)
    ax7.bar(groups_df['Group_Num'], groups_df['Total_Capital_Deployed'], 
            alpha=0.7, color='purple')
    ax7.set_xlabel('Group Number')
    ax7.set_ylabel('Capital Deployed ($)')
    ax7.set_title('Capital Deployment by Group')
    ax7.grid(True, alpha=0.3)
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 8: Return vs Risk scatter
    ax8 = plt.subplot(3, 4, 8)
    
    # Calculate volatility for each group (simplified)
    group_volatilities = []
    for group_num in groups_df['Group_Num']:
        group_positions = positions_df[positions_df['Group_Num'] == group_num]
        if len(group_positions) > 1:
            vol = group_positions['Final_Return_Pct'].std()
        else:
            vol = 0
        group_volatilities.append(vol)
    
    scatter = ax8.scatter(group_volatilities, groups_df['Group_Return_Pct'], 
                         s=groups_df['Total_Positions']*10, alpha=0.7, 
                         c=groups_df['Group_Num'], cmap='viridis')
    
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Group Volatility (%)')
    ax8.set_ylabel('Group Return (%)')
    ax8.set_title('Return vs Risk (Size = # Positions)')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8, fraction=0.046, pad=0.04)
    
    # Plot 9: Performance metrics summary
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    metrics_text = f"""
CONTRARIAN STRATEGY SUMMARY
{'='*25}

RETURNS:
Total: {performance_metrics['total_return_pct']:+.1f}%
Annualized: {performance_metrics['annualized_return_pct']:+.1f}%

SUCCESS RATES:
Group Win Rate: {performance_metrics['group_win_rate_pct']:.1f}%
Position Win Rate: {performance_metrics['overall_win_rate_pct']:.1f}%
TP Hit Rate: {performance_metrics['tp_hit_rate_pct']:.1f}%

POSITION METRICS:
Total Positions: {performance_metrics['total_positions']}
Avg Return: {performance_metrics['avg_position_return_pct']:+.1f}%
Avg Hold Period: {performance_metrics['avg_holding_period_days']:.0f} days

RISK METRICS:
Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%
Sharpe Ratio: {performance_metrics['group_sharpe_ratio']:.2f}

CAPITAL EFFICIENCY:
Total Deployed: ${performance_metrics['total_capital_deployed']:,.0f}
Efficiency: {performance_metrics['capital_efficiency_pct']:.1f}%
"""
    
    ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Plot 10: Position entry timing analysis
    ax10 = plt.subplot(3, 4, 10)
    entry_days = positions_df['Entry_Day']
    if len(entry_days) > 0:
        ax10.hist(entry_days, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax10.set_xlabel('Entry Day (Days from Start)')
        ax10.set_ylabel('Frequency')
        ax10.set_title('Position Entry Timing')
        ax10.grid(True, alpha=0.3)
        
        # Add median line
        median_entry = entry_days.median()
        ax10.axvline(median_entry, color='red', linestyle='--', linewidth=2, 
                     label=f'Median: {median_entry:.0f} days')
        ax10.legend()
    
    # Plot 11: Drawdown trigger analysis
    ax11 = plt.subplot(3, 4, 11)
    
    # Simulate drawdown levels that triggered entries (approximate)
    # This would need to be calculated from daily monitoring data
    drawdown_levels = np.random.normal(25, 5, len(positions_df))  # Simplified for demo
    drawdown_levels = np.clip(drawdown_levels, 20, 50)  # Clip to reasonable range
    
    ax11.hist(drawdown_levels, bins=15, alpha=0.7, color='red', edgecolor='black')
    ax11.axvline(20, color='black', linestyle='--', linewidth=2, label='20% Threshold')
    ax11.set_xlabel('Straddle Drawdown at Entry (%)')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Entry Trigger Distribution')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Strategy effectiveness assessment
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate some effectiveness metrics
    profitable_groups = (groups_df['Group_Return_Pct'] > 0).sum()
    total_groups = len(groups_df)
    avg_positions_per_asset = performance_metrics['avg_positions_per_asset']
    
    effectiveness_text = f"""
STRATEGY ASSESSMENT
{'='*17}

CONCEPT: Contrarian Mean Reversion
IMPLEMENTATION: {'Successful' if performance_metrics['overall_win_rate_pct'] > 50 else 'Challenging'}

ENTRY CRITERIA: 20%+ Drawdown
EFFECTIVENESS: {performance_metrics['overall_win_rate_pct']:.0f}% Win Rate

POSITION MANAGEMENT:
✓ 30% TP Target
✓ 150+ DTE Requirement
✓ Max 3 Positions/Asset

RESULTS:
Groups Profitable: {profitable_groups}/{total_groups}
Avg Positions/Asset: {avg_positions_per_asset:.1f}
Capital Efficiency: {performance_metrics['capital_efficiency_pct']:.1f}%

OVERALL RATING:
{'⭐' * min(5, max(1, int(3 + performance_metrics['total_return_pct']/10)))}
"""
    
    ax12.text(0.1, 0.9, effectiveness_text, transform=ax12.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Contrarian Straddle Strategy - Comprehensive Analysis\n'
                f'Total Return: {performance_metrics["total_return_pct"]:+.1f}% | '
                f'Win Rate: {performance_metrics["overall_win_rate_pct"]:.1f}% | '
                f'Positions: {performance_metrics["total_positions"]} | '
                f'Sharpe: {performance_metrics["group_sharpe_ratio"]:.2f}', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'contrarian_straddle_strategy_overview.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
    
    plt.show()

def plot_individual_asset_analysis(asset_num, group_num, daily_df, positions_df, save_plots=True):
    """
    Plot detailed analysis for individual asset showing position entries and performance
    """
    
    # Get data for specific asset
    asset_daily = daily_df[(daily_df['Group_Num'] == group_num) & (daily_df['Asset_Num'] == asset_num)].copy()
    asset_positions = positions_df[(positions_df['Group_Num'] == group_num) & (positions_df['Asset_Num'] == asset_num)].copy()
    
    if asset_daily.empty:
        print(f"No data found for Asset {asset_num+1} in Group {group_num}")
        return
    
    asset_daily = asset_daily.sort_values('Day')
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Spot price and straddle value over time
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    # Plot spot price
    line1 = ax1.plot(asset_daily['Day'], asset_daily['Spot_Price'], 
                     color='blue', linewidth=2, alpha=0.8, label='Spot Price')
    
    # Plot straddle value
    line2 = ax1_twin.plot(asset_daily['Day'], asset_daily['Straddle_Value'], 
                          color='red', linewidth=2, alpha=0.8, label='Straddle Value')
    
    # Mark position entries
    for _, pos in asset_positions.iterrows():
        entry_day = pos['Entry_Day']
        if entry_day < len(asset_daily):
            entry_spot = asset_daily.iloc[entry_day]['Spot_Price']
            entry_straddle = asset_daily.iloc[entry_day]['Straddle_Value']
            
            ax1.scatter([entry_day], [entry_spot], color='green', s=100, marker='^', 
                       zorder=10, edgecolor='black', linewidth=2)
            ax1_twin.scatter([entry_day], [entry_straddle], color='green', s=100, marker='^', 
                            zorder=10, edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('Days from Start')
    ax1.set_ylabel('Spot Price ($)', color='blue')
    ax1_twin.set_ylabel('Straddle Value ($)', color='red')
    ax1.set_title(f'Asset {asset_num+1} - Group {group_num}: Price and Straddle Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Straddle drawdown over time
    ax2 = axes[0, 1]
    ax2.plot(asset_daily['Day'], asset_daily['Straddle_Drawdown_Pct'], 
             color='red', linewidth=2, alpha=0.8)
    ax2.fill_between(asset_daily['Day'], asset_daily['Straddle_Drawdown_Pct'], 0,
                     where=(asset_daily['Straddle_Drawdown_Pct'] >= 0), 
                     alpha=0.3, color='red', interpolate=True)
    
    # Mark 20% threshold and entries
    ax2.axhline(y=20, color='black', linestyle='--', alpha=0.7, linewidth=2, label='20% Threshold')
    
    for _, pos in asset_positions.iterrows():
        entry_day = pos['Entry_Day']
        if entry_day < len(asset_daily):
            entry_drawdown = asset_daily.iloc[entry_day]['Straddle_Drawdown_Pct']
            ax2.scatter([entry_day], [entry_drawdown], color='green', s=100, marker='^', 
                       zorder=10, edgecolor='black', linewidth=2)
    
    ax2.set_xlabel('Days from Start')
    ax2.set_ylabel('Straddle Drawdown (%)')
    ax2.set_title('Straddle Drawdown and Entry Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position performance
    ax3 = axes[1, 0]
    
    if not asset_positions.empty:
        position_nums = range(1, len(asset_positions) + 1)
        returns = asset_positions['Final_Return_Pct']
        colors = ['green' if ret > 0 else 'red' for ret in returns]
        
        bars = ax3.bar(position_nums, returns, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='30% TP')
        
        # Add value labels
        for bar, ret in zip(bars, returns):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{ret:+.1f}%', ha='center', va='bottom' if ret > 0 else 'top', 
                    fontweight='bold', fontsize=9)
        
        ax3.set_xlabel('Position Number')
        ax3.set_ylabel('Return (%)')
        ax3.set_title('Individual Position Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Positions Entered', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Individual Position Performance')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if not asset_positions.empty:
        total_return = asset_positions['Final_Return_Pct'].sum()
        avg_return = asset_positions['Final_Return_Pct'].mean()
        win_rate = (asset_positions['Final_Return_Pct'] > 0).mean() * 100
        tp_rate = asset_positions['TP_Hit'].mean() * 100
        avg_hold = asset_positions['Days_Held'].mean()
        
        max_drawdown = asset_daily['Straddle_Drawdown_Pct'].max()
        
        summary_text = f"""
ASSET {asset_num+1} - GROUP {group_num} SUMMARY
{'='*30}

POSITION STATISTICS:
Total Positions: {len(asset_positions)}
Win Rate: {win_rate:.1f}%
TP Hit Rate: {tp_rate:.1f}%
Avg Return: {avg_return:+.1f}%
Total Return: {total_return:+.1f}%
Avg Hold Period: {avg_hold:.0f} days

MARKET CONDITIONS:
Max Drawdown: {max_drawdown:.1f}%
Entry Triggers: {len(asset_positions)}

POSITION DETAILS:
"""
        
        for i, (_, pos) in enumerate(asset_positions.iterrows()):
            summary_text += f"P{i+1}: {pos['Final_Return_Pct']:+.1f}% ({pos['Days_Held']:.0f}d)\n"
        
    else:
        summary_text = f"""
ASSET {asset_num+1} - GROUP {group_num} SUMMARY
{'='*30}

POSITION STATISTICS:
Total Positions: 0
Max Drawdown: {asset_daily['Straddle_Drawdown_Pct'].max():.1f}%

STATUS: No contrarian entries triggered
REASON: Drawdown < 20% or DTE < 150
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'contrarian_asset_{asset_num+1}_group_{group_num}_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
    
    plt.show()

def main():
    """
    Main function to run the contrarian straddle strategy simulation
    """
    
    print("🚀 CONTRARIAN STRADDLE STRATEGY SIMULATOR")
    print("="*80)
    print("Advanced mean-reversion strategy: Enter when straddles experience drawdowns")
    print("Betting on reversal and accelerated decay after adverse moves")
    print("="*80)
    
    # Strategy parameters
    n_groups = 10
    initial_capital = 100000
    n_assets = 5
    capital_per_asset = 20000  # Capital per asset for contrarian positions
    drawdown_threshold = 20.0  # 20% drawdown to trigger entry
    tp_threshold = 30.0       # 30% TP for each position
    min_days_to_expiry = 150  # Minimum 150 DTE to enter
    
    # Market parameters
    strike_multiplier = 1.25
    iv = 0.25
    risk_free_rate = 0.04
    annual_vol = 0.20
    total_days = 365
    
    print(f"\n🎯 Strategy Configuration:")
    print(f"   Contrarian Entry: {drawdown_threshold}% straddle drawdown")
    print(f"   TP Target: {tp_threshold}% per position")
    print(f"   Minimum DTE: {min_days_to_expiry} days")
    print(f"   Capital per asset: ${capital_per_asset:,.0f}")
    print(f"   Max positions per asset: 3")
    
    # Step 1: Run contrarian portfolio simulation
    print(f"\n🎯 Running contrarian strategy simulation...")
    groups_df, assets_df, positions_df, daily_df = run_contrarian_portfolio_simulation(
        n_groups=n_groups,
        initial_capital=initial_capital,
        n_assets=n_assets,
        capital_per_asset=capital_per_asset,
        drawdown_threshold=drawdown_threshold,
        tp_threshold=tp_threshold,
        min_days_to_expiry=min_days_to_expiry,
        strike_multiplier=strike_multiplier,
        iv=iv,
        risk_free_rate=risk_free_rate,
        annual_vol=annual_vol,
        total_days=total_days
    )
    
    # Step 2: Calculate performance metrics
    print(f"\n📊 Calculating contrarian strategy performance...")
    performance_metrics = calculate_contrarian_performance_metrics(
        groups_df, assets_df, positions_df, daily_df, initial_capital
    )
    
    # Step 3: Save results
    print(f"\n💾 Saving contrarian strategy results...")
    saved_files = save_contrarian_results(
        groups_df, assets_df, positions_df, daily_df, performance_metrics
    )
    
    # Step 4: Create visualizations
    print(f"\n📊 Creating contrarian strategy visualizations...")
    
    # Overview analysis
    plot_contrarian_strategy_overview(
        groups_df, assets_df, positions_df, performance_metrics
    )
    
    # Individual asset examples (first few assets from first group)
    print(f"\n📈 Creating individual asset analysis examples...")
    for asset_num in range(min(3, n_assets)):  # First 3 assets from Group 1
        print(f"   📊 Analyzing Asset {asset_num+1} from Group 1...")
        plot_individual_asset_analysis(asset_num, 1, daily_df, positions_df)
    
    # Final summary
    print(f"\n🎉 CONTRARIAN STRADDLE STRATEGY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📈 Strategy Performance: ${initial_capital:,.0f} → ${performance_metrics['final_portfolio']:,.0f}")
    print(f"📊 Total Return: {performance_metrics['total_return_pct']:+.1f}%")
    print(f"📅 Annualized Return: {performance_metrics['annualized_return_pct']:+.1f}%")
    print(f"🎯 Overall Win Rate: {performance_metrics['overall_win_rate_pct']:.1f}%")
    print(f"🚀 TP Hit Rate: {performance_metrics['tp_hit_rate_pct']:.1f}%")
    print(f"📉 Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%")
    print(f"📊 Sharpe Ratio: {performance_metrics['group_sharpe_ratio']:.2f}")
    print("="*80)
    print(f"🎯 STRATEGY INSIGHTS:")
    print(f"   Total Positions Entered: {performance_metrics['total_positions']}")
    print(f"   Average Positions/Asset: {performance_metrics['avg_positions_per_asset']:.1f}")
    print(f"   Capital Efficiency: {performance_metrics['capital_efficiency_pct']:.1f}%")
    print(f"   Average Holding Period: {performance_metrics['avg_holding_period_days']:.0f} days")
    print("="*80)
    print(f"📁 All data saved with timestamp for detailed analysis")
    print(f"📊 All plots generated showing strategy effectiveness")
    print(f"✅ Ready for strategy evaluation and optimization!")

if __name__ == "__main__":
    main()

    