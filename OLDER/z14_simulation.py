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
    
    Returns:
    - scenario: 'bull', 'sideways', or 'bear'  
    - drift: annual drift rate for the scenario
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Base probabilities: 70% bull, 20% sideways, 10% bear
    base_bull_prob = 0.70
    base_sideways_prob = 0.20
    base_bear_prob = 0.10
    
    # Adjust probabilities based on cumulative performance
    # If we're underperforming, increase bull probability slightly
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
        # Bull: 12% to 25% annual drift
        drift = np.random.uniform(0.12, 0.25)
    elif random_val < (bull_prob + sideways_prob):
        scenario = 'sideways'
        # Sideways: 2% to 12% annual drift  
        drift = np.random.uniform(0.02, 0.12)
    else:
        scenario = 'bear'
        # Bear: -10% to 5% annual drift
        drift = np.random.uniform(-0.10, 0.05)
    
    return scenario, drift

def generate_realistic_gbm_path(S0, annual_drift, days=365, annual_vol=0.20, dt=1/365, seed=None):
    """
    Generate pure Geometric Brownian Motion path with specified drift
    No target forcing - let it end wherever market forces take it
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

def simulate_single_trade(trade_num, entry_spot, portfolio_value, scenario, drift,
                         strike_multiplier=1.25, tp_threshold=20.0, days_to_expiry=365,
                         iv=0.25, risk_free_rate=0.04, annual_vol=0.20):
    """
    Simulate a single short straddle trade with realistic random walk
    """
    
    # Trade setup
    strike_price = entry_spot * strike_multiplier
    
    # Generate realistic price path - NO target forcing
    seed = 10000 + trade_num
    price_path = generate_realistic_gbm_path(
        S0=entry_spot,
        annual_drift=drift,
        days=days_to_expiry,
        annual_vol=annual_vol,
        seed=seed
    )
    
    # Calculate entry straddle value and position size
    entry_straddle_value = calculate_straddle_value(
        entry_spot, strike_price, 1.0, risk_free_rate, iv
    )
    
    # Position size: use 100% of portfolio (fractional contracts allowed)
    position_size = portfolio_value / entry_straddle_value
    
    # Track daily P&L and returns
    days_array = np.arange(days_to_expiry, -1, -1)
    daily_data = []
    
    exit_triggered = False
    exit_day = None
    exit_reason = "Expiry"
    
    for i, (day, spot) in enumerate(zip(days_array, price_path)):
        time_to_expiry = max(day / 365.0, 1/365)
        
        # Calculate current straddle value
        current_straddle_value = calculate_straddle_value(
            spot, strike_price, time_to_expiry, risk_free_rate, iv
        )
        
        # Short straddle P&L
        pnl_per_contract = entry_straddle_value - current_straddle_value
        total_pnl = pnl_per_contract * position_size
        current_portfolio_value = portfolio_value + total_pnl
        
        # Return calculation
        return_pct = (pnl_per_contract / entry_straddle_value) * 100
        
        # Check for 20% TP
        if return_pct >= tp_threshold and not exit_triggered:
            exit_triggered = True
            exit_day = i
            exit_reason = "20% TP Hit"
        
        # Store daily data
        daily_data.append({
            'Trade_Num': trade_num,
            'Day': i,
            'Days_to_Expiry': day,
            'Spot_Price': spot,
            'Straddle_Value': current_straddle_value,
            'Return_Pct': return_pct,
            'PnL_Total': total_pnl,
            'Portfolio_Value': current_portfolio_value,
            'Exit_Triggered': exit_triggered
        })
        
        # Exit if TP hit
        if exit_triggered and exit_day == i:
            break
    
    # Calculate actual return achieved by underlying
    actual_underlying_return = (price_path[-1] - entry_spot) / entry_spot
    
    # Final trade results
    final_data = daily_data[-1]
    trade_result = {
        'Trade_Num': trade_num,
        'Entry_Date': f"Trade_{trade_num}",
        'Entry_Spot': entry_spot,
        'Strike_Price': strike_price,
        'Scenario': scenario,
        'Annual_Drift': drift,
        'Actual_Underlying_Return': actual_underlying_return,
        'Final_Spot': final_data['Spot_Price'],
        'Entry_Portfolio': portfolio_value,
        'Exit_Portfolio': final_data['Portfolio_Value'],
        'Position_Size': position_size,
        'Entry_Straddle_Value': entry_straddle_value,
        'Exit_Straddle_Value': final_data['Straddle_Value'],
        'Trade_PnL': final_data['PnL_Total'],
        'Trade_Return_Pct': final_data['Return_Pct'],
        'Portfolio_Return_Pct': ((final_data['Portfolio_Value'] - portfolio_value) / portfolio_value) * 100,
        'Days_Held': len(daily_data),
        'Exit_Reason': exit_reason,
        'TP_Hit': exit_triggered
    }
    
    return trade_result, daily_data

def run_realistic_trading_simulation(n_trades=50, initial_capital=100000, target_annual_return=0.15):
    """
    Run realistic short straddle trading simulation with scenario-based paths
    """
    
    print(f"REALISTIC SHORT STRADDLE TRADING SIMULATION")
    print("="*80)
    print(f"Number of trades: {n_trades}")
    print(f"Initial capital: ${initial_capital:,.0f}")
    print(f"Target annual return: {target_annual_return*100:+.0f}%")
    print(f"Scenario probabilities: 70% Bull | 20% Sideways | 10% Bear")
    print(f"Take profit: 20%")
    print(f"Strike: 25% above spot (365 DTE)")
    print(f"Volatility: 20% (realistic daily movements)")
    print("="*80)
    
    # Initialize tracking
    portfolio_value = initial_capital
    entry_spot = 100.0  # Starting spot price
    cumulative_underlying_return = 0.0  # Track long-term performance
    
    all_trade_results = []
    all_daily_data = []
    
    # Run each trade
    for trade_num in range(1, n_trades + 1):
        print(f"\nTrade {trade_num}: Portfolio ${portfolio_value:,.0f}")
        
        # Calculate how far we are from target 15% annual return path
        target_cumulative_return = target_annual_return * (trade_num / n_trades)
        current_cumulative_return = cumulative_underlying_return / trade_num if trade_num > 0 else 0
        return_deficit = target_cumulative_return - current_cumulative_return
        
        # Select scenario based on performance and probability weighting
        scenario_seed = 5000 + trade_num
        scenario, drift = select_trade_scenario(trade_num, return_deficit, seed=scenario_seed)
        
        print(f"  Entry Spot: ${entry_spot:.2f}")
        print(f"  Scenario: {scenario.title()} (drift: {drift*100:+.1f}% annual)")
        print(f"  Cumulative Return Deficit: {return_deficit*100:+.1f}%")
        
        # Simulate the trade
        trade_result, daily_data = simulate_single_trade(
            trade_num=trade_num,
            entry_spot=entry_spot,
            portfolio_value=portfolio_value,
            scenario=scenario,
            drift=drift
        )
        
        # Update tracking
        portfolio_value = trade_result['Exit_Portfolio']
        entry_spot = trade_result['Final_Spot']  
        cumulative_underlying_return += trade_result['Actual_Underlying_Return']
        
        print(f"  Final Spot: ${trade_result['Final_Spot']:.2f} ({trade_result['Actual_Underlying_Return']*100:+.1f}%)")
        print(f"  Exit: {trade_result['Exit_Reason']}")
        print(f"  Days Held: {trade_result['Days_Held']}")
        print(f"  Trade Return: {trade_result['Trade_Return_Pct']:+.1f}%")
        print(f"  New Portfolio: ${portfolio_value:,.0f}")
        
        # Store results
        all_trade_results.append(trade_result)
        all_daily_data.extend(daily_data)
    
    # Convert to DataFrames
    trades_df = pd.DataFrame(all_trade_results)
    daily_df = pd.DataFrame(all_daily_data)
    
    return trades_df, daily_df

def analyze_realistic_simulation_results(trades_df, daily_df, initial_capital):
    """
    Enhanced analysis for realistic simulation results
    """
    
    print(f"\n" + "="*80)
    print("REALISTIC SIMULATION RESULTS ANALYSIS")
    print("="*80)
    
    final_portfolio = trades_df['Exit_Portfolio'].iloc[-1]
    total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
    
    # Overall performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Initial Capital: ${initial_capital:,.0f}")
    print(f"  Final Portfolio: ${final_portfolio:,.0f}")
    print(f"  Total Return: {total_return:+.1f}%")
    print(f"  Annualized Return: {total_return / len(trades_df) * 50:+.1f}%")
    print(f"  Total Trades: {len(trades_df)}")
    
    # Scenario analysis
    print(f"\nSCENARIO BREAKDOWN:")
    scenario_stats = trades_df.groupby('Scenario').agg({
        'Trade_Return_Pct': ['count', 'mean', 'std'],
        'TP_Hit': 'sum',
        'Days_Held': 'mean',
        'Actual_Underlying_Return': 'mean'
    }).round(2)
    
    for scenario in ['bull', 'sideways', 'bear']:
        if scenario in scenario_stats.index:
            stats = scenario_stats.loc[scenario]
            count = int(stats[('Trade_Return_Pct', 'count')])
            tp_hits = int(stats[('TP_Hit', 'sum')])
            
            print(f"  {scenario.title()}: {count} trades ({count/len(trades_df)*100:.0f}%)")
            print(f"    TP Hit Rate: {tp_hits}/{count} ({tp_hits/count*100:.0f}%)")
            print(f"    Avg Trade Return: {stats[('Trade_Return_Pct', 'mean')]:+.1f}%")
            print(f"    Avg Underlying Move: {stats[('Actual_Underlying_Return', 'mean')]*100:+.1f}%")
            print(f"    Avg Days Held: {stats[('Days_Held', 'mean')]:.0f}")
    
    # Trade statistics
    winning_trades = (trades_df['Trade_Return_Pct'] > 0).sum()
    tp_hits = trades_df['TP_Hit'].sum()
    
    print(f"\nTRADE STATISTICS:")
    print(f"  Winning Trades: {winning_trades}/{len(trades_df)} ({winning_trades/len(trades_df)*100:.1f}%)")
    print(f"  TP Hits: {tp_hits}/{len(trades_df)} ({tp_hits/len(trades_df)*100:.1f}%)")
    print(f"  Average Trade Return: {trades_df['Trade_Return_Pct'].mean():+.1f}%")
    print(f"  Best Trade: {trades_df['Trade_Return_Pct'].max():+.1f}%")
    print(f"  Worst Trade: {trades_df['Trade_Return_Pct'].min():+.1f}%")
    print(f"  Average Days Held: {trades_df['Days_Held'].mean():.0f}")
    
    # Underlying performance
    avg_underlying_return = trades_df['Actual_Underlying_Return'].mean()
    print(f"\nUNDERLYING PERFORMANCE:")
    print(f"  Average Underlying Return: {avg_underlying_return*100:+.1f}%")
    print(f"  Annualized Underlying: {avg_underlying_return * 50:+.1f}%")
    print(f"  Target was: 15.0% annually")
    
    # Portfolio metrics
    portfolio_returns = trades_df['Portfolio_Return_Pct']
    
    # Calculate max drawdown
    peak_portfolio = trades_df['Exit_Portfolio'].expanding().max()
    drawdown = (trades_df['Exit_Portfolio'] - peak_portfolio) / peak_portfolio * 100
    max_drawdown = drawdown.min()
    
    print(f"\nPORTFOLIO METRICS:")
    print(f"  Max Drawdown: {max_drawdown:.1f}%")
    print(f"  Return Volatility: {portfolio_returns.std():.1f}%")
    if portfolio_returns.std() > 0:
        print(f"  Sharpe Ratio: {portfolio_returns.mean()/portfolio_returns.std():.2f}")

def plot_realistic_simulation_results(trades_df, daily_df, initial_capital):
    """
    Enhanced plotting for realistic simulation results
    """
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: Portfolio equity curve
    ax1 = plt.subplot(4, 2, 1)
    portfolio_values = [initial_capital] + trades_df['Exit_Portfolio'].tolist()
    trade_numbers = list(range(len(portfolio_values)))
    
    ax1.plot(trade_numbers, portfolio_values, linewidth=3, color='blue', marker='o', markersize=4)
    ax1.fill_between(trade_numbers, portfolio_values, alpha=0.3, color='blue')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Equity Curve - Realistic Simulation')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Trade returns by scenario
    ax2 = plt.subplot(4, 2, 2)
    scenarios = trades_df['Scenario'].unique()
    colors = {'bull': 'green', 'sideways': 'orange', 'bear': 'red'}
    
    for scenario in scenarios:
        scenario_data = trades_df[trades_df['Scenario'] == scenario]['Trade_Return_Pct']
        ax2.hist(scenario_data, alpha=0.6, label=f'{scenario.title()} ({len(scenario_data)} trades)', 
                color=colors.get(scenario, 'gray'), bins=15)
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Breakeven')
    ax2.axvline(x=20, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='20% TP')
    ax2.set_xlabel('Trade Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Trade Returns by Scenario')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scenario distribution pie chart
    ax3 = plt.subplot(4, 2, 3)
    scenario_counts = trades_df['Scenario'].value_counts()
    colors_pie = [colors.get(scenario, 'gray') for scenario in scenario_counts.index]
    
    wedges, texts, autotexts = ax3.pie(scenario_counts.values, labels=scenario_counts.index, 
                                      autopct='%1.0f%%', colors=colors_pie, startangle=90)
    ax3.set_title('Actual Scenario Distribution')
    
    # Plot 4: Underlying returns vs straddle returns
    ax4 = plt.subplot(4, 2, 4)
    colors_scatter = [colors.get(scenario, 'gray') for scenario in trades_df['Scenario']]
    ax4.scatter(trades_df['Actual_Underlying_Return']*100, trades_df['Trade_Return_Pct'], 
               alpha=0.7, s=50, c=colors_scatter)
    ax4.set_xlabel('Underlying Asset Return (%)')
    ax4.set_ylabel('Straddle Trade Return (%)')
    ax4.set_title('Underlying vs Straddle Performance')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative returns
    ax5 = plt.subplot(4, 2, 5)
    portfolio_returns = trades_df['Portfolio_Return_Pct']
    cumulative_returns = (1 + portfolio_returns/100).cumprod() - 1
    
    ax5.plot(range(1, len(cumulative_returns)+1), cumulative_returns*100, 
             linewidth=3, color='purple', marker='o', markersize=3)
    ax5.set_xlabel('Trade Number')
    ax5.set_ylabel('Cumulative Return (%)')
    ax5.set_title('Cumulative Returns by Trade')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 6: Days held distribution by outcome
    ax6 = plt.subplot(4, 2, 6)
    tp_trades = trades_df[trades_df['TP_Hit'] == True]['Days_Held']
    expiry_trades = trades_df[trades_df['TP_Hit'] == False]['Days_Held']
    
    if len(tp_trades) > 0 and len(expiry_trades) > 0:
        ax6.hist([tp_trades, expiry_trades], bins=15, alpha=0.7, 
                 label=[f'TP Hits ({len(tp_trades)})', f'Held to Expiry ({len(expiry_trades)})'], 
                 color=['green', 'orange'])
    elif len(tp_trades) > 0:
        ax6.hist(tp_trades, bins=15, alpha=0.7, label=f'TP Hits ({len(tp_trades)})', color='green')
    elif len(expiry_trades) > 0:
        ax6.hist(expiry_trades, bins=15, alpha=0.7, label=f'Held to Expiry ({len(expiry_trades)})', color='orange')
    
    ax6.set_xlabel('Days Held')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Holding Period by Outcome')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Win/Loss by scenario
    ax7 = plt.subplot(4, 2, 7)
    scenario_performance = trades_df.groupby('Scenario')['Trade_Return_Pct'].apply(lambda x: (x > 0).sum() / len(x) * 100)
    
    bars = ax7.bar(scenario_performance.index, scenario_performance.values, 
                   color=[colors.get(scenario, 'gray') for scenario in scenario_performance.index],
                   alpha=0.7)
    ax7.set_xlabel('Scenario')
    ax7.set_ylabel('Win Rate (%)')
    ax7.set_title('Win Rate by Scenario')
    ax7.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, scenario_performance.values):
        ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 8: Drift vs Performance
    ax8 = plt.subplot(4, 2, 8)
    colors_drift = [colors.get(scenario, 'gray') for scenario in trades_df['Scenario']]
    ax8.scatter(trades_df['Annual_Drift']*100, trades_df['Trade_Return_Pct'], 
               alpha=0.7, s=50, c=colors_drift)
    ax8.set_xlabel('Annual Drift (%)')
    ax8.set_ylabel('Trade Return (%)')
    ax8.set_title('Drift Rate vs Trade Performance')
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'realistic_trading_simulation_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Realistic simulation plots saved: {filename}")
    
    plt.show()

def main():
    """Main function to run the realistic simulation"""
    
    # Run realistic simulation
    trades_df, daily_df = run_realistic_trading_simulation(
        n_trades=50,
        initial_capital=100000,
        target_annual_return=0.15
    )
    
    # Analyze results
    analyze_realistic_simulation_results(trades_df, daily_df, 100000)
    
    # Create plots
    plot_realistic_simulation_results(trades_df, daily_df, 100000)
    
    # Save results to CSV
    trades_df.to_csv('realistic_trading_simulation_trades.csv', index=False)
    daily_df.to_csv('realistic_trading_simulation_daily_data.csv', index=False)
    
    print(f"\n💾 Realistic trade results saved to: realistic_trading_simulation_trades.csv")
    print(f"💾 Daily data saved to: realistic_trading_simulation_daily_data.csv")
    print(f"🎉 Realistic simulation complete!")

if __name__ == "__main__":
    main()