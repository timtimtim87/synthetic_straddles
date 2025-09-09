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

def generate_weighted_random_outcome(target_mean=0.15, outcome_range=(-0.10, 0.35), seed=None):
    """
    Generate a random outcome weighted around target_mean (+15%)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use beta distribution to create weighting around target
    # Map beta distribution to our desired range
    min_val, max_val = outcome_range
    
    # Beta parameters to create distribution peaked around target
    alpha = 2.0
    beta = 2.0
    
    # Generate beta random variable (0 to 1)
    beta_sample = np.random.beta(alpha, beta)
    
    # Adjust to weight around our target
    target_position = (target_mean - min_val) / (max_val - min_val)
    
    # Blend the beta sample with target position
    weight = 0.6  # How much to weight toward target
    adjusted_sample = weight * target_position + (1 - weight) * beta_sample
    
    # Map to our range
    outcome = min_val + adjusted_sample * (max_val - min_val)
    
    return outcome

def generate_price_path_to_target(S0, target_return, days=365, annual_vol=0.20, seed=None):
    """
    Generate realistic price path that reaches target return at expiry
    """
    if seed is not None:
        np.random.seed(seed)
    
    target_price = S0 * (1 + target_return)
    T = days / 365.0
    dt = 1/365
    
    # Calculate drift to hit target
    required_log_return = np.log(target_price / S0)
    mu = (required_log_return + 0.5 * annual_vol**2 * T) / T
    
    # Generate GBM path
    n_steps = days
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)
    
    for i in range(n_steps):
        log_prices[i + 1] = log_prices[i] + (mu - 0.5 * annual_vol**2) * dt + annual_vol * dW[i]
    
    prices = np.exp(log_prices)
    prices[-1] = target_price  # Ensure exact target
    
    return prices

def simulate_single_trade(trade_num, entry_spot, portfolio_value, target_return, 
                         strike_multiplier=1.25, tp_threshold=20.0, days_to_expiry=365,
                         iv=0.25, risk_free_rate=0.04, annual_vol=0.20):
    """
    Simulate a single short straddle trade
    """
    
    # Trade setup
    strike_price = entry_spot * strike_multiplier
    
    # Generate price path for this trade
    seed = 10000 + trade_num  # Unique seed for each trade
    price_path = generate_price_path_to_target(
        S0=entry_spot,
        target_return=target_return,
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
    
    # Final trade results
    final_data = daily_data[-1]
    trade_result = {
        'Trade_Num': trade_num,
        'Entry_Date': f"Trade_{trade_num}",
        'Entry_Spot': entry_spot,
        'Strike_Price': strike_price,
        'Target_Return': target_return,
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

def run_trading_simulation(n_trades=50, initial_capital=100000, target_mean=0.15):
    """
    Run the complete 50-trade simulation
    """
    
    print(f"ADVANCED SHORT STRADDLE TRADING SIMULATION")
    print("="*80)
    print(f"Number of trades: {n_trades}")
    print(f"Initial capital: ${initial_capital:,.0f}")
    print(f"Target outcome mean: {target_mean*100:+.0f}%")
    print(f"Outcome range: -10% to +35%")
    print(f"Take profit: 20%")
    print(f"Strike: 25% above spot (365 DTE)")
    print("="*80)
    
    # Initialize tracking
    portfolio_value = initial_capital
    entry_spot = 100.0  # Starting spot price
    
    all_trade_results = []
    all_daily_data = []
    
    # Run each trade
    for trade_num in range(1, n_trades + 1):
        print(f"\nTrade {trade_num}: Portfolio ${portfolio_value:,.0f}")
        
        # Generate random outcome for this trade
        outcome_seed = 5000 + trade_num
        target_return = generate_weighted_random_outcome(
            target_mean=target_mean,
            seed=outcome_seed
        )
        
        print(f"  Entry Spot: ${entry_spot:.2f}")
        print(f"  Target Outcome: {target_return*100:+.1f}%")
        
        # Simulate the trade
        trade_result, daily_data = simulate_single_trade(
            trade_num=trade_num,
            entry_spot=entry_spot,
            portfolio_value=portfolio_value,
            target_return=target_return
        )
        
        # Update portfolio and spot for next trade
        portfolio_value = trade_result['Exit_Portfolio']
        entry_spot = trade_result['Final_Spot']  # Next trade starts where this one ended
        
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

def analyze_simulation_results(trades_df, daily_df, initial_capital):
    """
    Analyze and summarize simulation results
    """
    
    print(f"\n" + "="*80)
    print("SIMULATION RESULTS ANALYSIS")
    print("="*80)
    
    final_portfolio = trades_df['Exit_Portfolio'].iloc[-1]
    total_return = ((final_portfolio - initial_capital) / initial_capital) * 100
    
    # Overall performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Initial Capital: ${initial_capital:,.0f}")
    print(f"  Final Portfolio: ${final_portfolio:,.0f}")
    print(f"  Total Return: {total_return:+.1f}%")
    print(f"  Total Trades: {len(trades_df)}")
    
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
    
    # Portfolio metrics
    portfolio_returns = trades_df['Portfolio_Return_Pct']
    cumulative_returns = (1 + portfolio_returns/100).cumprod() - 1
    
    # Calculate max drawdown
    peak_portfolio = trades_df['Exit_Portfolio'].expanding().max()
    drawdown = (trades_df['Exit_Portfolio'] - peak_portfolio) / peak_portfolio * 100
    max_drawdown = drawdown.min()
    
    print(f"\nPORTFOLIO METRICS:")
    print(f"  Max Drawdown: {max_drawdown:.1f}%")
    print(f"  Volatility: {portfolio_returns.std():.1f}%")
    print(f"  Sharpe Ratio: {portfolio_returns.mean()/portfolio_returns.std():.2f}")
    
    # Outcome analysis
    print(f"\nOUTCOME ANALYSIS:")
    print(f"  Average Target Outcome: {trades_df['Target_Return'].mean()*100:+.1f}%")
    print(f"  Outcome Range: {trades_df['Target_Return'].min()*100:+.1f}% to {trades_df['Target_Return'].max()*100:+.1f}%")
    print(f"  Std Dev of Outcomes: {trades_df['Target_Return'].std()*100:.1f}%")

def plot_simulation_results(trades_df, daily_df, initial_capital):
    """
    Create comprehensive plots of simulation results
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
    ax1.set_title('Portfolio Equity Curve')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Trade returns distribution
    ax2 = plt.subplot(4, 2, 2)
    ax2.hist(trades_df['Trade_Return_Pct'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax2.axvline(x=20, color='blue', linestyle='--', linewidth=2, label='20% TP')
    ax2.set_xlabel('Trade Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Trade Returns Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Days held distribution
    ax3 = plt.subplot(4, 2, 3)
    tp_trades = trades_df[trades_df['TP_Hit'] == True]['Days_Held']
    expiry_trades = trades_df[trades_df['TP_Hit'] == False]['Days_Held']
    
    ax3.hist([tp_trades, expiry_trades], bins=20, alpha=0.7, 
             label=['TP Hits', 'Held to Expiry'], color=['green', 'orange'])
    ax3.set_xlabel('Days Held')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Holding Period Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Target outcomes vs actual returns
    ax4 = plt.subplot(4, 2, 4)
    ax4.scatter(trades_df['Target_Return']*100, trades_df['Trade_Return_Pct'], 
               alpha=0.7, s=50, c=trades_df['TP_Hit'], cmap='RdYlGn')
    ax4.set_xlabel('Target Price Outcome (%)')
    ax4.set_ylabel('Actual Trade Return (%)')
    ax4.set_title('Target Outcome vs Trade Return')
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    min_val = min(trades_df['Target_Return'].min()*100, trades_df['Trade_Return_Pct'].min())
    max_val = max(trades_df['Target_Return'].max()*100, trades_df['Trade_Return_Pct'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    ax4.legend()
    
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
    
    # Plot 6: Drawdown analysis
    ax6 = plt.subplot(4, 2, 6)
    peak_portfolio = trades_df['Exit_Portfolio'].expanding().max()
    drawdown = (trades_df['Exit_Portfolio'] - peak_portfolio) / peak_portfolio * 100
    
    ax6.fill_between(range(1, len(drawdown)+1), drawdown, 0, alpha=0.7, color='red')
    ax6.plot(range(1, len(drawdown)+1), drawdown, linewidth=2, color='darkred')
    ax6.set_xlabel('Trade Number')
    ax6.set_ylabel('Drawdown (%)')
    ax6.set_title('Portfolio Drawdown')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Win/Loss streak analysis
    ax7 = plt.subplot(4, 2, 7)
    wins = (trades_df['Trade_Return_Pct'] > 0).astype(int)
    trade_nums = range(1, len(wins)+1)
    colors = ['red' if w == 0 else 'green' for w in wins]
    
    ax7.bar(trade_nums, [1]*len(wins), color=colors, alpha=0.7, width=0.8)
    ax7.set_xlabel('Trade Number')
    ax7.set_ylabel('Win (Green) / Loss (Red)')
    ax7.set_title('Win/Loss Pattern')
    ax7.set_ylim(0, 1.2)
    
    # Plot 8: Rolling performance metrics
    ax8 = plt.subplot(4, 2, 8)
    window = 10
    if len(trades_df) >= window:
        rolling_winrate = wins.rolling(window=window).mean() * 100
        rolling_avg_return = trades_df['Trade_Return_Pct'].rolling(window=window).mean()
        
        ax8_twin = ax8.twinx()
        
        line1 = ax8.plot(range(window, len(rolling_winrate)+1), rolling_winrate[window-1:], 
                        linewidth=2, color='blue', label=f'{window}-Trade Win Rate')
        line2 = ax8_twin.plot(range(window, len(rolling_avg_return)+1), rolling_avg_return[window-1:], 
                             linewidth=2, color='orange', label=f'{window}-Trade Avg Return')
        
        ax8.set_xlabel('Trade Number')
        ax8.set_ylabel('Rolling Win Rate (%)', color='blue')
        ax8_twin.set_ylabel('Rolling Avg Return (%)', color='orange')
        ax8.set_title(f'Rolling {window}-Trade Performance')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax8.legend(lines, labels, loc='upper left')
        
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, f'Need at least {window} trades\nfor rolling analysis', 
                ha='center', va='center', transform=ax8.transAxes)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'trading_simulation_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Simulation plots saved: {filename}")
    
    plt.show()

def plot_individual_trades(daily_df, n_trades_to_show=10):
    """
    Plot individual trade return curves for first n trades
    """
    
    plt.style.use('default')
    
    # Calculate grid size
    n_cols = 5
    n_rows = (n_trades_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle(f'Individual Trade Return Curves (First {n_trades_to_show} Trades)', 
                 fontsize=16, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i in range(n_trades_to_show):
        trade_num = i + 1
        trade_data = daily_df[daily_df['Trade_Num'] == trade_num]
        
        if not trade_data.empty:
            ax = axes_flat[i]
            
            # Plot return curve
            ax.plot(trade_data['Day'], trade_data['Return_Pct'], 
                   linewidth=2, color='blue')
            
            # Add 20% TP line
            ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=1, label='20% TP')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Breakeven')
            
            # Fill profit/loss areas
            ax.fill_between(trade_data['Day'], trade_data['Return_Pct'], 0,
                           where=(trade_data['Return_Pct'] >= 0), alpha=0.3, color='green')
            ax.fill_between(trade_data['Day'], trade_data['Return_Pct'], 0,
                           where=(trade_data['Return_Pct'] < 0), alpha=0.3, color='red')
            
            # Formatting
            final_return = trade_data['Return_Pct'].iloc[-1]
            exit_triggered = trade_data['Exit_Triggered'].iloc[-1]
            exit_reason = "TP Hit" if exit_triggered else "Expiry"
            
            ax.set_title(f'Trade {trade_num}\nFinal: {final_return:+.1f}% ({exit_reason})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Days')
            ax.set_ylabel('Return (%)')
            ax.grid(True, alpha=0.3)
            
            # Highlight final point
            ax.scatter([trade_data['Day'].iloc[-1]], [final_return], 
                      color='red' if final_return < 0 else 'green', s=50, zorder=5)
    
    # Hide unused subplots
    for j in range(n_trades_to_show, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    filename = f'individual_trade_curves_{n_trades_to_show}trades.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Individual trade plots saved: {filename}")
    
    plt.show()

def main():
    """Main function to run the complete simulation"""
    
    # Run simulation
    trades_df, daily_df = run_trading_simulation(
        n_trades=50,
        initial_capital=100000,
        target_mean=0.15
    )
    
    # Analyze results
    analyze_simulation_results(trades_df, daily_df, 100000)
    
    # Create plots
    plot_simulation_results(trades_df, daily_df, 100000)
    plot_individual_trades(daily_df, n_trades_to_show=10)
    
    # Save results to CSV
    trades_df.to_csv('trading_simulation_trades.csv', index=False)
    daily_df.to_csv('trading_simulation_daily_data.csv', index=False)
    
    print(f"\n💾 Trade results saved to: trading_simulation_trades.csv")
    print(f"💾 Daily data saved to: trading_simulation_daily_data.csv")
    print(f"🎉 Simulation complete!")

if __name__ == "__main__":
    main()