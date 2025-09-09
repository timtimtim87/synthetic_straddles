import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_simulation_data():
    """
    Load the simulation results from z14 output files
    """
    try:
        trades_df = pd.read_csv('realistic_trading_simulation_trades.csv')
        daily_df = pd.read_csv('realistic_trading_simulation_daily_data.csv')
        
        print(f"Loaded {len(trades_df)} trades and {len(daily_df)} daily records")
        print(f"Date range: Trade {trades_df['Trade_Num'].min()} to {trades_df['Trade_Num'].max()}")
        
        return trades_df, daily_df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run z14_realistic_trading_simulation.py first!")
        return None, None

def create_continuous_daily_timeline(daily_df, trades_df):
    """
    Create a continuous daily timeline across all trades for portfolio analysis
    """
    
    print("Creating continuous daily timeline...")
    
    # Create a continuous timeline
    continuous_data = []
    global_day = 0
    
    # Process each trade in order
    for trade_num in sorted(trades_df['Trade_Num'].unique()):
        trade_daily = daily_df[daily_df['Trade_Num'] == trade_num].copy()
        trade_daily = trade_daily.sort_values('Day')
        
        # Add global day counter
        for _, row in trade_daily.iterrows():
            continuous_data.append({
                'Global_Day': global_day,
                'Trade_Num': trade_num,
                'Trade_Day': row['Day'],
                'Days_to_Expiry': row['Days_to_Expiry'],
                'Spot_Price': row['Spot_Price'],
                'Straddle_Value': row['Straddle_Value'],
                'Return_Pct': row['Return_Pct'],
                'PnL_Total': row['PnL_Total'],
                'Portfolio_Value': row['Portfolio_Value'],
                'Exit_Triggered': row['Exit_Triggered']
            })
            global_day += 1
    
    continuous_df = pd.DataFrame(continuous_data)
    
    # Calculate drawdown
    continuous_df['Peak_Portfolio'] = continuous_df['Portfolio_Value'].expanding().max()
    continuous_df['Drawdown'] = (continuous_df['Portfolio_Value'] - continuous_df['Peak_Portfolio']) / continuous_df['Peak_Portfolio'] * 100
    continuous_df['Drawdown_Dollar'] = continuous_df['Portfolio_Value'] - continuous_df['Peak_Portfolio']
    
    # Calculate daily returns
    continuous_df['Daily_Return'] = continuous_df['Portfolio_Value'].pct_change() * 100
    continuous_df['Daily_Return'] = continuous_df['Daily_Return'].fillna(0)
    
    print(f"Created continuous timeline with {len(continuous_df)} days")
    
    return continuous_df

def plot_all_individual_trade_timeseries(daily_df, trades_df):
    """
    Create time series plots for every single trade (10 trades per image)
    """
    
    plt.style.use('default')
    
    n_trades = len(trades_df)
    trades_per_image = 10
    n_images = (n_trades + trades_per_image - 1) // trades_per_image
    
    print(f"\n📊 Creating {n_images} time series images for all {n_trades} trades...")
    
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    for image_num in range(n_images):
        start_trade = image_num * trades_per_image + 1
        end_trade = min((image_num + 1) * trades_per_image, n_trades)
        
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        fig.suptitle(f'Trade Return Time Series - Trades {start_trade} to {end_trade}', 
                     fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i in range(trades_per_image):
            trade_num = start_trade + i
            ax = axes_flat[i]
            
            if trade_num <= n_trades:
                # Get trade data
                trade_data = daily_df[daily_df['Trade_Num'] == trade_num].copy()
                trade_info = trades_df[trades_df['Trade_Num'] == trade_num].iloc[0]
                
                if not trade_data.empty:
                    trade_data = trade_data.sort_values('Day')
                    
                    # Calculate days from start
                    days_from_start = trade_data['Day'].max() - trade_data['Day']
                    
                    # Get colors and info
                    scenario = trade_info['Scenario']
                    line_color = scenario_colors.get(scenario, '#1f77b4')
                    
                    # Plot the return curve
                    ax.plot(days_from_start, trade_data['Return_Pct'], 
                           linewidth=2, color=line_color, alpha=0.8)
                    
                    # Fill profit/loss areas
                    ax.fill_between(days_from_start, trade_data['Return_Pct'], 0,
                                   where=(trade_data['Return_Pct'] >= 0), 
                                   alpha=0.3, color='green', interpolate=True)
                    ax.fill_between(days_from_start, trade_data['Return_Pct'], 0,
                                   where=(trade_data['Return_Pct'] < 0), 
                                   alpha=0.3, color='red', interpolate=True)
                    
                    # Reference lines
                    ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=1)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
                    
                    # Trade details
                    final_return = trade_data['Return_Pct'].iloc[-1]
                    exit_reason = "TP" if trade_info['TP_Hit'] else "Exp"
                    underlying_return = trade_info['Actual_Underlying_Return'] * 100
                    days_held = trade_info['Days_Held']
                    
                    # Title and labels
                    ax.set_title(f'Trade {trade_num} - {scenario.title()} Scenario\n'
                               f'Final: {final_return:+.1f}% ({exit_reason}) | '
                               f'Underlying: {underlying_return:+.0f}% | {days_held}d', 
                               fontsize=10, fontweight='bold')
                    
                    ax.set_xlabel('Days from Start', fontsize=9)
                    ax.set_ylabel('Straddle Return (%)', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='both', labelsize=8)
                    
                    # Highlight final point
                    final_color = '#8B0000' if final_return < 0 else '#006400'
                    ax.scatter([days_from_start.iloc[-1]], [final_return], 
                              color=final_color, s=50, zorder=5, 
                              edgecolor='white', linewidth=1.5)
                    
                    # Set reasonable y-limits
                    y_min = min(-50, trade_data['Return_Pct'].min() * 1.1)
                    y_max = max(30, trade_data['Return_Pct'].max() * 1.1)
                    ax.set_ylim(y_min, y_max)
                    
            else:
                ax.set_visible(False)
        
        plt.tight_layout()
        
        filename = f'trade_timeseries_batch_{image_num+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Saved: {filename}")
        
        plt.show()

def plot_continuous_portfolio_analysis(continuous_df, trades_df):
    """
    Create comprehensive continuous portfolio analysis
    """
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: Daily Portfolio Value
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(continuous_df['Global_Day'], continuous_df['Portfolio_Value'], 
             linewidth=2, color='#1f77b4', alpha=0.8)
    ax1.fill_between(continuous_df['Global_Day'], continuous_df['Portfolio_Value'], 
                     continuous_df['Portfolio_Value'].min(), alpha=0.3, color='#1f77b4')
    
    # Add trade completion markers
    trade_end_days = []
    trade_end_values = []
    for trade_num in sorted(trades_df['Trade_Num'].unique()):
        trade_end_data = continuous_df[continuous_df['Trade_Num'] == trade_num].iloc[-1]
        trade_end_days.append(trade_end_data['Global_Day'])
        trade_end_values.append(trade_end_data['Portfolio_Value'])
    
    ax1.scatter(trade_end_days, trade_end_values, color='red', s=30, alpha=0.7, zorder=5)
    
    ax1.set_xlabel('Days Since Start')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Daily Portfolio Value Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Drawdown Analysis
    ax2 = plt.subplot(4, 2, 2)
    ax2.fill_between(continuous_df['Global_Day'], continuous_df['Drawdown'], 0, 
                     color='red', alpha=0.6)
    ax2.plot(continuous_df['Global_Day'], continuous_df['Drawdown'], 
             linewidth=1, color='darkred')
    
    # Mark maximum drawdown
    max_dd_idx = continuous_df['Drawdown'].idxmin()
    max_dd_day = continuous_df.loc[max_dd_idx, 'Global_Day']
    max_dd_value = continuous_df.loc[max_dd_idx, 'Drawdown']
    
    ax2.annotate(f'Max DD: {max_dd_value:.1f}%', 
                xy=(max_dd_day, max_dd_value), xytext=(10, 10),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    ax2.set_xlabel('Days Since Start')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Portfolio Drawdown Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Daily Returns Distribution
    ax3 = plt.subplot(4, 2, 3)
    daily_returns = continuous_df['Daily_Return'].dropna()
    
    ax3.hist(daily_returns, bins=50, alpha=0.7, color='#2E8B57', edgecolor='black')
    ax3.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {daily_returns.mean():.2f}%')
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Daily Returns Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling Statistics
    ax4 = plt.subplot(4, 2, 4)
    window = 30  # 30-day rolling window
    
    if len(continuous_df) >= window:
        rolling_mean = continuous_df['Daily_Return'].rolling(window=window).mean()
        rolling_std = continuous_df['Daily_Return'].rolling(window=window).std()
        
        ax4.plot(continuous_df['Global_Day'], rolling_mean, 
                label=f'{window}-Day Mean Return', linewidth=2, color='blue')
        ax4.plot(continuous_df['Global_Day'], rolling_std, 
                label=f'{window}-Day Volatility', linewidth=2, color='orange')
        
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Days Since Start')
        ax4.set_ylabel('Return (%)')
        ax4.set_title(f'{window}-Day Rolling Statistics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Underwater Curve (Time in Drawdown)
    ax5 = plt.subplot(4, 2, 5)
    ax5.fill_between(continuous_df['Global_Day'], continuous_df['Drawdown'], 0, 
                     color='blue', alpha=0.4)
    ax5.plot(continuous_df['Global_Day'], continuous_df['Drawdown'], 
             linewidth=1, color='navy')
    
    # Calculate time in drawdown statistics
    in_drawdown = (continuous_df['Drawdown'] < -1).sum()  # More than 1% drawdown
    total_days = len(continuous_df)
    pct_in_drawdown = (in_drawdown / total_days) * 100
    
    ax5.text(0.02, 0.95, f'Time in Drawdown (>1%): {pct_in_drawdown:.1f}%\n'
                         f'Max Drawdown: {max_dd_value:.1f}%\n'
                         f'Current Drawdown: {continuous_df["Drawdown"].iloc[-1]:.1f}%',
             transform=ax5.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
             verticalalignment='top')
    
    ax5.set_xlabel('Days Since Start')
    ax5.set_ylabel('Drawdown (%)')
    ax5.set_title('Underwater Curve (Time in Drawdown)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Trade Frequency and Size Over Time
    ax6 = plt.subplot(4, 2, 6)
    
    # Calculate portfolio value at start of each trade
    trade_starts = []
    trade_values = []
    trade_returns = []
    
    for trade_num in sorted(trades_df['Trade_Num'].unique()):
        trade_start_data = continuous_df[continuous_df['Trade_Num'] == trade_num].iloc[0]
        trade_info = trades_df[trades_df['Trade_Num'] == trade_num].iloc[0]
        
        trade_starts.append(trade_start_data['Global_Day'])
        trade_values.append(trade_start_data['Portfolio_Value'])
        trade_returns.append(trade_info['Trade_Return_Pct'])
    
    # Color code by return
    colors = ['green' if r > 0 else 'red' for r in trade_returns]
    sizes = [abs(r) * 3 + 20 for r in trade_returns]  # Size by absolute return
    
    scatter = ax6.scatter(trade_starts, trade_values, c=colors, s=sizes, alpha=0.7, 
                         edgecolors='black', linewidth=0.5)
    
    ax6.set_xlabel('Days Since Start')
    ax6.set_ylabel('Portfolio Value at Trade Start ($)')
    ax6.set_title('Trade Performance Over Time (Size = Return Magnitude)')
    ax6.grid(True, alpha=0.3)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 7: Risk Metrics Over Time
    ax7 = plt.subplot(4, 2, 7)
    
    # Calculate rolling Sharpe ratio
    window_sharpe = 60
    if len(continuous_df) >= window_sharpe:
        rolling_returns = continuous_df['Daily_Return'].rolling(window=window_sharpe)
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)  # Annualized
        
        ax7.plot(continuous_df['Global_Day'], rolling_sharpe, 
                linewidth=2, color='purple', label=f'{window_sharpe}-Day Sharpe')
        ax7.axhline(1.0, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        ax7.axhline(0.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 0.0')
        
        ax7.set_xlabel('Days Since Start')
        ax7.set_ylabel('Rolling Sharpe Ratio')
        ax7.set_title(f'{window_sharpe}-Day Rolling Sharpe Ratio')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: Portfolio Statistics Summary
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Calculate key statistics
    total_return = (continuous_df['Portfolio_Value'].iloc[-1] / continuous_df['Portfolio_Value'].iloc[0] - 1) * 100
    total_days = len(continuous_df)
    annualized_return = ((continuous_df['Portfolio_Value'].iloc[-1] / continuous_df['Portfolio_Value'].iloc[0]) ** (365.25 / total_days) - 1) * 100
    volatility = continuous_df['Daily_Return'].std() * np.sqrt(252)
    sharpe_ratio = (continuous_df['Daily_Return'].mean() / continuous_df['Daily_Return'].std() * np.sqrt(252)) if continuous_df['Daily_Return'].std() > 0 else 0
    
    stats_text = f"""
PORTFOLIO STATISTICS
{'='*30}

Total Return: {total_return:+.1f}%
Annualized Return: {annualized_return:+.1f}%
Total Days: {total_days:,}
Volatility: {volatility:.1f}%
Sharpe Ratio: {sharpe_ratio:.2f}

Max Drawdown: {max_dd_value:.1f}%
Time in Drawdown: {pct_in_drawdown:.1f}%
Current Value: ${continuous_df['Portfolio_Value'].iloc[-1]:,.0f}

Best Day: {continuous_df['Daily_Return'].max():+.2f}%
Worst Day: {continuous_df['Daily_Return'].min():+.2f}%
Avg Daily Return: {continuous_df['Daily_Return'].mean():+.3f}%

Winning Days: {(continuous_df['Daily_Return'] > 0).sum()} ({(continuous_df['Daily_Return'] > 0).mean()*100:.1f}%)
"""
    
    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=12,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    filename = 'continuous_portfolio_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Continuous portfolio analysis saved: {filename}")
    
    plt.show()

def plot_trade_correlation_analysis(continuous_df, trades_df):
    """
    Analyze correlations and patterns across trades
    """
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trade Correlation and Pattern Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Return vs Days Held
    ax1 = axes[0, 0]
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = trades_df[trades_df['Scenario'] == scenario]
        if not scenario_data.empty:
            ax1.scatter(scenario_data['Days_Held'], scenario_data['Trade_Return_Pct'],
                       c=scenario_colors[scenario], label=scenario.title(), 
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(20, color='green', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Days Held')
    ax1.set_ylabel('Trade Return (%)')
    ax1.set_title('Return vs Holding Period by Scenario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sequential Trade Performance
    ax2 = axes[0, 1]
    
    # Calculate rolling average performance
    window = 5
    rolling_avg = trades_df['Trade_Return_Pct'].rolling(window=window).mean()
    
    ax2.bar(trades_df['Trade_Num'], trades_df['Trade_Return_Pct'], 
           color=['green' if x > 0 else 'red' for x in trades_df['Trade_Return_Pct']],
           alpha=0.6)
    ax2.plot(trades_df['Trade_Num'], rolling_avg, 
            color='blue', linewidth=3, label=f'{window}-Trade Moving Average')
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Trade Return (%)')
    ax2.set_title('Sequential Trade Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scenario Performance Over Time
    ax3 = axes[0, 2]
    
    scenario_performance = []
    for i in range(1, len(trades_df) + 1):
        subset = trades_df.iloc[:i]
        bull_avg = subset[subset['Scenario'] == 'bull']['Trade_Return_Pct'].mean()
        sideways_avg = subset[subset['Scenario'] == 'sideways']['Trade_Return_Pct'].mean()
        bear_avg = subset[subset['Scenario'] == 'bear']['Trade_Return_Pct'].mean()
        
        scenario_performance.append({
            'Trade_Num': i,
            'Bull_Avg': bull_avg if not pd.isna(bull_avg) else 0,
            'Sideways_Avg': sideways_avg if not pd.isna(sideways_avg) else 0,
            'Bear_Avg': bear_avg if not pd.isna(bear_avg) else 0
        })
    
    scenario_perf_df = pd.DataFrame(scenario_performance)
    
    ax3.plot(scenario_perf_df['Trade_Num'], scenario_perf_df['Bull_Avg'], 
            color=scenario_colors['bull'], linewidth=2, label='Bull Avg')
    ax3.plot(scenario_perf_df['Trade_Num'], scenario_perf_df['Sideways_Avg'], 
            color=scenario_colors['sideways'], linewidth=2, label='Sideways Avg')
    ax3.plot(scenario_perf_df['Trade_Num'], scenario_perf_df['Bear_Avg'], 
            color=scenario_colors['bear'], linewidth=2, label='Bear Avg')
    
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative Average Return (%)')
    ax3.set_title('Scenario Performance Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility vs Return Analysis
    ax4 = axes[1, 0]
    
    # Calculate realized volatility for each trade period
    trade_volatilities = []
    for trade_num in trades_df['Trade_Num']:
        trade_daily = continuous_df[continuous_df['Trade_Num'] == trade_num]
        if len(trade_daily) > 1:
            daily_rets = trade_daily['Daily_Return']
            vol = daily_rets.std() * np.sqrt(252)  # Annualized
            trade_volatilities.append(vol)
        else:
            trade_volatilities.append(0)
    
    trades_df_temp = trades_df.copy()
    trades_df_temp['Trade_Volatility'] = trade_volatilities
    
    colors = [scenario_colors.get(s, 'blue') for s in trades_df_temp['Scenario']]
    ax4.scatter(trades_df_temp['Trade_Volatility'], trades_df_temp['Trade_Return_Pct'],
               c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax4.set_xlabel('Trade Period Volatility (%)')
    ax4.set_ylabel('Trade Return (%)')
    ax4.set_title('Volatility vs Return by Scenario')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Drawdown Recovery Time
    ax5 = axes[1, 1]
    
    # Find drawdown periods and recovery times
    drawdowns = continuous_df['Drawdown'] < -1  # More than 1% drawdown
    drawdown_periods = []
    
    in_drawdown = False
    drawdown_start = None
    
    for idx, is_dd in enumerate(drawdowns):
        if is_dd and not in_drawdown:
            in_drawdown = True
            drawdown_start = idx
        elif not is_dd and in_drawdown:
            in_drawdown = False
            if drawdown_start is not None:
                drawdown_periods.append({
                    'Start': drawdown_start,
                    'End': idx,
                    'Length': idx - drawdown_start,
                    'Max_DD': continuous_df.iloc[drawdown_start:idx]['Drawdown'].min()
                })
    
    if drawdown_periods:
        dd_df = pd.DataFrame(drawdown_periods)
        ax5.scatter(dd_df['Max_DD'], dd_df['Length'], s=80, alpha=0.7, 
                   color='red', edgecolors='black', linewidth=1)
        
        # Add trend line
        if len(dd_df) > 1:
            z = np.polyfit(dd_df['Max_DD'], dd_df['Length'], 1)
            p = np.poly1d(z)
            ax5.plot(dd_df['Max_DD'], p(dd_df['Max_DD']), "r--", alpha=0.8)
        
        ax5.set_xlabel('Maximum Drawdown (%)')
        ax5.set_ylabel('Recovery Time (Days)')
        ax5.set_title('Drawdown Magnitude vs Recovery Time')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No significant\ndrawdown periods\nfound', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    
    # Plot 6: Performance Metrics Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate correlation statistics
    corr_returns_days = trades_df['Trade_Return_Pct'].corr(trades_df['Days_Held'])
    corr_returns_underlying = trades_df['Trade_Return_Pct'].corr(trades_df['Actual_Underlying_Return'])
    
    # Calculate win rates manually to avoid lambda issues
    scenario_win_rates = {}
    scenario_counts = {}
    
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = trades_df[trades_df['Scenario'] == scenario]
        if not scenario_data.empty:
            wins = (scenario_data['Trade_Return_Pct'] > 0).sum()
            total = len(scenario_data)
            scenario_win_rates[scenario] = wins
            scenario_counts[scenario] = total
        else:
            scenario_win_rates[scenario] = 0
            scenario_counts[scenario] = 0
    
    summary_text = f"""
CORRELATION ANALYSIS
{'='*25}

Return vs Days Held: {corr_returns_days:.3f}
Return vs Underlying: {corr_returns_underlying:.3f}

SCENARIO WIN RATES:
Bull: {scenario_win_rates['bull']}/{scenario_counts['bull']} ({scenario_win_rates['bull']/max(scenario_counts['bull'],1)*100:.0f}%)
Sideways: {scenario_win_rates['sideways']}/{scenario_counts['sideways']} ({scenario_win_rates['sideways']/max(scenario_counts['sideways'],1)*100:.0f}%)
Bear: {scenario_win_rates['bear']}/{scenario_counts['bear']} ({scenario_win_rates['bear']/max(scenario_counts['bear'],1)*100:.0f}%)

TRADE PATTERNS:
Longest Win Streak: {calculate_longest_streak(trades_df['Trade_Return_Pct'] > 0)}
Longest Loss Streak: {calculate_longest_streak(trades_df['Trade_Return_Pct'] <= 0)}

Best Trade: {trades_df['Trade_Return_Pct'].max():+.1f}%
Worst Trade: {trades_df['Trade_Return_Pct'].min():+.1f}%
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    filename = 'trade_correlation_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Trade correlation analysis saved: {filename}")
    
    plt.show()

def calculate_longest_streak(boolean_series):
    """Calculate the longest consecutive streak of True values"""
    streaks = []
    current_streak = 0
    
    for value in boolean_series:
        if value:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    
    if current_streak > 0:
        streaks.append(current_streak)
    
    return max(streaks) if streaks else 0

def plot_risk_analysis(continuous_df, trades_df):
    """
    Detailed risk analysis plots
    """
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Value at Risk (VaR) Analysis
    ax1 = axes[0, 0]
    
    daily_returns = continuous_df['Daily_Return'].dropna()
    
    # Calculate VaR at different confidence levels
    var_95 = np.percentile(daily_returns, 5)
    var_99 = np.percentile(daily_returns, 1)
    
    ax1.hist(daily_returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    
    # Add VaR lines
    ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.2f}%')
    ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR: {var_99:.2f}%')
    ax1.axvline(daily_returns.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
    
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Value at Risk (VaR) Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling Risk Metrics
    ax2 = axes[0, 1]
    
    window = 30
    if len(continuous_df) >= window:
        rolling_vol = continuous_df['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
        rolling_var95 = continuous_df['Daily_Return'].rolling(window=window).quantile(0.05)
        
        ax2.plot(continuous_df['Global_Day'], rolling_vol, label=f'{window}D Volatility', linewidth=2, color='blue')
        ax2.plot(continuous_df['Global_Day'], rolling_var95, label=f'{window}D VaR(95%)', linewidth=2, color='red')
        
        ax2.set_xlabel('Days Since Start')
        ax2.set_ylabel('Risk Metric')
        ax2.set_title(f'{window}-Day Rolling Risk Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Maximum Adverse Excursion (MAE) Analysis
    ax3 = axes[1, 0]
    
    # Calculate MAE for each trade
    mae_data = []
    for trade_num in trades_df['Trade_Num']:
        trade_daily = continuous_df[continuous_df['Trade_Num'] == trade_num]
        if not trade_daily.empty:
            min_return = trade_daily['Return_Pct'].min()
            final_return = trade_daily['Return_Pct'].iloc[-1]
            mae_data.append({
                'Trade_Num': trade_num,
                'MAE': min_return,
                'Final_Return': final_return,
                'Scenario': trades_df[trades_df['Trade_Num'] == trade_num]['Scenario'].iloc[0]
            })
    
    mae_df = pd.DataFrame(mae_data)
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = mae_df[mae_df['Scenario'] == scenario]
        if not scenario_data.empty:
            ax3.scatter(scenario_data['MAE'], scenario_data['Final_Return'],
                       c=scenario_colors[scenario], label=scenario.title(), 
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(mae_df['MAE'].min(), mae_df['Final_Return'].min())
    max_val = max(mae_df['MAE'].max(), mae_df['Final_Return'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='MAE = Final')
    
    ax3.set_xlabel('Maximum Adverse Excursion (%)')
    ax3.set_ylabel('Final Return (%)')
    ax3.set_title('MAE vs Final Return Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk-Adjusted Returns
    ax4 = axes[1, 1]
    
    # Calculate trade-level Sharpe ratios
    trade_sharpes = []
    for trade_num in trades_df['Trade_Num']:
        trade_daily = continuous_df[continuous_df['Trade_Num'] == trade_num]
        if len(trade_daily) > 1:
            trade_returns = trade_daily['Daily_Return']
            if trade_returns.std() > 0:
                sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252)
                trade_sharpes.append(sharpe)
            else:
                trade_sharpes.append(0)
        else:
            trade_sharpes.append(0)
    
    trades_df_temp = trades_df.copy()
    trades_df_temp['Trade_Sharpe'] = trade_sharpes
    
    # Create bins for Sharpe ratios
    bins = [-np.inf, -1, 0, 1, 2, np.inf]
    labels = ['Very Poor (<-1)', 'Poor (-1 to 0)', 'Fair (0 to 1)', 'Good (1 to 2)', 'Excellent (>2)']
    trades_df_temp['Sharpe_Category'] = pd.cut(trades_df_temp['Trade_Sharpe'], bins=bins, labels=labels)
    
    sharpe_counts = trades_df_temp['Sharpe_Category'].value_counts()
    colors = ['darkred', 'red', 'orange', 'lightgreen', 'darkgreen']
    
    wedges, texts, autotexts = ax4.pie(sharpe_counts.values, labels=sharpe_counts.index, 
                                      autopct='%1.0f%%', colors=colors[:len(sharpe_counts)], 
                                      startangle=90)
    ax4.set_title('Distribution of Trade-Level Sharpe Ratios')
    
    plt.tight_layout()
    
    filename = 'risk_analysis_dashboard.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Risk analysis dashboard saved: {filename}")
    
    plt.show()

def generate_simulation_report(continuous_df, trades_df):
    """
    Generate a comprehensive text report of simulation results
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SIMULATION ANALYSIS REPORT")
    print("="*80)
    
    # Basic Statistics
    total_days = len(continuous_df)
    initial_value = continuous_df['Portfolio_Value'].iloc[0]
    final_value = continuous_df['Portfolio_Value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    annualized_return = ((final_value / initial_value) ** (365.25 / total_days) - 1) * 100
    
    print(f"\n📊 PORTFOLIO PERFORMANCE:")
    print(f"   Initial Value: ${initial_value:,.0f}")
    print(f"   Final Value: ${final_value:,.0f}")
    print(f"   Total Return: {total_return:+.1f}%")
    print(f"   Annualized Return: {annualized_return:+.1f}%")
    print(f"   Total Days: {total_days:,}")
    print(f"   Trading Period: {total_days/365.25:.1f} years")
    
    # Risk Metrics
    daily_returns = continuous_df['Daily_Return'].dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    max_dd = continuous_df['Drawdown'].min()
    var_95 = np.percentile(daily_returns, 5)
    
    print(f"\n📈 RISK METRICS:")
    print(f"   Annualized Volatility: {volatility:.1f}%")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"   Maximum Drawdown: {max_dd:.1f}%")
    print(f"   VaR (95%): {var_95:.2f}%")
    print(f"   Best Day: {daily_returns.max():+.2f}%")
    print(f"   Worst Day: {daily_returns.min():+.2f}%")
    
    # Trade Statistics
    winning_trades = (trades_df['Trade_Return_Pct'] > 0).sum()
    tp_hits = trades_df['TP_Hit'].sum()
    
    print(f"\n🎯 TRADE STATISTICS:")
    print(f"   Total Trades: {len(trades_df)}")
    print(f"   Winning Trades: {winning_trades} ({winning_trades/len(trades_df)*100:.1f}%)")
    print(f"   TP Hits: {tp_hits} ({tp_hits/len(trades_df)*100:.1f}%)")
    print(f"   Average Trade Return: {trades_df['Trade_Return_Pct'].mean():+.1f}%")
    print(f"   Best Trade: {trades_df['Trade_Return_Pct'].max():+.1f}%")
    print(f"   Worst Trade: {trades_df['Trade_Return_Pct'].min():+.1f}%")
    print(f"   Average Days Held: {trades_df['Days_Held'].mean():.0f}")
    
    # Scenario Analysis
    print(f"\n🎲 SCENARIO ANALYSIS:")
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = trades_df[trades_df['Scenario'] == scenario]
        if not scenario_data.empty:
            count = len(scenario_data)
            win_rate = (scenario_data['Trade_Return_Pct'] > 0).mean() * 100
            avg_return = scenario_data['Trade_Return_Pct'].mean()
            tp_rate = scenario_data['TP_Hit'].mean() * 100
            
            print(f"   {scenario.title()}: {count} trades ({count/len(trades_df)*100:.0f}%)")
            print(f"     Win Rate: {win_rate:.0f}%")
            print(f"     TP Rate: {tp_rate:.0f}%") 
            print(f"     Avg Return: {avg_return:+.1f}%")
    
    # Streaks and Patterns
    win_streak = calculate_longest_streak(trades_df['Trade_Return_Pct'] > 0)
    loss_streak = calculate_longest_streak(trades_df['Trade_Return_Pct'] <= 0)
    
    print(f"\n🔄 STREAKS AND PATTERNS:")
    print(f"   Longest Win Streak: {win_streak} trades")
    print(f"   Longest Loss Streak: {loss_streak} trades")
    
    # Time in Drawdown
    in_drawdown_days = (continuous_df['Drawdown'] < -1).sum()
    pct_in_drawdown = (in_drawdown_days / total_days) * 100
    
    print(f"\n⬇️  DRAWDOWN ANALYSIS:")
    print(f"   Time in Drawdown (>1%): {pct_in_drawdown:.1f}%")
    print(f"   Days in Drawdown: {in_drawdown_days}")
    print(f"   Current Drawdown: {continuous_df['Drawdown'].iloc[-1]:.1f}%")
    
    print("\n" + "="*80)

def main():
    """
    Main function to run all detailed analysis
    """
    
    print("🔍 DETAILED TRADING SIMULATION ANALYSIS")
    print("="*60)
    print("Loading simulation data and creating comprehensive analysis...")
    
    # Load data
    trades_df, daily_df = load_simulation_data()
    
    if trades_df is None or daily_df is None:
        return
    
    # Create continuous timeline
    continuous_df = create_continuous_daily_timeline(daily_df, trades_df)
    
    print("\n📊 Creating analysis plots...")
    
    # Generate all plots
    print("\n1. Individual trade time series...")
    plot_all_individual_trade_timeseries(daily_df, trades_df)
    
    print("\n2. Continuous portfolio analysis...")
    plot_continuous_portfolio_analysis(continuous_df, trades_df)
    
    print("\n3. Trade correlation analysis...")
    plot_trade_correlation_analysis(continuous_df, trades_df)
    
    print("\n4. Risk analysis dashboard...")
    plot_risk_analysis(continuous_df, trades_df)
    
    # Generate comprehensive report
    generate_simulation_report(continuous_df, trades_df)
    
    # Save enhanced data
    continuous_df.to_csv('continuous_portfolio_timeline.csv', index=False)
    
    print(f"\n💾 Enhanced data saved:")
    print(f"   - continuous_portfolio_timeline.csv ({len(continuous_df)} daily records)")
    
    print(f"\n📊 Generated analysis files:")
    print(f"   - trade_timeseries_batch_*.png (individual trade curves)")
    print(f"   - continuous_portfolio_analysis.png (daily portfolio metrics)")
    print(f"   - trade_correlation_analysis.png (pattern analysis)")
    print(f"   - risk_analysis_dashboard.png (risk metrics)")
    
    print(f"\n🎉 Detailed analysis complete!")
    print(f"📈 Portfolio grew from ${continuous_df['Portfolio_Value'].iloc[0]:,.0f} to ${continuous_df['Portfolio_Value'].iloc[-1]:,.0f}")
    print(f"📊 Maximum drawdown: {continuous_df['Drawdown'].min():.1f}%")

if __name__ == "__main__":
    main()