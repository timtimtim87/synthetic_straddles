import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_simulation_data():
    """
    Load the saved simulation CSV files
    """
    try:
        print("Loading simulation data files...")
        
        groups_df = pd.read_csv('portfolio_5yr_simulation_groups.csv')
        assets_df = pd.read_csv('portfolio_5yr_simulation_assets.csv')
        continuous_df = pd.read_csv('portfolio_5yr_continuous_timeline.csv')
        detailed_daily_df = pd.read_csv('portfolio_5yr_simulation_daily_detailed.csv')
        
        # Convert date columns if they exist
        date_columns = ['Start_Date', 'End_Date', 'Calendar_Date']
        for df, name in [(groups_df, 'groups'), (detailed_daily_df, 'detailed_daily')]:
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        print(f"  Converted {col} in {name} to datetime")
                    except:
                        print(f"  Could not convert {col} in {name} to datetime")
        
        print(f"✅ Data loaded successfully:")
        print(f"  Groups: {len(groups_df)} records")
        print(f"  Assets: {len(assets_df)} records") 
        print(f"  Continuous: {len(continuous_df)} daily records")
        print(f"  Detailed: {len(detailed_daily_df)} detailed records")
        
        return groups_df, assets_df, continuous_df, detailed_daily_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find simulation data files")
        print(f"Missing file: {e}")
        print("Make sure you've run z16_portfolio_trading_simulation.py first!")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None

def plot_individual_trade_groups(detailed_daily_df, groups_df, assets_df, max_groups_to_plot=18):
    """
    Plot individual trade group progressions - all 5 assets per group
    """
    
    if detailed_daily_df is None or groups_df is None:
        print("❌ No data for individual trade plots")
        return
    
    print(f"\n📊 Creating individual trade group plots...")
    
    plt.style.use('default')
    
    # Get unique groups and limit to reasonable number
    unique_groups = sorted(detailed_daily_df['Group_Num'].unique())
    n_groups_to_plot = min(len(unique_groups), max_groups_to_plot)
    groups_to_plot = unique_groups[:n_groups_to_plot]
    
    print(f"   Plotting first {n_groups_to_plot} groups out of {len(unique_groups)} total")
    
    # Calculate grid: 6 groups per image
    groups_per_image = 6
    n_images = (n_groups_to_plot + groups_per_image - 1) // groups_per_image
    
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    asset_markers = ['o', 's', '^', 'D', 'v']
    
    for image_num in range(n_images):
        start_idx = image_num * groups_per_image
        end_idx = min(start_idx + groups_per_image, n_groups_to_plot)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Trade Group Progressions - Groups {groups_to_plot[start_idx]} to {groups_to_plot[end_idx-1]}', 
                     fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for plot_idx in range(groups_per_image):
            ax = axes_flat[plot_idx]
            
            if start_idx + plot_idx < n_groups_to_plot:
                group_num = groups_to_plot[start_idx + plot_idx]
                
                # Get data for this group
                group_daily_data = detailed_daily_df[detailed_daily_df['Group_Num'] == group_num]
                group_info = groups_df[groups_df['Group_Num'] == group_num]
                
                if group_daily_data.empty or group_info.empty:
                    ax.text(0.5, 0.5, f'Group {group_num}\nNo Data', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    continue
                
                group_info = group_info.iloc[0]
                
                # Plot each asset (0-4) in different colors
                asset_data_dict = {}
                for asset_num in range(5):
                    asset_data = group_daily_data[group_daily_data['Asset_Num'] == asset_num].copy()
                    
                    if not asset_data.empty:
                        asset_data = asset_data.sort_values('Day')
                        
                        # Get asset scenario for coloring
                        try:
                            asset_info = assets_df[(assets_df['Group_Num'] == group_num) & 
                                                 (assets_df['Asset_Num'] == asset_num)]
                            if not asset_info.empty:
                                scenario = asset_info.iloc[0]['Scenario']
                                color = scenario_colors.get(scenario, '#1f77b4')
                            else:
                                scenario = 'unknown'
                                color = '#1f77b4'
                        except:
                            scenario = 'unknown'
                            color = '#1f77b4'
                        
                        # Calculate days from start for x-axis
                        days_from_start = asset_data['Day'].max() - asset_data['Day']
                        
                        # Plot asset return curve
                        ax.plot(days_from_start, asset_data['Asset_Return_Pct'], 
                               color=color, linewidth=1.5, alpha=0.7,
                               marker=asset_markers[asset_num], markersize=2,
                               label=f'A{asset_num} ({scenario[:4]})')
                        
                        # Store for group average calculation
                        asset_data_dict[asset_num] = {
                            'days': days_from_start.values,
                            'returns': asset_data['Asset_Return_Pct'].values
                        }
                
                # Calculate and plot group average
                if asset_data_dict:
                    # Get group-level returns from data (already calculated)
                    group_returns_data = group_daily_data.groupby('Day').first()['Group_Return_Pct']
                    group_days_data = group_returns_data.index
                    
                    if not group_returns_data.empty:
                        days_from_start_group = group_days_data.max() - group_days_data
                        
                        ax.plot(days_from_start_group, group_returns_data.values, 
                               color='black', linewidth=3, alpha=0.9, label='Group Avg')
                        
                        # Fill profit/loss areas
                        ax.fill_between(days_from_start_group, group_returns_data.values, 0,
                                       where=(group_returns_data.values >= 0), 
                                       alpha=0.2, color='green', interpolate=True)
                        ax.fill_between(days_from_start_group, group_returns_data.values, 0,
                                       where=(group_returns_data.values < 0), 
                                       alpha=0.2, color='red', interpolate=True)
                
                # Reference lines
                ax.axhline(y=20, color='green', linestyle='--', alpha=0.6, linewidth=1)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
                
                # Format plot
                final_return = group_info['Group_Return_Pct']
                exit_reason = "TP" if group_info['TP_Hit'] else "Exp"
                days_held = group_info['Days_Held']
                
                ax.set_title(f'Group {group_num}\nReturn: {final_return:+.1f}% ({exit_reason}) | {days_held}d', 
                           fontsize=10, fontweight='bold')
                ax.set_xlabel('Days from Start', fontsize=9)
                ax.set_ylabel('Return (%)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc='upper left')
                
                # Set reasonable y-limits
                ax.set_ylim(-40, 30)
                
            else:
                ax.set_visible(False)
        
        plt.tight_layout()
        
        filename = f'individual_trade_groups_batch_{image_num+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
        
        plt.show()

def plot_portfolio_performance_dashboard(continuous_df, groups_df):
    """
    Main portfolio performance dashboard
    """
    
    if continuous_df is None or groups_df is None:
        print("❌ No data for portfolio dashboard")
        return
    
    print(f"\n📊 Creating portfolio performance dashboard...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # Calculate metrics
    initial_value = continuous_df['Total_Portfolio_Value'].iloc[0]
    final_value = continuous_df['Total_Portfolio_Value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate daily returns and drawdown
    continuous_df['Daily_Return'] = continuous_df['Total_Portfolio_Value'].pct_change() * 100
    continuous_df['Daily_Return'] = continuous_df['Daily_Return'].fillna(0)
    continuous_df['Peak_Portfolio'] = continuous_df['Total_Portfolio_Value'].expanding().max()
    continuous_df['Drawdown'] = (continuous_df['Total_Portfolio_Value'] - continuous_df['Peak_Portfolio']) / continuous_df['Peak_Portfolio'] * 100
    
    # Plot 1: Portfolio equity curve
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(continuous_df['Global_Day'], continuous_df['Total_Portfolio_Value'], 
             linewidth=2, color='blue', alpha=0.8)
    ax1.fill_between(continuous_df['Global_Day'], continuous_df['Total_Portfolio_Value'], 
                     initial_value, alpha=0.3, color='blue')
    
    ax1.set_xlabel('Days Since Start')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Drawdown
    ax2 = plt.subplot(3, 3, 2)
    ax2.fill_between(continuous_df['Global_Day'], continuous_df['Drawdown'], 0, 
                     color='red', alpha=0.6)
    ax2.plot(continuous_df['Global_Day'], continuous_df['Drawdown'], 
             linewidth=1, color='darkred')
    
    max_dd = continuous_df['Drawdown'].min()
    max_dd_day = continuous_df.loc[continuous_df['Drawdown'].idxmin(), 'Global_Day']
    ax2.annotate(f'Max: {max_dd:.1f}%', xy=(max_dd_day, max_dd), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('Days Since Start')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Portfolio Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Daily returns distribution
    ax3 = plt.subplot(3, 3, 3)
    daily_returns = continuous_df['Daily_Return'].dropna()
    ax3.hist(daily_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(daily_returns.mean(), color='red', linestyle='--', 
               label=f'Mean: {daily_returns.mean():.3f}%')
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Daily Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Daily Returns Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Group returns
    ax4 = plt.subplot(3, 3, 4)
    tp_groups = groups_df[groups_df['TP_Hit'] == True]['Group_Return_Pct']
    exp_groups = groups_df[groups_df['TP_Hit'] == False]['Group_Return_Pct']
    
    if len(tp_groups) > 0:
        ax4.hist(tp_groups, alpha=0.7, label=f'TP Hits ({len(tp_groups)})', 
                color='green', bins=15)
    if len(exp_groups) > 0:
        ax4.hist(exp_groups, alpha=0.7, label=f'Expiry ({len(exp_groups)})', 
                color='orange', bins=15)
    
    ax4.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(20, color='green', linestyle='--', alpha=0.7, label='20% TP')
    ax4.set_xlabel('Group Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Group Returns Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Rolling volatility
    ax5 = plt.subplot(3, 3, 5)
    window = 30
    if len(continuous_df) >= window:
        rolling_vol = continuous_df['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
        ax5.plot(continuous_df['Global_Day'], rolling_vol, linewidth=2, color='purple')
        ax5.set_xlabel('Days Since Start')
        ax5.set_ylabel('Annualized Volatility (%)')
        ax5.set_title(f'{window}-Day Rolling Volatility')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative returns
    ax6 = plt.subplot(3, 3, 6)
    cumulative_returns = (1 + continuous_df['Daily_Return']/100).cumprod() - 1
    ax6.plot(continuous_df['Global_Day'], cumulative_returns * 100, 
             linewidth=2, color='orange')
    ax6.set_xlabel('Days Since Start')
    ax6.set_ylabel('Cumulative Return (%)')
    ax6.set_title('Cumulative Returns')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Win/Loss by group
    ax7 = plt.subplot(3, 3, 7)
    group_outcomes = ['Win' if ret > 0 else 'Loss' for ret in groups_df['Group_Return_Pct']]
    win_counts = pd.Series(group_outcomes).value_counts()
    
    colors = ['green' if x == 'Win' else 'red' for x in win_counts.index]
    bars = ax7.bar(win_counts.index, win_counts.values, color=colors, alpha=0.7)
    
    for bar, count in zip(bars, win_counts.values):
        ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax7.set_ylabel('Number of Groups')
    ax7.set_title('Win/Loss Distribution')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Performance over time
    ax8 = plt.subplot(3, 3, 8)
    colors = ['green' if ret > 0 else 'red' for ret in groups_df['Group_Return_Pct']]
    ax8.bar(groups_df['Group_Num'], groups_df['Group_Return_Pct'], 
           color=colors, alpha=0.7)
    ax8.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax8.axhline(20, color='green', linestyle='--', alpha=0.7)
    ax8.set_xlabel('Group Number')
    ax8.set_ylabel('Group Return (%)')
    ax8.set_title('Group Performance Over Time')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Key statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate key metrics
    simulation_days = len(continuous_df)
    simulation_years = simulation_days / 365.25
    annualized_return = ((final_value / initial_value) ** (1/simulation_years) - 1) * 100
    
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    win_rate = (groups_df['Group_Return_Pct'] > 0).mean() * 100
    tp_rate = groups_df['TP_Hit'].mean() * 100
    avg_days_held = groups_df['Days_Held'].mean()
    
    stats_text = f"""
PORTFOLIO PERFORMANCE
{'='*25}

RETURNS:
Total Return: {total_return:+.1f}%
Annualized Return: {annualized_return:+.1f}%
Simulation Period: {simulation_years:.1f} years

RISK METRICS:
Volatility: {volatility:.1f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.1f}%

TRADING STATS:
Total Groups: {len(groups_df)}
Win Rate: {win_rate:.0f}%
TP Hit Rate: {tp_rate:.0f}%
Avg Days Held: {avg_days_held:.0f}

PORTFOLIO:
Initial: ${initial_value:,.0f}
Final: ${final_value:,.0f}
"""
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    filename = 'portfolio_performance_dashboard.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: {filename}")
    
    plt.show()

def plot_asset_analysis(assets_df):
    """
    Asset-level analysis plots
    """
    
    if assets_df is None or len(assets_df) == 0:
        print("❌ No asset data for analysis")
        return
    
    print(f"\n📊 Creating asset analysis plots...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Asset-Level Analysis', fontsize=16, fontweight='bold')
    
    scenario_colors = {'bull': '#2E8B57', 'sideways': '#FF8C00', 'bear': '#DC143C'}
    
    # Plot 1: Returns by scenario
    ax1 = axes[0, 0]
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = assets_df[assets_df['Scenario'] == scenario]['Asset_Return_Pct']
        if not scenario_data.empty:
            ax1.hist(scenario_data, alpha=0.6, label=f'{scenario.title()} ({len(scenario_data)})', 
                    color=scenario_colors[scenario], bins=15)
    
    ax1.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Asset Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Asset Returns by Scenario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance by asset position
    ax2 = axes[0, 1]
    asset_stats = assets_df.groupby('Asset_Num')['Asset_Return_Pct'].agg(['mean', 'std', 'count'])
    
    if not asset_stats.empty:
        bars = ax2.bar(asset_stats.index, asset_stats['mean'], 
                      yerr=asset_stats['std'], capsize=5, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Asset Position')
        ax2.set_ylabel('Average Return (%)')
        ax2.set_title('Performance by Asset Position')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, asset_stats['mean']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top')
    
    # Plot 3: Scenario distribution
    ax3 = axes[0, 2]
    scenario_counts = assets_df['Scenario'].value_counts()
    colors = [scenario_colors[scenario] for scenario in scenario_counts.index]
    
    wedges, texts, autotexts = ax3.pie(scenario_counts.values, labels=scenario_counts.index,
                                      autopct='%1.0f%%', colors=colors, startangle=90)
    ax3.set_title('Scenario Distribution')
    
    # Plot 4: Win rates by scenario
    ax4 = axes[1, 0]
    scenario_win_rates = []
    scenario_names = []
    
    for scenario in ['bull', 'sideways', 'bear']:
        scenario_data = assets_df[assets_df['Scenario'] == scenario]
        if not scenario_data.empty:
            win_rate = (scenario_data['Asset_Return_Pct'] > 0).mean() * 100
            scenario_win_rates.append(win_rate)
            scenario_names.append(scenario.title())
    
    if scenario_win_rates:
        bars = ax4.bar(scenario_names, scenario_win_rates, 
                      color=[scenario_colors[s.lower()] for s in scenario_names], alpha=0.7)
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Win Rate by Scenario')
        ax4.grid(True, alpha=0.3)
        
        for bar, rate in zip(bars, scenario_win_rates):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Return vs underlying movement
    ax5 = axes[1, 1]
    colors = [scenario_colors.get(scenario, 'blue') for scenario in assets_df['Scenario']]
    ax5.scatter(assets_df['Actual_Underlying_Return'] * 100, assets_df['Asset_Return_Pct'],
               c=colors, alpha=0.6, s=30)
    ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Underlying Asset Return (%)')
    ax5.set_ylabel('Straddle Return (%)')
    ax5.set_title('Straddle vs Underlying Performance')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    total_assets = len(assets_df)
    avg_return = assets_df['Asset_Return_Pct'].mean()
    return_std = assets_df['Asset_Return_Pct'].std()
    
    scenario_stats = assets_df.groupby('Scenario').agg({
        'Asset_Return_Pct': ['count', 'mean'],
        'Actual_Underlying_Return': 'mean'
    }).round(2)
    
    stats_text = f"""
ASSET SUMMARY
{'='*15}

Total Assets: {total_assets}
Avg Return: {avg_return:+.1f}%
Return Std: {return_std:.1f}%

BY SCENARIO:
"""
    
    for scenario in scenario_stats.index:
        count = int(scenario_stats.loc[scenario, ('Asset_Return_Pct', 'count')])
        mean_ret = scenario_stats.loc[scenario, ('Asset_Return_Pct', 'mean')]
        underlying = scenario_stats.loc[scenario, ('Actual_Underlying_Return', 'mean')] * 100
        
        stats_text += f"\n{scenario.title()}:\n"
        stats_text += f"  Count: {count}\n"
        stats_text += f"  Avg Return: {mean_ret:+.1f}%\n"
        stats_text += f"  Underlying: {underlying:+.1f}%\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    filename = 'asset_level_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: {filename}")
    
    plt.show()

def main():
    """
    Main function - just loads data and creates visualizations
    """
    
    print("🎨 PORTFOLIO VISUALIZATION SCRIPT")
    print("="*50)
    print("Pure visualization script - loads saved CSV files and creates plots")
    print("="*50)
    
    # Load the saved simulation data
    groups_df, assets_df, continuous_df, detailed_daily_df = load_simulation_data()
    
    if groups_df is None:
        print("❌ Cannot proceed without data files")
        print("Run z16_portfolio_trading_simulation.py first to generate the data!")
        return
    
    # Create all visualizations
    print(f"\n🎨 Creating visualization suite...")
    
    # 1. Individual trade group progressions
    plot_individual_trade_groups(detailed_daily_df, groups_df, assets_df, max_groups_to_plot=18)
    
    # 2. Main portfolio performance dashboard  
    plot_portfolio_performance_dashboard(continuous_df, groups_df)
    
    # 3. Asset-level analysis
    plot_asset_analysis(assets_df)
    
    # Summary
    if continuous_df is not None and len(continuous_df) > 0:
        initial_value = continuous_df['Total_Portfolio_Value'].iloc[0]
        final_value = continuous_df['Total_Portfolio_Value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        print(f"\n🎉 Visualization Complete!")
        print(f"📈 Portfolio Performance: ${initial_value:,.0f} → ${final_value:,.0f} ({total_return:+.1f}%)")
        print(f"📊 Generated plots:")
        print(f"   • individual_trade_groups_batch_*.png")
        print(f"   • portfolio_performance_dashboard.png")
        print(f"   • asset_level_analysis.png")
        print(f"📁 Check your directory for the plot files!")
    else:
        print(f"\n✅ Visualization script completed!")

if __name__ == "__main__":
    main()