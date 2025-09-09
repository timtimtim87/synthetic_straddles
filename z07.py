import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_short_straddle_backtest(csv_file_path, output_file="short_straddle_results.csv"):
    """
    Run short straddle backtest with exit conditions:
    1. Asset reaches strike price (spot >= strike)
    2. Contract reaches 60 DTE
    
    For each trade, track:
    - Max gain within period
    - Max drawdown within period  
    - Period duration
    - Actual return at exit
    """
    
    try:
        # Read the cleaned straddle data
        print(f"Reading straddle data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Loaded {len(df):,} straddle records")
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return None
    
    # Get unique straddles
    unique_straddles = df[['Ticker', 'Expiry_Year']].drop_duplicates()
    print(f"Found {len(unique_straddles)} unique straddles to backtest")
    
    print(f"\n" + "="*80)
    print("RUNNING SHORT STRADDLE BACKTEST")
    print("="*80)
    print("Exit Conditions:")
    print("  1. Asset reaches strike price (spot >= strike)")
    print("  2. Contract reaches 60 DTE")
    print("  3. Expiration (if neither condition met)")
    print("="*80)
    
    backtest_results = []
    
    for idx, (_, straddle_row) in enumerate(unique_straddles.iterrows(), 1):
        ticker = straddle_row['Ticker']
        expiry_year = straddle_row['Expiry_Year']
        
        print(f"\n[{idx}/{len(unique_straddles)}] Backtesting {ticker} {expiry_year} LEAPS...")
        
        # Get data for this straddle
        straddle_data = df[(df['Ticker'] == ticker) & (df['Expiry_Year'] == expiry_year)].copy()
        straddle_data = straddle_data.sort_values('Date').reset_index(drop=True)
        
        if len(straddle_data) < 2:
            print(f"  ⚠️  Insufficient data ({len(straddle_data)} records)")
            continue
        
        # Trade setup
        entry_date = straddle_data.iloc[0]['Date']
        entry_dte = straddle_data.iloc[0]['Days_to_Expiry']
        entry_spot = straddle_data.iloc[0]['Spot_Price']
        strike_price = straddle_data.iloc[0]['Strike_Price']
        entry_straddle_value = straddle_data.iloc[0]['Straddle_Value']
        
        print(f"  Entry: {entry_date.strftime('%Y-%m-%d')} ({entry_dte} DTE)")
        print(f"  Entry Spot: ${entry_spot:.2f}, Strike: ${strike_price:.0f}")
        print(f"  Entry Straddle Value: ${entry_straddle_value:.2f}")
        
        # Short straddle: we collect premium upfront
        # P&L = entry_premium - current_straddle_value
        straddle_data['PnL'] = entry_straddle_value - straddle_data['Straddle_Value']
        straddle_data['PnL_Pct'] = (straddle_data['PnL'] / entry_straddle_value) * 100
        
        # Track max gain and max drawdown during the trade
        straddle_data['Running_Max_PnL'] = straddle_data['PnL'].expanding().max()
        straddle_data['Running_Min_PnL'] = straddle_data['PnL'].expanding().min()
        straddle_data['Drawdown'] = straddle_data['PnL'] - straddle_data['Running_Max_PnL']
        
        # Find exit point
        exit_reason = "Expiration"
        exit_idx = len(straddle_data) - 1  # Default to last day
        
        # Check for strike hit (spot >= strike)
        strike_hit = straddle_data['Spot_Price'] >= strike_price
        if strike_hit.any():
            first_strike_hit_idx = strike_hit.idxmax()
            exit_idx = first_strike_hit_idx
            exit_reason = "Strike Hit"
        
        # Check for 60 DTE exit (only if strike not hit first)
        elif (straddle_data['Days_to_Expiry'] <= 60).any():
            dte_60_idx = (straddle_data['Days_to_Expiry'] <= 60).idxmax()
            exit_idx = dte_60_idx
            exit_reason = "60 DTE"
        
        # Get exit data
        exit_data = straddle_data.iloc[exit_idx]
        exit_date = exit_data['Date']
        exit_dte = exit_data['Days_to_Expiry']
        exit_spot = exit_data['Spot_Price']
        exit_straddle_value = exit_data['Straddle_Value']
        
        # Calculate trade metrics
        trade_duration_days = (exit_date - entry_date).days
        actual_pnl = exit_data['PnL']
        actual_return_pct = exit_data['PnL_Pct']
        
        # Get max gain and max drawdown during the trade period
        trade_period_data = straddle_data.iloc[:exit_idx + 1]
        max_gain = trade_period_data['PnL'].max()
        max_gain_pct = (max_gain / entry_straddle_value) * 100
        max_drawdown = trade_period_data['Drawdown'].min()  # Most negative drawdown
        max_drawdown_pct = (max_drawdown / entry_straddle_value) * 100
        
        print(f"  Exit: {exit_date.strftime('%Y-%m-%d')} ({exit_dte} DTE) - {exit_reason}")
        print(f"  Exit Spot: ${exit_spot:.2f}")
        print(f"  Duration: {trade_duration_days} days")
        print(f"  Final P&L: ${actual_pnl:.2f} ({actual_return_pct:+.1f}%)")
        print(f"  Max Gain: ${max_gain:.2f} ({max_gain_pct:+.1f}%)")
        print(f"  Max Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:+.1f}%)")
        
        # Store results
        result = {
            'Ticker': ticker,
            'Expiry_Year': expiry_year,
            'Entry_Date': entry_date,
            'Exit_Date': exit_date,
            'Entry_DTE': entry_dte,
            'Exit_DTE': exit_dte,
            'Entry_Spot': entry_spot,
            'Exit_Spot': exit_spot,
            'Strike_Price': strike_price,
            'Entry_Straddle_Value': entry_straddle_value,
            'Exit_Straddle_Value': exit_straddle_value,
            'Trade_Duration_Days': trade_duration_days,
            'Exit_Reason': exit_reason,
            'Actual_PnL_Dollar': actual_pnl,
            'Actual_Return_Pct': actual_return_pct,
            'Max_Gain_Dollar': max_gain,
            'Max_Gain_Pct': max_gain_pct,
            'Max_Drawdown_Dollar': max_drawdown,
            'Max_Drawdown_Pct': max_drawdown_pct,
            'Entry_Moneyness': entry_spot / strike_price,
            'Exit_Moneyness': exit_spot / strike_price
        }
        
        backtest_results.append(result)
    
    # Convert to DataFrame
    if backtest_results:
        results_df = pd.DataFrame(backtest_results)
        
        # Generate summary statistics
        print_backtest_summary(results_df)
        
        # Save detailed results
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return results_df
    else:
        print("❌ No trades were executed")
        return None

def print_backtest_summary(results_df):
    """Print comprehensive summary of backtest results"""
    
    print(f"\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    
    total_trades = len(results_df)
    winning_trades = (results_df['Actual_Return_Pct'] > 0).sum()
    losing_trades = (results_df['Actual_Return_Pct'] < 0).sum()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total trades: {total_trades}")
    print(f"  Winning trades: {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
    print(f"  Losing trades: {losing_trades} ({losing_trades/total_trades*100:.1f}%)")
    
    print(f"\nRETURN STATISTICS:")
    print(f"  Mean return: {results_df['Actual_Return_Pct'].mean():+.1f}%")
    print(f"  Median return: {results_df['Actual_Return_Pct'].median():+.1f}%")
    print(f"  Best trade: {results_df['Actual_Return_Pct'].max():+.1f}%")
    print(f"  Worst trade: {results_df['Actual_Return_Pct'].min():+.1f}%")
    print(f"  Standard deviation: {results_df['Actual_Return_Pct'].std():.1f}%")
    
    print(f"\nMAX GAIN STATISTICS:")
    print(f"  Mean max gain: {results_df['Max_Gain_Pct'].mean():+.1f}%")
    print(f"  Median max gain: {results_df['Max_Gain_Pct'].median():+.1f}%")
    print(f"  Best max gain: {results_df['Max_Gain_Pct'].max():+.1f}%")
    
    print(f"\nMAX DRAWDOWN STATISTICS:")
    print(f"  Mean max drawdown: {results_df['Max_Drawdown_Pct'].mean():+.1f}%")
    print(f"  Median max drawdown: {results_df['Max_Drawdown_Pct'].median():+.1f}%")
    print(f"  Worst max drawdown: {results_df['Max_Drawdown_Pct'].min():+.1f}%")
    
    print(f"\nTRADE DURATION:")
    print(f"  Mean duration: {results_df['Trade_Duration_Days'].mean():.0f} days")
    print(f"  Median duration: {results_df['Trade_Duration_Days'].median():.0f} days")
    print(f"  Shortest trade: {results_df['Trade_Duration_Days'].min()} days")
    print(f"  Longest trade: {results_df['Trade_Duration_Days'].max()} days")
    
    print(f"\nEXIT REASONS:")
    exit_reasons = results_df['Exit_Reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} trades ({count/total_trades*100:.1f}%)")
    
    # Show best and worst trades
    print(f"\nBEST TRADES (by return %):")
    best_trades = results_df.nlargest(5, 'Actual_Return_Pct')
    for _, trade in best_trades.iterrows():
        print(f"  {trade['Ticker']} {trade['Expiry_Year']}: {trade['Actual_Return_Pct']:+.1f}% in {trade['Trade_Duration_Days']} days ({trade['Exit_Reason']})")
    
    print(f"\nWORST TRADES (by return %):")
    worst_trades = results_df.nsmallest(5, 'Actual_Return_Pct')
    for _, trade in worst_trades.iterrows():
        print(f"  {trade['Ticker']} {trade['Expiry_Year']}: {trade['Actual_Return_Pct']:+.1f}% in {trade['Trade_Duration_Days']} days ({trade['Exit_Reason']})")

def analyze_by_exit_reason(results_df):
    """Analyze performance by exit reason"""
    
    print(f"\n" + "="*60)
    print("PERFORMANCE BY EXIT REASON")
    print("="*60)
    
    for exit_reason in results_df['Exit_Reason'].unique():
        subset = results_df[results_df['Exit_Reason'] == exit_reason]
        
        print(f"\n{exit_reason.upper()} EXITS ({len(subset)} trades):")
        print(f"  Win rate: {(subset['Actual_Return_Pct'] > 0).mean()*100:.1f}%")
        print(f"  Mean return: {subset['Actual_Return_Pct'].mean():+.1f}%")
        print(f"  Mean duration: {subset['Trade_Duration_Days'].mean():.0f} days")
        print(f"  Mean max gain: {subset['Max_Gain_Pct'].mean():+.1f}%")
        print(f"  Mean max drawdown: {subset['Max_Drawdown_Pct'].mean():+.1f}%")

if __name__ == "__main__":
    # File paths
    input_file = "straddle/cleaned_straddle_values.csv"
    output_file = "straddle/short_straddle_backtest_results.csv"
    
    print("SHORT STRADDLE BACKTEST")
    print("="*80)
    print("Strategy: Sell straddles, exit on strike hit or 60 DTE")
    print("="*80)
    
    # Run the backtest
    results = run_short_straddle_backtest(input_file, output_file)
    
    if results is not None:
        # Additional analysis
        analyze_by_exit_reason(results)
        
        print(f"\n🎉 Backtest complete!")
        print(f"📊 Results saved to: {output_file}")
        print(f"📈 {len(results)} trades analyzed")
    else:
        print("❌ Backtest failed")