import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
from datetime import datetime, timedelta
import os
warnings.filterwarnings('ignore')

# ============================================================================
# BLACK-SCHOLES OPTION PRICING FUNCTIONS
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
    call_value = black_scholes_call(S, K, T, r, sigma)
    put_value = black_scholes_put(S, K, T, r, sigma)
    return call_value + put_value, call_value, put_value

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def load_and_process_data(file_path):
    """Load and process the daily stock price data"""
    
    print(f"📊 Loading data from: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get all asset columns (exclude date)
    asset_columns = [col for col in df.columns if col != 'date']
    
    # Clean asset names (remove ** prefix/suffix)
    clean_asset_names = {}
    for col in asset_columns:
        clean_name = col.replace('**', '').strip()
        clean_asset_names[col] = clean_name
    
    print(f"✅ Loaded {len(df)} daily records for {len(asset_columns)} assets")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df, asset_columns, clean_asset_names

def filter_data_from_2010(df):
    """Filter data to start from 2010 onwards"""
    
    start_date = pd.to_datetime('2010-01-01')
    filtered_df = df[df['date'] >= start_date].copy()
    
    print(f"📅 Filtered to 2010+ data: {len(filtered_df)} records")
    print(f"📅 New date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
    
    return filtered_df

def generate_expiry_dates(start_date, end_date, frequency_months=6):
    """Generate expiry dates every 6 months"""
    
    expiry_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Add 365 days for 1-year expiry from current date
        expiry_date = current_date + timedelta(days=365)
        if expiry_date <= end_date:
            expiry_dates.append({
                'start_date': current_date,
                'expiry_date': expiry_date
            })
        
        # Move forward by frequency_months
        current_date = current_date + timedelta(days=frequency_months * 30)  # Approximate
    
    return expiry_dates

# ============================================================================
# STRADDLE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_asset_straddles(df, asset_col, clean_name, expiry_dates, 
                          strike_multiplier=1.25, iv=0.25, risk_free_rate=0.04):
    """Analyze straddle performance for a single asset across multiple expiry periods"""
    
    print(f"\n🔍 Analyzing straddles for {clean_name} ({asset_col})...")
    
    all_straddle_data = []
    straddle_summary = []
    
    for i, expiry_info in enumerate(expiry_dates):
        start_date = expiry_info['start_date']
        expiry_date = expiry_info['expiry_date']
        
        # Get price data for this period
        period_data = df[(df['date'] >= start_date) & (df['date'] <= expiry_date)].copy()
        
        if len(period_data) < 100:  # Need at least 100 days of data
            continue
        
        # Check if asset has valid data for this period
        period_data = period_data.dropna(subset=[asset_col])
        if len(period_data) == 0:
            continue
        
        # Get entry spot price (first valid price in period)
        entry_spot = period_data[asset_col].iloc[0]
        if pd.isna(entry_spot) or entry_spot <= 0:
            continue
        
        # Calculate strike price (25% above entry spot)
        strike_price = entry_spot * strike_multiplier
        
        print(f"  📅 Period {i+1}: {start_date.strftime('%Y-%m-%d')} to {expiry_date.strftime('%Y-%m-%d')}")
        print(f"      Entry spot: ${entry_spot:.2f}, Strike: ${strike_price:.2f}")
        
        # Calculate straddle values for each day in the period
        period_straddle_data = []
        
        for _, row in period_data.iterrows():
            current_date = row['date']
            current_spot = row[asset_col]
            
            if pd.isna(current_spot) or current_spot <= 0:
                continue
            
            # Calculate days to expiry
            days_to_expiry = (expiry_date - current_date).days
            time_to_expiry = max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day
            
            # Calculate straddle value
            straddle_value, call_value, put_value = calculate_straddle_value(
                current_spot, strike_price, time_to_expiry, risk_free_rate, iv
            )
            
            # Calculate moneyness
            moneyness = current_spot / strike_price
            
            period_straddle_data.append({
                'date': current_date,
                'spot_price': current_spot,
                'strike_price': strike_price,
                'days_to_expiry': days_to_expiry,
                'time_to_expiry': time_to_expiry,
                'straddle_value': straddle_value,
                'call_value': call_value,
                'put_value': put_value,
                'moneyness': moneyness,
                'period': i + 1,
                'entry_spot': entry_spot,
                'asset': clean_name
            })
        
        if len(period_straddle_data) == 0:
            continue
        
        # Convert to DataFrame for easier analysis
        period_df = pd.DataFrame(period_straddle_data)
        
        # Calculate short straddle returns
        entry_straddle = period_df['straddle_value'].iloc[0]
        period_df['short_return_pct'] = ((entry_straddle - period_df['straddle_value']) / entry_straddle) * 100
        
        # Calculate summary statistics for this period
        final_return = period_df['short_return_pct'].iloc[-1]
        max_return = period_df['short_return_pct'].max()
        min_return = period_df['short_return_pct'].min()
        final_spot = period_df['spot_price'].iloc[-1]
        total_move_pct = ((final_spot - entry_spot) / entry_spot) * 100
        
        straddle_summary.append({
            'period': i + 1,
            'start_date': start_date,
            'expiry_date': expiry_date,
            'entry_spot': entry_spot,
            'final_spot': final_spot,
            'strike_price': strike_price,
            'entry_straddle': entry_straddle,
            'final_straddle': period_df['straddle_value'].iloc[-1],
            'final_return_pct': final_return,
            'max_return_pct': max_return,
            'min_return_pct': min_return,
            'total_move_pct': total_move_pct,
            'days_tracked': len(period_df),
            'asset': clean_name
        })
        
        # Add to master data
        all_straddle_data.extend(period_straddle_data)
        
        print(f"      ✅ Final short straddle return: {final_return:+.1f}%")
    
    print(f"  📊 Completed {len(straddle_summary)} straddle periods for {clean_name}")
    
    return all_straddle_data, straddle_summary

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_asset_straddle_analysis(all_straddle_data, straddle_summary, asset_name, save_plots=True):
    """Create comprehensive straddle analysis plot for a single asset"""
    
    if len(all_straddle_data) == 0:
        print(f"⚠️  No data to plot for {asset_name}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_straddle_data)
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Straddle Values Over Time
    ax1_colors = plt.cm.viridis(np.linspace(0, 1, len(df['period'].unique())))
    
    for i, period in enumerate(sorted(df['period'].unique())):
        period_data = df[df['period'] == period].copy()
        period_data = period_data.sort_values('date')
        
        ax1.plot(period_data['days_to_expiry'], period_data['straddle_value'], 
                linewidth=2, alpha=0.8, color=ax1_colors[i], 
                label=f'Period {period} (Entry: ${period_data["entry_spot"].iloc[0]:.1f})')
    
    ax1.set_xlabel('Days to Expiry')
    ax1.set_ylabel('Straddle Value ($)')
    ax1.set_title(f'{asset_name} - Straddle Values Over Time (25% OTM, 25% IV)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Higher DTE on left, expiry on right
    
    # Plot 2: Moneyness Evolution
    for i, period in enumerate(sorted(df['period'].unique())):
        period_data = df[df['period'] == period].copy()
        period_data = period_data.sort_values('date')
        
        ax2.plot(period_data['days_to_expiry'], period_data['moneyness'], 
                linewidth=2, alpha=0.8, color=ax1_colors[i],
                label=f'Period {period}')
    
    # Add reference lines for moneyness
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM (1.0)')
    ax2.axhline(y=0.8, color='orange', linestyle=':', alpha=0.6, linewidth=1, label='Deep ITM Put (0.8)')
    ax2.axhline(y=1.2, color='green', linestyle=':', alpha=0.6, linewidth=1, label='Deep ITM Call (1.2)')
    ax2.fill_between([0, 365], 0.9, 1.1, alpha=0.2, color='yellow', label='Near ATM Zone')
    
    ax2.set_xlabel('Days to Expiry')
    ax2.set_ylabel('Moneyness (Spot/Strike)')
    ax2.set_title(f'{asset_name} - Moneyness Evolution (How Close to Strike)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    ax2.set_ylim(0.5, 1.8)  # Reasonable range for moneyness
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'{asset_name.lower()}_straddle_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
    
    plt.show()

def create_summary_report(all_summaries):
    """Create summary report across all assets"""
    
    if len(all_summaries) == 0:
        print("❌ No summary data to report")
        return
    
    summary_df = pd.DataFrame(all_summaries)
    
    print(f"\n" + "="*80)
    print("📊 COMPREHENSIVE STRADDLE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Assets Analyzed: {summary_df['asset'].nunique()}")
    print(f"Total Straddle Periods: {len(summary_df)}")
    print(f"Date Range: {summary_df['start_date'].min()} to {summary_df['expiry_date'].max()}")
    
    # Overall performance statistics
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"  Average Final Return: {summary_df['final_return_pct'].mean():+.1f}%")
    print(f"  Median Final Return: {summary_df['final_return_pct'].median():+.1f}%")
    print(f"  Best Performance: {summary_df['final_return_pct'].max():+.1f}%")
    print(f"  Worst Performance: {summary_df['final_return_pct'].min():+.1f}%")
    print(f"  Win Rate: {(summary_df['final_return_pct'] > 0).mean()*100:.1f}%")
    
    # Performance by asset
    print(f"\n📈 PERFORMANCE BY ASSET:")
    asset_perf = summary_df.groupby('asset').agg({
        'final_return_pct': ['count', 'mean', 'std'],
        'total_move_pct': 'mean'
    }).round(2)
    
    asset_perf.columns = ['Count', 'Avg_Return', 'Std_Return', 'Avg_Price_Move']
    asset_perf = asset_perf.sort_values('Avg_Return', ascending=False)
    
    print(asset_perf.head(10).to_string())
    
    # Save summary to CSV
    summary_df.to_csv('straddle_analysis_summary.csv', index=False)
    asset_perf.to_csv('asset_performance_summary.csv')
    
    print(f"\n💾 Summary data saved:")
    print(f"  📄 straddle_analysis_summary.csv ({len(summary_df)} records)")
    print(f"  📄 asset_performance_summary.csv ({len(asset_perf)} assets)")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to process all assets and generate comprehensive straddle datasets"""
    
    print("🚀 REAL ASSET STRADDLE DATA GENERATOR")
    print("="*80)
    print("Calculating comprehensive straddle datasets using real market data")
    print("Strategy: 25% OTM strikes, 365 DTE, overlapping 6-month entries")
    print("Focus: Rich datasets for future analysis and visualization")
    print("="*80)
    
    # Configuration
    data_file = 'DAILY_INTERPOLATED_STOCK_DATA.csv'
    strike_multiplier = 1.25  # 25% above spot
    iv = 0.25  # 25% implied volatility
    risk_free_rate = 0.04  # 4% risk-free rate
    frequency_months = 6  # New expiry every 6 months
    
    # Step 1: Load and process data
    try:
        df, asset_columns, clean_asset_names = load_and_process_data(data_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_file}")
        print("   Please make sure the file exists in the current directory")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Filter to 2010+ data
    df = filter_data_from_2010(df)
    
    if len(df) == 0:
        print("❌ No data available from 2010 onwards")
        return
    
    # Step 3: Generate expiry dates
    start_date = df['date'].min()
    end_date = df['date'].max()
    expiry_dates = generate_expiry_dates(start_date, end_date, frequency_months)
    
    print(f"📅 Generated {len(expiry_dates)} overlapping expiry periods")
    
    # Step 4: Process each asset and build comprehensive dataset
    all_summaries = []
    all_master_data = []
    successful_assets = 0
    
    print(f"\n🔄 Processing {len(asset_columns)} assets - generating comprehensive datasets...")
    
    for i, asset_col in enumerate(asset_columns):
        clean_name = clean_asset_names[asset_col]
        
        # Progress indicator
        progress = (i + 1) / len(asset_columns) * 100
        print(f"\n[{progress:5.1f}%] Processing {clean_name} ({i+1}/{len(asset_columns)})")
        
        try:
            # Check if asset has sufficient data
            valid_data = df[asset_col].dropna()
            if len(valid_data) < 365:  # Need at least 1 year of data
                print(f"         ⚠️  Insufficient data ({len(valid_data)} records) - skipping")
                continue
            
            # Analyze straddles for this asset
            all_straddle_data, straddle_summary = analyze_asset_straddles(
                df, asset_col, clean_name, expiry_dates, 
                strike_multiplier, iv, risk_free_rate
            )
            
            if len(straddle_summary) == 0:
                print(f"         ⚠️  No valid straddle periods - skipping")
                continue
            
            # Save detailed dataset for this asset
            save_detailed_straddle_data(all_straddle_data, clean_name)
            
            # Add to master dataset
            all_master_data.extend(all_straddle_data)
            
            # Add to summary
            all_summaries.extend(straddle_summary)
            successful_assets += 1
            
            # Brief summary for this asset
            avg_return = np.mean([s['final_return_pct'] for s in straddle_summary])
            total_records = len(all_straddle_data)
            print(f"         ✅ {len(straddle_summary)} periods, {total_records:,} data points, avg return: {avg_return:+.1f}%")
            
        except Exception as e:
            print(f"         ❌ Error: {str(e)[:100]}...")
            continue
    
    # Step 5: Save master dataset and create comprehensive summary
    print(f"\n" + "="*80)
    print(f"🎉 DATASET GENERATION COMPLETE!")
    print(f"✅ Successfully processed {successful_assets}/{len(asset_columns)} assets")
    
    if len(all_master_data) > 0:
        # Save the comprehensive master dataset
        master_filename = save_master_dataset(all_master_data)
        
        # Create summary reports
        create_summary_report(all_summaries)
        
        # Final statistics
        total_records = len(all_master_data)
        total_periods = len(all_summaries)
        all_returns = [s['final_return_pct'] for s in all_summaries]
        
        print(f"\n📊 COMPREHENSIVE DATASET STATISTICS:")
        print(f"   📈 Total data points: {total_records:,}")
        print(f"   📅 Total straddle periods: {total_periods}")
        print(f"   🏢 Assets with data: {successful_assets}")
        print(f"   📊 Average return: {np.mean(all_returns):+.1f}%")
        print(f"   🎯 Win rate: {(np.array(all_returns) > 0).mean()*100:.1f}%")
        print(f"   📈 Best performance: {max(all_returns):+.1f}%")
        print(f"   📉 Worst performance: {min(all_returns):+.1f}%")
    else:
        print("❌ No valid straddle data generated")
    
    print(f"\n📁 DATASET FILES GENERATED:")
    print(f"   📊 master_straddle_dataset.csv - Comprehensive dataset with all data points")
    print(f"   📄 [asset]_detailed_straddle_data.csv - Individual asset datasets ({successful_assets} files)")
    print(f"   📊 straddle_analysis_summary.csv - Period-by-period summary")
    print(f"   📊 asset_performance_summary.csv - Asset performance rankings")
    print(f"\n✅ Rich datasets ready for analysis and visualization!")
    print(f"💡 Use these datasets to create custom plots and analysis in a separate script")
    print("="*80)

if __name__ == "__main__":
    main()