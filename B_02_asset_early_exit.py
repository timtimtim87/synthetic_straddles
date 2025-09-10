import pandas as pd
import numpy as np
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
# FOLDER MANAGEMENT FUNCTIONS
# ============================================================================

def create_output_folders():
    """Create organized folder structure for outputs"""
    
    # Get current timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    main_output_dir = f"straddle_early_exit_{timestamp}"
    
    # Create subdirectories
    folders = {
        'main': main_output_dir,
        'individual': os.path.join(main_output_dir, 'individual_assets'),
        'summaries': os.path.join(main_output_dir, 'summaries'),
        'master': os.path.join(main_output_dir, 'master_data'),
        'analysis': os.path.join(main_output_dir, 'exit_analysis')
    }
    
    # Create all directories
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"📁 Created output folder structure: {main_output_dir}/")
    print(f"   📊 Master data: {folders['master']}")
    print(f"   📄 Individual assets: {folders['individual']}")
    print(f"   📋 Summary reports: {folders['summaries']}")
    print(f"   🚪 Exit analysis: {folders['analysis']}")
    
    return folders

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def load_and_process_data(file_path):
    """Load and process the daily stock price data"""
    
    print(f"📊 Loading data from: {file_path}")
    
    try:
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
        print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df, asset_columns, clean_asset_names
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def filter_data_from_2010(df):
    """Filter data to start from 2010 onwards"""
    
    start_date = pd.to_datetime('2010-01-01')
    filtered_df = df[df['date'] >= start_date].copy()
    
    print(f"📅 Filtered to 2010+ data: {len(filtered_df)} records")
    if len(filtered_df) > 0:
        print(f"📅 New date range: {filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}")
    
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
        year = current_date.year
        month = current_date.month + frequency_months
        
        if month > 12:
            year += (month - 1) // 12
            month = ((month - 1) % 12) + 1
        
        try:
            current_date = pd.to_datetime(f"{year}-{month:02d}-01")
        except:
            # If date is invalid, move to next valid date
            if month == 2:
                current_date = pd.to_datetime(f"{year}-03-01")
            else:
                current_date = pd.to_datetime(f"{year}-{month:02d}-01")
    
    return expiry_dates

# ============================================================================
# EARLY EXIT STRADDLE ANALYSIS FUNCTIONS
# ============================================================================

def check_early_exit_condition(current_spot, strike_price, exit_buffer=0.02):
    """
    Check if early exit condition is met (spot crosses strike)
    
    Parameters:
    - current_spot: Current spot price
    - strike_price: Strike price of straddle
    - exit_buffer: Small buffer around strike (2% default)
    
    Returns:
    - should_exit: Boolean
    - exit_reason: String description
    """
    
    # Since we're short straddles with strikes ABOVE entry spot,
    # we exit when spot moves UP and crosses the strike
    upper_exit_threshold = strike_price * (1 - exit_buffer)  # Exit slightly before strike
    
    if current_spot >= upper_exit_threshold:
        return True, f"Strike Breach: Spot ${current_spot:.2f} >= ${upper_exit_threshold:.2f} (Strike: ${strike_price:.2f})"
    
    return False, None

def analyze_asset_straddles_with_early_exit(df, asset_col, clean_name, expiry_dates, 
                                           strike_multiplier=1.25, iv=0.25, risk_free_rate=0.04,
                                           exit_buffer=0.02):
    """Analyze straddle performance with early exit conditions"""
    
    print(f"\n🔍 Analyzing straddles with early exit for {clean_name} ({asset_col})...")
    print(f"    Exit condition: Spot crosses {(1-exit_buffer)*100:.0f}% of strike price")
    
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
        
        # Track position status
        position_active = True
        exit_day = None
        exit_reason = "Expiry"
        exit_spot = None
        exit_straddle_value = None
        
        # Calculate straddle values for each day in the period
        period_straddle_data = []
        
        for day_idx, (_, row) in enumerate(period_data.iterrows()):
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
            
            # Check for early exit condition (only if position is still active)
            if position_active:
                should_exit, exit_desc = check_early_exit_condition(
                    current_spot, strike_price, exit_buffer
                )
                
                if should_exit:
                    position_active = False
                    exit_day = day_idx
                    exit_reason = exit_desc
                    exit_spot = current_spot
                    exit_straddle_value = straddle_value
                    print(f"      🚪 EARLY EXIT on day {day_idx}: {exit_desc}")
            
            period_straddle_data.append({
                'date': current_date,
                'day_index': day_idx,
                'spot_price': current_spot,
                'strike_price': strike_price,
                'days_to_expiry': days_to_expiry,
                'time_to_expiry': time_to_expiry,
                'straddle_value': straddle_value,
                'call_value': call_value,
                'put_value': put_value,
                'moneyness': moneyness,
                'position_active': position_active,
                'period': i + 1,
                'entry_spot': entry_spot,
                'asset': clean_name
            })
            
            # If position was closed early, stop tracking
            if not position_active:
                break
        
        if len(period_straddle_data) == 0:
            continue
        
        # Convert to DataFrame for easier analysis
        period_df = pd.DataFrame(period_straddle_data)
        
        # Calculate short straddle returns
        entry_straddle = period_df['straddle_value'].iloc[0]
        period_df['short_return_pct'] = ((entry_straddle - period_df['straddle_value']) / entry_straddle) * 100
        
        # Determine final values
        if exit_day is not None:
            # Early exit
            final_return = period_df['short_return_pct'].iloc[-1]
            final_spot = exit_spot
            final_straddle = exit_straddle_value
            days_held = exit_day + 1
            early_exit = True
        else:
            # Held to expiry
            final_return = period_df['short_return_pct'].iloc[-1]
            final_spot = period_df['spot_price'].iloc[-1]
            final_straddle = period_df['straddle_value'].iloc[-1]
            days_held = len(period_df)
            early_exit = False
        
        # Calculate additional metrics
        max_return = period_df['short_return_pct'].max()
        min_return = period_df['short_return_pct'].min()
        total_move_pct = ((final_spot - entry_spot) / entry_spot) * 100
        
        # Create summary record
        straddle_summary.append({
            'period': i + 1,
            'start_date': start_date,
            'expiry_date': expiry_date,
            'entry_spot': entry_spot,
            'final_spot': final_spot,
            'strike_price': strike_price,
            'entry_straddle': entry_straddle,
            'final_straddle': final_straddle,
            'final_return_pct': final_return,
            'max_return_pct': max_return,
            'min_return_pct': min_return,
            'total_move_pct': total_move_pct,
            'days_held': days_held,
            'days_tracked': len(period_df),
            'early_exit': early_exit,
            'exit_reason': exit_reason,
            'exit_day': exit_day,
            'days_to_natural_expiry': 365,
            'time_saved_days': 365 - days_held if early_exit else 0,
            'asset': clean_name
        })
        
        # Add to master data
        all_straddle_data.extend(period_straddle_data)
        
        exit_status = "EARLY EXIT" if early_exit else "HELD TO EXPIRY"
        print(f"      ✅ {exit_status}: {final_return:+.1f}% return in {days_held} days")
    
    print(f"  📊 Completed {len(straddle_summary)} straddle periods for {clean_name}")
    
    return all_straddle_data, straddle_summary

# ============================================================================
# END OF PART 1
# ============================================================================



# ============================================================================
# DATA SAVING FUNCTIONS
# ============================================================================

def save_detailed_straddle_data(all_straddle_data, asset_name, folders):
    """Save detailed straddle data for individual asset"""
    
    if len(all_straddle_data) == 0:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_straddle_data)
    
    # Create filename
    filename = f'{asset_name.lower().replace(" ", "_")}_early_exit_straddle_data.csv'
    full_path = os.path.join(folders['individual'], filename)
    
    # Save to CSV
    df.to_csv(full_path, index=False)
    
    print(f"  💾 Saved detailed data: {filename} ({len(df)} records)")

def save_master_dataset(all_master_data, folders):
    """Save the comprehensive master dataset"""
    
    if len(all_master_data) == 0:
        return None
    
    # Convert to DataFrame
    master_df = pd.DataFrame(all_master_data)
    
    # Create filename
    filename = 'master_early_exit_straddle_dataset.csv'
    full_path = os.path.join(folders['master'], filename)
    
    # Save to CSV
    master_df.to_csv(full_path, index=False)
    
    print(f"💾 Saved master dataset: {full_path} ({len(master_df):,} records)")
    
    return filename

def create_early_exit_analysis(all_summaries, folders):
    """Create comprehensive early exit analysis"""
    
    if len(all_summaries) == 0:
        print("❌ No summary data for early exit analysis")
        return
    
    summary_df = pd.DataFrame(all_summaries)
    
    print(f"\n" + "="*80)
    print("📊 EARLY EXIT STRADDLE ANALYSIS")
    print("="*80)
    
    # Basic statistics
    total_periods = len(summary_df)
    early_exits = summary_df['early_exit'].sum()
    held_to_expiry = total_periods - early_exits
    early_exit_rate = (early_exits / total_periods) * 100
    
    print(f"Total Straddle Periods: {total_periods}")
    print(f"Early Exits: {early_exits} ({early_exit_rate:.1f}%)")
    print(f"Held to Expiry: {held_to_expiry} ({100-early_exit_rate:.1f}%)")
    
    # Performance comparison
    early_exit_data = summary_df[summary_df['early_exit'] == True]
    expiry_data = summary_df[summary_df['early_exit'] == False]
    
    print(f"\n🚪 EARLY EXIT PERFORMANCE:")
    if len(early_exit_data) > 0:
        print(f"  Count: {len(early_exit_data)}")
        print(f"  Average Return: {early_exit_data['final_return_pct'].mean():+.1f}%")
        print(f"  Median Return: {early_exit_data['final_return_pct'].median():+.1f}%")
        print(f"  Average Days Held: {early_exit_data['days_held'].mean():.0f}")
        print(f"  Average Time Saved: {early_exit_data['time_saved_days'].mean():.0f} days")
        print(f"  Win Rate: {(early_exit_data['final_return_pct'] > 0).mean()*100:.1f}%")
    
    print(f"\n📅 HELD TO EXPIRY PERFORMANCE:")
    if len(expiry_data) > 0:
        print(f"  Count: {len(expiry_data)}")
        print(f"  Average Return: {expiry_data['final_return_pct'].mean():+.1f}%")
        print(f"  Median Return: {expiry_data['final_return_pct'].median():+.1f}%")
        print(f"  Average Days Held: {expiry_data['days_held'].mean():.0f}")
        print(f"  Win Rate: {(expiry_data['final_return_pct'] > 0).mean()*100:.1f}%")
    
    # Asset-level analysis
    print(f"\n📈 EARLY EXIT BY ASSET:")
    asset_exit_analysis = summary_df.groupby('asset').agg({
        'early_exit': ['count', 'sum', 'mean'],
        'final_return_pct': 'mean',
        'days_held': 'mean',
        'time_saved_days': 'mean'
    }).round(2)
    
    asset_exit_analysis.columns = ['Total_Periods', 'Early_Exits', 'Early_Exit_Rate', 'Avg_Return', 'Avg_Days_Held', 'Avg_Time_Saved']
    asset_exit_analysis = asset_exit_analysis.sort_values('Early_Exit_Rate', ascending=False)
    
    print(asset_exit_analysis.head(15).to_string())
    
    # Time analysis
    if len(early_exit_data) > 0:
        print(f"\n⏰ EARLY EXIT TIMING ANALYSIS:")
        print(f"  Earliest Exit: {early_exit_data['days_held'].min()} days")
        print(f"  Latest Exit: {early_exit_data['days_held'].max()} days")
        print(f"  Most Common Exit Time: {early_exit_data['days_held'].mode().iloc[0]} days")
        
        # Exit timing buckets
        exit_buckets = pd.cut(early_exit_data['days_held'], 
                             bins=[0, 30, 90, 180, 270, 365], 
                             labels=['0-30d', '31-90d', '91-180d', '181-270d', '271-365d'])
        bucket_analysis = exit_buckets.value_counts().sort_index()
        
        print(f"\n  Exit Timing Distribution:")
        for bucket, count in bucket_analysis.items():
            pct = (count / len(early_exit_data)) * 100
            print(f"    {bucket}: {count} exits ({pct:.1f}%)")
    
    # Save analysis files
    summary_path = os.path.join(folders['summaries'], 'early_exit_straddle_summary.csv')
    asset_analysis_path = os.path.join(folders['analysis'], 'asset_early_exit_analysis.csv')
    
    summary_df.to_csv(summary_path, index=False)
    asset_exit_analysis.to_csv(asset_analysis_path)
    
    # Create detailed early exit analysis
    if len(early_exit_data) > 0:
        early_exit_detail_path = os.path.join(folders['analysis'], 'early_exit_details.csv')
        early_exit_data.to_csv(early_exit_detail_path, index=False)
        
        # Create exit timing analysis
        exit_timing_path = os.path.join(folders['analysis'], 'exit_timing_analysis.csv')
        timing_analysis = early_exit_data.groupby('days_held').agg({
            'final_return_pct': ['count', 'mean', 'std'],
            'time_saved_days': 'mean'
        }).round(2)
        timing_analysis.to_csv(exit_timing_path)
    
    print(f"\n💾 Early exit analysis saved:")
    print(f"  📄 {summary_path} ({len(summary_df)} records)")
    print(f"  📄 {asset_analysis_path} ({len(asset_exit_analysis)} assets)")
    if len(early_exit_data) > 0:
        print(f"  📄 {early_exit_detail_path} ({len(early_exit_data)} early exits)")
        print(f"  📄 {exit_timing_path} (timing analysis)")

def save_run_metadata(folders, successful_assets, total_assets, all_summaries, all_master_data):
    """Save metadata about this analysis run"""
    
    if len(all_summaries) == 0:
        return
    
    summary_df = pd.DataFrame(all_summaries)
    early_exits = summary_df['early_exit'].sum()
    
    metadata = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_assets_processed': total_assets,
        'successful_assets': successful_assets,
        'success_rate_pct': (successful_assets / total_assets * 100) if total_assets > 0 else 0,
        'total_straddle_periods': len(all_summaries),
        'total_data_points': len(all_master_data),
        'early_exits': int(early_exits),
        'early_exit_rate_pct': (early_exits / len(all_summaries) * 100) if len(all_summaries) > 0 else 0,
        'date_range_start': min([s['start_date'] for s in all_summaries]).strftime('%Y-%m-%d') if all_summaries else None,
        'date_range_end': max([s['expiry_date'] for s in all_summaries]).strftime('%Y-%m-%d') if all_summaries else None,
        'strategy_parameters': {
            'strike_multiplier': 1.25,
            'implied_volatility': 0.25,
            'risk_free_rate': 0.04,
            'days_to_expiry': 365,
            'frequency_months': 6,
            'exit_buffer': 0.02
        }
    }
    
    if all_summaries:
        all_returns = [s['final_return_pct'] for s in all_summaries]
        early_exit_data = summary_df[summary_df['early_exit'] == True]
        expiry_data = summary_df[summary_df['early_exit'] == False]
        
        metadata.update({
            'average_return_pct': np.mean(all_returns),
            'median_return_pct': np.median(all_returns),
            'best_return_pct': max(all_returns),
            'worst_return_pct': min(all_returns),
            'win_rate_pct': (np.array(all_returns) > 0).mean() * 100,
            'early_exit_avg_return_pct': early_exit_data['final_return_pct'].mean() if len(early_exit_data) > 0 else 0,
            'expiry_avg_return_pct': expiry_data['final_return_pct'].mean() if len(expiry_data) > 0 else 0,
            'avg_days_held_early_exit': early_exit_data['days_held'].mean() if len(early_exit_data) > 0 else 0,
            'avg_time_saved_days': early_exit_data['time_saved_days'].mean() if len(early_exit_data) > 0 else 0
        })
    
    # Save metadata
    metadata_path = os.path.join(folders['main'], 'run_metadata.csv')
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(metadata_path, index=False)
    
    # Also save as readable text file
    metadata_txt_path = os.path.join(folders['main'], 'run_summary.txt')
    with open(metadata_txt_path, 'w') as f:
        f.write("EARLY EXIT STRADDLE ANALYSIS RUN SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run Time: {metadata['run_timestamp']}\n")
        f.write(f"Assets Processed: {successful_assets}/{total_assets} ({metadata['success_rate_pct']:.1f}%)\n")
        f.write(f"Total Periods: {metadata['total_straddle_periods']}\n")
        f.write(f"Early Exits: {metadata['early_exits']} ({metadata['early_exit_rate_pct']:.1f}%)\n")
        f.write(f"Total Data Points: {metadata['total_data_points']:,}\n")
        
        if all_summaries:
            f.write(f"\nPerformance Summary:\n")
            f.write(f"  Overall Average Return: {metadata['average_return_pct']:+.1f}%\n")
            f.write(f"  Early Exit Average Return: {metadata['early_exit_avg_return_pct']:+.1f}%\n")
            f.write(f"  Expiry Average Return: {metadata['expiry_avg_return_pct']:+.1f}%\n")
            f.write(f"  Overall Win Rate: {metadata['win_rate_pct']:.1f}%\n")
            f.write(f"  Average Days Held (Early Exit): {metadata['avg_days_held_early_exit']:.0f}\n")
            f.write(f"  Average Time Saved: {metadata['avg_time_saved_days']:.0f} days\n")
        
        f.write(f"\nStrategy Parameters:\n")
        for key, value in metadata['strategy_parameters'].items():
            f.write(f"  {key}: {value}\n")
    
    print(f"📋 Saved run metadata: {metadata_path}")
    print(f"📋 Saved run summary: {metadata_txt_path}")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to process all assets with early exit conditions"""
    
    print("🚀 STRADDLE EARLY EXIT ANALYSIS")
    print("="*80)
    print("Testing early exit strategy: Close when spot crosses strike price")
    print("Compare performance: Early exit vs Hold to expiry")
    print("Strategy: 25% OTM strikes, 365 DTE, exit when spot hits 98% of strike")
    print("="*80)
    
    # Configuration
    data_file = 'DAILY_INTERPOLATED_STOCK_DATA.csv'
    strike_multiplier = 1.25  # 25% above spot
    iv = 0.25  # 25% implied volatility
    risk_free_rate = 0.04  # 4% risk-free rate
    frequency_months = 6  # New expiry every 6 months
    exit_buffer = 0.02  # Exit when spot hits 98% of strike (2% buffer)
    
    # Step 1: Create organized folder structure
    folders = create_output_folders()
    
    # Step 2: Load and process data
    try:
        df, asset_columns, clean_asset_names = load_and_process_data(data_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_file}")
        print("   Please make sure the file exists in the current directory")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 3: Filter to 2010+ data
    df = filter_data_from_2010(df)
    
    if len(df) == 0:
        print("❌ No data available from 2010 onwards")
        return
    
    # Step 4: Generate expiry dates
    start_date = df['date'].min()
    end_date = df['date'].max()
    expiry_dates = generate_expiry_dates(start_date, end_date, frequency_months)
    
    print(f"📅 Generated {len(expiry_dates)} overlapping expiry periods")
    
    # Step 5: Process each asset with early exit analysis
    all_summaries = []
    all_master_data = []
    successful_assets = 0
    
    print(f"\n🔄 Processing {len(asset_columns)} assets with early exit conditions...")
    
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
            
            # Analyze straddles with early exit for this asset
            all_straddle_data, straddle_summary = analyze_asset_straddles_with_early_exit(
                df, asset_col, clean_name, expiry_dates, 
                strike_multiplier, iv, risk_free_rate, exit_buffer
            )
            
            if len(straddle_summary) == 0:
                print(f"         ⚠️  No valid straddle periods - skipping")
                continue
            
            # Save detailed dataset for this asset
            save_detailed_straddle_data(all_straddle_data, clean_name, folders)
            
            # Add to master dataset
            all_master_data.extend(all_straddle_data)
            
            # Add to summary
            all_summaries.extend(straddle_summary)
            successful_assets += 1
            
            # Brief summary for this asset
            early_exits = sum(1 for s in straddle_summary if s['early_exit'])
            total_periods = len(straddle_summary)
            early_exit_rate = (early_exits / total_periods) * 100
            avg_return = np.mean([s['final_return_pct'] for s in straddle_summary])
            total_records = len(all_straddle_data)
            
            print(f"         ✅ {total_periods} periods, {early_exits} early exits ({early_exit_rate:.0f}%)")
            print(f"            {total_records:,} data points, avg return: {avg_return:+.1f}%")
            
        except Exception as e:
            print(f"         ❌ Error: {str(e)[:100]}...")
            continue
    
    # Step 6: Save master dataset and create comprehensive analysis
    print(f"\n" + "="*80)
    print(f"🎉 EARLY EXIT ANALYSIS COMPLETE!")
    print(f"✅ Successfully processed {successful_assets}/{len(asset_columns)} assets")
    
    if len(all_master_data) > 0:
        # Save the comprehensive master dataset
        master_filename = save_master_dataset(all_master_data, folders)
        
        # Create comprehensive early exit analysis
        create_early_exit_analysis(all_summaries, folders)
        
        # Save run metadata
        save_run_metadata(folders, successful_assets, len(asset_columns), all_summaries, all_master_data)
        
        # Final statistics
        summary_df = pd.DataFrame(all_summaries)
        total_records = len(all_master_data)
        total_periods = len(all_summaries)
        early_exits = summary_df['early_exit'].sum()
        early_exit_rate = (early_exits / total_periods) * 100
        
        all_returns = [s['final_return_pct'] for s in all_summaries]
        early_exit_data = summary_df[summary_df['early_exit'] == True]
        expiry_data = summary_df[summary_df['early_exit'] == False]
        
        print(f"\n📊 COMPREHENSIVE RESULTS:")
        print(f"   📈 Total data points: {total_records:,}")
        print(f"   📅 Total straddle periods: {total_periods}")
        print(f"   🚪 Early exits: {early_exits} ({early_exit_rate:.1f}%)")
        print(f"   📅 Held to expiry: {total_periods - early_exits} ({100-early_exit_rate:.1f}%)")
        print(f"   🏢 Assets with data: {successful_assets}")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   📊 Overall average return: {np.mean(all_returns):+.1f}%")
        print(f"   🎯 Overall win rate: {(np.array(all_returns) > 0).mean()*100:.1f}%")
        
        if len(early_exit_data) > 0:
            print(f"   🚪 Early exit average return: {early_exit_data['final_return_pct'].mean():+.1f}%")
            print(f"   🚪 Early exit win rate: {(early_exit_data['final_return_pct'] > 0).mean()*100:.1f}%")
            print(f"   🚪 Average days held (early exit): {early_exit_data['days_held'].mean():.0f}")
            print(f"   🚪 Average time saved: {early_exit_data['time_saved_days'].mean():.0f} days")
        
        if len(expiry_data) > 0:
            print(f"   📅 Expiry average return: {expiry_data['final_return_pct'].mean():+.1f}%")
            print(f"   📅 Expiry win rate: {(expiry_data['final_return_pct'] > 0).mean()*100:.1f}%")
            print(f"   📅 Average days held (expiry): {expiry_data['days_held'].mean():.0f}")
        
        # Identify most volatile assets (high early exit rates)
        asset_exit_rates = summary_df.groupby('asset')['early_exit'].agg(['count', 'sum', 'mean']).reset_index()
        asset_exit_rates.columns = ['asset', 'total_periods', 'early_exits', 'early_exit_rate']
        asset_exit_rates = asset_exit_rates[asset_exit_rates['total_periods'] >= 5]  # At least 5 periods
        high_exit_assets = asset_exit_rates.nlargest(10, 'early_exit_rate')
        
        print(f"\n🔥 TOP 10 ASSETS BY EARLY EXIT RATE:")
        for _, row in high_exit_assets.iterrows():
            print(f"   {row['asset']:>6}: {row['early_exit_rate']*100:5.1f}% ({row['early_exits']:.0f}/{row['total_periods']:.0f})")
        
    else:
        print("❌ No valid straddle data generated")
    
    print(f"\n📁 ORGANIZED OUTPUT STRUCTURE:")
    print(f"   📂 Main folder: {folders['main']}/")
    print(f"      📊 master_data/ - Comprehensive master dataset")
    print(f"      📄 individual_assets/ - Individual asset CSVs ({successful_assets} files)")
    print(f"      📋 summaries/ - Performance summaries")
    print(f"      🚪 exit_analysis/ - Early exit detailed analysis")
    print(f"      📋 run_metadata.csv - Analysis run details")
    print(f"      📋 run_summary.txt - Human-readable summary")
    print(f"\n✅ Early exit analysis complete - ready for comparison!")
    print("="*80)

if __name__ == "__main__":
    main()