import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_first_day_moneyness(csv_file_path):
    """
    Analyze the moneyness on the first day of each straddle to verify 
    that strikes are correctly set 25% above spot price (365 days before expiry)
    """
    
    try:
        # Read the straddle results CSV
        print(f"Reading straddle results from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Strike_Determination_Date'] = pd.to_datetime(df['Strike_Determination_Date'])
        
        print(f"Loaded {len(df):,} straddle records")
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file_path}")
        print("Make sure you've run the straddle analysis script first!")
        return None
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return None
    
    # Calculate moneyness
    df['Moneyness'] = df['Spot_Price'] / df['Strike_Price']
    
    # Get unique combinations of ticker and expiry year
    unique_straddles = df[['Ticker', 'Expiry_Year']].drop_duplicates()
    
    print(f"\nFound {len(unique_straddles)} unique straddles to analyze")
    print("="*80)
    
    analysis_results = []
    
    for _, row in unique_straddles.iterrows():
        ticker = row['Ticker']
        expiry_year = row['Expiry_Year']
        
        # Get data for this specific straddle
        straddle_data = df[(df['Ticker'] == ticker) & (df['Expiry_Year'] == expiry_year)].copy()
        straddle_data = straddle_data.sort_values('Date')
        
        if straddle_data.empty:
            continue
        
        # Get first day data
        first_day = straddle_data.iloc[0]
        
        # Get strike determination info
        strike_determination_date = first_day['Strike_Determination_Date']
        spot_at_strike_date = first_day['Spot_At_Strike_Date']
        strike_price = first_day['Strike_Price']
        
        # Calculate what the strike SHOULD be (25% above spot)
        expected_strike_raw = spot_at_strike_date * 1.25
        expected_strike_rounded = round(expected_strike_raw / 10) * 10
        
        # First day info
        first_day_date = first_day['Date']
        first_day_spot = first_day['Spot_Price']
        first_day_moneyness = first_day['Moneyness']
        
        # Expected moneyness if strike was exactly 25% above
        expected_moneyness_exact = spot_at_strike_date / expected_strike_raw
        expected_moneyness_rounded = spot_at_strike_date / expected_strike_rounded
        
        # Days between strike determination and first trading day
        days_between = (first_day_date - strike_determination_date).days
        
        # Store results
        result = {
            'Ticker': ticker,
            'Expiry_Year': expiry_year,
            'Strike_Determination_Date': strike_determination_date,
            'Spot_At_Strike_Date': spot_at_strike_date,
            'Expected_Strike_Raw': expected_strike_raw,
            'Expected_Strike_Rounded': expected_strike_rounded,
            'Actual_Strike': strike_price,
            'First_Trading_Date': first_day_date,
            'First_Day_Spot': first_day_spot,
            'First_Day_Moneyness': first_day_moneyness,
            'Expected_Moneyness_Exact': expected_moneyness_exact,
            'Expected_Moneyness_Rounded': expected_moneyness_rounded,
            'Days_Between': days_between,
            'Strike_Difference': strike_price - expected_strike_rounded,
            'Moneyness_Difference': first_day_moneyness - expected_moneyness_rounded
        }
        
        analysis_results.append(result)
        
        # Print detailed info for this straddle
        print(f"\n{ticker} {expiry_year} LEAPS:")
        print(f"  Strike Determination Date: {strike_determination_date.strftime('%Y-%m-%d')}")
        print(f"  Spot Price (365 DTE): ${spot_at_strike_date:.2f}")
        print(f"  Expected Strike (25% higher): ${expected_strike_raw:.2f} → ${expected_strike_rounded:.0f} (rounded)")
        print(f"  Actual Strike Used: ${strike_price:.0f}")
        print(f"  Strike Difference: ${strike_price - expected_strike_rounded:.0f}")
        print(f"  ") 
        print(f"  First Trading Date: {first_day_date.strftime('%Y-%m-%d')} ({days_between} days later)")
        print(f"  First Day Spot Price: ${first_day_spot:.2f}")
        print(f"  First Day Moneyness: {first_day_moneyness:.4f}")
        print(f"  Expected Moneyness: {expected_moneyness_rounded:.4f}")
        print(f"  Moneyness Difference: {first_day_moneyness - expected_moneyness_rounded:+.4f}")
        
        # Flag any issues
        if abs(strike_price - expected_strike_rounded) > 0.01:
            print(f"  ⚠️  WARNING: Strike price mismatch!")
        
        if abs(first_day_moneyness - 0.8) > 0.1:  # Should be around 0.8 if 25% above
            print(f"  ⚠️  WARNING: Moneyness far from expected 0.8!")
    
    # Convert to DataFrame for analysis
    if analysis_results:
        results_df = pd.DataFrame(analysis_results)
        
        print(f"\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        print(f"\nTotal Straddles Analyzed: {len(results_df)}")
        
        print(f"\nFirst Day Moneyness Statistics:")
        print(f"  Mean: {results_df['First_Day_Moneyness'].mean():.4f}")
        print(f"  Median: {results_df['First_Day_Moneyness'].median():.4f}")
        print(f"  Std Dev: {results_df['First_Day_Moneyness'].std():.4f}")
        print(f"  Min: {results_df['First_Day_Moneyness'].min():.4f}")
        print(f"  Max: {results_df['First_Day_Moneyness'].max():.4f}")
        
        print(f"\nExpected vs Actual Strike Differences:")
        print(f"  Mean Difference: ${results_df['Strike_Difference'].mean():.2f}")
        print(f"  Std Dev: ${results_df['Strike_Difference'].std():.2f}")
        print(f"  Range: ${results_df['Strike_Difference'].min():.0f} to ${results_df['Strike_Difference'].max():.0f}")
        
        print(f"\nMoneyness Differences (Actual - Expected):")
        print(f"  Mean Difference: {results_df['Moneyness_Difference'].mean():+.4f}")
        print(f"  Std Dev: {results_df['Moneyness_Difference'].std():.4f}")
        print(f"  Range: {results_df['Moneyness_Difference'].min():+.4f} to {results_df['Moneyness_Difference'].max():+.4f}")
        
        # Check how many are close to expected 0.8 moneyness
        close_to_08 = (results_df['First_Day_Moneyness'] >= 0.75) & (results_df['First_Day_Moneyness'] <= 0.85)
        print(f"\nStraddles with moneyness between 0.75-0.85: {close_to_08.sum()} / {len(results_df)} ({close_to_08.mean()*100:.1f}%)")
        
        # Identify outliers
        outliers = results_df[abs(results_df['First_Day_Moneyness'] - 0.8) > 0.1]
        if not outliers.empty:
            print(f"\nOUTLIERS (moneyness far from 0.8):")
            for _, outlier in outliers.iterrows():
                print(f"  {outlier['Ticker']} {outlier['Expiry_Year']}: Moneyness = {outlier['First_Day_Moneyness']:.4f}")
        
        # Save detailed analysis
        output_file = "first_day_moneyness_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed analysis saved to: {output_file}")
        
        return results_df
    
    else:
        print("No results to analyze")
        return None

def quick_check_moneyness_distribution(csv_file_path):
    """Quick check of moneyness distribution across all days"""
    
    try:
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Moneyness'] = df['Spot_Price'] / df['Strike_Price']
        
        print(f"\nQUICK MONEYNESS DISTRIBUTION CHECK")
        print("="*50)
        print(f"Total records: {len(df):,}")
        
        # Overall moneyness stats
        print(f"\nOverall Moneyness Statistics:")
        print(f"  Mean: {df['Moneyness'].mean():.4f}")
        print(f"  Median: {df['Moneyness'].median():.4f}")
        print(f"  Std Dev: {df['Moneyness'].std():.4f}")
        print(f"  Min: {df['Moneyness'].min():.4f}")
        print(f"  Max: {df['Moneyness'].max():.4f}")
        
        # Moneyness ranges
        print(f"\nMoneyness Ranges:")
        print(f"  < 0.5 (Deep OTM): {(df['Moneyness'] < 0.5).sum():,} ({(df['Moneyness'] < 0.5).mean()*100:.1f}%)")
        print(f"  0.5-0.8 (OTM): {((df['Moneyness'] >= 0.5) & (df['Moneyness'] < 0.8)).sum():,} ({((df['Moneyness'] >= 0.5) & (df['Moneyness'] < 0.8)).mean()*100:.1f}%)")
        print(f"  0.8-1.2 (Near ATM): {((df['Moneyness'] >= 0.8) & (df['Moneyness'] <= 1.2)).sum():,} ({((df['Moneyness'] >= 0.8) & (df['Moneyness'] <= 1.2)).mean()*100:.1f}%)")
        print(f"  > 1.2 (ITM): {(df['Moneyness'] > 1.2).sum():,} ({(df['Moneyness'] > 1.2).mean()*100:.1f}%)")
        
    except Exception as e:
        print(f"Error in quick check: {str(e)}")

if __name__ == "__main__":
    # File path to your straddle results
    straddle_results_file = "all_assets_straddle_values.csv"
    
    print("FIRST DAY MONEYNESS ANALYSIS")
    print("="*80)
    print("Analyzing whether strikes are correctly set 25% above spot price...")
    print("(Expected first day moneyness should be ~0.8)")
    
    # Run the detailed analysis
    results = analyze_first_day_moneyness(straddle_results_file)
    
    # Quick distribution check
    quick_check_moneyness_distribution(straddle_results_file)
    
    print(f"\n✅ Analysis complete!")
    print(f"Check 'first_day_moneyness_analysis.csv' for detailed results")