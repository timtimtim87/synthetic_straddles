import pandas as pd
import os
import glob

def find_and_analyze_straddle_data():
    """
    Find available straddle CSV files and analyze their first day moneyness
    """
    
    print("CHECKING AVAILABLE STRADDLE DATA FILES")
    print("="*60)
    
    # Look for common straddle CSV files in current directory AND straddle folder
    possible_files = [
        "cleaned_straddle_values.csv",
        "all_assets_straddle_values.csv", 
        "aapl_straddle_values.csv",
        "*straddle*.csv",
        "straddle/cleaned_straddle_values.csv",
        "straddle/all_assets_straddle_values.csv",
        "straddle/*straddle*.csv",
        "straddle/*.csv"
    ]
    
    found_files = []
    
    # Check each possible file
    for pattern in possible_files:
        if "*" in pattern:
            # Use glob for wildcard patterns
            files = glob.glob(pattern)
            found_files.extend(files)
        else:
            # Check exact filename
            if os.path.exists(pattern):
                found_files.append(pattern)
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    
    if not found_files:
        print("❌ No straddle CSV files found!")
        print("\nLooking for these patterns:")
        for pattern in possible_files:
            print(f"  - {pattern}")
        print("\nMake sure you've run the straddle analysis script first.")
        return
    
    print(f"Found {len(found_files)} straddle CSV file(s):")
    for i, file in enumerate(found_files, 1):
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"  {i}. {file} ({file_size:.1f} KB)")
    
    # Analyze each file
    for file_path in found_files:
        print(f"\n" + "="*60)
        print(f"ANALYZING: {file_path}")
        print("="*60)
        
        try:
            # Read the file
            df = pd.read_csv(file_path)
            print(f"Total records: {len(df):,}")
            
            # Check if required columns exist
            required_cols = ['Ticker', 'Expiry_Year', 'Spot_Price', 'Strike_Price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"Available columns: {list(df.columns)}")
                continue
            
            # Calculate moneyness
            df['Moneyness'] = df['Spot_Price'] / df['Strike_Price']
            
            # Get unique straddles and their first day moneyness
            print(f"\nUnique straddles analysis:")
            unique_straddles = df.groupby(['Ticker', 'Expiry_Year']).agg({
                'Date': ['min', 'count'],
                'Moneyness': 'first',
                'Spot_Price': 'first',
                'Strike_Price': 'first'
            }).round(4)
            
            # Flatten column names
            unique_straddles.columns = ['First_Date', 'Record_Count', 'First_Day_Moneyness', 'First_Spot', 'First_Strike']
            unique_straddles = unique_straddles.reset_index()
            
            print(f"Number of unique straddles: {len(unique_straddles)}")
            
            # Analyze first day moneyness distribution
            first_day_moneyness = unique_straddles['First_Day_Moneyness']
            
            print(f"\nFIRST DAY MONEYNESS STATISTICS:")
            print(f"  Mean: {first_day_moneyness.mean():.4f}")
            print(f"  Median: {first_day_moneyness.median():.4f}")
            print(f"  Min: {first_day_moneyness.min():.4f}")
            print(f"  Max: {first_day_moneyness.max():.4f}")
            print(f"  Std Dev: {first_day_moneyness.std():.4f}")
            
            # Check for ATM/ITM straddles
            atm_itm_count = (first_day_moneyness >= 0.8).sum()
            close_to_atm_count = (first_day_moneyness >= 0.9).sum()
            very_close_to_atm_count = (first_day_moneyness >= 0.95).sum()
            
            print(f"\nMONEYNESS DISTRIBUTION:")
            print(f"  < 0.8 (>20% OTM): {(first_day_moneyness < 0.8).sum()} straddles")
            print(f"  0.8-0.9 (10-20% OTM): {((first_day_moneyness >= 0.8) & (first_day_moneyness < 0.9)).sum()} straddles")
            print(f"  0.9-0.95 (5-10% OTM): {((first_day_moneyness >= 0.9) & (first_day_moneyness < 0.95)).sum()} straddles")
            print(f"  0.95-1.0 (0-5% OTM): {((first_day_moneyness >= 0.95) & (first_day_moneyness < 1.0)).sum()} straddles")
            print(f"  >= 1.0 (ATM/ITM): {(first_day_moneyness >= 1.0).sum()} straddles")
            
            # Show problematic straddles
            if atm_itm_count > 0:
                print(f"\n⚠️  STRADDLES WITH MONEYNESS >= 0.8 (should be removed):")
                problematic = unique_straddles[unique_straddles['First_Day_Moneyness'] >= 0.8]
                for _, row in problematic.iterrows():
                    print(f"    {row['Ticker']} {row['Expiry_Year']}: Moneyness {row['First_Day_Moneyness']:.4f} (Spot: ${row['First_Spot']:.2f}, Strike: ${row['First_Strike']:.0f})")
            else:
                print(f"\n✅ NO straddles with moneyness >= 0.8 (all are sufficiently OTM)")
            
            # Show best examples (most OTM)
            print(f"\nMOST OTM STRADDLES (lowest moneyness):")
            best_otm = unique_straddles.nsmallest(5, 'First_Day_Moneyness')
            for _, row in best_otm.iterrows():
                print(f"    {row['Ticker']} {row['Expiry_Year']}: Moneyness {row['First_Day_Moneyness']:.4f} (Spot: ${row['First_Spot']:.2f}, Strike: ${row['First_Strike']:.0f})")
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {str(e)}")
            continue
    
    print(f"\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    if 'cleaned_straddle_values.csv' in found_files:
        print("✅ You have cleaned data - this should show no ATM/ITM straddles")
    elif 'all_assets_straddle_values.csv' in found_files:
        print("⚠️  You have original data - run the cleanup script to remove ATM/ITM straddles")
        print("   File: all_assets_straddle_values.csv contains all original straddles")
    else:
        print("❓ Run the main straddle analysis script first to generate straddle data")

if __name__ == "__main__":
    find_and_analyze_straddle_data()