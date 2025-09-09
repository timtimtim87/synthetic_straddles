import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

def identify_straddles_to_remove(csv_file_path, moneyness_threshold=0.8):
    """
    Identify straddles that start within 20% OTM (moneyness >= 0.8)
    These include ATM and ITM straddles that should be removed
    """
    
    try:
        # Read the straddle results CSV
        print(f"Reading straddle results from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Strike_Determination_Date'] = pd.to_datetime(df['Strike_Determination_Date'])
        
        print(f"Loaded {len(df):,} total straddle records")
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return None, None
    
    # Calculate moneyness
    df['Moneyness'] = df['Spot_Price'] / df['Strike_Price']
    
    # Get unique combinations of ticker and expiry year
    unique_straddles = df[['Ticker', 'Expiry_Year']].drop_duplicates()
    
    print(f"\nAnalyzing {len(unique_straddles)} unique straddles...")
    print(f"Threshold: Removing straddles with starting moneyness >= {moneyness_threshold} (within 20% OTM)")
    print("="*80)
    
    straddles_to_remove = []
    straddles_to_keep = []
    
    for _, row in unique_straddles.iterrows():
        ticker = row['Ticker']
        expiry_year = row['Expiry_Year']
        
        # Get data for this specific straddle
        straddle_data = df[(df['Ticker'] == ticker) & (df['Expiry_Year'] == expiry_year)].copy()
        straddle_data = straddle_data.sort_values('Date')
        
        if straddle_data.empty:
            continue
        
        # Get first day moneyness
        first_day_moneyness = straddle_data.iloc[0]['Moneyness']
        
        # Determine if this straddle should be removed
        if first_day_moneyness >= moneyness_threshold:
            straddles_to_remove.append({
                'Ticker': ticker,
                'Expiry_Year': expiry_year,
                'First_Day_Moneyness': first_day_moneyness,
                'Records': len(straddle_data)
            })
            print(f"❌ REMOVE: {ticker} {expiry_year} - Moneyness: {first_day_moneyness:.4f} (too close to ATM)")
        else:
            straddles_to_keep.append({
                'Ticker': ticker,
                'Expiry_Year': expiry_year,
                'First_Day_Moneyness': first_day_moneyness,
                'Records': len(straddle_data)
            })
            print(f"✅ KEEP:   {ticker} {expiry_year} - Moneyness: {first_day_moneyness:.4f} (sufficiently OTM)")
    
    return straddles_to_remove, straddles_to_keep

def remove_plot_files(straddles_to_remove):
    """Remove PNG plot files for straddles that should be deleted"""
    
    print(f"\n" + "="*50)
    print("REMOVING PLOT FILES")
    print("="*50)
    
    removed_plots = 0
    
    for straddle in straddles_to_remove:
        ticker = straddle['Ticker']
        
        # Plot filename pattern
        plot_filename = f"{ticker.lower()}_straddle_analysis.png"
        
        if os.path.exists(plot_filename):
            try:
                os.remove(plot_filename)
                print(f"🗑️  Removed plot: {plot_filename}")
                removed_plots += 1
            except Exception as e:
                print(f"❌ Error removing {plot_filename}: {str(e)}")
        else:
            print(f"⚠️  Plot not found: {plot_filename}")
    
    print(f"\nRemoved {removed_plots} plot files")

def clean_csv_data(csv_file_path, straddles_to_remove, output_file="cleaned_straddle_values.csv"):
    """Remove data for specified straddles from CSV and save cleaned version"""
    
    print(f"\n" + "="*50)
    print("CLEANING CSV DATA")
    print("="*50)
    
    try:
        # Read original data
        df = pd.read_csv(csv_file_path)
        original_count = len(df)
        
        print(f"Original records: {original_count:,}")
        
        # Create list of (ticker, expiry_year) tuples to remove
        remove_combinations = [(s['Ticker'], s['Expiry_Year']) for s in straddles_to_remove]
        
        # Filter out the straddles to remove
        mask = ~df.apply(lambda row: (row['Ticker'], row['Expiry_Year']) in remove_combinations, axis=1)
        cleaned_df = df[mask].copy()
        
        final_count = len(cleaned_df)
        removed_count = original_count - final_count
        
        print(f"Records removed: {removed_count:,}")
        print(f"Records remaining: {final_count:,}")
        print(f"Removal percentage: {(removed_count/original_count)*100:.1f}%")
        
        # Save cleaned data
        cleaned_df.to_csv(output_file, index=False)
        print(f"\n💾 Cleaned data saved to: {output_file}")
        
        # Show summary by ticker
        print(f"\nREMAINING DATA BY TICKER:")
        remaining_summary = cleaned_df.groupby('Ticker').agg({
            'Expiry_Year': 'nunique',
            'Date': 'count'
        }).rename(columns={'Expiry_Year': 'Expiry_Years', 'Date': 'Records'})
        
        for ticker, row in remaining_summary.iterrows():
            print(f"  {ticker}: {row['Expiry_Years']} expiry years, {row['Records']:,} records")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error cleaning CSV data: {str(e)}")
        return None

def create_summary_report(straddles_to_remove, straddles_to_keep):
    """Create a summary report of the cleanup operation"""
    
    print(f"\n" + "="*80)
    print("CLEANUP SUMMARY REPORT")
    print("="*80)
    
    # Overall statistics
    total_straddles = len(straddles_to_remove) + len(straddles_to_keep)
    removed_count = len(straddles_to_remove)
    kept_count = len(straddles_to_keep)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total straddles analyzed: {total_straddles}")
    print(f"  Straddles removed: {removed_count} ({(removed_count/total_straddles)*100:.1f}%)")
    print(f"  Straddles kept: {kept_count} ({(kept_count/total_straddles)*100:.1f}%)")
    
    # Removed straddles details
    if straddles_to_remove:
        removed_df = pd.DataFrame(straddles_to_remove)
        total_removed_records = removed_df['Records'].sum()
        
        print(f"\nREMOVED STRADDLES:")
        print(f"  Total records removed: {total_removed_records:,}")
        print(f"  Average moneyness: {removed_df['First_Day_Moneyness'].mean():.4f}")
        print(f"  Moneyness range: {removed_df['First_Day_Moneyness'].min():.4f} - {removed_df['First_Day_Moneyness'].max():.4f}")
        
        print(f"\n  Details:")
        for _, straddle in removed_df.iterrows():
            print(f"    {straddle['Ticker']} {straddle['Expiry_Year']}: Moneyness {straddle['First_Day_Moneyness']:.4f}, {straddle['Records']:,} records")
    
    # Kept straddles details
    if straddles_to_keep:
        kept_df = pd.DataFrame(straddles_to_keep)
        total_kept_records = kept_df['Records'].sum()
        
        print(f"\nKEPT STRADDLES (Sufficiently OTM):")
        print(f"  Total records kept: {total_kept_records:,}")
        print(f"  Average moneyness: {kept_df['First_Day_Moneyness'].mean():.4f}")
        print(f"  Moneyness range: {kept_df['First_Day_Moneyness'].min():.4f} - {kept_df['First_Day_Moneyness'].max():.4f}")
        
        print(f"\n  Details:")
        for _, straddle in kept_df.iterrows():
            print(f"    {straddle['Ticker']} {straddle['Expiry_Year']}: Moneyness {straddle['First_Day_Moneyness']:.4f}, {straddle['Records']:,} records")
    
    # Save detailed report
    report_data = {
        'Removed': straddles_to_remove,
        'Kept': straddles_to_keep
    }
    
    # Save removed straddles list
    if straddles_to_remove:
        removed_df = pd.DataFrame(straddles_to_remove)
        removed_df.to_csv("removed_straddles_report.csv", index=False)
        print(f"\n📋 Removed straddles report saved to: removed_straddles_report.csv")
    
    # Save kept straddles list
    if straddles_to_keep:
        kept_df = pd.DataFrame(straddles_to_keep)
        kept_df.to_csv("kept_straddles_report.csv", index=False)
        print(f"📋 Kept straddles report saved to: kept_straddles_report.csv")

def main():
    """Main function to execute the cleanup"""
    
    # Configuration
    csv_file_path = "all_assets_straddle_values.csv"
    moneyness_threshold = 0.8  # Remove straddles with starting moneyness >= 0.8
    
    print("STRADDLE CLEANUP SCRIPT")
    print("="*80)
    print(f"Removing straddles that start within 20% OTM (moneyness >= {moneyness_threshold})")
    print("This includes ATM and ITM straddles")
    print("="*80)
    
    # Step 1: Identify straddles to remove
    straddles_to_remove, straddles_to_keep = identify_straddles_to_remove(csv_file_path, moneyness_threshold)
    
    if straddles_to_remove is None:
        print("❌ Failed to analyze straddles. Exiting.")
        return
    
    if not straddles_to_remove:
        print("✅ No straddles found that need to be removed!")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  WARNING: About to remove {len(straddles_to_remove)} straddles and their files!")
    response = input("Do you want to proceed? (yes/no): ").lower().strip()
    
    if response not in ['yes', 'y']:
        print("❌ Operation cancelled by user")
        return
    
    print("\n🚀 Starting cleanup process...")
    
    # Step 2: Remove plot files
    remove_plot_files(straddles_to_remove)
    
    # Step 3: Clean CSV data
    cleaned_df = clean_csv_data(csv_file_path, straddles_to_remove)
    
    # Step 4: Create summary report
    create_summary_report(straddles_to_remove, straddles_to_keep)
    
    print(f"\n🎉 Cleanup complete!")
    print(f"📁 Check 'cleaned_straddle_values.csv' for the filtered data")
    print(f"📊 Only straddles with starting moneyness < {moneyness_threshold} remain")

if __name__ == "__main__":
    main()