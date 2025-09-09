import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Black-Scholes functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    if K <= 0:  # Safety check for zero strike
        return S
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call_price, 0)

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    if K <= 0:  # Safety check for zero strike
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put_price, 0)

def calculate_straddle_value(S, K, T, r, sigma):
    """Calculate straddle value (call + put)"""
    if K <= 0 or S <= 0:  # Safety checks
        return 0
    
    call_value = black_scholes_call(S, K, T, r, sigma)
    put_value = black_scholes_put(S, K, T, r, sigma)
    return call_value + put_value

def round_to_nearest_10(price):
    """Round price to nearest $10, with minimum of $10"""
    rounded = round(price / 10) * 10
    return max(rounded, 10)  # Ensure minimum strike of $10

def plot_straddle_analysis(ticker, results_df):
    """
    Create plots showing straddle prices and moneyness over time for each LEAPS expiry
    """
    try:
        # Calculate moneyness (Spot/Strike)
        results_df['Moneyness'] = results_df['Spot_Price'] / results_df['Strike_Price']
        
        # Get unique expiry years
        expiry_years = sorted(results_df['Expiry_Year'].unique())
        
        if len(expiry_years) == 0:
            print(f"  No valid expiry years for {ticker}")
            return
        
        # Create subplots - 2 rows (price and moneyness) x number of expiry years
        fig, axes = plt.subplots(2, len(expiry_years), figsize=(6*len(expiry_years), 10))
        
        # If only one expiry year, axes won't be 2D
        if len(expiry_years) == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, year in enumerate(expiry_years):
            year_data = results_df[results_df['Expiry_Year'] == year].sort_values('Date')
            
            if year_data.empty:
                continue
                
            color = colors[i % len(colors)]
            strike = year_data['Strike_Price'].iloc[0]
            
            # Top row: Straddle Price over time
            axes[0, i].plot(year_data['Date'], year_data['Straddle_Value'], 
                           color=color, linewidth=2)
            axes[0, i].set_title(f'{year} LEAPS - Straddle Price\n(Strike: ${strike})', fontsize=12, fontweight='bold')
            axes[0, i].set_ylabel('Straddle Value ($)', fontsize=10)
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # Add min/max annotations
            min_val = year_data['Straddle_Value'].min()
            max_val = year_data['Straddle_Value'].max()
            min_date = year_data.loc[year_data['Straddle_Value'].idxmin(), 'Date']
            max_date = year_data.loc[year_data['Straddle_Value'].idxmax(), 'Date']
            
            axes[0, i].annotate(f'Max: ${max_val:.2f}', 
                               xy=(max_date, max_val), xytext=(10, 10),
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            axes[0, i].annotate(f'Min: ${min_val:.2f}', 
                               xy=(min_date, min_val), xytext=(10, -15),
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
            # Bottom row: Moneyness over time
            axes[1, i].plot(year_data['Date'], year_data['Moneyness'], 
                           color=color, linewidth=2)
            axes[1, i].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1, i].fill_between(year_data['Date'], year_data['Moneyness'], 1, 
                                   where=(year_data['Moneyness'] >= 1), alpha=0.3, color='green', label='ITM')
            axes[1, i].fill_between(year_data['Date'], year_data['Moneyness'], 1, 
                                   where=(year_data['Moneyness'] < 1), alpha=0.3, color='red', label='OTM')
            
            axes[1, i].set_title(f'{year} LEAPS - Moneyness', fontsize=12, fontweight='bold')
            axes[1, i].set_ylabel('Moneyness (S/K)', fontsize=10)
            axes[1, i].set_xlabel('Date', fontsize=10)
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].tick_params(axis='x', rotation=45)
            axes[1, i].legend(fontsize=8)
            
            # Add current moneyness annotation
            current_moneyness = year_data['Moneyness'].iloc[-1]
            current_date = year_data['Date'].iloc[-1]
            axes[1, i].annotate(f'Final: {current_moneyness:.3f}', 
                               xy=(current_date, current_moneyness), xytext=(-50, 10),
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        plt.suptitle(f'{ticker} Straddle Analysis - Strike 25% Above Spot (365 Days Prior)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        filename = f'{ticker.lower()}_straddle_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        print(f"  Plot saved: {filename}")
        
    except Exception as e:
        print(f"  Error creating plot for {ticker}: {str(e)}")
        plt.close()

def process_ticker_straddles(ticker, ticker_df, expiration_dates):
    """Process straddle calculations for a single ticker"""
    
    IV = 0.25  # 25% implied volatility
    RISK_FREE_RATE = 0.04  # 4% risk-free rate
    DAYS_DATA = 365  # 365 days of straddle data
    
    print(f"\n=== Processing {ticker} ===")
    print(f"Found {len(ticker_df)} {ticker} records from {ticker_df['Date'].min()} to {ticker_df['Date'].max()}")
    
    all_results = []
    
    for year, expiry_date in expiration_dates.items():
        print(f"\nProcessing {year} LEAPS (expires {expiry_date.strftime('%Y-%m-%d')})...")
        
        # Find date 365 days before expiration to determine strike
        strike_determination_date = expiry_date - timedelta(days=365)
        
        # Find the closest trading day to 365 days before expiry
        date_diffs = abs(ticker_df['Date'] - strike_determination_date)
        
        if len(date_diffs) == 0:
            print(f"  No data available for {ticker} around strike determination date")
            continue
            
        closest_idx = date_diffs.idxmin()
        strike_date_data = ticker_df.loc[closest_idx]
        strike_determination_actual_date = strike_date_data['Date']
        spot_price_365_days_before = strike_date_data['Close']
        
        # Skip if spot price is invalid
        if pd.isna(spot_price_365_days_before) or spot_price_365_days_before <= 0:
            print(f"  Invalid spot price 365 days before: ${spot_price_365_days_before}")
            continue
        
        # Calculate strike as 25% higher than spot price 365 days before expiry
        target_strike = spot_price_365_days_before * 1.25
        strike_price = round_to_nearest_10(target_strike)
        
        # Skip if strike price is less than $50
        if strike_price < 50:
            print(f"  Skipping {year} LEAPS - strike price too low (Strike: ${strike_price}, Spot: ${spot_price_365_days_before:.2f})")
            continue
        
        print(f"  Strike determination date: {strike_determination_actual_date.strftime('%Y-%m-%d')}")
        print(f"  Spot price 365 days before: ${spot_price_365_days_before:.2f}")
        print(f"  Strike price (25% higher, rounded): ${strike_price}")
        
        # Get exactly 365 days of data leading up to expiration
        # Find the closest date to expiration
        expiry_diffs = abs(ticker_df['Date'] - expiry_date)
        if len(expiry_diffs) == 0:
            print(f"  No data available near expiration date")
            continue
            
        expiry_idx = expiry_diffs.idxmin()
        
        # Get the 365 trading days before expiration
        end_idx = expiry_idx
        start_idx = max(0, end_idx - DAYS_DATA + 1)
        
        period_data = ticker_df.iloc[start_idx:end_idx + 1].copy()
        
        if len(period_data) < 50:  # Minimum threshold
            print(f"  Insufficient data for {year} LEAPS (only {len(period_data)} records)")
            continue
            
        print(f"  Using {len(period_data)} records from {period_data['Date'].min().strftime('%Y-%m-%d')} to {period_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Calculate straddle values
        straddle_values = []
        
        for idx, row in period_data.iterrows():
            spot_price = row['Close']
            
            # Skip invalid spot prices
            if pd.isna(spot_price) or spot_price <= 0:
                continue
            
            # Find actual expiration date in data (closest to target expiry)
            actual_expiry_date = period_data['Date'].max()  # Use last date in our period
            days_to_expiry = (actual_expiry_date - row['Date']).days
            time_to_expiry = max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day
            
            try:
                straddle_value = calculate_straddle_value(
                    S=spot_price,
                    K=strike_price, 
                    T=time_to_expiry,
                    r=RISK_FREE_RATE,
                    sigma=IV
                )
                
                if pd.isna(straddle_value) or straddle_value < 0:
                    continue
                
                result = {
                    'Ticker': ticker,
                    'Date': row['Date'],
                    'Expiry_Year': year,
                    'Days_to_Expiry': days_to_expiry,
                    'Spot_Price': spot_price,
                    'Strike_Price': strike_price,
                    'Straddle_Value': straddle_value,
                    'Strike_Determination_Date': strike_determination_actual_date,
                    'Spot_At_Strike_Date': spot_price_365_days_before
                }
                
                straddle_values.append(result)
                
            except Exception as e:
                print(f"    Error calculating straddle for {row['Date']}: {str(e)}")
                continue
        
        all_results.extend(straddle_values)
        print(f"  Calculated {len(straddle_values)} straddle values")
    
    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Display summary
        print(f"\n=== {ticker} SUMMARY ===")
        print(f"Total records: {len(results_df)}")
        
        for year in results_df['Expiry_Year'].unique():
            year_data = results_df[results_df['Expiry_Year'] == year]
            print(f"{year} LEAPS: {len(year_data)} records")
            print(f"  Date range: {year_data['Date'].min().strftime('%Y-%m-%d')} to {year_data['Date'].max().strftime('%Y-%m-%d')}")
            print(f"  Strike determination: {year_data['Strike_Determination_Date'].iloc[0].strftime('%Y-%m-%d')} (Spot: ${year_data['Spot_At_Strike_Date'].iloc[0]:.2f})")
            print(f"  Strike: ${year_data['Strike_Price'].iloc[0]} (25% above ${year_data['Spot_At_Strike_Date'].iloc[0]:.2f})")
            print(f"  Straddle value range: ${year_data['Straddle_Value'].min():.2f} - ${year_data['Straddle_Value'].max():.2f}")
        
        return results_df
    else:
        print(f"No valid results generated for {ticker}")
        return None

def process_all_assets_straddles(csv_file_path):
    """Main function to process straddles for all assets in CSV"""
    
    # Parameters
    expiration_dates = {
        '2023': datetime(2023, 1, 20),
        '2024': datetime(2024, 1, 19), 
        '2025': datetime(2025, 1, 17)
    }
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Loaded {len(df):,} total records")
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None
    
    # Get all unique tickers
    tickers = sorted(df['Ticker'].unique())
    print(f"Found {len(tickers)} tickers: {tickers}")
    
    all_results = []
    successful_tickers = []
    
    for ticker in tickers:
        try:
            # Filter for current ticker
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
            
            if ticker_df.empty:
                print(f"No data found for {ticker}")
                continue
            
            # Process this ticker
            ticker_results = process_ticker_straddles(ticker, ticker_df, expiration_dates)
            
            if ticker_results is not None and not ticker_results.empty:
                all_results.append(ticker_results)
                successful_tickers.append(ticker)
                
                # Create the plot for this ticker
                plot_straddle_analysis(ticker, ticker_results)
            else:
                print(f"Failed to process {ticker} - no valid straddle data")
        
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        try:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            print(f"\n=== OVERALL SUMMARY ===")
            print(f"Successfully processed {len(successful_tickers)} tickers: {successful_tickers}")
            print(f"Total records across all tickers: {len(combined_results)}")
            print(f"Date range: {combined_results['Date'].min().strftime('%Y-%m-%d')} to {combined_results['Date'].max().strftime('%Y-%m-%d')}")
            
            # Save combined results
            output_file = "all_assets_straddle_values.csv"
            combined_results.to_csv(output_file, index=False)
            print(f"\nCombined results saved to: {output_file}")
            
            return combined_results
            
        except Exception as e:
            print(f"Error combining results: {str(e)}")
            return None
    else:
        print("No successful results from any ticker")
        return None

# Usage example:
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = "/Users/tim/CODE_PROJECTS/synthetic_straddles/spot_prices.csv"
    
    # Process all assets
    print("Starting straddle analysis for all assets...")
    results = process_all_assets_straddles(csv_file_path)
    
    if results is not None:
        print(f"\n🎉 Processing complete!")
        print(f"📊 Generated plot files for each qualifying ticker")
        print(f"📁 Check directory for individual ticker PNG files")
        print(f"💾 Combined data saved to 'all_assets_straddle_values.csv'")
    else:
        print("❌ Processing failed - no valid results generated")