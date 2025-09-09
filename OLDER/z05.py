import pandas as pd

# Read cleaned data
df = pd.read_csv("cleaned_straddle_values.csv")
df['Moneyness'] = df['Spot_Price'] / df['Strike_Price']

# Get first day moneyness for each straddle
first_days = df.groupby(['Ticker', 'Expiry_Year']).first()['Moneyness']

print(f"Remaining straddles: {len(first_days)}")
print(f"First day moneyness range: {first_days.min():.4f} - {first_days.max():.4f}")
print(f"Any >= 0.8? {(first_days >= 0.8).any()}")