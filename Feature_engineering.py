import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('BTCUSDT_1h.csv')

# Ensure the columns are in the correct order and type
df['open'] = pd.to_numeric(df['open'], errors='coerce')
df['high'] = pd.to_numeric(df['high'], errors='coerce')
df['low'] = pd.to_numeric(df['low'], errors='coerce')
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Sort by time if not already sorted
df = df.sort_values(by='close_time').reset_index(drop=True)

# 1. Short-Term Moving Averages
df['MA20'] = df['close'].rolling(window=20).mean()
df['MA10'] = df['close'].rolling(window=10).mean()

# 2. Deviation from Moving Averages
df['DIFF-MA20-CLOSE'] = df['MA20'] - df['close']
df['DIFF-MA10-CLOSE'] = df['MA10'] - df['close']

# 3. Rolling Min/Max (Support & Resistance Levels)
df['MA14_low'] = df['low'].rolling(window=14).min()
df['MA14_high'] = df['high'].rolling(window=14).max()

# 4. Bollinger Bands
df['MA20_std'] = df['close'].rolling(window=20).std()
df['Bollinger_Upper'] = df['MA20'] + (df['MA20_std'] * 2)
df['Bollinger_Lower'] = df['MA20'] - (df['MA20_std'] * 2)

# 5. RSI Calculation
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_rsi(df['close'], window=14)

# 6. MACD Calculation
df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
df['EMA17'] = df['close'].ewm(span=17, adjust=False).mean()
df['MACD'] = df['EMA8'] - df['EMA17']

# Drop rows with NaN values generated during rolling window calculations
df.dropna(inplace=True)

# Save the engineered features to a new CSV
df.to_csv('BTCUSDT_1h_engineered.csv', index=False)

print("✅ Feature engineering completed and saved to 'BTCUSDT_1h_engineered.csv'")