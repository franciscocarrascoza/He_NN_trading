import pandas as pd
from binance.client import Client
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle event loop in threads
nest_asyncio.apply()

def fetch_binance_data(api_key, api_secret, symbol='BTCUSDT', interval='1m', limit=500):
    """Fetch candlestick data from Binance."""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        client = Client(api_key, api_secret)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"Fetched candlestick data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Fetch binance data error: {e}")
        return pd.DataFrame()
    finally:
        if 'loop' in locals():
            loop.close()
