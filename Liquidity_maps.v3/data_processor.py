import pandas as pd

def process_data(df):
    """Process candlestick data."""
    try:
        if df.empty:
            print("Process data: Empty DataFrame received")
            return df
        df_processed = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        print(f"Processed candlestick data shape: {df_processed.shape}")
        return df_processed
    except Exception as e:
        print(f"Process data error: {e}")
        return pd.DataFrame()

def process_order_book(bids_df, asks_df):
    """Process order book data."""
    try:
        bids_processed = bids_df.copy()
        asks_processed = asks_df.copy()
        bids_processed['cum_quantity'] = bids_processed['quantity'].cumsum()
        asks_processed['cum_quantity'] = asks_processed['quantity'].cumsum()
        print(f"Processed order book: Bids shape={bids_processed.shape}, Asks shape={asks_processed.shape}")
        return bids_processed, asks_processed
    except Exception as e:
        print(f"Process order book error: {e}")
        return pd.DataFrame(columns=['price', 'quantity', 'cum_quantity']), pd.DataFrame(columns=['price', 'quantity', 'cum_quantity'])
