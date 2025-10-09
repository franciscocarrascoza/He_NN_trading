import plotly.graph_objects as go

def create_candlestick_figure(df, symbol='BTCUSDT'):
    """Create candlestick figure."""
    try:
        if df.empty:
            print("Candlestick: Empty DataFrame, returning empty figure")
            return go.Figure()
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Candlestick'
            )
        ])
        fig.update_layout(
            title=f'{symbol} Candlestick Chart',
            xaxis_title='Time',
            yaxis_title='Price (USDT)',
            height=600,
            width=1000,
            plot_bgcolor='white',
            font=dict(size=14)
        )
        print("Candlestick figure created successfully")
        return fig
    except Exception as e:
        print(f"Candlestick figure creation error: {e}")
        return go.Figure()

def create_order_book_figure(bids_processed, asks_processed):
    """Create order book figure."""
    try:
        fig = go.Figure()
        if not bids_processed.empty:
            fig.add_trace(go.Scatter(
                x=bids_processed['price'],
                y=bids_processed['cum_quantity'],
                mode='lines',
                name='Bids',
                line=dict(color='green')
            ))
        if not asks_processed.empty:
            fig.add_trace(go.Scatter(
                x=asks_processed['price'],
                y=asks_processed['cum_quantity'],
                mode='lines',
                name='Asks',
                line=dict(color='red')
            ))
        fig.update_layout(
            title='Order Book (Cumulative)',
            xaxis_title='Price (USDT)',
            yaxis_title='Cumulative Quantity',
            height=600,
            width=1000,
            plot_bgcolor='white',
            font=dict(size=14)
        )
        print("Order book figure created successfully")
        return fig
    except Exception as e:
        print(f"Order book figure creation error: {e}")
        return go.Figure()
