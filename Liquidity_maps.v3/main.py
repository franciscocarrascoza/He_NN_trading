import os
import threading
import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager
from data_fetcher import fetch_binance_data
from data_processor import process_data, process_order_book
from data_plotter import create_candlestick_figure, create_order_book_figure
from liqs_map import create_coinglass_liq_map, create_liq_map_figure
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Configuration variables
SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
LIMIT = 500
BINS = 200  # Total liquidation bars (±100 up/down)
PRICE_RANGE = 0.75  # ±75% price range for liquidation bins
WEBSOCKET_INTERVAL = 5000  # Update interval (ms)

# Binance API credentials
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Initialize websocket manager
twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
bids = {}
asks = {}
lock = threading.Lock()

def handle_book_message(msg):
    """Handle order book updates from websocket."""
    try:
        if 'e' in msg and msg['e'] == 'depthUpdate':
            with lock:
                for bid in msg['b']:
                    price, qty = float(bid[0]), float(bid[1])
                    if qty == 0:
                        bids.pop(price, None)
                    else:
                        bids[price] = qty
                for ask in msg['a']:
                    price, qty = float(ask[0]), float(ask[1])
                    if qty == 0:
                        asks.pop(price, None)
                    else:
                        asks[price] = qty
        elif 'lastUpdateId' in msg:
            with lock:
                bids.clear()
                asks.clear()
                for bid in msg['bids']:
                    price, qty = float(bid[0]), float(bid[1])
                    if qty > 0:
                        bids[price] = qty
                for ask in msg['asks']:
                    price, qty = float(ask[0]), float(ask[1])
                    if qty > 0:
                        asks[price] = qty
    except Exception as e:
        print(f"Error in handle_book_message: {e}")

# Start websocket
twm.start()
twm.start_depth_socket(callback=handle_book_message, symbol=SYMBOL, depth=20)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'])

# Layout
app.layout = html.Div([
    html.H1('Real-Time Trading Dashboard', className='text-2xl font-bold text-center my-4'),
    dcc.Graph(id='candlestick-graph', className='mb-4'),
    dcc.Graph(id='order-book-graph', className='mb-4'),
    dcc.Graph(id='liq-map-graph', className='mb-4'),
    dcc.Graph(id='liq-map-monte-carlo-graph', className='mb-4'),
    dcc.Interval(id='interval-component', interval=WEBSOCKET_INTERVAL, n_intervals=0)
], className='container mx-auto p-4')

@app.callback(
    [Output('candlestick-graph', 'figure'),
     Output('order-book-graph', 'figure'),
     Output('liq-map-graph', 'figure'),
     Output('liq-map-monte-carlo-graph', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    try:
        df = fetch_binance_data(API_KEY, API_SECRET, symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
        df_processed = process_data(df)
    except Exception as e:
        print(f"Error fetching candlestick data: {e}")
        df_processed = pd.DataFrame()

    with lock:
        bids_df = pd.DataFrame(list(bids.items()), columns=['price', 'quantity']).sort_values('price', ascending=False)
        asks_df = pd.DataFrame(list(asks.items()), columns=['price', 'quantity']).sort_values('price', ascending=True)

    try:
        bids_processed, asks_processed = process_order_book(bids_df, asks_df)
    except Exception as e:
        print(f"Error processing order book: {e}")
        bids_processed = pd.DataFrame(columns=['price', 'quantity', 'cum_quantity'])
        asks_processed = pd.DataFrame(columns=['price', 'quantity', 'cum_quantity'])

    try:
        candlestick_fig = create_candlestick_figure(df_processed, symbol=SYMBOL)
    except Exception as e:
        print(f"Error creating candlestick figure: {e}")
        candlestick_fig = go.Figure()

    try:
        order_book_fig = create_order_book_figure(bids_processed, asks_processed)
    except Exception as e:
        print(f"Error creating order book figure: {e}")
        order_book_fig = go.Figure()

    try:
        liq_map_fig = create_coinglass_liq_map(client, symbol=SYMBOL, bins=BINS, price_range=PRICE_RANGE)
    except Exception as e:
        print(f"Error creating Coinglass liquidation map figure: {e}")
        liq_map_fig = go.Figure()

    try:
        liq_map_monte_carlo_fig = create_liq_map_figure(client, symbol=SYMBOL, bins=BINS, price_range=PRICE_RANGE)
    except Exception as e:
        print(f"Error creating Monte Carlo liquidation map figure: {e}")
        liq_map_monte_carlo_fig = go.Figure()

    return candlestick_fig, order_book_fig, liq_map_fig, liq_map_monte_carlo_fig

if __name__ == '__main__':
    app.run(debug=True)
    # Await the coroutine properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(twm.stop_client())
    loop.close()
