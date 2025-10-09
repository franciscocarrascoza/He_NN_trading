import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d

# Constants
MMR = 0.004
LEVERAGE_TIERS = [5, 10, 20, 50, 75, 100]

def fetch_futures_data(client, symbol='BTCUSDT'):
    ticker = client.futures_mark_price(symbol=symbol)
    current_price = float(ticker['markPrice'])
    
    oi_data = client.futures_open_interest(symbol=symbol)
    total_oi = float(oi_data['openInterest'])
    
    ls_data = client.futures_global_longshort_ratio(symbol=symbol, period='5m')
    ls_ratio = ls_data[0]
    long_ratio = float(ls_ratio['longAccount'])
    short_ratio = 1 - long_ratio
    
    funding = client.futures_funding_rate(symbol=symbol, limit=1)
    funding_rate = float(funding[0]['fundingRate'])
    
    klines = client.futures_klines(symbol=symbol, interval='1h', limit=24)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
               'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored']
    df_k = pd.DataFrame(klines, columns=columns)
    df_k['close'] = df_k['close'].astype(float)
    volatility = df_k['close'].pct_change().std()
    
    return current_price, total_oi, long_ratio, short_ratio, funding_rate, volatility

def estimate_liq_levels(client, symbol='BTCUSDT', num_samples=100000):
    current_price, total_oi, long_ratio, short_ratio, funding_rate, volatility = fetch_futures_data(client, symbol)
    
    leverage_weights = np.array([0.3, 0.3, 0.25, 0.15, 0.1, 0.1])  # Fixed: 6 elements to match LEVERAGE_TIERS
    if funding_rate > 0.0001:
        leverage_weights[3] += 0.1
    leverage_weights /= leverage_weights.sum()
    
    np.random.seed(42)
    long_entries = np.random.normal(current_price * 1.005, current_price * volatility, int(num_samples * long_ratio))
    short_entries = np.random.normal(current_price * 0.995, current_price * volatility, int(num_samples * short_ratio))
    
    liq_prices = []
    weights = []
    
    for i, leverage in enumerate(LEVERAGE_TIERS):
        prob = leverage_weights[i]
        
        long_liq = long_entries * (1 - (1 / leverage) + MMR)
        liq_prices.extend(long_liq)
        if len(long_entries) > 0:
            weights.extend([total_oi * long_ratio * prob / len(long_entries)] * len(long_entries))
        
        short_liq = short_entries * (1 + (1 / leverage) - MMR)
        liq_prices.extend(short_liq)
        if len(short_entries) > 0:
            weights.extend([total_oi * short_ratio * prob / len(short_entries)] * len(short_entries))
    
    return pd.DataFrame({'liq_price': liq_prices, 'weight': weights}), current_price

def create_coinglass_liq_map(client, symbol, bins, price_range):
    ticker = client.futures_mark_price(symbol=symbol)
    P = float(ticker['markPrice'])
    oi_data = client.futures_open_interest(symbol=symbol)
    OI = float(oi_data['openInterest'])
    ls_data = client.futures_global_longshort_ratio(symbol=symbol, period='5m')
    long_ratio = float(ls_data[0]['longAccount'])
    short_ratio = 1 - long_ratio
    funding = client.futures_funding_rate(symbol=symbol, limit=1)
    funding_rate = float(funding[0]['fundingRate'])
    
    klines = client.futures_klines(symbol=symbol, interval='1h', limit=24)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
               'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored']
    df = pd.DataFrame(klines, columns=columns)
    df[['close', 'volume']] = df[['close', 'volume']].astype(float)
    
    price_min, price_max = P * (1 - price_range), P * (1 + price_range)
    bins_edges = np.linspace(price_min, price_max, bins + 1)
    df['price_bin'] = pd.cut(df['close'], bins=bins_edges, labels=bins_edges[:-1], include_lowest=True)
    volume_by_price = df.groupby('price_bin', observed=True)['volume'].sum().reset_index()
    volume_by_price['price_bin'] = volume_by_price['price_bin'].astype(float)
    total_volume = volume_by_price['volume'].sum()
    volume_by_price['weight'] = volume_by_price['volume'] / (total_volume + 1e-10)
    
    leverage_weights = np.array([0.3, 0.3, 0.25, 0.15, 0.1, 0.1])
    if funding_rate > 0.0001:
        leverage_weights[-2:] += 0.1
    leverage_weights /= leverage_weights.sum()
    
    density = np.zeros(bins)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    
    for i, L in enumerate(LEVERAGE_TIERS):
        offset_long = P * (1 - 1/L + MMR)
        offset_short = P * (1 + 1/L - MMR)
        
        bin_long = np.digitize(offset_long, bins_edges) - 1
        bin_short = np.digitize(offset_short, bins_edges) - 1
        
        for j, price in enumerate(bin_centers):
            vol_weight = volume_by_price[volume_by_price['price_bin'] == price]['weight'].iloc[0] if price in volume_by_price['price_bin'].values else 1.0 / bins
            w = OI * leverage_weights[i] * vol_weight * 1000000
            if 0 <= bin_long < bins:
                density[bin_long] += w * long_ratio
            if 0 <= bin_short < bins:
                density[bin_short] += w * short_ratio
    
    density = gaussian_filter1d(density, sigma=5)
    
    density_df = pd.DataFrame({'price': bin_centers, 'liq_density': density})
    max_density = density_df['liq_density'].max()
    min_density = density_df['liq_density'].min()
    norm_density = (density_df['liq_density'] - min_density) / (max_density - min_density + 1e-10)
    norm_density = np.power(norm_density, 0.5)
    
    colors = [f'rgb({int(128 * (1 - d))}, 0, {int(255 * d)})' for d in norm_density]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=density_df['price'],
        y=density_df['liq_density'],
        marker_color=colors,
        marker_line_color='darkred',
        opacity=0.9,
        width=(price_max - price_min) / bins * 1.2,
        name='Liq Density'
    ))
    fig.add_vline(
        x=P,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"${P:,.2f}",
        annotation_position="top left",
        annotation_y=0.5,
        annotation_xshift=10,
        annotation_font=dict(size=12, color="blue")
    )
    
    fig.update_layout(
        title=f'{symbol} Coinglass-like Liquidation Heatmap',
        xaxis_title='Price (USDT)',
        yaxis_title='Potential Liq Volume (USDT)',
        height=600,
        width=1000,
        xaxis=dict(range=[price_min, price_max]),
        yaxis=dict(range=[0, max_density * 1.5]),
        plot_bgcolor='white',
        font=dict(size=14)
    )
    
    return fig

def create_liq_map_figure(client, symbol='BTCUSDT', bins=200, price_range=0.75):
    """Create the liquidation heatmap figure."""
    df_liq, current_price = estimate_liq_levels(client, symbol)
    
    price_min, price_max = current_price * (1 - price_range), current_price * (1 + price_range)
    bins = np.linspace(price_min, price_max, bins + 1)
    df_liq['bin'] = pd.cut(df_liq['liq_price'], bins=bins, labels=bins[:-1], include_lowest=True)
    
    density = df_liq.groupby('bin', observed=True)['weight'].sum().reset_index()
    density['bin'] = density['bin'].astype(float)
    density = density.sort_values('bin')
    
    max_weight = density['weight'].max()
    min_weight = density['weight'].min()
    norm_weights = (density['weight'] - min_weight) / (max_weight - min_weight + 1e-10)
    
    colors = [
        f'rgb({int(255 * w)}, {int(100 * (1 - w))}, {int(100 * (1 - w))})'
        for w in norm_weights
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=density['bin'],
        y=density['weight'],
        marker_color=colors,
        marker_line_color='darkred',
        opacity=0.9,
        width=(price_max - price_min) / bins * 1.2,
        name='Liq Density'
    ))
    
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"${current_price:,.2f}",
        annotation_position="top left",
        annotation_y=0.5,
        annotation_xshift=10,
        annotation_font=dict(size=12, color="blue")
    )
    
    fig.update_layout(
        title=f'{symbol} Perpetual Liquidation Heatmap (Monte Carlo)',
        xaxis_title='Price (USDT)',
        yaxis_title='Potential Liq Volume (USDT)',
        height=600,
        width=1000,
        xaxis=dict(range=[price_min, price_max]),
        yaxis=dict(range=[0, max_weight * 1.5]),
        plot_bgcolor='white',
        font=dict(size=14)
    )
    
    return fig
