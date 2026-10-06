import os
import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm
from dotenv import load_dotenv
from schwab import auth

load_dotenv()

st.set_page_config(page_title="SPY 0-DTE Credit Spread & GEX Monitor", layout="wide")
st.title("🎯 SPY 0-DTE GEX & Credit Spread Analyzer")
st.caption("Identify Call/Put Walls, Volatility Regimes, Delta Strikes, and Compute Risk-Reward Metrics for SPY")

@st.cache_resource
def get_schwab_client():
    api_key = os.getenv("SCHWAB_API_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    token_path = "token.json"
    
    if not os.path.exists(token_path):
        st.error("Missing token.json! Run 'python auth_test.py' first.")
        st.stop()
        
    return auth.client_from_token_file(token_path, api_key, app_secret)

def calculate_greeks(S, K, T, r, sigma, is_call=True):
    """Calculates Black-Scholes Gamma and Delta."""
    T = max(T, 0.0001)       # Avoid division by zero
    sigma = max(sigma, 0.05)  # Enforce 5% IV floor
    if S <= 0 or K <= 0:
        return 0.0, 0.0
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if is_call:
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0  # Put delta is negative
        
    return gamma, delta

@st.cache_data(ttl=15)
def fetch_0dte_gex():
    client = get_schwab_client()
    symbol = "SPY"  # Equity ticker
    
    # 1. Fetch Spot Price
    try:
        quote_resp = client.get_quote(symbol)
        if quote_resp.status_code != 200:
            st.error(f"Failed to fetch {symbol} quote: Status {quote_resp.status_code}")
            return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, {}, {}
            
        quote_data = quote_resp.json().get(symbol, {}).get('quote', {})
        spot_price = float(quote_data.get('lastPrice', quote_data.get('closePrice', 0.0)))
    except Exception as e:
        st.error(f"Error fetching quote from Schwab API: {e}")
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, {}, {}

    if spot_price == 0.0:
        st.warning("Underlying spot price is 0.0. Market data may be unavailable.")
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, {}, {}

    today = datetime.date.today()
    to_date = today + datetime.timedelta(days=7)
    
    # 2. Fetch Option Chain
    try:
        chain_resp = client.get_option_chain(
            symbol=symbol,
            contract_type=client.Options.ContractType.ALL,
            strike_count=100,
            from_date=today,
            to_date=to_date,
            include_underlying_quote=False
        )
    except Exception as e:
        st.error(f"Error fetching option chain from Schwab API: {e}")
        return pd.DataFrame(), spot_price, spot_price + 3.0, spot_price - 3.0, spot_price, {}, {}
    
    if chain_resp.status_code != 200:
        st.error(f"Failed to fetch {symbol} option chain: Status {chain_resp.status_code}")
        return pd.DataFrame(), spot_price, spot_price + 3.0, spot_price - 3.0, spot_price, {}, {}
        
    chain_data = chain_resp.json()
    call_map = chain_data.get('callExpDateMap', {})
    put_map = chain_data.get('putExpDateMap', {})
    
    if not call_map:
        st.warning(f"No active option expirations returned for {symbol}.")
        return pd.DataFrame(), spot_price, spot_price + 3.0, spot_price - 3.0, spot_price, {}, {}
        
    target_exp_date = sorted(list(call_map.keys()))[0]
    records = []
    
    def process_exp_map(exp_map, is_call=True):
        if target_exp_date not in exp_map:
            return
        strikes = exp_map[target_exp_date]
        for strike_str, info_list in strikes.items():
            for opt in info_list:
                strike = float(opt['strikePrice'])
                oi = float(opt.get('openInterest', 0))
                vol = float(opt.get('totalVolume', 0))
                
                contract_weight = oi if oi > 0 else (vol if vol > 0 else 10.0)
                
                raw_iv = float(opt.get('volatility', 0))
                iv = (raw_iv / 100.0) if raw_iv > 1.0 else raw_iv
                if iv <= 0:
                    iv = 0.15
                
                dte_raw = float(opt.get('daysToExpiration', 0))
                dte = max(dte_raw, 0.1) / 365.0
                
                gamma, delta = calculate_greeks(spot_price, strike, dte, 0.045, iv, is_call=is_call)
                gex_sign = 1.0 if is_call else -1.0
                gex = gex_sign * gamma * contract_weight * 100 * (spot_price ** 2) * 0.01
                
                records.append({
                    'strike': strike,
                    'gex': gex,
                    'call_gex': gex if is_call else 0.0,
                    'put_gex': gex if not is_call else 0.0,
                    'call_vol': vol if is_call else 0,
                    'put_vol': vol if not is_call else 0,
                    'call_oi': oi if is_call else 0,
                    'put_oi': oi if not is_call else 0,
                    'is_call': is_call,
                    'delta': delta,
                    'abs_delta': abs(delta)
                })

    process_exp_map(call_map, is_call=True)
    process_exp_map(put_map, is_call=False)
    
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        return pd.DataFrame(), spot_price, spot_price + 3.0, spot_price - 3.0, spot_price, {}, {}
        
    # Aggregate metrics by strike
    grouped = df_raw.groupby('strike').agg({
        'gex': 'sum',
        'call_gex': 'sum',
        'put_gex': 'sum',
        'call_vol': 'sum',
        'put_vol': 'sum',
        'call_oi': 'sum',
        'put_oi': 'sum'
    }).reset_index()
    
    # Total Strike Volume and Open Interest
    grouped['total_vol'] = grouped['call_vol'] + grouped['put_vol']
    grouped['total_oi'] = grouped['call_oi'] + grouped['put_oi']
    
    # Combined Vol/OI Ratio
    grouped['vol_oi_ratio'] = np.where(
        grouped['total_oi'] > 0, 
        grouped['total_vol'] / grouped['total_oi'], 
        grouped['total_vol']
    )
    
    # Identify Spikes (Vol/OI >= 1.5 AND total volume >= 500 contracts)
    grouped['is_spike'] = (grouped['vol_oi_ratio'] >= 1.5) & (grouped['total_vol'] >= 500)
    
    # Strike window filter for SPY (± $15 around spot)
    window = 15.0
    grouped = grouped[(grouped['strike'] >= spot_price - window) & (grouped['strike'] <= spot_price + window)]
    
    # Wall calculation logic
    calls_above = grouped[grouped['strike'] >= spot_price].copy()
    puts_below = grouped[grouped['strike'] <= spot_price].copy()
    
    call_wall = calls_above.loc[calls_above['call_gex'].abs().idxmax()]['strike'] if not calls_above.empty else spot_price + 3.0
    put_wall = puts_below.loc[puts_below['put_gex'].abs().idxmax()]['strike'] if not puts_below.empty else spot_price - 3.0
    
    flip_idx = np.where(np.diff(np.sign(grouped['gex'])))[0]
    flip_level = grouped['strike'].iloc[flip_idx[0]] if len(flip_idx) > 0 else spot_price
    
    put_df = df_raw[df_raw['is_call'] == False].copy()
    call_df = df_raw[df_raw['is_call'] == True].copy()
    
    def get_closest_strike(df_sub, target_delta):
        if df_sub.empty:
            return spot_price
        df_sub['delta_diff'] = (df_sub['abs_delta'] - target_delta).abs()
        return df_sub.loc[df_sub['delta_diff'].idxmin()]['strike']

    put_deltas = {
        '5d': get_closest_strike(put_df, 0.05),
        '10d': get_closest_strike(put_df, 0.10),
        '15d': get_closest_strike(put_df, 0.15)
    }
    
    call_deltas = {
        '5d': get_closest_strike(call_df, 0.05),
        '10d': get_closest_strike(call_df, 0.10),
        '15d': get_closest_strike(call_df, 0.15)
    }

    return grouped, spot_price, call_wall, put_wall, flip_level, put_deltas, call_deltas

# Load Data
with st.spinner("Analyzing SPY 0-DTE Option Chain & Delta Profiles..."):
    df, spot_price, call_wall, put_wall, flip_level, put_deltas, call_deltas = fetch_0dte_gex()

# --- SIDEBAR: CREDIT SPREAD RISK-REWARD CALCULATOR ---
st.sidebar.header("🧮 SPY Trade Risk-Reward Calculator")
st.sidebar.caption("Evaluate 0-DTE SPY Credit Spread Risk Metrics")

spread_type = st.sidebar.radio("Spread Type", ["Put Credit Spread (Bullish)", "Call Credit Spread (Bearish)"])

if not df.empty:
    default_short_put = put_deltas.get('10d', spot_price - 3.0)
    default_short_call = call_deltas.get('10d', spot_price + 3.0)
else:
    default_short_put = 550.0
    default_short_call = 555.0

if spread_type == "Put Credit Spread (Bullish)":
    short_strike = st.sidebar.number_input("Short Put Strike", value=float(default_short_put), step=1.0)
    long_strike = st.sidebar.number_input("Long Put Strike", value=float(default_short_put - 1.0), step=1.0)
else:
    short_strike = st.sidebar.number_input("Short Call Strike", value=float(default_short_call), step=1.0)
    long_strike = st.sidebar.number_input("Long Call Strike", value=float(default_short_call + 1.0), step=1.0)

credit_received = st.sidebar.number_input("Net Credit Received ($ per share)", value=0.15, step=0.01)
contracts = st.sidebar.number_input("Contract Quantity", value=10, min_value=1, step=1)

# Math Calculations
spread_width = abs(short_strike - long_strike)
max_profit_per_contract = credit_received * 100
max_loss_per_contract = (spread_width - credit_received) * 100

total_max_profit = max_profit_per_contract * contracts
total_max_loss = max_loss_per_contract * contracts
return_on_risk = (max_profit_per_contract / max_loss_per_contract * 100) if max_loss_per_contract > 0 else 0.0

if spread_type == "Put Credit Spread (Bullish)":
    breakeven = short_strike - credit_received
else:
    breakeven = short_strike + credit_received

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Calculator Results")
st.sidebar.metric("Max Profit", f"${total_max_profit:,.2f}")
st.sidebar.metric("Max Risk / Loss", f"${total_max_loss:,.2f}")
st.sidebar.metric("Return on Risk (RoR)", f"{return_on_risk:.2f}%")
st.sidebar.metric("Break-Even SPY Level", f"${breakeven:,.2f}")

# Main Page Rendering
if not df.empty:
    # 1. Macro Overview Bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPY Spot Price", f"${spot_price:,.2f}")
    c2.metric("0-DTE Call Wall (Resistance)", f"${call_wall:,.2f}")
    c3.metric("0-DTE Put Wall (Support)", f"${put_wall:,.2f}")
    c4.metric("Gamma Flip Level", f"${flip_level:,.2f}")

    st.markdown("---")
    
    # 2. Delta Target Selection Cards
    st.subheader("🎯 Short Strike Delta Targets")
    col_put_deltas, col_call_deltas = st.columns(2)
    
    with col_put_deltas:
        st.markdown("#### 🟢 Put Credit Spread Targets (Short Put Strikes)")
        d1, d2, d3 = st.columns(3)
        d1.metric("5-Delta Put", f"${put_deltas.get('5d', 0):,.2f}", help="Conservative (~95% Prob OTM)")
        d2.metric("10-Delta Put", f"${put_deltas.get('10d', 0):,.2f}", help="Balanced (~90% Prob OTM)")
        d3.metric("15-Delta Put", f"${put_deltas.get('15d', 0):,.2f}", help="Aggressive (~85% Prob OTM)")
        
        st.caption(f"**Structural Context:** Put Wall sits at **${put_wall:,.2f}**. Ideal short puts sit at or below the Put Wall.")

    with col_call_deltas:
        st.markdown("#### 🔴 Call Credit Spread Targets (Short Call Strikes)")
        c1, c2, c3 = st.columns(3)
        c1.metric("15-Delta Call", f"${call_deltas.get('15d', 0):,.2f}", help="Aggressive (~85% Prob OTM)")
        c2.metric("10-Delta Call", f"${call_deltas.get('10d', 0):,.2f}", help="Balanced (~90% Prob OTM)")
        c3.metric("5-Delta Call", f"${call_deltas.get('5d', 0):,.2f}", help="Conservative (~95% Prob OTM)")
        
        st.caption(f"**Structural Context:** Call Wall sits at **${call_wall:,.2f}**. Ideal short calls sit at or above the Call Wall.")

    st.markdown("---")

    # 3. Dynamic Scaling Logic for Plotly Chart
    max_gex = max(df['call_gex'].abs().max(), df['put_gex'].abs().max())
    if max_gex >= 1e6:
        scale_factor = 1e6
        unit_label = "$ Millions"
    elif max_gex >= 1e3:
        scale_factor = 1e3
        unit_label = "$ Thousands"
    else:
        scale_factor = 1.0
        unit_label = "$ Raw Units"

    fig = go.Figure()

    # Call GEX (Green)
    fig.add_trace(go.Bar(
        x=df['strike'], 
        y=df['call_gex'] / scale_factor,
        name='Call GEX (Positive / Resistance)', 
        marker_color='#26a69a'
    ))

    # Put GEX (Red)
    fig.add_trace(go.Bar(
        x=df['strike'], 
        y=df['put_gex'] / scale_factor,
        name='Put GEX (Negative / Acceleration)', 
        marker_color='#ef5350'
    ))

    # Structural Key Vertical Lines
    fig.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot_price:,.2f}")
    fig.add_vline(x=call_wall, line_dash="dot", line_color="#00e676", annotation_text=f"Call Wall: {call_wall:,.2f}")
    fig.add_vline(x=put_wall, line_dash="dot", line_color="#ff1744", annotation_text=f"Put Wall: {put_wall:,.2f}")
    fig.add_vline(x=flip_level, line_dash="dashdot", line_color="yellow", annotation_text=f"Flip: {flip_level:,.2f}")

    # Delta Target Overlay Lines
    fig.add_vline(x=put_deltas.get('10d', 0), line_dash="solid", line_color="#ff8a80", annotation_text=f"10Δ Put: {put_deltas.get('10d', 0):,.2f}")
    fig.add_vline(x=call_deltas.get('10d', 0), line_dash="solid", line_color="#b9f6ca", annotation_text=f"10Δ Call: {call_deltas.get('10d', 0):,.2f}")

    # Sidebar Trade Overlay Line
    fig.add_vline(x=short_strike, line_dash="dash", line_color="#e040fb", annotation_text=f"Trade Short Strike: {short_strike:,.2f}")

    fig.update_layout(
        title=f"0-DTE SPY Gamma Profile & Delta Targets ({unit_label} per Strike)",
        xaxis=dict(title="Strike Price", dtick=1.0),
        yaxis=dict(title=f"Net GEX ({unit_label})", zeroline=True, zerolinecolor="gray", zerolinewidth=1.5),
        barmode='relative',
        template='plotly_dark',
        height=550
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔥 Intraday Volume & Vol/OI Spike Detector")

    # Filter strikes with active volume from the populated DataFrame
    spikes_df = df[df['total_vol'] > 100].copy()

    if not spikes_df.empty:
        spikes_df = spikes_df.sort_values(by='vol_oi_ratio', ascending=False)
        
        display_df = spikes_df[[
            'strike', 'total_vol', 'total_oi', 'vol_oi_ratio', 'call_vol', 'put_vol', 'is_spike'
        ]].copy()
        
        display_df.columns = [
            'Strike', 'Total Volume', 'Open Interest', 'Vol / OI Ratio', 'Call Vol', 'Put Vol', 'Spike Signal'
        ]
        
        st.dataframe(
            display_df.style.format({
                'Strike': '${:.2f}',
                'Total Volume': '{:,.0f}',
                'Open Interest': '{:,.0f}',
                'Vol / OI Ratio': '{:.2f}x',
                'Call Vol': '{:,.0f}',
                'Put Vol': '{:,.0f}'
            }).map(lambda v: 'background-color: #2e7d32; color: white;' if v else '', subset=['Spike Signal']),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No significant intraday volume detected yet for this session.")

else:
    st.warning("No option chain data returned from Schwab API. Check authentication or active market hours.")