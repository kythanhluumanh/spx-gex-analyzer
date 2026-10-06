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

st.set_page_config(page_title="SPX 0-DTE Credit Spread & GEX Monitor", layout="wide")
st.title("🎯 SPX 0-DTE GEX & Credit Spread Analyzer")
st.caption("Identify Call/Put Walls, Volatility Regimes, Delta Strikes, IV Rank, Expected Move, and Compute Risk-Reward Metrics for $SPX")

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

def fetch_vix_metrics(client):
    """Fetches current VIX quote and historical 52-week high/low to calculate IV Rank."""
    try:
        vix_resp = client.get_quote("$VIX")
        if vix_resp.status_code != 200:
            return 15.0, 50.0
            
        vix_data = vix_resp.json().get("$VIX", {}).get('quote', {})
        vix_current = float(vix_data.get('lastPrice', vix_data.get('closePrice', 15.0)))
        
        vix_52_high = float(vix_data.get('52WkHigh', 30.0))
        vix_52_low = float(vix_data.get('52WkLow', 12.0))
        
        if vix_52_high > vix_52_low:
            iv_rank = ((vix_current - vix_52_low) / (vix_52_high - vix_52_low)) * 100.0
        else:
            iv_rank = 50.0
            
        return vix_current, iv_rank
    except Exception as e:
        return 15.0, 50.0

def calculate_0dte_expected_move(spot_price, atm_iv):
    """Calculates 1-Standard Deviation (68%) Expected Move for 0-DTE session."""
    dte_fraction = 1.0 / 365.0
    expected_move = spot_price * atm_iv * np.sqrt(dte_fraction)
    
    upper_expected = spot_price + expected_move
    lower_expected = spot_price - expected_move
    
    return expected_move, upper_expected, lower_expected

@st.cache_data(ttl=15)
def fetch_0dte_gex():
    client = get_schwab_client()
    symbol = "$SPX"  # Index ticker
    
    # 1. Fetch Spot Price & VIX Metrics
    vix_current, iv_rank = fetch_vix_metrics(client)

    try:
        quote_resp = client.get_quote(symbol)
        if quote_resp.status_code != 200:
            st.error(f"Failed to fetch {symbol} quote: Status {quote_resp.status_code}")
            return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0
            
        quote_data = quote_resp.json().get(symbol, {}).get('quote', {})
        spot_price = float(quote_data.get('lastPrice', quote_data.get('closePrice', 0.0)))
    except Exception as e:
        st.error(f"Error fetching quote from Schwab API: {e}")
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0

    if spot_price == 0.0:
        st.warning("Underlying spot price is 0.0. Market data may be unavailable.")
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0

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
        return pd.DataFrame(), spot_price, spot_price + 20.0, spot_price - 20.0, spot_price, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0
    
    if chain_resp.status_code != 200:
        st.error(f"Failed to fetch {symbol} option chain: Status {chain_resp.status_code}")
        return pd.DataFrame(), spot_price, spot_price + 20.0, spot_price - 20.0, spot_price, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0
        
    chain_data = chain_resp.json()
    call_map = chain_data.get('callExpDateMap', {})
    put_map = chain_data.get('putExpDateMap', {})
    
    if not call_map:
        st.warning(f"No active option expirations returned for {symbol}.")
        return pd.DataFrame(), spot_price, spot_price + 20.0, spot_price - 20.0, spot_price, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0
        
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
                    'abs_delta': abs(delta),
                    'iv': iv
                })

    process_exp_map(call_map, is_call=True)
    process_exp_map(put_map, is_call=False)
    
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        return pd.DataFrame(), spot_price, spot_price + 20.0, spot_price - 20.0, spot_price, {}, {}, vix_current, iv_rank, 0.0, 0.0, 0.0, 0.0
        
    # Extract ATM Implied Volatility
    atm_idx = (df_raw['strike'] - spot_price).abs().idxmin()
    atm_iv = df_raw.loc[atm_idx, 'iv'] if 'iv' in df_raw.columns else 0.15
    
    # Calculate Expected Move
    exp_move, exp_upper, exp_lower = calculate_0dte_expected_move(spot_price, atm_iv)

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
    
    # Strike window filter for SPX (± $120 around spot)
    window = 120.0
    grouped = grouped[(grouped['strike'] >= spot_price - window) & (grouped['strike'] <= spot_price + window)]
    
    # Wall calculation logic
    calls_above = grouped[grouped['strike'] >= spot_price].copy()
    puts_below = grouped[grouped['strike'] <= spot_price].copy()
    
    call_wall = calls_above.loc[calls_above['call_gex'].abs().idxmax()]['strike'] if not calls_above.empty else spot_price + 20.0
    put_wall = puts_below.loc[puts_below['put_gex'].abs().idxmax()]['strike'] if not puts_below.empty else spot_price - 20.0
    
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

    return grouped, spot_price, call_wall, put_wall, flip_level, put_deltas, call_deltas, vix_current, iv_rank, atm_iv, exp_move, exp_upper, exp_lower
# --- SYSTEMATIC 0-DTE DECISION-MAKING FRAMEWORK ---
with st.expander("📘 4-Step 0-DTE Decision-Making Framework (Execution Playbook)", expanded=False):
    st.markdown("""
    ### Step 1: Volatility & Regime Filter (Are conditions favorable?)
    Check **VIX**, **IV Rank**, and **ATM 0-DTE IV** first to determine strategy selection and execution style.
    """)
    
    # Render Step 1 Regime Table
    regime_data = {
        "Metric / Threshold": [
            "IV Rank < 20%",
            "IV Rank 20% – 60%",
            "IV Rank > 60% / VIX > 25",
            "ATM IV > VIX × 1.2"
        ],
        "Environment": [
            "Low Premium Environment",
            "Optimal Selling Zone",
            "High Volatility Regime",
            "Event / Intraday Stress"
        ],
        "Operational Meaning": [
            "Option premiums are cheap; low margin of safety.",
            "Normal mean-reverting environment.",
            "Expanded range, high premium, but elevated tail risk.",
            "Event risk or intraday stress priced into 0-DTE."
        ],
        "Strategic Action": [
            "Avoid iron condors. Tighten spreads (5-pt width) or skip credit selling.",
            "Standard 0-DTE credit spread setup (10-point spread width).",
            "Widen strikes further out (5-delta instead of 10-delta) or reduce contract size by 50%.",
            "Premium is rich. Favors credit spreads if structural walls hold."
        ]
    }
    st.table(pd.DataFrame(regime_data))

    st.markdown("""
    ---
    ### Step 2: Directional Bias & Volatility Bounds (Where is price constrained?)
    Locate the spot price relative to the **Gamma Flip Level** and **Intraday Vol/OI Spikes** to choose between Put Spreads, Call Spreads, or Neutral Iron Condors.

    * **Spot > Gamma Flip Level (Positive Gamma Regime):** Market makers are long gamma (buying dips, selling rallies). Price action is sticky and mean-reverting.  
      * **Action:** Favors non-directional Iron Condors or Put Credit Spreads on pullbacks.
    * **Spot < Gamma Flip Level (Negative Gamma Regime):** Market makers are short gamma (selling dips, buying rallies). Volatility accelerates rapidly.  
      * **Action:** Avoid selling Put Credit Spreads near spot. Favor Call Credit Spreads or wait for spot to reclaim the Gamma Flip Level.
    * **Intraday Vol/OI Spike Check:** Look at the **Spike Detector table**. If heavy volume (>2x OI) hits out-of-the-money puts/calls, institutional hedging or directional speculation is taking place at that strike. Treat that strike as active dynamic support/resistance.

    ---
    ### Step 3: Confluence Gate (Where is the entry boundary?)
    Verify that your target **10-Delta short strike** passes the **Double Barrier Rule** before placing an order.
    """)

    st.latex(r"\text{Put Spread Target Strike} \le \min(\text{Put Wall}, \text{Spot} - 1\sigma \text{ Expected Move})")
    st.latex(r"\text{Call Spread Target Strike} \ge \max(\text{Call Wall}, \text{Spot} + 1\sigma \text{ Expected Move})")

    st.markdown("""
    * 🟢 **Pass (`st.success`):** High probability setup. Enter trade.
    * 🟡 **Fail (`st.warning`):** The 10-delta strike sits inside either barrier.  
      * **Adjustment:** Move down to the 5-delta strike or pass on the trade.

    ---
    ### Step 4: Trade Execution & Risk-Reward Sizing
    Use the **Sidebar Risk-Reward Calculator** to finalize contract sizing and structural risk limits.

    * 💵 **Credit Threshold:** Collect at least **10% – 15% of the spread width** in credit (e.g., $1.00 to $1.50 credit on a 10-point SPX spread).
    * 🛡️ **Max Loss Sizing:** Ensure total maximum loss does not exceed **1% – 2% of total portfolio capital** per trade.
    * 🚪 **Exit Management:**
      * **Take Profit:** Set a limit order to buy back the spread at **50% – 75% max profit**.
      * **Stop Loss:** Cut the trade if spot price crosses the **Short Strike** or if credit expands to **2x – 3x initial credit received**.
    """)
# Load Data
with st.spinner("Analyzing SPX 0-DTE Option Chain, Volatility & Delta Profiles..."):
    df, spot_price, call_wall, put_wall, flip_level, put_deltas, call_deltas, vix_current, iv_rank, atm_iv, exp_move, exp_upper, exp_lower = fetch_0dte_gex()

# --- SIDEBAR: CREDIT SPREAD RISK-REWARD CALCULATOR ---
st.sidebar.header("🧮 SPX Trade Risk-Reward Calculator")
st.sidebar.caption("Evaluate 0-DTE SPX Credit Spread Risk Metrics")

spread_type = st.sidebar.radio("Spread Type", ["Put Credit Spread (Bullish)", "Call Credit Spread (Bearish)"])

if not df.empty:
    default_short_put = put_deltas.get('10d', spot_price - 20.0)
    default_short_call = call_deltas.get('10d', spot_price + 20.0)
else:
    default_short_put = 5500.0
    default_short_call = 5550.0

if spread_type == "Put Credit Spread (Bullish)":
    short_strike = st.sidebar.number_input("Short Put Strike", value=float(default_short_put), step=5.0)
    long_strike = st.sidebar.number_input("Long Put Strike", value=float(default_short_put - 10.0), step=5.0)
else:
    short_strike = st.sidebar.number_input("Short Call Strike", value=float(default_short_call), step=5.0)
    long_strike = st.sidebar.number_input("Long Call Strike", value=float(default_short_call + 10.0), step=5.0)

credit_received = st.sidebar.number_input("Net Credit Received ($ per index pt)", value=1.20, step=0.05)
contracts = st.sidebar.number_input("Contract Quantity", value=1, min_value=1, step=1)

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
st.sidebar.metric("Break-Even SPX Level", f"${breakeven:,.2f}")



# Main Page Rendering
if not df.empty:
    # 1. Macro Overview Bar & Volatility Bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPX Spot Price", f"${spot_price:,.2f}")
    c2.metric("0-DTE Call Wall", f"${call_wall:,.2f}")
    c3.metric("0-DTE Put Wall", f"${put_wall:,.2f}")
    c4.metric("Gamma Flip Level", f"${flip_level:,.2f}")

    st.markdown("---")
    
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("VIX Index", f"{vix_current:.2f}")
    v2.metric("VIX IV Rank", f"{iv_rank:.1f}%")
    v3.metric("ATM 0-DTE IV", f"{atm_iv * 100:.1f}%")
    v4.metric("0-DTE Expected Move", f"±${exp_move:,.2f}")
    v5.metric("1σ Bounds", f"${exp_lower:,.0f} - ${exp_upper:,.0f}")

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
        
        st.caption(f"**Structural Context:** Put Wall sits at **${put_wall:,.2f}** | Lower 1σ Move sits at **${exp_lower:,.2f}**.")

    with col_call_deltas:
        st.markdown("#### 🔴 Call Credit Spread Targets (Short Call Strikes)")
        c1, c2, c3 = st.columns(3)
        c1.metric("15-Delta Call", f"${call_deltas.get('15d', 0):,.2f}", help="Aggressive (~85% Prob OTM)")
        c2.metric("10-Delta Call", f"${call_deltas.get('10d', 0):,.2f}", help="Balanced (~90% Prob OTM)")
        c3.metric("5-Delta Call", f"${call_deltas.get('5d', 0):,.2f}", help="Conservative (~95% Prob OTM)")
        
        st.caption(f"**Structural Context:** Call Wall sits at **${call_wall:,.2f}** | Upper 1σ Move sits at **${exp_upper:,.2f}**.")

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

    # Expected Move Vertical Lines
    fig.add_vline(x=exp_upper, line_dash="dot", line_color="#00bcd4", annotation_text=f"+1σ Exp: {exp_upper:,.2f}")
    fig.add_vline(x=exp_lower, line_dash="dot", line_color="#00bcd4", annotation_text=f"-1σ Exp: {exp_lower:,.2f}")

    # Delta Target Overlay Lines
    fig.add_vline(x=put_deltas.get('10d', 0), line_dash="solid", line_color="#ff8a80", annotation_text=f"10Δ Put: {put_deltas.get('10d', 0):,.2f}")
    fig.add_vline(x=call_deltas.get('10d', 0), line_dash="solid", line_color="#b9f6ca", annotation_text=f"10Δ Call: {call_deltas.get('10d', 0):,.2f}")

    # Sidebar Trade Overlay Line
    fig.add_vline(x=short_strike, line_dash="dash", line_color="#e040fb", annotation_text=f"Trade Short Strike: {short_strike:,.2f}")

    fig.update_layout(
        title=f"0-DTE SPX Gamma Profile, Expected Move & Delta Targets ({unit_label} per Strike)",
        xaxis=dict(title="Strike Price", dtick=5.0),
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
                'Strike': '${:,.2f}',
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



    