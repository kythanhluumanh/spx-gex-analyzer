import os
import time
import datetime
import streamlit as st
from schwab import auth
from authlib.integrations.base_client.errors import InvalidTokenError

st.set_page_config(page_title="0-DTE GEX & Credit Spread Monitor", layout="wide")

@st.cache_resource
def get_schwab_client():
    api_key = os.getenv("SCHWAB_API_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    token_path = "token.json"
    
    if not os.path.exists(token_path):
        return None
        
    try:
        return auth.client_from_token_file(token_path, api_key, app_secret)
    except (InvalidTokenError, Exception):
        if os.path.exists(token_path):
            os.remove(token_path)
        return None

# --- TOKEN HEALTH CHECK (Runs on every page load) ---
token_path = "token.json"

if os.path.exists(token_path):
    file_mod_time = os.path.getmtime(token_path)
    created_dt = datetime.datetime.fromtimestamp(file_mod_time)
    age_hours = (time.time() - file_mod_time) / 3600.0
    time_remaining_hours = max(0.0, 168.0 - age_hours)
    
    # Display status in the sidebar
    with st.sidebar:
        st.subheader("🔑 Auth Status")
        st.caption(f"**Last Authed:** {created_dt.strftime('%Y-%m-%d %H:%M')}")
        
        if age_hours >= 168.0:
            st.error("🛑 Token Expired (>7 days). Run `python auth_test.py`.")
        elif age_hours >= 144.0:
            st.warning(f"⚠️ Token expires in {time_remaining_hours:.1f} hrs!")
        else:
            st.success(f"🟢 Token Active ({time_remaining_hours/24:.1f} days left)")
else:
    with st.sidebar:
        st.error("🔒 `token.json` Missing")

# --- APP UI ---
st.title("🎯 0-DTE GEX & Credit Spread Monitor")
st.write("Welcome! Select a dashboard from the sidebar menu on the left to begin:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 SPX 0-DTE Analyzer")
    st.write("Track SPX Gamma Exposure, Call/Put Walls, Gamma Flip levels, and 5/10/15 Delta short strike targets.")
    st.page_link("pages/1_SPX.py", label="Open SPX Dashboard", icon="📈")

with col2:
    st.subheader("📊 SPY 0-DTE Analyzer")
    st.write("Track SPY ETF Gamma Exposure, native dollar walls, and credit spread risk-reward calculations.")
    st.page_link("pages/2_SPY.py", label="Open SPY Dashboard", icon="📊")