import os
import time
import datetime
import streamlit as st
from schwab import auth
from authlib.integrations.base_client.errors import InvalidTokenError

@st.cache_resource
def get_schwab_client():
    """Initializes and returns the authenticated Schwab API client."""
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

def check_auth_status():
    """
    Renders an Auth Status badge in the Streamlit sidebar.
    Returns the active client or stops execution if token is missing/expired.
    """
    token_path = "token.json"
    
    with st.sidebar:
        st.divider()
        st.subheader("🔑 Auth Status")
        
        if not os.path.exists(token_path):
            st.error("🔒 `token.json` missing!")
            st.caption("Run `python auth_test.py` in terminal.")
            st.stop()
            
        file_mod_time = os.path.getmtime(token_path)
        created_dt = datetime.datetime.fromtimestamp(file_mod_time)
        age_hours = (time.time() - file_mod_time) / 3600.0
        time_remaining_hours = max(0.0, 168.0 - age_hours)
        
        st.caption(f"**Last Authed:** {created_dt.strftime('%Y-%m-%d %H:%M')}")
        
        if age_hours >= 168.0:
            st.error("🛑 Token Expired (>7 days)")
            st.caption("Run `python auth_test.py` to re-authenticate.")
            st.stop()
        elif age_hours >= 144.0:
            st.warning(f"⚠️ Expires in {time_remaining_hours:.1f} hrs")
        else:
            st.success(f"🟢 Active ({time_remaining_hours/24:.1f} days left)")

    client = get_schwab_client()
    if not client:
        st.error("🔑 Failed to connect to Schwab API. Please re-authenticate.")
        st.stop()
        
    return client