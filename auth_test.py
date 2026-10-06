import os
from dotenv import load_dotenv
from schwab import auth, client

# Load credentials from .env
load_dotenv()

api_key = os.getenv("SCHWAB_API_KEY")
app_secret = os.getenv("SCHWAB_APP_SECRET")
callback_url = os.getenv("SCHWAB_CALLBACK_URL")
token_path = "token.json"

print("Connecting to Schwab API...")

# Initiates browser flow for manual authorization
c = auth.client_from_manual_flow(
    api_key=api_key,
    app_secret=app_secret,
    callback_url=callback_url,
    token_path=token_path
)

# Test API call by fetching SPX quote
response = c.get_quote('$SPX')

if response.status_code == 200:
    print("Success! SPX Quote Data received:")
    data = response.json()
    print(data)
else:
    print(f"Error fetching quote: Status Code {response.status_code}")