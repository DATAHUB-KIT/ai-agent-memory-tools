# AI Agent with Memory & Tools

A beginner-friendly Streamlit app using Google Gemini, session-based conversation memory, and Python tools.

## Features
- Gemini response generation
- Conversation memory during the current Streamlit session
- Safe basic arithmetic calculator
- Temperature conversion (C/F/K)
- Demonstration currency conversion (USD/INR/EUR/GBP; fixed example rates, not live)

## Run locally
1. Install Python and Git.
2. Open this folder in VS Code.
3. Create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.streamlit/secrets.toml` and add:
   `GEMINI_API_KEY = "your_actual_key"`
6. Run: `streamlit run app.py`

## Deploy to Streamlit Community Cloud
Push this repository to GitHub. On https://share.streamlit.io choose the repository, branch `main`, and file `app.py`. In Advanced settings / Secrets, add:
`GEMINI_API_KEY = "your_actual_key"`

Never commit your actual API key or `.streamlit/secrets.toml`.

## Memory note
This starter project uses `st.session_state`, so memory lasts for the active session. It is not durable cross-session storage or a shared user database.
