import json
import ast
import operator
import streamlit as st
from google import genai

st.set_page_config(page_title="AI Agent with Memory & Tools", page_icon="🤖", layout="centered")
st.title("🤖 AI Agent with Memory & Tools")
st.caption("Gemini + conversation memory + Python tools")

# API key: Streamlit Cloud uses app Settings > Secrets.
# Locally, put GEMINI_API_KEY in .streamlit/secrets.toml.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = ""

if not API_KEY:
    st.error("Gemini API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml locally or Streamlit Cloud Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.5-flash"

# --- Tools available to the agent ---
def safe_calculate(expression: str) -> str:
    """Evaluate basic arithmetic only. No Python function calls or variables."""
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent too large")
            return allowed_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic is allowed.")
    try:
        return str(visit(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Calculation error: {exc}"

def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    from_unit, to_unit = from_unit.upper(), to_unit.upper()
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        return "Unsupported source unit. Use C, F, or K."
    if to_unit == "C":
        result = celsius
    elif to_unit == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit == "K":
        result = celsius + 273.15
    else:
        return "Unsupported target unit. Use C, F, or K."
    return f"{value:g}°{from_unit} = {result:.2f}°{to_unit}"

def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    # EDUCATIONAL DEMO ONLY: fixed example rates, not live market data.
    rates_to_usd = {"USD": 1.0, "INR": 0.012, "EUR": 1.09, "GBP": 1.28}
    source, target = from_currency.upper(), to_currency.upper()
    if source not in rates_to_usd or target not in rates_to_usd:
        return "Supported currencies: USD, INR, EUR, GBP."
    converted = amount * rates_to_usd[source] / rates_to_usd[target]
    return (f"{amount:.2f} {source} ≈ {converted:.2f} {target} "
            "(demo rate only; not a live exchange rate).")

TOOL_GUIDE = """
You are a helpful AI assistant with access to three tools.
Choose a tool only when the user request needs it. Return ONLY valid JSON in one of these formats:
{"tool":"calculate","args":{"expression":"(12+8)*3"}}
{"tool":"temperature","args":{"value":25,"from_unit":"C","to_unit":"F"}}
{"tool":"currency","args":{"amount":1000,"from_currency":"INR","to_currency":"USD"}}
{"tool":"none","answer":"your direct answer"}
Tools:
- calculate: basic arithmetic expression.
- temperature: convert between C, F, K.
- currency: convert between USD, INR, EUR, GBP using demonstration rates.
If the request is not a tool request, use tool=none. Do not claim currency rates are live.
Conversation history is supplied below. Use it to understand references such as 'that', 'the previous answer', and follow-up questions.
"""

def ask_agent(user_text: str, history: list[dict]) -> tuple[str, str]:
    # Keep recent conversation context in the prompt.
    recent = history[-12:]
    history_text = "\n".join(
        f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
        for item in recent
    ) or "(No previous conversation.)"

    decision_prompt = (
        f"{TOOL_GUIDE}\n\nRecent conversation:\n{history_text}\n\n"
        f"Current user request: {user_text}\nJSON response:"
    )
    response = client.models.generate_content(model=MODEL, contents=decision_prompt)
    raw = (response.text or "").strip()
    # Be tolerant if the model wraps JSON in a markdown fence.
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return raw, "No tool used (model returned non-JSON)."

    tool = decision.get("tool", "none")
    if tool == "calculate":
        result = safe_calculate(str(decision.get("args", {}).get("expression", "")))
    elif tool == "temperature":
        args = decision.get("args", {})
        result = convert_temperature(float(args["value"]), str(args["from_unit"]), str(args["to_unit"]))
    elif tool == "currency":
        args = decision.get("args", {})
        result = currency_convert(float(args["amount"]), str(args["from_currency"]), str(args["to_currency"]))
    elif tool == "none":
        return str(decision.get("answer", "I couldn't form an answer.")), "No tool used"
    else:
        return "I couldn't identify a supported tool for that request.", f"Unknown tool: {tool}"

    final_prompt = (
        "Answer the user clearly using the tool result below. Do not alter numeric results. "
        "For currency, preserve the demo-rate warning.\n"
        f"User request: {user_text}\nTool used: {tool}\nTool result: {result}"
    )
    final_response = client.models.generate_content(model=MODEL, contents=final_prompt)
    return (final_response.text or result), f"Tool used: {tool} → {result}"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_log" not in st.session_state:
    st.session_state.tool_log = []

with st.sidebar:
    st.subheader("Agent controls")
    st.write("**Memory:** current browser session")
    st.write("**Model:**", MODEL)
    if st.button("Clear conversation memory"):
        st.session_state.messages = []
        st.session_state.tool_log = []
        st.rerun()
    st.caption("Currency conversion uses example rates, not live exchange rates.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_note"):
            with st.expander("Agent tool activity"):
                st.write(message["tool_note"])

user_input = st.chat_input("Ask a question, calculate something, or convert units...")
if user_input:
    previous_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                answer, tool_note = ask_agent(user_input, previous_history)
            except Exception as exc:
                answer = f"Something went wrong: {exc}\n\nCheck your API key and try again."
                tool_note = "Agent request failed."
        st.markdown(answer)
        with st.expander("Agent tool activity"):
            st.write(tool_note)
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "tool_note": tool_note
    })
    st.session_state.tool_log.append(tool_note)
