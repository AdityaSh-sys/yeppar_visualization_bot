import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import google.generativeai as genai

# ========== Load API key from .env ==========
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ========== Helper Functions ==========
def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(('xlsx', 'xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None

        for col in df.columns:
            if any(kw in col.lower() for kw in ["date", "time", "year"]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def ask_generative_model(instruction, columns, preview_summary, preview_sample, viz_type=None):
    viz_clause = "Only generate a chart if explicitly asked. Otherwise, answer in text or tables."
    if viz_type and viz_type.lower() != "let assistant pick":
        viz_clause = f"Always use a {viz_type.lower()} chart if visualization is required."

    prompt = f"""
You are a helpful data assistant. The user may ask questions about their uploaded data in plain language.
You have a pandas DataFrame named df with columns: {columns}
Summary: {preview_summary}
Preview:
{preview_sample}

User: {instruction}
Chart type preference: {viz_type}
{viz_clause}

- If asked for a chart, respond with valid Python code inside triple backticks (```python ... ```).
- If asked for a table, reply with the pandas code that returns the table (e.g. df[...] or df.query(...))
- If asked a simple question, respond with a short 1-line text.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"GEN_AI_ERROR: {e}"

def extract_code_and_kind(response):
    if response.lstrip().startswith("```"):
        code_block = response.split("```")[1]
        if code_block.lstrip().startswith("python"):
            code_block = code_block[len("python"):].lstrip("\n")
        return "code", code_block.strip()
    elif "plt.show()" in response or "df[" in response or "df." in response:
        return "code", response
    return "text", response.strip()

def try_execute_code(code, df):
    buf = io.BytesIO()
    plt.close("all")
    locs = {'df': df.copy(), 'plt': plt, 'sns': sns, 'pd': pd}
    try:
        exec(code, {}, locs)
        if "plt.show()" in code:
            fig = plt.gcf()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return "plot", buf, None
        elif isinstance(locs.get("df"), pd.DataFrame):
            return "table", locs["df"], None
        return None, None, "No output generated"
    except Exception as e:
        return None, None, str(e)

# ========== Streamlit UI ==========
st.set_page_config(page_title="📊 Data Chatbot", page_icon="📊", layout="wide")

st.sidebar.header("📂 Upload your data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if st.sidebar.button("Clear Session", use_container_width=True):
    st.session_state.clear()
    st.experimental_rerun()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_df' not in st.session_state:
    st.session_state.data_df = None

if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None:
        st.session_state.data_df = df
        st.sidebar.success(f"📄 {uploaded_file.name}")
        st.sidebar.caption(f"{round(uploaded_file.size / 1024, 1)} KB")
        st.sidebar.markdown("**Preview:**")
        st.sidebar.dataframe(df.head(), use_container_width=True, height=200)
        st.sidebar.info(f"**Rows:** {df.shape[0]}  \n**Cols:** {df.shape[1]}")
else:
    st.sidebar.info("Upload a file to begin.")

st.markdown(
    """
    <h1 style='display:flex; align-items:center; gap:12px;'>📊 Data Chatbot</h1>
    <p>Ask anything about your data. Get insights, charts, or tables with natural language.</p>
    """,
    unsafe_allow_html=True
)

if st.session_state.data_df is not None:
    st.divider()
    st.subheader("Chat with your data:")
    chat_area = st.container()
    icons = {
        "user": "🧑‍💻",
        "bot": "🤖",
        "plot": "📈",
        "table": "📋"
    }
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_area.markdown(
                f"<div style='background-color:#f8f9fa; padding:10px 15px; border-radius:10px;'>"
                f"<b>{icons['user']} You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "bot":
            if msg["type"] == "text":
                chat_area.markdown(
                    f"<div style='background-color:#fffbe7; padding:10px 15px; border-radius:10px;'>"
                    f"<b>{icons['bot']} Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)
            elif msg["type"] == "plot":
                chat_area.markdown(
                    f"<div style='background-color:#eafbf7; padding:10px 15px; border-radius:10px;'>"
                    f"<b>{icons['plot']} Chart:</b></div>", unsafe_allow_html=True)
                chat_area.image(msg["img"], use_container_width=True)
                if msg.get("summary"):
                    chat_area.markdown(
                        f"<div style='color:gray; font-size:14px; padding-top:5px;'>💡 {msg['summary']}</div>",
                        unsafe_allow_html=True)
            elif msg["type"] == "table":
                chat_area.markdown(f"<b>{icons['table']} Table Output:</b>")
                chat_area.dataframe(msg["table"], use_container_width=True)

    st.divider()
    viz_options = ["Let Assistant pick", "Bar", "Line", "Pie", "Scatter"]
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 2])
        with col1:
            user_query = st.text_input("Ask a question or describe a chart...", placeholder="e.g. Compare sales by year")
        with col2:
            viz_type = st.selectbox("Chart Style", viz_options)
        submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            df = st.session_state.data_df
            columns = ', '.join(df.columns)
            preview_summary = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
            preview_sample = df.sample(min(50, len(df))).to_string(index=False)
            with st.spinner("Assistant is thinking..."):
                response = ask_generative_model(user_query, columns, preview_summary, preview_sample, viz_type)

            kind, output = extract_code_and_kind(response)
            if kind == "code":
                result_type, result_data, err = try_execute_code(output, df)
                if result_type == "plot":
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "plot", "img": result_data,
                        "summary": f"Here's a visualization based on your query: '{user_query}'."
                    })
                elif result_type == "table":
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "table", "table": result_data
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "text",
                        "content": f"Sorry, code failed: {err}"})
            else:
                st.session_state.chat_history.append({
                    "role": "bot", "type": "text", "content": output})
            st.rerun()
else:
    st.markdown(
        "<div style='margin-top:40px; color:gray; text-align:center;'>"
        "⬅️ Upload a file in the sidebar to get started.<br><i>Example: Show cases of typhoid in 2020</i></div>",
        unsafe_allow_html=True
    )
