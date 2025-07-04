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
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def ask_gemini(instruction, columns, preview, viz_type=None):
    if viz_type and viz_type.lower() != "let gemini pick":
        viz_clause = f"Always use a {viz_type.lower()} chart if you answer with a visualization."
    else:
        viz_clause = "If a visual is better than a text answer, pick the best chart shape yourself."

    prompt = f"""
You are a helpful data assistant. The user is not technical and may type very simple or vague questions about their data.
You have a pandas DataFrame named df with columns: {columns}
Preview:
{preview}

User: {instruction}
Chart type: {viz_type}
{viz_clause}

- If answering with visualization, OUTPUT ONLY valid Python code using matplotlib/seaborn inside ```python code blocks```.
- If answering with summary, return ONLY a one-line plain text.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"GEN_AI_ERROR: {e}"

def extract_code_and_kind(gemini_response):
    if gemini_response.lstrip().startswith("```"):
        code_block = gemini_response.split("```")[1]
        if code_block.lstrip().startswith("python"):
            code_block = code_block[len("python"):].lstrip("\n")
        return "code", code_block.strip()
    if "plt.show()" in gemini_response:
        return "code", gemini_response
    return "text", gemini_response.strip()

def try_execute_code(code, df):
    buf = io.BytesIO()
    plt.close("all")
    locs = {'df': df, 'plt': plt, 'sns': sns}
    try:
        exec(code, {}, locs)
        fig = plt.gcf()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf, None
    except Exception as e:
        return None, str(e)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Chat Data Explorer", page_icon="📊", layout="wide")

# -- Sidebar Setup --
st.sidebar.header("📂 Upload your data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if st.sidebar.button("Clear Session", use_container_width=True):
    st.session_state.clear()
    st.experimental_rerun()

# -- Session State --
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_df' not in st.session_state:
    st.session_state.data_df = None

# -- Load & Preview File --
if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None:
        st.session_state.data_df = df
        st.sidebar.success(f"📄 {uploaded_file.name}")
        st.sidebar.caption(f"{round(uploaded_file.size/1024, 1)} KB")
        st.sidebar.markdown("**Preview:**")
        st.sidebar.dataframe(df.head(), use_container_width=True, height=200)
        st.sidebar.info(f"**Rows:** {df.shape[0]}  \n**Cols:** {df.shape[1]}")
else:
    st.sidebar.info("Upload a file to start.")

# -- Main Title --
st.markdown(
    """
    <h1 style='display:flex; align-items:center; gap:12px;'>📊 Data Chatbot</h1>
    <p>Interact with your data using natural language.<br>
    <span style='color:gray;font-size:14px;'>Ask questions or request charts in plain English.</span></p>
    """,
    unsafe_allow_html=True
)

# -- Main Chat Logic --
if st.session_state.data_df is not None:
    st.divider()
    st.subheader("Chat about your data:")
    chat_area = st.container()
    icons = {
        "user": "🧑‍💻",
        "bot": "🤖",
        "plot": "🖼️",
        "code-debug": "🛠️"
    }
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with chat_area:
                st.markdown(
                    f"<div style='background-color:#f8f9fa; padding:10px 15px; border-radius:10px;'>"
                    f"<b>{icons['user']} You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "bot":
            if msg.get("type") == "text":
                with chat_area:
                    st.markdown(
                        f"<div style='background-color:#fffbe7; padding:10px 15px; border-radius:10px;'>"
                        f"<b>{icons['bot']} Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
            elif msg.get("type") == "plot":
                with chat_area:
                    st.markdown(
                        f"<div style='background-color:#eafbf7; padding:10px 15px; border-radius:10px;'>"
                        f"<b>{icons['plot']} Chart:</b></div>", unsafe_allow_html=True)
                    st.image(msg["img"], use_column_width=True)
            elif msg.get("type") == "code-debug":
                with chat_area:
                    with st.expander("🔧 Code generated by Gemini"):
                        st.code(msg["code"], language="python")

    st.divider()

    # -- Chat Input Form --
    viz_options = ["Let Gemini pick", "Bar", "Line", "Pie", "Scatter"]
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 2])
        with col1:
            user_query = st.text_input("Ask a question or describe a chart...", placeholder="e.g. Trend of revenue by year")
        with col2:
            viz_type = st.selectbox("Chart Style", viz_options)
        submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            df = st.session_state.data_df
            columns = ', '.join(df.columns)
            preview = df.head(5).to_string(index=False)
            with st.spinner("Gemini is thinking..."):
                gemini_response = ask_gemini(user_query, columns, preview, viz_type)
            kind, out = extract_code_and_kind(gemini_response)
            if kind == "code":
                st.session_state.chat_history.append({"role": "bot", "type": "code-debug", "code": out})
                fig_data, err = try_execute_code(out, df)
                if fig_data:
                    st.session_state.chat_history.append({"role": "bot", "type": "plot", "img": fig_data})
                else:
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "text",
                        "content": f"Sorry, chart code failed: {err}"
                    })
            else:
                st.session_state.chat_history.append({"role": "bot", "type": "text", "content": out})
            st.rerun()

else:
    st.markdown(
        "<div style='margin-top:40px; color:gray; text-align:center;'>"
        "⬅️ Start by uploading a table in the sidebar.<br>"
        "Sample questions: <br> <i>Show profits by year, Compare two items, Trend of sales over time</i></div>",
        unsafe_allow_html=True
    )
