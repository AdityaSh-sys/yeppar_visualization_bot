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

def ask_model(instruction, columns, preview):
    prompt = f"""
You are a helpful data assistant. The user may ask questions about a dataset without specifying a chart. 
You have a pandas DataFrame named df with columns: {columns}
Preview:
{preview}

User: {instruction}

- If the user asks for a chart or visualization, OUTPUT ONLY valid Python code using matplotlib/seaborn inside ```python code blocks```.
- If the user asks to see a specific part of the data (like a table or filter), return ONLY a pandas DataFrame expression that selects the desired rows/columns, inside a python code block.
- Otherwise, respond with a one-line summary, insight, or statistic from the data.
- Do not output both text and code together.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"GEN_AI_ERROR: {e}"

def generate_chart_summary(code):
    summary_prompt = f"""
You just created a chart using the following matplotlib/seaborn code:
{code}

Now write a short 1–2 line description of what this chart likely shows in plain English.
Be clear and simple. Do not mention code or the user.
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        summary_response = model.generate_content(summary_prompt)
        return summary_response.text.strip()
    except Exception as e:
        return "Summary generation failed."

def extract_code_and_kind(response):
    if response.lstrip().startswith("```"):
        code_block = response.split("```")[1]
        if code_block.lstrip().startswith("python"):
            code_block = code_block[len("python"):].lstrip("\n")
        return "code", code_block.strip()
    if "plt.show()" in response or "pd." in response or "df[" in response:
        return "code", response
    return "text", response.strip()

def try_execute_code(code, df):
    buf = io.BytesIO()
    plt.close("all")
    locs = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
    try:
        exec(code, {}, locs)
        if 'plt' in code or 'sns' in code:
            fig = plt.gcf()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return 'plot', buf, None
        elif 'df' in code:
            result_df = eval(code.strip().split("=")[-1], {}, locs)
            return 'table', result_df, None
    except Exception as e:
        return None, None, str(e)

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
    <span style='color:gray;font-size:14px;'>Ask questions, request charts or tables in plain English.</span></p>
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
        "table": "📋"
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
                        f"<b>{icons['bot']} Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)
            elif msg.get("type") == "plot":
                with chat_area:
                    st.markdown(
                        f"<div style='background-color:#eafbf7; padding:10px 15px; border-radius:10px;'>"
                        f"<b>{icons['plot']} Chart:</b></div>", unsafe_allow_html=True)
                    st.image(msg["img"], use_column_width=True)

                    if "summary" in msg:
                        st.markdown(
                            f"""
                            <div style='
                                background-color: #f2f2f2;
                                padding: 12px 16px;
                                margin-top: 10px;
                                border-left: 4px solid #4CAF50;
                                border-radius: 6px;
                                font-size: 15px;
                                color: #333;
                            '>
                            <b>📌 Summary:</b><br>{msg['summary']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            elif msg.get("type") == "table":
                with chat_area:
                    st.markdown(
                        f"<div style='background-color:#eef2ff; padding:10px 15px; border-radius:10px;'>"
                        f"<b>{icons['table']} Table:</b></div>", unsafe_allow_html=True)
                    st.dataframe(msg["df"], use_container_width=True)

    st.divider()

    # -- Chat Input Form --
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question, request a chart or table...", placeholder="e.g. Show top 5 rows where sales > 1000")
        submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            df = st.session_state.data_df
            columns = ', '.join(df.columns)
            preview = df.head(5).to_string(index=False)
            with st.spinner("Thinking..."):
                model_response = ask_model(user_query, columns, preview)
            kind, out = extract_code_and_kind(model_response)
            if kind == "code":
                output_type, result, err = try_execute_code(out, df)
                if output_type == 'plot':
                    summary = generate_chart_summary(out)
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "type": "plot",
                        "img": result,
                        "summary": summary
                    })
                elif output_type == 'table':
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "type": "table",
                        "df": result
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "text",
                        "content": f"Sorry, code failed: {err}"
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
