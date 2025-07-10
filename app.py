import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import google.generativeai as genai

# ========== Load API key ==========
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
        # Attempt to parse date-like columns
        for col in df.columns:
            if any(kw in col.lower() for kw in ["date", "time", "year"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Could not convert column '{col}' to datetime: {e}")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def ask_generative_model(instruction, columns, preview_summary, preview_sample, viz_type=None):
    viz_clause = "Only generate code, tables, or charts in triple backticks if the user specifically asks for data analysis or visualization. Otherwise, answer briefly in plain English."
    if viz_type and viz_type.lower() != "let assistant pick":
        viz_clause = f"If a chart is requested, prefer a {viz_type.lower()} chart."

    prompt = f"""
You are a helpful data assistant.
You have a Pandas DataFrame named `df` with columns: {columns}

Instructions:

- If the user query is about data content (showing rows, a certain row, columns, values, statistics, counts, totals, minimum, maximum, average, filtering by column/date), DO NOT answer in English; instead, always reply with the exact Python code that gets the value or DataFrame, assigning the result to a variable named `result`, inside triple backticks (```python ... ```).
- For calculation queries ("total", "count", "sum", "average"), always output as a one-row DataFrame/Table.
- For chart/drawing requests, output code for the plot, assign any table result to `result`, and include plt.show().
- For vague/description queries ("what is this data", "describe the data", "summary"), use a short conversational answer in English.
- NEVER reply with "I don't have access to the data" or "Please provide the DataFrame"—the DataFrame is available as `df`.
- If the query is "last row", reply with code for `result = df.tail(1)` (not with an explanation).
- Only respond with either a code block or a text block; never mix explanations and code.
- Never use placeholders; always use the real values from the data when possible.

Data Summary: {preview_summary}
Sample Data (first 10 rows): {preview_sample}
User Query: "{instruction}"
Chart preference: {viz_type}
{viz_clause}
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
    elif "plt.show()" in response or "df[" in response or "df." in response or "result" in response:
        return "code", response
    return "text", response.strip()

def try_execute_code(code, df):
    buf = io.BytesIO()
    plt.close("all")
    locs = {'df': df.copy(), 'plt': plt, 'sns': sns, 'pd': pd}
    try:
        exec(code, {}, locs)
        plot, table = None, None
        if "plt.show()" in code:
            fig = plt.gcf()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plot = buf
        result = locs.get("result")
        if isinstance(result, (pd.DataFrame, pd.Series)):
            table = result if isinstance(result, pd.DataFrame) else result.to_frame()
        # If a series with unnamed index, better to put to_frame().T for single values (e.g., count)
        if isinstance(table, pd.DataFrame) and table.shape == (1, 1) and table.columns[0] == 0:
            table.columns = ["Value"]
        if plot and table is not None:
            return "both", {"plot": plot, "table": table}, None
        elif plot:
            return "plot", plot, None
        elif table is not None:
            return "table", table, None
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
    "<h1 style='display:flex; align-items:center; gap:12px;'>📊 Data Chatbot</h1>"
    "<p>Ask anything about your data. Get insights, charts, or tables with natural language.<br/>"
    "<i>For best results, be specific, like: 'Show total revenue by region as bar chart', or just ask 'What does this data describe?'</i></p>",
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
    # Show chat including download/export for tables/plots
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            chat_area.markdown(
                f"<div style='background-color:#f8f9fa; padding:10px 15px; border-radius:10px;'>"
                f"<b>{icons['user']} You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "bot":
            if msg["type"] == "text":
                chat_area.markdown(
                    f"<div style='background-color:#fffbe7; padding:10px 15px; border-radius:10px;'>"
                    f"{icons['bot']} Assistant: {msg['content']}</div>", unsafe_allow_html=True)
            elif msg["type"] == "plot":
                chat_area.markdown(f"{icons['plot']} Chart Output:")
                chat_area.image(msg["img"], use_container_width=True)
                chat_area.download_button(
                    label="🖼️ Download Chart as PNG",
                    data=msg["img"].getvalue(),
                    file_name=f"chart_{i}.png",
                    mime="image/png",
                    use_container_width=True
                )
            elif msg["type"] == "table":
                chat_area.markdown(f"{icons['bot']} Assistant: Here is the data you asked for:")
                chat_area.dataframe(msg["table"], use_container_width=True)
                csv = msg["table"].to_csv(index=False).encode('utf-8')
                chat_area.download_button(
                    label="⬇️ Export Table as CSV",
                    data=csv,
                    file_name=f"exported_table_{i}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            elif msg["type"] == "both":
                chat_area.markdown(f"{icons['plot']} Chart Output:")
                chat_area.image(msg["plot"], use_container_width=True)
                chat_area.download_button(
                    label="🖼️ Download Chart as PNG",
                    data=msg["plot"].getvalue(),
                    file_name=f"chart_{i}.png",
                    mime="image/png",
                    use_container_width=True
                )
                chat_area.markdown(f"{icons['bot']} Assistant: Here is the data you asked for:")
                chat_area.dataframe(msg["table"], use_container_width=True)
                csv = msg["table"].to_csv(index=False).encode('utf-8')
                chat_area.download_button(
                    label="⬇️ Export Table as CSV",
                    data=csv,
                    file_name=f"exported_table_{i}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    # Export full chat history
    if st.session_state.chat_history:
        chat_log = "\n\n".join([
            f"User: {msg['content']}" if msg["role"] == "user" else
            f"Assistant: {msg['content']}" if msg["type"] == "text" else
            f"Assistant: [Chart/Table Output]"
            for msg in st.session_state.chat_history
        ])
        st.download_button(
            label="📝 Export Chat History as TXT",
            data=chat_log.encode("utf-8"),
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

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
            preview_sample = df.head(10).to_string(index=False)  # head for speed
            with st.spinner("Assistant is thinking..."):
                response = ask_generative_model(user_query, columns, preview_summary, preview_sample, viz_type)

            kind, output = extract_code_and_kind(response)
            if kind == "code":
                result_type, result_data, err = try_execute_code(output, df)
                if result_type == "plot":
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "plot", "img": result_data
                    })
                elif result_type == "table":
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "table", "table": result_data
                    })
                elif result_type == "both":
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "both", "plot": result_data["plot"], "table": result_data["table"]
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "bot", "type": "text", "content": f"Sorry, code failed: {err}"
                    })
            else:
                st.session_state.chat_history.append({
                    "role": "bot", "type": "text", "content": output
                })
            st.rerun()
else:
    st.markdown(
        "<div style='margin-top:40px; color:gray; text-align:center;'>"
        "⬅️ Upload a file in the sidebar to get started.<br><i>Example: Show cases of typhoid in 2020</i></div>",
        unsafe_allow_html=True
    )