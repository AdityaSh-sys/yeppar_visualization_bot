import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import google.generativeai as genai

# ======= GEMINI API KEY =======
GEMINI_API_KEY = 'AIzaSyCDVT34vxFU_q3ce24_qQYow3ZdE2TDasI'  # Replace with your real Gemini API Key!
# ==============================

genai.configure(api_key=GEMINI_API_KEY)

def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(('xlsx', 'xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def ask_gemini(instruction, columns, preview, viz_type=None):
    if viz_type and viz_type.lower() != "let gemini pick":
        viz_clause = f"Always use a {viz_type.lower()} chart if you answer with a visualization."
    else:
        viz_clause = "If a visual is better than a text answer, pick the best chart shape yourself."

    prompt = f"""
You are a helpful data assistant. The user is not technical and may type very simple or vague questions about their data, with or without selecting a chart-type preference. 
You have a pandas DataFrame named df with columns: {columns}
Preview:
{preview}

User: {instruction}
Chart type: {viz_type}

{viz_clause}
- If you answer with a visualization, OUTPUT ONLY valid, executable Python plotting code (using matplotlib or seaborn, plt and sns are already imported). Output code inside a markdown code block. Do NOT add any explanation, comments, or extra plain text.
- If you answer with a summary or text, OUTPUT ONLY a one-sentence plain answer. Never return code and text in the same response.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"GEN_AI_ERROR: {e}"

def extract_code_and_kind(gemini_response):
    # Try to detect code block (prefers python code block, but also plain code block)
    if gemini_response.lstrip().startswith("```"):
        code_block = gemini_response.split("```")[1]
        # Remove 'python' if present
        if code_block.lstrip().startswith("python"):
            code_block = code_block[len("python"):].lstrip("\n")
        return "code", code_block.strip()
    # Check for plt.show() in plain responses (sometimes Gemini omits code block)
    if "plt.show()" in gemini_response:
        return "code", gemini_response
    return "text", gemini_response.strip()

def try_execute_code(code, df):
    buf = io.BytesIO()
    import matplotlib.pyplot as plt
    import seaborn as sns
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

# Streamlit UI
st.set_page_config(page_title="Chat Data Explorer", page_icon="📊", layout="centered")
st.title("📊 Chat with Your Data")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_df' not in st.session_state:
    st.session_state.data_df = None

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None:
        st.session_state.data_df = df
        st.success(f"File loaded: {uploaded_file.name}")
        st.dataframe(df.head())
        st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
else:
    st.info("Upload a file to start.")

if st.session_state.data_df is not None:
    st.divider()
    st.subheader("Ask about your data (and optionally pick a chart style)")
    for msg in st.session_state.chat_history:
        if msg['role'] == "user":
            st.markdown(f"🧑‍💻 **You:** {msg['content']}")
        elif msg['role'] == "bot":
            if msg["type"] == "text":
                st.markdown(f"😊 **Bot:** {msg['content']}")
            elif msg["type"] == "plot":
                st.markdown(f"😊 **Bot:** Here’s what I found!")
                st.image(msg["img"], use_column_width=True)
            elif msg["type"] == "code-debug":
                with st.expander("Debug: See the exact code from Gemini"):
                    st.code(msg["code"], language="python")

    viz_options = [
        "Let Gemini pick", "Bar", "Line", "Pie", "Scatter"
    ]
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 2])
        with col1:
            user_query = st.text_input("Type your question (e.g. 'compare the regions', or just 'show changes')")
        with col2:
            viz_type = st.selectbox("Chart type (optional)", viz_options)
        submitted = st.form_submit_button("Send")

        if submitted and user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            df = st.session_state.data_df
            columns = ', '.join(df.columns)
            preview = df.head(5).to_string(index=False)
            with st.spinner("Gemini is thinking..."):
                gemini_response = ask_gemini(user_query, columns, preview, viz_type)
            kind, out = extract_code_and_kind(gemini_response)
            if kind == "code":
                # Always show Gemini code for debugging under an expander
                st.session_state.chat_history.append({"role": "bot", "type": "code-debug", "code": out})
                fig_data, err = try_execute_code(out, df)
                if fig_data:
                    st.session_state.chat_history.append({"role": "bot", "type": "plot", "img": fig_data})
                else:
                    st.session_state.chat_history.append(
                        {"role": "bot", "type": "text",
                         "content": f"Sorry, I couldn't make the chart: {err}"}
                    )
            elif kind == "text":
                st.session_state.chat_history.append({"role": "bot", "type": "text", "content": out})
            st.rerun()

st.markdown("""
---
**Example questions:**  
- "Show how sales changed"  
- "Compare the regions"  
- "Who sold the most?"  
- (Choose "Bar", "Line", "Pie" or "Scatter" if you want a style!)

_You can upload `.csv`, `.xlsx`, or `.xls` files._
""")