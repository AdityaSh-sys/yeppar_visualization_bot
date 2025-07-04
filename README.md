# Yeppar Visualization Bot

# 📊 AI-Powered Data Chatbot

Interact with your CSV or Excel data using natural language! This Streamlit app uses the Gemini API to answer your questions, create visualizations, and summarize your data — all with simple English queries.

---

## 🚀 Features

- 🔍 Upload `.csv`, `.xlsx`, or `.xls` files
- 💬 Ask natural language questions like:
  - `"Show sales trends over time"`
  - `"Compare regions in a pie chart"`
- 📈 Automatically generates visualizations (Bar, Line, Pie, Scatter)
- 🧠 Intelligent chart summarization in plain English
- 📦 Built with Streamlit, Pandas, Matplotlib, and Gemini API

---

## 🛠️ Installation

1. **Clone the repo**  
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt


🔐 API Key Setup

Create a .env file in the root directory:
GEMINI_API_KEY=your_actual_gemini_api_key

Or if deploying to Streamlit Cloud, go to:

App Settings → Secrets
Add this:
GEMINI_API_KEY = "your_actual_gemini_api_key"


🧪 Run the App

streamlit run app.py
Then open http://localhost:8501 in your browser.


🧪 Example Prompts
Show sales over time

Compare region-wise performance

Create a pie chart of categories

Trend in customer growth


