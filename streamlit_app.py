import os
import tempfile
import hashlib
import pickle
import joblib
import time

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from file_checker import checkFile, extract_info

# ───────────────────────────────────────────────────────────────────────────────
# Page Setup
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyst Malware Platform",
    layout="wide",
    page_icon="🛡️"
)

# ───────────────────────────────────────────────────────────────────────────────
# Theme Definitions & Dynamic Theming
# ───────────────────────────────────────────────────────────────────────────────
THEMES = {
    "Light": {"body_bg": "#f8f9fa", "text_color": "#212529", "sidebar_bg": "#ffffff"},
    "Dark": {"body_bg": "#121212", "text_color": "#e0e0e0", "sidebar_bg": "#1e1e1e"},
    "Solarized Light": {"body_bg": "#fdf6e3", "text_color": "#657b83", "sidebar_bg": "#eee8d5"},
    "Solarized Dark": {"body_bg": "#002b36", "text_color": "#839496", "sidebar_bg": "#073642"}
}

def apply_theme(theme_name: str):
    theme = THEMES.get(theme_name, THEMES["Light"])
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{ background-color: {theme['body_bg']} !important; color: {theme['text_color']} !important; }}
    [data-testid="stSidebar"] {{ background-color: {theme['sidebar_bg']} !important; color: {theme['text_color']} !important; }}
    [role="tab"] {{ background-color: {theme['sidebar_bg']} !important; color: {theme['text_color']} !important; border: none !important; }}
    [role="tab"][aria-selected="true"] {{ border-bottom: 2px solid #0d6efd !important; color: #0d6efd !important; }}
    h1, h2, h3, h4 {{ color: {theme['text_color']} !important; }}
    .markdown-text-container p, .markdown-text-container li {{ color: {theme['text_color']} !important; }}
    .stDataFrame th, .stDataFrame td {{ color: {theme['text_color']} !important; }}
    button {{ background-color: {theme['sidebar_bg']} !important; color: {theme['text_color']} !important; }}
    </style>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar: Settings
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
theme_choice = st.sidebar.selectbox("Select Theme", options=list(THEMES.keys()), index=1)
apply_theme(theme_choice)

# ───────────────────────────────────────────────────────────────────────────────
# Custom CSS for Result Labels
# ───────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .result-label { font-size: 18px; margin: 0.5em 0; }
    .legit { color: #198754; font-weight: bold; }
    .malw { color: #dc3545; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────────────────────────────────────
# Load Model & Features
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/model.pkl")
    feature_list = pickle.load(open("model/features.pkl", "rb"))
    return model, feature_list

model, feature_list = load_artifacts()

# ───────────────────────────────────────────────────────────────────────────────
# Initialize In-Memory Logs
# ───────────────────────────────────────────────────────────────────────────────
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# ───────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ───────────────────────────────────────────────────────────────────────────────
tab_scan, tab_features, tab_dashboard, tab_reports, tab_about = st.tabs([
    "🔬 File Scanner", "🔍 Feature Explorer", "📈 Dashboard", "📝 Reports", "ℹ️ About"
])

# ───────────────────────────────────────────────────────────────────────────────
# Tab 1: File Scanner
# ───────────────────────────────────────────────────────────────────────────────
with tab_scan:
    st.header("🔬 File Scanner")
    st.write("Upload one or more Windows PE files (.exe, .dll) to classify them as legitimate or malware.")
    files = st.file_uploader("Select files:", type=["exe", "dll"], accept_multiple_files=True)
    if files:
        for f in files:
            suffix = os.path.splitext(f.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.getvalue()); tmp.close()

            file_hash = hashlib.sha256(f.getvalue()).hexdigest()
            features = extract_info(tmp.name)
            os.remove(tmp.name)
            if not features:
                st.error(f"Invalid PE file: {f.name}")
                continue

            df = pd.DataFrame([features])[feature_list]
            pred = model.predict(df)[0]
            label = "LEGITIMATE" if pred else "MALWARE"
            css = "legit" if pred else "malw"
            st.session_state["logs"].append({"time": time.time(), "file": f.name, "hash": file_hash, "label": label, **features})

            col1, col2 = st.columns([4,1])
            col1.markdown(f"**{f.name}** — <span class='{css} result-label'>{label}</span>", unsafe_allow_html=True)
            col2.code(file_hash[:16] + "…")

# ───────────────────────────────────────────────────────────────────────────────
# Tab 2: Feature Explorer
# ───────────────────────────────────────────────────────────────────────────────
with tab_features:
    st.header("🔍 Feature Explorer")
    st.write("View the extracted PE file features for any previously scanned file.")
    logs = st.session_state["logs"]
    if not logs:
        st.info("Scan files first to explore features.")
    else:
        files_list = [e["file"] for e in logs]
        sel = st.selectbox("Select file:", files_list)
        entry = next(e for e in logs if e["file"] == sel)
        df_feat = pd.DataFrame.from_dict({k: entry[k] for k in feature_list}, orient='index', columns=['Value'])
        df_feat.index.name = 'Feature'
        st.dataframe(df_feat, use_container_width=True)
        csv = df_feat.to_csv().encode('utf-8')
        st.download_button("📥 Download features as CSV", csv, file_name=f"{sel}_features.csv")

# ───────────────────────────────────────────────────────────────────────────────
# Tab 3: Dashboard
# ───────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.header("📈 Dashboard")
    st.write("Monitor scan counts and feature trends over time to detect patterns and anomalies.")
    df_logs = pd.DataFrame(st.session_state["logs"])
    if df_logs.empty:
        st.info("No scans to display.")
    else:
        df_logs['datetime'] = pd.to_datetime(df_logs['time'], unit='s')
        df_logs['date'] = df_logs['datetime'].dt.date
        counts = df_logs['label'].value_counts()
        st.metric("Total Scans", len(df_logs))
        st.metric("Malware Detected", counts.get('MALWARE', 0))

        st.subheader("Scan Trends Over Time")
        grouped = df_logs.groupby(['date', 'label']).size().unstack(fill_value=0)
        fig1, ax1 = plt.subplots(figsize=(6,3))
        grouped.plot(ax=ax1, marker='o')
        ax1.set_xlabel('Date'); ax1.set_ylabel('Count')
        ax1.legend(title='Label')
        col1, col2 = st.columns(2)
        col1.pyplot(fig1)
        col1.write("Daily counts of legitimate vs malware scans." )

        st.subheader("Feature Trend Analysis")
        feat = st.selectbox("Select feature to track:", feature_list)
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(df_logs['datetime'], df_logs[feat], marker='x')
        ax2.set_xlabel('Time'); ax2.set_ylabel(feat)
        col2.pyplot(fig2)
        col2.write("Tracks selected feature values across scans to spot anomalies.")

# ───────────────────────────────────────────────────────────────────────────────
# Tab 4: Reports
# ───────────────────────────────────────────────────────────────────────────────
with tab_reports:
    st.header("📝 Reports")
    st.write("Generate a consolidated PDF report of all scans performed in this session.")
    if st.button("Download PDF Report"):
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "Malware Analysis Report", ln=True)
        for ent in st.session_state["logs"]:
            pdf.set_font("Arial", size=12)
            line = f"File: {ent['file']} | Hash: {ent['hash'][:12]}... | Label: {ent['label']}"
            pdf.multi_cell(0, 8, line)
        path = "malware_report.pdf"; pdf.output(path)
        with open(path, 'rb') as f: st.download_button("Download PDF Report", f, file_name="malware_report.pdf")

# ───────────────────────────────────────────────────────────────────────────────
# Tab 5: About
# ───────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.header("ℹ️ About")
    st.write(
        "The Analyst Malware Platform provides static analysis of Windows PE files, feature exploration, "
        "real-time dashboards for detection trends, and automated report generation—all in an open-source, multi-theme environment."
    )
    st.markdown("""
    **Tabs Overview**:
    
    - **File Scanner**: Upload and classify `.exe`/`.dll` files with a single click.
    - **Feature Explorer**: Inspect the complete set of extracted features for each scan.
    - **Dashboard**: Visualize daily scan counts and track individual feature trends over time.
    - **Reports**: Export session data into a forensic-grade PDF summary.
    - **About**: Learn about the platform, theming, and next steps.
    
    **Theme Options**: Light, Dark, Solarized Light, Solarized Dark.
    **Threat Intelligence**: Integrate external APIs by setting secrets like `VT_API_KEY`.
    **Next Steps**: Data persistence, CI/CD, team collaboration tools.
    """)
