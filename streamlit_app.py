import os
import streamlit as st
from file_checker import checkFile

# Page layout
st.set_page_config(page_title="Malware Detection", layout="centered")

# Header section with custom styling
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #ff6347;
        text-align: center;
    }
    .subheader {
        font-size: 20px;
        font-weight: 600;
        color: #20b2aa;
        text-align: center;
    }
    .upload-box {
        border: 2px solid #20b2aa;
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8ff;
        text-align: center;
    }
    .button {
        background-color: #ff6347;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .result {
        font-size: 16px;
        color: #333;
        font-weight: 500;
    }
    .legitimate {
        color: #32cd32;
    }
    .malware {
        color: #ff4500;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="title">Malware Detection using Random Forest Algorithm</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subheader">Upload a file to verify</h2>', unsafe_allow_html=True)

# File uploader with attractive styling
file = st.file_uploader("Choose a file to scan for malware:", accept_multiple_files=True)
if file:
    with st.spinner("Analyzing files..."):
        for i in file:
            # Saving the uploaded file temporarily
            with open('malwares/tempFile', 'wb') as f:
                f.write(i.getvalue())
            
            # Checking the file
            legitimate = checkFile("malwares/tempFile")
            os.remove("malwares/tempFile")

            # Display results with custom colors and icons
            if legitimate:
                st.markdown(f'<p class="result legitimate">File "{i.name}" is <strong>LEGITIMATE</strong> 👍</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="result malware">File "{i.name}" is <strong>PROBABLY MALWARE</strong> ⚠️</p>', unsafe_allow_html=True)
