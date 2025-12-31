#!/bin/bash
# Fixed run script to prevent PyTorch/Streamlit conflict
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
streamlit run app.py
