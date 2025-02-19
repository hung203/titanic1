import streamlit as st
import mlflow
import pandas as pd
mlflow_url = "http://localhost:5000"  # Thay đổi nếu chạy trên server khác
st.markdown(f'<iframe src="{mlflow_url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
