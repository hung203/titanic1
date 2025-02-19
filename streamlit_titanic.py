import streamlit as st
import mlflow
import pandas as pd

# Cấu hình MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

st.title("Chi tiết Run trong MLflow")

# Chọn Experiment
experiments = client.search_experiments()
experiment_dict = {exp.name: exp.experiment_id for exp in experiments}
experiment_name = st.selectbox("Chọn Experiment:", list(experiment_dict.keys()))
experiment_id = experiment_dict[experiment_name]

# Chọn Run
runs = client.search_runs(experiment_ids=[experiment_id])
df_runs = pd.DataFrame([{"Run ID": run.info.run_id} for run in runs])
selected_run = st.selectbox("Chọn Run:", df_runs["Run ID"].tolist())

# Lấy thông tin Run
run_info = client.get_run(selected_run)

# Hiển thị thông tin Run
st.write("### Parameters:")
st.json(run_info.data.params)

st.write("### Metrics:")
st.json(run_info.data.metrics)

st.write("### Artifacts:")
artifact_uri = run_info.info.artifact_uri
st.write(artifact_uri)
