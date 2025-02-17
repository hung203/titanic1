# app.py
import streamlit as st
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
from processing_titanic import process

# Load model
model = mlflow.sklearn.load_model("model/")
# Load data để lấy feature names
_, _, _, _, _, X_test, _, _ = process()
feature_names = X_test.columns.tolist()

# Giao diện
st.title("📊 Titanic Survival Prediction Dashboard")

# Section 1: Hiển thị metrics
st.header("📈 Model Performance Metrics")
client = mlflow.tracking.MlflowClient()
latest_run = client.search_runs(experiment_ids=["0"], max_results=1)[0]

st.subheader("Cross-Validation Metrics")
col_cv1, col_cv2 = st.columns(2)
with col_cv1:
    st.metric("Mean CV Accuracy", f"{latest_run.data.metrics['mean_cv_accuracy']:.2%}")

st.subheader("Validation Set Metrics")
col_val1, col_val2, col_val3, col_val4 = st.columns(4)
with col_val1:
    st.metric("Accuracy", f"{latest_run.data.metrics['valid_accuracy']:.2%}")
with col_val2:
    st.metric("Precision", f"{latest_run.data.metrics['valid_precision']:.2%}")
with col_val3:
    st.metric("Recall", f"{latest_run.data.metrics['valid_recall']:.2%}")
with col_val4:
    st.metric("F1 Score", f"{latest_run.data.metrics['valid_f1']:.2%}")

st.subheader("Test Set Metrics")
col_test1, col_test2, col_test3, col_test4 = st.columns(4)
with col_test1:
    st.metric("Accuracy", f"{latest_run.data.metrics['test_accuracy']:.2%}")
with col_test2:
    st.metric("Precision", f"{latest_run.data.metrics['test_precision']:.2%}")
with col_test3:
    st.metric("Recall", f"{latest_run.data.metrics['test_recall']:.2%}")
with col_test4:
    st.metric("F1 Score", f"{latest_run.data.metrics['test_f1']:.2%}")
# Section 2: Feature importance
st.header("🔍 Feature Importances")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], align='center')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_title("Random Forest Feature Importances")
st.pyplot(fig)
# Phần dự đoán
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 25)
        
    with col2:
        sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
        parch = st.number_input("Parents/Children", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 50.0)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    if st.form_submit_button("Predict Survival"):
        # Xử lý input giống hệt processing_titanic
        sex_encoded = 1 if sex == "female" else 0
        embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]
        
        # Tạo input data đúng thứ tự features
        input_data = [[
            int(pclass),
            sex_encoded,
            float(age),
            int(sibsp),
            int(parch),
            float(fare),
            embarked_encoded
        ]]
        
        try:
            prediction = model.predict(input_data)
            result = "🌟 Survived!" if prediction[0] == 1 else "💀 Did Not Survive"
            st.subheader(f"Prediction Result: {result}")
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")