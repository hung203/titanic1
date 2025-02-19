import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import mlflow
import mlflow.sklearn
# Tiêu đề ứng dụng
st.title("Tiền xử lý dữ liệu Titanic cho Multiple Regression")

# Tải dữ liệu Titanic
st.header("1. Tải dữ liệu Titanic")
df = pd.read_csv("titanic.csv")
st.write("Dữ liệu ban đầu:")
st.write(df)

# Xử lý giá trị thiếu
st.header("2. Xử lý giá trị thiếu")

# Tính số lượng giá trị thiếu cho mỗi cột và chuyển thành DataFrame
missing_data = df.isnull().sum().reset_index()
missing_data.columns = ['Column', 'Missing Count']
st.write(missing_data)

st.write("### Điền giá trị thiếu cho cột Age, Fare, và Embarked:")

st.write("Đối với cột Age thay thế các giá trị thiếu (NaN) trong cột Age bằng giá trị trung vị vừa tính được.")
st.write("Đối với cột Fare thay thế các giá trị thiếu trong cột Fare bằng giá trị trung bình.")
st.write("Đối với cột Embarked thay thế các giá trị thiếu trong cột Embarked bằng giá trị mode vừa lấy được.")

# Lọc chỉ những cột có giá trị thiếu
missing_data = missing_data[missing_data['Missing Count'] > 0]
# Điền giá trị thiếu
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Xóa cột Cabin
df.drop("Cabin", axis=1, inplace=True)
st.write("#### Dữ liệu sau khi xử lý giá trị thiếu:")
st.write(df)

# Mã hóa dữ liệu
st.header("3. Mã hóa dữ liệu")
st.write("Mã hóa cột Sex và one-hot encoding cho Embarked và Title:")

# Mã hóa cột Sex
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encoding cho Embarked và Title
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

st.write("Dữ liệu sau khi mã hóa:")
st.write(df)

# Xóa các cột không cần thiết
st.header("4. Xóa các cột không cần thiết")
st.write("Xóa các cột PassengerId, Name, và Ticket:")

# Xóa các cột không cần thiết
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

st.write("Dữ liệu sau khi xóa các cột không cần thiết:")
st.write(df)

# Chuẩn hóa dữ liệu
st.header("5. Chuẩn hóa dữ liệu")
st.write("Chuẩn hóa các cột số (Age, Fare, SibSp, Parch:")

# Chuẩn hóa các cột số
scaler = StandardScaler()
numerical_features = ["Age", "Fare", "SibSp", "Parch"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

st.write("Dữ liệu sau khi chuẩn hóa:")
st.write(df)

#chia dữ liệu và train mô hình
st.header("6. Chia dữ liệu")
# Chia dữ liệu
X = df.drop("Survived", axis=1)  # Đặc trưng
y = df["Survived"]               # Mục tiêu

st.write("#### Một số dòng dữ liệu của các feature", X.head())
st.write("#### Một số dòng dữ liệu của target", y.head())

st.header("Chia tách dữ liệu")
st.write("Nhập tỉ lệ (phần trăm) chia dữ liệu cho Train, Validation và Test (tổng phải = 100).")

col1, col2, col3 = st.columns(3)
with col1:
    train_ratio = st.number_input("Train (%)", min_value=1, max_value=100, value=70)
with col2:
    valid_ratio = st.number_input("Validation (%)", min_value=1, max_value=100, value=15)
with col3:
    test_ratio = st.number_input("Test (%)", min_value=1, max_value=100, value=15)
    
total = train_ratio + valid_ratio + test_ratio
if total != 100:
    st.warning(f"Tổng tỉ lệ hiện tại là {total}, vui lòng đảm bảo tổng bằng 100.")
else:
    # Tách dữ liệu: Đầu tiên tách ra Test, sau đó tách train & validation từ phần còn lại
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=42)
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_ratio_adjusted, random_state=42)
    
    st.write("Hình dạng của tập Train:", X_train.shape)
    st.write("Hình dạng của tập Validation:", X_valid.shape)
    st.write("Hình dạng của tập Test:", X_test.shape)
    
    # Gộp tập Train và Validation để thực hiện Cross Validation
    X_train_valid = pd.concat([X_train, X_valid])
    y_train_valid = pd.concat([y_train, y_valid])
    
    st.header("7. Huấn luyện & Kiểm thử mô hình")
    st.write("Chọn thuật toán huấn luyện:")
    algorithm = st.selectbox("Thuật toán:", ["Multiple Regression", "Polynomial Regression"])
    
    if algorithm == "Polynomial Regression":
        degree = st.number_input("Chọn bậc của đa thức:", min_value=2, max_value=5, value=2)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
    else:
        model = LinearRegression()

    #MLFlow
    ex=mlflow.set_experiment(experiment_name='experiment2')

    with mlflow.start_run(experiment_id=ex.experiment_id):
        st.subheader("Cross Validation (5-fold) trên tập Train+Validation")

        # Định nghĩa KFold cho cross validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Sử dụng R2 score để tính điểm Cross Validation
        scores = cross_val_score(model, X_train_valid, y_train_valid, cv=cv, scoring='r2')
        st.write("Điểm Cross Validation (R2):", scores)
        st.write("Điểm R2 trung bình:", np.mean(scores))

        # Huấn luyện trên tập Train+Validation và đánh giá trên tập Test
        model.fit(X_train_valid, y_train_valid)
        test_score = model.score(X_test, y_test)

        # Tính thêm các độ đo: R-squared, Adjusted R-squared, MSE
        y_pred = model.predict(X_test)

        # R-squared (có thể dùng lại test_score hoặc tính lại bằng hàm r2_score)
        r2 = r2_score(y_test, y_pred)
        st.write("R-squared:", r2)

        # Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error (MSE):", mse)

        # Tính Adjusted R-squared
        n = len(y_test)         # Số mẫu trong tập test
        p = X_test.shape[1]     # Số đặc trưng
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        st.write("Adjusted R-squared:", adj_r2)

        mlflow.log_metric("test_r2_score",r2)
        mlflow.log_metric("test_MSE",mse)
        mlflow.log_metric("test_R-squared",adj_r2)

        # Vẽ biểu đồ Actual vs Predicted trên tập Test
        st.subheader("Biểu đồ Actual vs Predicted trên tập Test")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Giá trị thực (Survive)")
        ax.set_ylabel("Giá trị dự đoán (Survive)")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

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
        # Bao gồm tùy chọn "C" vì khi drop_first=True, "C" là baseline
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    if st.form_submit_button("Predict Survival"):
        # Encode biến Sex: female -> 1, male -> 0
        sex_encoded = 1 if sex == "female" else 0
        
        # One-hot encoding cho biến embarked (với "C" là baseline: cả 2 dummy = 0)
        if embarked == "C":
            embarked_Q = 0
            embarked_S = 0
        elif embarked == "Q":
            embarked_Q = 1
            embarked_S = 0
        else:  # embarked == "S"
            embarked_Q = 0
            embarked_S = 1
        
        # Tạo DataFrame input theo thứ tự đặc trưng khi train:
        # ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
        input_df = pd.DataFrame({
            "Pclass": [int(pclass)],
            "Sex": [sex_encoded],
            "Age": [float(age)],
            "SibSp": [int(sibsp)],
            "Parch": [int(parch)],
            "Fare": [float(fare)],
            "Embarked_Q": [embarked_Q],
            "Embarked_S": [embarked_S]
        })
        
        # Áp dụng scaling cho các cột số sử dụng scaler đã fit trước đó
        numerical_features = ["Age", "Fare", "SibSp", "Parch"]
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        try:
            # Dự đoán sử dụng mô hình (LinearRegression hoặc Polynomial Regression)
            prediction = model.predict(input_df)
            # Với mô hình hồi quy, đặt ngưỡng 0.5 để chuyển thành phân lớp
            predicted_class = 1 if prediction[0] >= 0.5 else 0
            result = "🌟 Survived!" if predicted_class == 1 else "💀 Did Not Survive"
            
            # Kiểm tra xem input đã đưa vào có tồn tại trong bộ dữ liệu (df) hay không
            # Sử dụng các cột đặc trưng đã được xử lý
            features_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
            # So sánh với tolerance nhỏ để đảm bảo khớp với giá trị float sau scaling
            matching_rows = df[features_cols].apply(
                lambda row: np.allclose(row.values, input_df.iloc[0].values, atol=1e-6),
                axis=1
            )
            
            if matching_rows.any():
                # Nếu tìm thấy, lấy dòng đầu tiên khớp
                idx = matching_rows.idxmax()
                actual_survived = df.loc[idx, "Survived"]
                if actual_survived == predicted_class:
                    annotation = "Dự đoán đúng với thực tế"
                else:
                    annotation = "Dự đoán sai với thực tế"
                st.subheader(f"Prediction Result: {result} ({annotation})")
            else:
                st.subheader(f"Prediction Result: {result} (Input không có trong bộ dữ liệu)")
            
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
import os

# Lấy URL của MLflow Tracking Server từ biến môi trường, nếu không có sẽ dùng mặc định
mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Tạo button dưới dạng HTML với liên kết đến MLflow Tracking UI
button_html = f"""
    <div style="text-align: center; margin-top: 20px;">
        <a href="{mlflow_url}" target="_blank">
            <button style="padding: 10px 20px; font-size: 16px; cursor: pointer;">
                Go to MLflow UI
            </button>
        </a>
    </div>
"""
st.markdown(button_html, unsafe_allow_html=True)


