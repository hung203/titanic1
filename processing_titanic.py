from sklearn.model_selection import train_test_split
import pandas as pd

def process():
    # Đọc dữ liệu
    df = pd.read_csv('titanic.txt')

    # Làm sạch dữ liệu
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Thay thế giá trị thiếu trong cột 'Age' bằng giá trị trung vị
    df['Cabin'] = df['Cabin'].fillna('Unknown')  # Thay thế giá trị thiếu trong cột 'Cabin' bằng 'Unknown'
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Thay thế giá trị thiếu trong cột 'Embarked' bằng giá trị phổ biến nhất

    # Chuyển đổi kiểu dữ liệu
    df['Survived'] = df['Survived'].astype('category')
    df['Pclass'] = df['Pclass'].astype('category')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Chuẩn bị dữ liệu cho mô hình
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    # Chia tập dữ liệu thành train/valid/test theo tỷ lệ 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_temp, y_train, y_temp, X_valid, X_test, y_valid, y_test
