# File: Membangun_model/modelling.py
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Muat data yang sudah diproses
df = pd.read_csv('dataset_preprocessing/Telco-Customer-Churn_preprocessing.csv')

# Pisahkan fitur dan target
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan autolog dari MLflow
mlflow.autolog()

# Mulai MLflow Run
with mlflow.start_run():
    # Buat model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Lakukan prediksi
    y_pred = model.predict(X_test)
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

print("Model training selesai. Cek MLflow UI.")