import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Inisialisasi DagsHub
dagshub.init(repo_owner='rifkyadiii', repo_name='SMSML_Moch-Rifky-Aulia-Adikusumah', mlflow=True)

# 2. Muat Data
df = pd.read_csv('Membangun_model/dataset_preprocessing/Telco-Customer-Churn_preprocessing.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 4. Mulai MLflow Run
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.set_tag("developer", "Moch Rifky Aulia Adikusumah")
    mlflow.set_tag("model", "RandomForestClassifier_Tuned")

    # 5. Log Parameter dan Metrik (Manual)
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")

    # 6. Log Artefak Kustom
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = "Membangun_model/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "visuals")

    # Feature Importance
    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    fi_path = "Membangun_model/feature_importance.png"
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path, "visuals")
    
    # 7. Log Model
    mlflow.sklearn.log_model(model, "churn_model")