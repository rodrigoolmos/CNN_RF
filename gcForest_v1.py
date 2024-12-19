import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar MNIST desde sklearn
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# Convertir y a entero
y = y.astype(int)

y = y - y.min()

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y_train))
print(f"Número de clases: {num_classes}")

# Crear el dataset para LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Configuración de parámetros del modelo
params = {
    'objective': 'multiclass',  # Cambia esto según tu problema ('multiclass' para clasificación multiclase)
    'metric': 'multi_logloss',  # Usa 'multi_logloss' para problemas multiclase
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 255,
    'num_trees' : 128,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'num_class': num_classes
}

# Entrenar el modelo
print("Entrenando el modelo...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data]
)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Convertir las probabilidades a etiquetas de clase
y_pred_labels = np.argmax(y_pred, axis=1)  # Selecciona la clase con mayor probabilidad

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\nAccuracy en el conjunto de prueba: {accuracy:.4f}")

# Reporte de clasificación opcional
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_labels))
