from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. Definir la CNN optimizada para clasificar números
def create_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid')(inputs)  # 32 filtros
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)       # 64 filtros
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)      # 128 filtros
    x = GlobalAveragePooling2D()(x)  # Genera el vector de características
    model = Model(inputs, x)
    return model

# 2. Generar características con la CNN
def extract_features(model, X_images):
    features = model.predict(X_images)
    print(f"Tamaño de las características extraídas: {features.shape}")
    return features

# 3. Cargar y preparar los datos (MNIST)
# Cargar dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesar datos
X_train = X_train 
X_test = X_test   

# Redimensionar para adaptarse a la CNN (64x64x1)
X_train = np.expand_dims(X_train, axis=-1)  # Convertir a formato (28,28,1)
X_test = np.expand_dims(X_test, axis=-1)

# Dividir en entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Crear la CNN y extraer características
cnn_model = create_cnn(input_shape=(28, 28, 1))
print("Extrayendo características con la CNN...")
X_train_features = extract_features(cnn_model, X_train)
X_val_features = extract_features(cnn_model, X_val)
X_test_features = extract_features(cnn_model, X_test)

# 4. Entrenar el Random Forest
print("Entrenando el Random Forest...")
rf = RandomForestClassifier(n_estimators=1024, random_state=42, n_jobs=32)
rf.fit(X_train_features, y_train)

# 5. Evaluar el RF
val_accuracy = rf.score(X_val_features, y_val)
test_accuracy = rf.score(X_test_features, y_test)
print(f"Precisión del modelo CNN + RF en validación: {val_accuracy:.4f}")
print(f"Precisión del modelo CNN + RF en prueba: {test_accuracy:.4f}")

# 5. Entrenar una Red Neuronal
print("Entrenando la Red Neuronal...")
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 clases para MNIST
])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_features, y_train, validation_data=(X_val_features, y_val), epochs=10, batch_size=32)

# Evaluar la Red Neuronal
test_loss, val_accuracy_nn = nn_model.evaluate(X_val_features, y_val, verbose=0)
test_loss, test_accuracy_nn = nn_model.evaluate(X_test_features, y_test, verbose=0)
print(f"Precisión del modelo CNN + NN en validacion: {val_accuracy_nn:.4f}")
print(f"Precisión del modelo CNN + NN en prueba: {test_accuracy_nn:.4f}")
