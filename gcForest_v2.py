from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class GcForest:
    def __init__(self, n_estimators=100, max_leaf_nodes=None, max_layers=5, early_stopping_rounds=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.layers = []
        self.feature_names_by_layer = []  # Mantener registro de los nombres de características
        self.best_score = 0
        self.no_improve_counter = 0

    def _train_layer(self, X_train, y_train, X_val, y_val, max_leaf_nodes=None):
        models = []
        outputs = []
        for Model in [RandomForestClassifier, ExtraTreesClassifier]:
            model = Model(
                n_estimators=self.n_estimators,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)
            models.append(model)
            outputs.append(model.predict_proba(X_val))  # Salida probabilística en validación
        return models, np.hstack(outputs)  # Concatenar salidas

    def _update_feature_names(self, X, predictions, layer_idx):
        """
        Actualiza los nombres de las características añadiendo los nombres de las predicciones de la capa.
        """
        pred_columns = [f"layer{layer_idx}_model{j}_class{k}" 
                        for j in range(len(predictions)) 
                        for k in range(predictions[j].shape[1])]
        pred_df = pd.DataFrame(np.hstack(predictions), columns=pred_columns, index=X.index)
        return pd.concat([X, pred_df], axis=1)

    def fit(self, X, y):
        # Dividir el conjunto de datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        current_input_train = X_train
        current_input_val = X_val

        for layer_idx in range(self.max_layers):
            print(f"Entrenando capa {layer_idx + 1}")
            models, layer_output_val = self._train_layer(current_input_train, y_train, current_input_val, y_val)
            self.layers.append(models)

            # Actualizar características enriquecidas
            current_input_train = self._update_feature_names(
                current_input_train,
                [model.predict_proba(current_input_train) for model in models],
                layer_idx + 1
            )
            current_input_val = self._update_feature_names(
                current_input_val,
                [model.predict_proba(current_input_val) for model in models],
                layer_idx + 1
            )

            # Guardar los nombres de las características
            self.feature_names_by_layer.append(current_input_train.columns.tolist())

            # Calcular precisión en validación
            y_pred_layer = np.argmax(layer_output_val, axis=1)
            accuracy_layer = accuracy_score(y_val, y_pred_layer)
            print(f"Precisión en capa {layer_idx + 1}: {accuracy_layer:.4f}")

            # Early stopping
            if accuracy_layer > self.best_score:
                self.best_score = accuracy_layer
                self.no_improve_counter = 0
            else:
                self.no_improve_counter += 1

            if self.no_improve_counter >= self.early_stopping_rounds:
                print("Detención temprana activada.")
                break

    def predict(self, X):
        # Creamos una copia de X para no modificar el original
        current_input = X.copy()

        # Recorrer las capas en el mismo orden que se entrenaron
        for layer_idx, models in enumerate(self.layers):
            # Obtener las probabilidades de cada modelo de la capa actual
            predictions = [model.predict_proba(current_input) for model in models]
            
            # Actualizar las características con la misma función usada en fit
            # Notar que en fit se usa (layer_idx + 1), aquí repetimos lo mismo
            current_input = self._update_feature_names(current_input, predictions, layer_idx + 1)

        # Después de la última capa, 'predictions' tendrá las probabilidades del último conjunto de modelos
        # Podemos tomar las últimas 'predictions' para hacer la predicción final
        final_probs = np.hstack(predictions)
        y_pred = np.argmax(final_probs, axis=1)

        return y_pred





# Main script
if __name__ == "__main__":
    # Cargar MNIST desde sklearn
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Convertir y a entero
    y = y.astype(int)

    # Para agilizar, podríamos tomar una muestra más pequeña (opcional):
    idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
    X = X.iloc[idx]
    y = y.iloc[idx]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo gcForest
    gcforest = GcForest(n_estimators=1024, max_leaf_nodes=32, max_layers=12, early_stopping_rounds=2, random_state=42)
    gcforest.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = gcforest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.4f}")