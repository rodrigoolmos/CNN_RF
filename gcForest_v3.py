import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiGrainedScanner:
    """
    Realiza un 'multi-grained scanning' simplificado sobre datos tabulares.
    En vez de ventanas secuenciales, se usan subconjuntos aleatorios de características.
    """
    def __init__(self, window_sizes=[0.5, 0.75], n_estimators=100, random_state=42):
        self.window_sizes = window_sizes
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scanners = []
    
    def fit_transform(self, X, y):
        self.scanners = []
        n_features = X.shape[1]
        
        outputs = []
        for w in self.window_sizes:
            size = int(n_features * w)
            rng = np.random.RandomState(self.random_state)
            feature_idxs = rng.choice(n_features, size=size, replace=False)
            subX = X.iloc[:, feature_idxs]

            rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            et = ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            rf.fit(subX, y)
            et.fit(subX, y)

            self.scanners.append((feature_idxs, [rf, et]))

            rf_proba = rf.predict_proba(subX)
            et_proba = et.predict_proba(subX)
            outputs.append(np.hstack([rf_proba, et_proba]))

        mg_features = np.hstack(outputs)
        return mg_features
    
    def transform(self, X):
        outputs = []
        for feature_idxs, models in self.scanners:
            subX = X.iloc[:, feature_idxs]
            rf, et = models
            rf_proba = rf.predict_proba(subX)
            et_proba = et.predict_proba(subX)
            outputs.append(np.hstack([rf_proba, et_proba]))
        mg_features = np.hstack(outputs)
        return mg_features

class CascadeForest:
    """
    Construye una cascada de niveles. Cada nivel entrena varios modelos (por ejemplo rf y et),
    concatena sus salidas con las características existentes, y alimenta el siguiente nivel.
    """
    def __init__(self, n_estimators=100, max_layers=10, early_stopping_rounds=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.layers = []
        self.best_score = 0
        self.no_improve_counter = 0
        self.classes_ = None
    
    def _train_layer(self, X_train, y_train, X_val, y_val):
        rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        et = ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        
        rf.fit(X_train, y_train)
        et.fit(X_train, y_train)
        
        self.layers.append([rf, et])
        
        rf_proba_val = rf.predict_proba(X_val)
        et_proba_val = et.predict_proba(X_val)
        
        val_output = np.hstack([rf_proba_val, et_proba_val])
        return val_output

    def fit(self, X_train, y_train, X_val, y_val):
        self.classes_ = np.unique(y_train)
        current_train = X_train
        current_val = X_val
        
        for layer_idx in range(self.max_layers):
            print(f"Entrenando capa {layer_idx + 1}")
            val_output = self._train_layer(current_train, y_train, current_val, y_val)
            
            # Generar salidas para el entrenamiento usando la capa recién entrenada
            rf, et = self.layers[-1]
            rf_proba_train = rf.predict_proba(current_train)
            et_proba_train = et.predict_proba(current_train)
            
            train_output = np.hstack([rf_proba_train, et_proba_train])
            
            # Actualizar las características
            current_train = np.hstack([current_train, train_output])
            current_val = np.hstack([current_val, val_output])

            # Promedio de las probabilidades entre los modelos antes del argmax
            n_classes = len(self.classes_)
            n_models = 2
            val_output_reshaped = val_output.reshape(val_output.shape[0], n_models, n_classes)
            val_output_mean = val_output_reshaped.mean(axis=1)

            y_pred_layer = self.classes_[np.argmax(val_output_mean, axis=1)]
            accuracy_layer = accuracy_score(y_val, y_pred_layer)
            print(f"Precisión en capa {layer_idx + 1}: {accuracy_layer:.4f}")

            if accuracy_layer > self.best_score:
                self.best_score = accuracy_layer
                self.no_improve_counter = 0
            else:
                self.no_improve_counter += 1

            if self.no_improve_counter >= self.early_stopping_rounds:
                print("Detención temprana activada.")
                break
        
        self.final_feature_count_ = current_train.shape[1]

    def predict(self, X):
        current_input = X
        for (rf, et) in self.layers:
            rf_proba = rf.predict_proba(current_input)
            et_proba = et.predict_proba(current_input)
            output = np.hstack([rf_proba, et_proba])
            current_input = np.hstack([current_input, output])
        
        # Promedio entre modelos antes del argmax
        n_classes = len(self.classes_)
        n_models = 2
        output_reshaped = output.reshape(output.shape[0], n_models, n_classes)
        output_mean = output_reshaped.mean(axis=1)
        
        return self.classes_[np.argmax(output_mean, axis=1)]

class gcForestComplete:
    """
    gcForest completo:
    - Aplica multi-grained scanning al conjunto de datos.
    - Construye una cascada de bosques.
    """
    def __init__(self, 
                 mg_window_sizes=[0.5, 0.75],
                 mg_n_estimators=100,
                 cascade_n_estimators=100,
                 cascade_max_layers=5,
                 cascade_early_stopping_rounds=2,
                 random_state=42):
        
        self.mg_window_sizes = mg_window_sizes
        self.mg_n_estimators = mg_n_estimators
        self.cascade_n_estimators = cascade_n_estimators
        self.cascade_max_layers = cascade_max_layers
        self.cascade_early_stopping_rounds = cascade_early_stopping_rounds
        self.random_state = random_state

    def fit(self, X, y):
        # Dividir en entrenamiento/validación
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        # Multi-grained scanning
        self.scanner_ = MultiGrainedScanner(
            window_sizes=self.mg_window_sizes,
            n_estimators=self.mg_n_estimators,
            random_state=self.random_state
        )
        
        # Convertir a DataFrame si no lo es
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            X_val = pd.DataFrame(X_val)
        
        mg_features_train = self.scanner_.fit_transform(X_train, y_train)
        mg_features_val = self.scanner_.transform(X_val)
        
        full_train = np.hstack([X_train.values, mg_features_train])
        full_val = np.hstack([X_val.values, mg_features_val])
        
        self.cascade_ = CascadeForest(
            n_estimators=self.cascade_n_estimators,
            max_layers=self.cascade_max_layers,
            early_stopping_rounds=self.cascade_early_stopping_rounds,
            random_state=self.random_state
        )
        
        self.cascade_.fit(full_train, y_train, full_val, y_val)
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        mg_features = self.scanner_.transform(X)
        full_X = np.hstack([X.values, mg_features])
        return self.cascade_.predict(full_X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


if __name__ == "__main__":
    # Cargar MNIST desde sklearn
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Convertir y a entero
    y = y.astype(int)

    # Para agilizar, podríamos tomar una muestra más pequeña (opcional):
    # idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
    # X = X.iloc[idx]
    # y = y.iloc[idx]

    # Ajustar el gcForest completo
    gcforest = gcForestComplete(
        mg_window_sizes=[0.1,0.5,0.2,1],
        mg_n_estimators=128,  # reducir n_estimators para hacerlo más rápido
        cascade_n_estimators=1024,
        cascade_max_layers=3,
        cascade_early_stopping_rounds=1,
        random_state=42
    )
    gcforest.fit(X, y)

    # División final de prueba (se puede reutilizar el train_test_split)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    accuracy = gcforest.score(X_test, y_test)
    print(f"Precisión final del modelo gcForest completo en MNIST: {accuracy:.4f}")