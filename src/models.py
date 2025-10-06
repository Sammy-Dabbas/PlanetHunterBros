import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json

# TensorFlow is optional
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    keras = None
    layers = None


class ExoplanetClassifier:
    """Ensemble of ML models for exoplanet detection."""

    def __init__(self, model_type='random_forest'):
        """
        Initialize classifier.

        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm', 'neural_net', 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self.feature_importance = None
        self.training_history = []

    def create_model(self, input_dim=None, **hyperparams):
        """Create model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 200),
                max_depth=hyperparams.get('max_depth', 20),
                min_samples_split=hyperparams.get('min_samples_split', 5),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 2),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=hyperparams.get('n_estimators', 200),
                max_depth=hyperparams.get('max_depth', 10),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                subsample=hyperparams.get('subsample', 0.8),
                colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
                scale_pos_weight=hyperparams.get('scale_pos_weight', 3),
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=hyperparams.get('n_estimators', 200),
                max_depth=hyperparams.get('max_depth', 10),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                num_leaves=hyperparams.get('num_leaves', 31),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == 'neural_net':
            if not HAS_TENSORFLOW:
                raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
            if input_dim is None:
                raise ValueError("input_dim required for neural network")

            self.model = self._create_neural_network(
                input_dim,
                hidden_layers=hyperparams.get('hidden_layers', [128, 64, 32]),
                dropout_rate=hyperparams.get('dropout_rate', 0.3),
                learning_rate=hyperparams.get('learning_rate', 0.001)
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_neural_network(self, input_dim, hidden_layers, dropout_rate, learning_rate):
        """Create neural network architecture."""
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        if self.model is None:
            input_dim = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
            self.create_model(input_dim=input_dim, **kwargs)

        if self.model_type == 'neural_net':
            # Calculate class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

            # Neural network training
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32),
                class_weight=class_weights,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss' if X_val is not None else 'loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            self.training_history = history.history

        else:
            # Tree-based model training
            self.model.fit(X_train, y_train)

            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_

    def predict(self, X):
        """Make predictions."""
        if self.model_type == 'neural_net':
            # Use optimal threshold if available, otherwise 0.5
            threshold = getattr(self, 'optimal_threshold', 0.5)
            predictions = (self.model.predict(X, verbose=0) > threshold).astype(int).flatten()
        else:
            predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model_type == 'neural_net':
            proba = self.model.predict(X, verbose=0).flatten()
            return np.column_stack([1 - proba, proba])
        else:
            return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        return self.metrics

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        return {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores.tolist()
        }

    def get_feature_importance(self, feature_names):
        """Get feature importance for tree-based models."""
        if self.feature_importance is None:
            return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df.to_dict('records')

    def save_model(self, filepath='models/exoplanet_model.pkl'):
        """Save trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if self.model_type == 'neural_net':
            # Save Keras model separately
            model_path = filepath.replace('.pkl', '.h5')
            self.model.save(model_path)
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'metrics': self.metrics,
                'training_history': self.training_history,
                'model_path': model_path
            }
            joblib.dump(metadata, filepath)
        else:
            # Save sklearn/xgboost/lightgbm models
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }, filepath)

    def load_model(self, filepath='models/exoplanet_model.pkl'):
        """Load trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        data = joblib.load(filepath)
        self.model_type = data['model_type']
        self.metrics = data.get('metrics', {})

        if self.model_type == 'neural_net':
            # Load Keras model
            model_path = data['model_path']
            self.model = keras.models.load_model(model_path)
            self.training_history = data.get('training_history', [])
        else:
            # Load sklearn/xgboost/lightgbm models
            self.model = data['model']
            self.feature_importance = data.get('feature_importance')


class ModelEnsemble:
    """Ensemble of multiple models for improved predictions."""

    def __init__(self):
        self.models = []
        self.weights = []

    def add_model(self, model, weight=1.0):
        """Add model to ensemble."""
        self.models.append(model)
        self.weights.append(weight)

    def predict_proba(self, X):
        """Weighted average of model predictions."""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X)[:, 1] * weight
            predictions.append(pred)

        avg_pred = np.sum(predictions, axis=0) / sum(self.weights)
        return np.column_stack([1 - avg_pred, avg_pred])

    def predict(self, X):
        """Make binary predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
