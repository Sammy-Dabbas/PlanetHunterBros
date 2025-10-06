import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os


class ExoplanetDataProcessor:
    """Process and prepare exoplanet data for ML models."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.target_column = None

    def load_data(self, filepath, target_col='koi_disposition'):
        """
        Load exoplanet data from CSV file.

        Args:
            filepath: Path to CSV file
            target_col: Name of target column (default: 'koi_disposition')

        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(filepath)
        self.target_column = target_col
        return df

    def select_features(self, df):
        """
        Select relevant features for exoplanet detection.

        Key features based on transit method:
        - Orbital period
        - Transit duration
        - Planetary radius
        - Transit depth
        - Stellar parameters
        """
        # Common feature patterns across Kepler/K2/TESS datasets
        feature_patterns = [
            'koi_period',      # Orbital period
            'koi_duration',    # Transit duration
            'koi_depth',       # Transit depth
            'koi_prad',        # Planetary radius
            'koi_teq',         # Equilibrium temperature
            'koi_insol',       # Insolation flux
            'koi_model_snr',   # Signal-to-noise ratio
            'koi_steff',       # Stellar effective temperature
            'koi_slogg',       # Stellar surface gravity
            'koi_srad',        # Stellar radius
            'ra',              # Right ascension
            'dec',             # Declination
        ]

        # Find available features
        available_features = []
        for pattern in feature_patterns:
            matching_cols = [col for col in df.columns if pattern in col.lower()]
            available_features.extend(matching_cols)

        # Remove duplicates and keep numeric columns only
        available_features = list(set(available_features))
        numeric_features = df[available_features].select_dtypes(include=[np.number]).columns.tolist()

        self.feature_columns = numeric_features
        return numeric_features

    def encode_target(self, df, target_col):
        """
        Encode target variable into binary classification.

        - CONFIRMED/CANDIDATE -> 1 (Exoplanet)
        - FALSE POSITIVE -> 0 (Not an exoplanet)
        """
        target_mapping = {
            'CONFIRMED': 1,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0,
            'FALSE_POSITIVE': 0,
            'NOT DISPOSITIONED': 0,
        }

        # Handle different target column formats
        if target_col in df.columns:
            df['target'] = df[target_col].map(lambda x: target_mapping.get(str(x).upper(), 0))
        else:
            # Try to infer from available columns
            disposition_cols = [col for col in df.columns if 'disposition' in col.lower()]
            if disposition_cols:
                df['target'] = df[disposition_cols[0]].map(lambda x: target_mapping.get(str(x).upper(), 0))
            else:
                raise ValueError(f"Target column '{target_col}' not found")

        return df

    def preprocess(self, df, fit=True):
        """
        Preprocess data: handle missing values, scale features.

        Args:
            df: Input DataFrame
            fit: Whether to fit scalers (True for training, False for inference)

        Returns:
            X (features), y (target)
        """
        # Select features if not already done
        if self.feature_columns is None:
            self.select_features(df)

        # Extract features
        X = df[self.feature_columns].copy()

        # Remove columns that are all NaN
        if fit:
            valid_cols = X.columns[X.notna().any()].tolist()
            self.feature_columns = valid_cols
            X = X[valid_cols]

        # Handle missing values
        if fit:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)

        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)

        # Convert back to DataFrame
        X_processed = pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)

        # Get target if available
        y = df['target'] if 'target' in df.columns else None

        return X_processed, y

    def handle_imbalance(self, X, y, method='smote'):
        """
        Handle class imbalance using SMOTE or other techniques.

        Args:
            X: Features (DataFrame)
            y: Target (Series)
            method: Resampling method ('smote', 'none')

        Returns:
            Resampled X (DataFrame), y (Series)
        """
        if method == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Convert back to DataFrame/Series to preserve column names
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name='target')

            return X_resampled, y_resampled
        else:
            return X, y

    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save fitted preprocessor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }, filepath)

    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load fitted preprocessor."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.feature_columns = data['feature_columns']
        self.target_column = data['target_column']

    def get_feature_importance_data(self, X):
        """Get statistics about features for visualization."""
        stats = {
            'feature_names': self.feature_columns,
            'means': X.mean().tolist(),
            'stds': X.std().tolist(),
            'missing_counts': X.isnull().sum().tolist()
        }
        return stats
