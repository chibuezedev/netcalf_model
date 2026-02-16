import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import joblib
import warnings

warnings.filterwarnings("ignore")


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline with multiple balancing techniques
    """

    def __init__(
        self,
        scaling_method="standard",
        balance_method="smote_tomek",
        feature_selection=True,
        n_features=50,
        apply_pca=False,
        pca_variance=0.95,
    ):
        """
        Initialize preprocessor

        Args:
            scaling_method: 'standard', 'robust', 'minmax'
            balance_method: 'smote', 'adasyn', 'smote_tomek', 'smote_enn', 'borderline_smote'
            feature_selection: Whether to apply feature selection
            n_features: Number of features to select
            apply_pca: Whether to apply PCA
            pca_variance: Variance to retain for PCA
        """
        self.scaling_method = scaling_method
        self.balance_method = balance_method
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.apply_pca = apply_pca
        self.pca_variance = pca_variance

        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        self.balancer = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize preprocessing components"""

        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "robust":
            self.scaler = RobustScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()

        if self.feature_selection:
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif, k=min(self.n_features, 50)
            )

        if self.apply_pca:
            self.pca = PCA(n_components=self.pca_variance)

        if self.balance_method == "smote":
            self.balancer = SMOTE(random_state=42, k_neighbors=5)
        elif self.balance_method == "adasyn":
            self.balancer = ADASYN(random_state=42, n_neighbors=5) 
        elif self.balance_method == "smote_tomek":
            self.balancer = SMOTETomek(random_state=42)
        elif self.balance_method == "smote_enn":
            self.balancer = SMOTEENN(random_state=42)  
        elif self.balance_method == "borderline_smote":
            self.balancer = BorderlineSMOTE(
                random_state=42, k_neighbors=5
            ) 

    def fit_transform(self, X, y):
        """
        Fit and transform training data

        Args:
            X: Features (numpy array or pandas DataFrame)
            y: Labels

        Returns:
            X_processed, y_processed
        """
        print("Starting advanced preprocessing pipeline...")

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X = self._handle_missing_values(X)

        # infinite values
        X = self._handle_infinite_values(X)

        # Scaling
        print(f"Applying {self.scaling_method} scaling...")
        X_scaled = self.scaler.fit_transform(X)

        if self.feature_selection:
            print(f"Selecting top {self.n_features} features...")
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [feature_names[i] for i in selected_indices]
            print(f"Selected features: {len(self.selected_features)}")

        # PCA
        if self.apply_pca:
            print(f"Applying PCA (variance retained: {self.pca_variance})...")
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"PCA components: {X_scaled.shape[1]}")

        print(f"Applying {self.balance_method} balancing...")
        print(f"Original class distribution: {np.bincount(y)}")
        # X_balanced, y_balanced = self.balancer.fit_resample(X_scaled, y)
        # print(f"Balanced class distribution: {np.bincount(y_balanced)}")

        return X_scaled, y

    def transform(self, X):
        """
        Transform new data (for testing/inference)

        Args:
            X: Features

        Returns:
            X_processed
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = self._handle_missing_values(X)
        X = self._handle_infinite_values(X)

        # Scaling
        X_scaled = self.scaler.transform(X)

        # Feature selection
        if self.feature_selection:
            X_scaled = self.feature_selector.transform(X_scaled)

        # PCA
        if self.apply_pca:
            X_scaled = self.pca.transform(X_scaled)

        return X_scaled

    def _handle_missing_values(self, X):
        """Handle missing values by replacing with column mean"""
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        return X

    def _handle_infinite_values(self, X):
        """Replace infinite values with column max/min"""
        X = np.where(np.isposinf(X), np.nan, X)
        X = np.where(np.isneginf(X), np.nan, X)
        col_max = np.nanmax(X, axis=0)
        col_min = np.nanmin(X, axis=0)

        for col in range(X.shape[1]):
            pos_inf_mask = np.isposinf(X[:, col])
            neg_inf_mask = np.isneginf(X[:, col])
            X[pos_inf_mask, col] = col_max[col]
            X[neg_inf_mask, col] = col_min[col]

        # NaN check
        X = self._handle_missing_values(X)
        return X

    def save(self, path):
        """Save preprocessor"""
        preprocessor_state = {
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "pca": self.pca,
            "selected_features": self.selected_features,
            "balancer": self.balancer,
            "scaling_method": self.scaling_method,
            "balance_method": self.balance_method,
            "feature_selection": self.feature_selection,
            "n_features": self.n_features,
            "apply_pca": self.apply_pca,
            "pca_variance": self.pca_variance,
        }
        joblib.dump(preprocessor_state, path)
        print(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path):
        """Load preprocessor"""
        state = joblib.load(path)
        preprocessor = cls(
            scaling_method=state["scaling_method"],
            balance_method=state["balance_method"],
            feature_selection=state["feature_selection"],
            n_features=state["n_features"],
            apply_pca=state["apply_pca"],
            pca_variance=state["pca_variance"],
        )
        preprocessor.scaler = state["scaler"]
        preprocessor.feature_selector = state["feature_selector"]
        preprocessor.pca = state["pca"]
        preprocessor.selected_features = state["selected_features"]
        preprocessor.balancer = state["balancer"]
        print(f"Preprocessor loaded from {path}")
        return preprocessor
