import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys

sys.path.append("./network_ids_system")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

from preprocessors.preprocessor import AdvancedPreprocessor
from models.architectures import IDSArchitectures, get_callbacks, compile_model
from utils.evaluation import IDSEvaluator

np.random.seed(42)
tf.random.set_seed(42)


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Using GPU: {len(gpus)} device(s) available")
    except RuntimeError as e:
        print(e)
else:
    print("✓ Using CPU")


class IDSTrainer:
    """Comprehensive training pipeline for IDS models"""

    def __init__(self, data_path, labels_path, output_dir="outputs"):
        """
        Initialize trainer

        Args:
            data_path: Path to Data.csv
            labels_path: Path to Label.csv
            output_dir: Directory for outputs
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.output_dir = output_dir

        self.models_dir = os.path.join(output_dir, "models")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        self.preprocessors_dir = os.path.join(output_dir, "preprocessors")

        for dir_path in [self.models_dir, self.viz_dir, self.preprocessors_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Class
        self.class_names = [
            "Benign",
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ]

        self.results = {}

    def load_data(self):
        """Load and prepare data"""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        # Load data
        print(f"Loading data from {self.data_path}...")
        data = pd.read_csv(self.data_path)
        print(f"Loading labels from {self.labels_path}...")
        labels = pd.read_csv(self.labels_path)

        print(f"✓ Data shape: {data.shape}")
        print(f"✓ Labels shape: {labels.shape}")

        X = data.values
        y = labels.values.ravel()

        print("\nClass Distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(
                f"  {self.class_names[label]:15s}: {count:7d} ({count / len(y) * 100:.2f}%)"
            )

        return X, y

    def preprocess_data(self, X, y):
        """Preprocess and split data"""
        print("\n" + "=" * 70)
        print("PREPROCESSING DATA")
        print("=" * 70)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        preprocessor = AdvancedPreprocessor(
            scaling_method="standard",
            balance_method="smote_tomek",
            feature_selection=True,
            n_features=50,
            apply_pca=False,
        )

        X_train_processed, y_train_processed = preprocessor.fit_transform(
            X_train, y_train
        )

        X_test_processed = preprocessor.transform(X_test)

        preprocessor_path = os.path.join(self.preprocessors_dir, "preprocessor.pkl")
        preprocessor.save(preprocessor_path)

        print(f"\n✓ Training data shape after preprocessing: {X_train_processed.shape}")
        print(f"✓ Test data shape after preprocessing: {X_test_processed.shape}")

        return (
            X_train_processed,
            X_test_processed,
            y_train_processed,
            y_test,
            preprocessor,
        )

    def train_model(self, model, model_name, X_train, y_train, X_val, y_val):
        """Train a single model"""
        print(f"\n{'=' * 70}")
        print(f"TRAINING {model_name}")
        print(f"{'=' * 70}")

        print("DEBUG - BEFORE ravel():")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_val shape: {y_val.shape}")

        y_train = y_train.ravel()
        y_val = y_val.ravel()

        print("DEBUG - AFTER ravel():")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_val shape: {y_val.shape}")
        print(f"  y_train sample values: {y_train[:5]}")
        print(f"  y_val sample values: {y_val[:5]}")

        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"✓ Class weights computed: {class_weight_dict}")

        model = compile_model(model, learning_rate=0.001)
        model.summary()

        model_path = os.path.join(self.models_dir, f"{model_name}_best.h5")
        callbacks = get_callbacks(model_path, patience=15)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=128,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        model.load_weights(model_path)
        return model, history

    def evaluate_model(self, model, model_name, X_test, y_test, history=None):
        """
        Evaluate model and generate visualizations

        Args:
            model: Trained model
            model_name: Name of the model
            X_test, y_test: Test data
            history: Training history

        Returns:
            Metrics dictionary
        """
        print(f"\n{'=' * 70}")
        print(f"EVALUATING {model_name}")
        print(f"{'=' * 70}")

        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        evaluator = IDSEvaluator(
            class_names=self.class_names,
            save_dir=os.path.join(self.viz_dir, model_name),
        )

        metrics = evaluator.evaluate_all(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            history=history,
            model_name=model_name,
        )

        return metrics

    def train_all_architectures(self, X_train, y_train, X_test, y_test):
        """Train all available architectures"""

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )

        if len(y_tr.shape) > 1:
            y_tr = y_tr.ravel()
        if len(y_val.shape) > 1:
            y_val = y_val.ravel()

        input_dim = X_train.shape[1]
        num_classes = len(self.class_names)

        architectures = {
            "DeepMLP_Attention": IDSArchitectures.deep_mlp_with_attention(
                input_dim, num_classes
            ),
            "CNN1D": IDSArchitectures.cnn_1d_architecture(input_dim, num_classes),
            "BiLSTM_Attention": IDSArchitectures.bidirectional_lstm(
                input_dim, num_classes
            ),
            "Transformer": IDSArchitectures.transformer_encoder(input_dim, num_classes),
            "ResNet": IDSArchitectures.residual_network(input_dim, num_classes),
            "Ensemble": IDSArchitectures.ensemble_model(input_dim, num_classes),
        }

        for model_name, model in architectures.items():
            try:
                if model_name in ["CNN1D", "BiLSTM_Attention"]:
                    X_tr_input = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
                    X_val_input = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                    X_test_input = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                else:
                    X_tr_input = X_tr
                    X_val_input = X_val
                    X_test_input = X_test

                trained_model, history = self.train_model(
                    model, model_name, X_tr_input, y_tr, X_val_input, y_val
                )

                metrics = self.evaluate_model(
                    trained_model, model_name, X_test_input, y_test, history
                )

                self.results[model_name] = {
                    "metrics": metrics,
                    "model_path": os.path.join(
                        self.models_dir, f"{model_name}_best.h5"
                    ),
                }

                tf.keras.backend.clear_session()

            except Exception as e:
                print(f"\n✗ Error training {model_name}: {str(e)}")
                continue

    def select_best_model(self):
        """Select best model based on F1-score"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result["metrics"]
            comparison_data.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "F1-Score (Macro)": metrics["f1_macro"],
                    "F1-Score (Weighted)": metrics["f1_weighted"],
                    "Precision (Macro)": metrics["precision_macro"],
                    "Recall (Macro)": metrics["recall_macro"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(
            "F1-Score (Weighted)", ascending=False
        )

        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))

        comparison_df.to_csv(
            os.path.join(self.output_dir, "model_comparison.csv"), index=False
        )

        best_model_name = comparison_df.iloc[0]["Model"]
        print(f"\n✓ Best Model: {best_model_name}")
        print(
            f"  F1-Score (Weighted): {comparison_df.iloc[0]['F1-Score (Weighted)']:.4f}"
        )

        return best_model_name

    def save_final_model(self, best_model_name, preprocessor):
        """Save final production model"""
        print("\n" + "=" * 70)
        print("SAVING PRODUCTION MODEL")
        print("=" * 70)

        best_model_path = os.path.join(self.models_dir, f"{best_model_name}_best.h5")
        production_model_path = os.path.join(self.output_dir, "production_model.h5")

        import shutil

        shutil.copy(best_model_path, production_model_path)

        metadata = {
            "model_name": best_model_name,
            "class_names": self.class_names,
            "input_shape": self.results[best_model_name]["metrics"].get(
                "input_shape", None
            ),
            "num_classes": len(self.class_names),
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "accuracy": self.results[best_model_name]["metrics"]["accuracy"],
                "f1_macro": self.results[best_model_name]["metrics"]["f1_macro"],
                "f1_weighted": self.results[best_model_name]["metrics"]["f1_weighted"],
            },
        }

        with open(os.path.join(self.output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"✓ Production model saved: {production_model_path}")
        print(
            f"✓ Preprocessor saved: {os.path.join(self.preprocessors_dir, 'preprocessor.pkl')}"
        )
        print(
            f"✓ Metadata saved: {os.path.join(self.output_dir, 'model_metadata.json')}"
        )

    def run(self):
        """Run complete training pipeline"""
        print("\n" + "=" * 70)
        print("NETWORK IDS - DEEP LEARNING TRAINING PIPELINE")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        X, y = self.load_data()

        X_train, X_test, y_train, y_test, preprocessor = self.preprocess_data(X, y)

        self.train_all_architectures(X_train, y_train, X_test, y_test)

        best_model_name = self.select_best_model()

        self.save_final_model(best_model_name, preprocessor)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAll outputs saved to: {self.output_dir}")


if __name__ == "__main__":
    DATA_PATH = "./dataset/Data.csv"
    LABELS_PATH = "./dataset/Label.csv"
    OUTPUT_DIR = "./outputs"

    trainer = IDSTrainer(
        data_path=DATA_PATH, labels_path=LABELS_PATH, output_dir=OUTPUT_DIR
    )

    trainer.run()
