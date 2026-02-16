import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize
import json
import os


class IDSEvaluator:
    """Comprehensive evaluation and visualization for IDS models"""

    def __init__(self, class_names, save_dir="visualizations"):
        """
        Initialize evaluator

        Args:
            class_names: List of class names
            save_dir: Directory to save visualizations
        """
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def evaluate_all(
        self, y_true, y_pred, y_pred_proba, history=None, model_name="Model"
    ):
        """
        Run complete evaluation suite

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            history: Training history
            model_name: Name of the model

        Returns:
            Dictionary of metrics
        """
        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 70}\n")

        metrics = {}

        metrics.update(self._compute_basic_metrics(y_true, y_pred))

        metrics["per_class"] = self._compute_per_class_metrics(y_true, y_pred)

        metrics.update(self._compute_advanced_metrics(y_true, y_pred))

        self._plot_confusion_matrix(y_true, y_pred, model_name)
        self._plot_normalized_confusion_matrix(y_true, y_pred, model_name)
        self._plot_classification_report_heatmap(y_true, y_pred, model_name)
        self._plot_roc_curves(y_true, y_pred_proba, model_name)
        self._plot_precision_recall_curves(y_true, y_pred_proba, model_name)
        self._plot_class_distribution(y_true, y_pred, model_name)
        self._plot_prediction_confidence(y_pred_proba, y_true, model_name)

        if history is not None:
            self._plot_training_history(history, model_name)

        self._save_metrics(metrics, model_name)

        self._print_summary(metrics, model_name)

        return metrics

    def _compute_basic_metrics(self, y_true, y_pred):
        """Compute basic classification metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

    def _compute_per_class_metrics(self, y_true, y_pred):
        """Compute per-class metrics"""
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        return report

    def _compute_advanced_metrics(self, y_true, y_pred):
        """Compute advanced metrics"""
        return {
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        }

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Confusion matrix saved")

    def _plot_normalized_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot normalized confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Percentage"},
        )
        plt.title(
            f"Normalized Confusion Matrix - {model_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_confusion_matrix_normalized.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Normalized confusion matrix saved")

    def _plot_classification_report_heatmap(self, y_true, y_pred, model_name):
        """Plot classification report as heatmap"""
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        metrics_data = []
        for class_name in self.class_names:
            if class_name in report:
                metrics_data.append(
                    [
                        report[class_name]["precision"],
                        report[class_name]["recall"],
                        report[class_name]["f1-score"],
                    ]
                )

        metrics_df = pd.DataFrame(
            metrics_data,
            columns=["Precision", "Recall", "F1-Score"],
            index=self.class_names,
        )

        plt.figure(figsize=(10, 10))
        sns.heatmap(
            metrics_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Score"},
        )
        plt.title(f"Per-Class Metrics - {model_name}", fontsize=16, fontweight="bold")
        plt.ylabel("Class", fontsize=12)
        plt.xlabel("Metric", fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_classification_report.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Classification report heatmap saved")

    def _plot_roc_curves(self, y_true, y_pred_proba, model_name):
        """Plot ROC curves for all classes"""
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"{self.class_names[i]} (AUC = {roc_auc[i]:.3f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"ROC Curves - {model_name}", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_roc_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ ROC curves saved")

    def _plot_precision_recall_curves(self, y_true, y_pred_proba, model_name):
        """Plot Precision-Recall curves"""
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            pr_auc = auc(recall, precision)
            plt.plot(
                recall,
                precision,
                color=color,
                lw=2,
                label=f"{self.class_names[i]} (AUC = {pr_auc:.3f})",
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(
            f"Precision-Recall Curves - {model_name}", fontsize=16, fontweight="bold"
        )
        plt.legend(loc="lower left", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_precision_recall_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Precision-Recall curves saved")

    def _plot_class_distribution(self, y_true, y_pred, model_name):
        """Plot class distribution comparison"""
        
        true_counts = pd.Series(y_true).value_counts().reindex(range(len(self.class_names)), fill_value=0).sort_index()
        pred_counts = pd.Series(y_pred).value_counts().reindex(range(len(self.class_names)), fill_value=0).sort_index()


        x = np.arange(len(self.class_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(
            x - width / 2,
            true_counts.values,
            width,
            label="True",
            alpha=0.8,
            color="steelblue",
        )
        bars2 = ax.bar(
            x + width / 2,
            pred_counts.values,
            width,
            label="Predicted",
            alpha=0.8,
            color="coral",
        )

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            f"Class Distribution Comparison - {model_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_class_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Class distribution plot saved")

    def _plot_prediction_confidence(self, y_pred_proba, y_true, model_name):
        """Plot prediction confidence distribution"""
        max_proba = np.max(y_pred_proba, axis=1)
        correct = np.argmax(y_pred_proba, axis=1) == y_true

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # confidence distribution
        axes[0].hist(
            max_proba, bins=50, alpha=0.7, color="steelblue", edgecolor="black"
        )
        axes[0].axvline(
            max_proba.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {max_proba.mean():.3f}",
        )
        axes[0].set_xlabel("Prediction Confidence", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title(
            "Overall Confidence Distribution", fontsize=14, fontweight="bold"
        )
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(
            max_proba[correct],
            bins=50,
            alpha=0.7,
            label="Correct",
            color="green",
            edgecolor="black",
        )
        axes[1].hist(
            max_proba[~correct],
            bins=50,
            alpha=0.7,
            label="Incorrect",
            color="red",
            edgecolor="black",
        )
        axes[1].set_xlabel("Prediction Confidence", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title(
            "Confidence by Prediction Correctness", fontsize=14, fontweight="bold"
        )
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Prediction Confidence Analysis - {model_name}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_prediction_confidence.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Prediction confidence plot saved")

    def _plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].plot(history.history["accuracy"], label="Training", linewidth=2)
        axes[0, 0].plot(
            history.history["val_accuracy"], label="Validation", linewidth=2
        )
        axes[0, 0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch", fontsize=12)
        axes[0, 0].set_ylabel("Accuracy", fontsize=12)
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history.history["loss"], label="Training", linewidth=2)
        axes[0, 1].plot(history.history["val_loss"], label="Validation", linewidth=2)
        axes[0, 1].set_title("Model Loss", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch", fontsize=12)
        axes[0, 1].set_ylabel("Loss", fontsize=12)
        axes[0, 1].legend(fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        if "precision" in history.history:
            axes[1, 0].plot(history.history["precision"], label="Training", linewidth=2)
            axes[1, 0].plot(
                history.history["val_precision"], label="Validation", linewidth=2
            )
            axes[1, 0].set_title("Model Precision", fontsize=14, fontweight="bold")
            axes[1, 0].set_xlabel("Epoch", fontsize=12)
            axes[1, 0].set_ylabel("Precision", fontsize=12)
            axes[1, 0].legend(fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)

        if "recall" in history.history:
            axes[1, 1].plot(history.history["recall"], label="Training", linewidth=2)
            axes[1, 1].plot(
                history.history["val_recall"], label="Validation", linewidth=2
            )
            axes[1, 1].set_title("Model Recall", fontsize=14, fontweight="bold")
            axes[1, 1].set_xlabel("Epoch", fontsize=12)
            axes[1, 1].set_ylabel("Recall", fontsize=12)
            axes[1, 1].legend(fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Training History - {model_name}", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{model_name}_training_history.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Training history plot saved")

    def _save_metrics(self, metrics, model_name):
        """Save metrics to JSON file"""

        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        metrics_converted = convert_types(metrics)

        with open(f"{self.save_dir}/{model_name}_metrics.json", "w") as f:
            json.dump(metrics_converted, f, indent=4)

        print("✓ Metrics saved to JSON")

    def _print_summary(self, metrics, model_name):
        """Print evaluation summary"""
        print(f"\n{'=' * 70}")
        print(f"EVALUATION SUMMARY - {model_name}")
        print(f"{'=' * 70}")
        print(f"Accuracy:              {metrics['accuracy']:.4f}")
        print(f"Precision (Macro):     {metrics['precision_macro']:.4f}")
        print(f"Precision (Weighted):  {metrics['precision_weighted']:.4f}")
        print(f"Recall (Macro):        {metrics['recall_macro']:.4f}")
        print(f"Recall (Weighted):     {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (Macro):      {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weighted):   {metrics['f1_weighted']:.4f}")
        print(f"Matthews Corr Coef:    {metrics['matthews_corrcoef']:.4f}")
        print(f"Cohen's Kappa:         {metrics['cohen_kappa']:.4f}")
        print(f"{'=' * 70}\n")
