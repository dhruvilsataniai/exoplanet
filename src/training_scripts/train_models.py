"""
Machine learning model training for exoplanet classification.
Implements multiple algorithms and model comparison.
"""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetModelTrainer:
    """Trains and evaluates multiple ML models for exoplanet classification."""

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all ML models with default parameters."""
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=42, multi_class="ovr"
            ),
            "svm": SVC(kernel="rbf", probability=True, random_state=42),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        }

    def train_all_models(self, X_train, y_train, X_test, y_test, use_smote=True):
        """Train all models and evaluate performance."""
        logger.info("Training all models...")

        # Apply SMOTE for class imbalance if requested
        if use_smote:
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {X_train_balanced.shape[0]} training samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            try:
                # Train model
                model.fit(X_train_balanced, y_train_balanced)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = (
                    model.predict_proba(X_test)
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted"
                )

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train_balanced, y_train_balanced, cv=5, scoring="accuracy"
                )

                # Store results
                results[name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                }

                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue

        self.model_scores = results

        # Find best model based on F1 score
        best_f1 = 0
        for name, metrics in results.items():
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                self.best_model = metrics["model"]
                self.best_model_name = name

        logger.info(f"Best model: {self.best_model_name} (F1: {best_f1:.4f})")

        return results

    def hyperparameter_tuning(self, X_train, y_train, model_name="random_forest"):
        """Perform hyperparameter tuning for a specific model."""
        logger.info(f"Performing hyperparameter tuning for {model_name}...")

        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
            },
        }

        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.models[model_name]

        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Detailed evaluation of a single model."""
        logger.info(f"Evaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        print(f"\n{model_name} Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
        }

    def plot_model_comparison(self, save_path=None):
        """Plot comparison of all trained models."""
        if not self.model_scores:
            logger.warning("No model scores available. Train models first.")
            return

        # Prepare data for plotting
        models = list(self.model_scores.keys())
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [self.model_scores[model][metric] for model in models]

            bars = axes[i].bar(models, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def get_feature_importance(self, model_name=None):
        """Get feature importance from tree-based models."""
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.model_scores:
            logger.warning(f"Model {model_name} not found in trained models.")
            return None

        model = self.model_scores[model_name]["model"]

        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            return np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} does not have feature importance.")
            return None

    def save_models(self):
        """Save all trained models."""
        logger.info("Saving trained models...")

        for name, results in self.model_scores.items():
            model_path = self.models_dir / f"{name}_model.pkl"
            joblib.dump(results["model"], model_path)
            logger.info(f"Saved {name} to {model_path}")

        # Save best model separately
        if self.best_model:
            best_model_path = self.models_dir / "best_model.pkl"
            joblib.dump(self.best_model, best_model_path)

            # Save model metadata
            metadata = {
                "best_model_name": self.best_model_name,
                "model_scores": {
                    name: {
                        k: v
                        for k, v in scores.items()
                        if k not in ["model", "predictions", "probabilities"]
                    }
                    for name, scores in self.model_scores.items()
                },
            }

            metadata_path = self.models_dir / "model_metadata.pkl"
            joblib.dump(metadata, metadata_path)

            logger.info(
                f"Best model ({self.best_model_name}) saved to {best_model_path}"
            )

    def load_best_model(self):
        """Load the best trained model."""
        best_model_path = self.models_dir / "best_model.pkl"
        metadata_path = self.models_dir / "model_metadata.pkl"

        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_model_path}")

        self.best_model = joblib.load(best_model_path)

        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.best_model_name = metadata["best_model_name"]
            logger.info(f"Loaded best model: {self.best_model_name}")

        return self.best_model


def main():
    """Main function to demonstrate model training."""
    # Add parent directory to path for imports
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Load preprocessed data
    from src.data_processing.preprocess import ExoplanetPreprocessor

    preprocessor = ExoplanetPreprocessor()

    try:
        # Load and preprocess data
        combined_df = preprocessor.load_and_combine_datasets()
        cleaned_df = preprocessor.clean_and_engineer_features(combined_df)
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_ml(cleaned_df)

        # Train models
        trainer = ExoplanetModelTrainer()
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)

        # Print results summary
        print("\nModel Training Results:")
        print("-" * 50)
        for name, metrics in results.items():
            print(
                f"{name:20s} | Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1_score']:.4f} | CV: {metrics['cv_mean']:.4f}"
            )

        # Hyperparameter tuning for best model
        if trainer.best_model_name in ["random_forest", "xgboost", "lightgbm"]:
            print(
                f"\nPerforming hyperparameter tuning for {trainer.best_model_name}..."
            )
            tuned_model = trainer.hyperparameter_tuning(
                X_train, y_train, trainer.best_model_name
            )

            # Re-evaluate tuned model
            tuned_results = trainer.evaluate_model(
                tuned_model, X_test, y_test, f"Tuned {trainer.best_model_name}"
            )

        # Save models
        trainer.save_models()
        preprocessor.save_preprocessors()

        # Plot comparisons
        trainer.plot_model_comparison(save_path="models/model_comparison.png")

        print(f"\nTraining complete! Best model: {trainer.best_model_name}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have run fetch_data.py and preprocess.py first.")


if __name__ == "__main__":
    main()
