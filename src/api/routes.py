"""
Flask API routes for exoplanet detection system.
Provides endpoints for predictions, data upload, and model management.
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint("api", __name__, url_prefix="/api")


class PredictionService:
    """Service class for handling ML predictions."""

    def __init__(self, models_dir="models"):
        # Get absolute path relative to project root
        project_root = Path(__file__).parent.parent.parent
        self.models_dir = project_root / models_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.imputer = None
        self.feature_columns = None
        self.load_model_components()

    def load_model_components(self):
        """Load trained model and preprocessors."""
        try:
            # Load model components
            self.model = joblib.load(self.models_dir / "best_model.pkl")
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            self.label_encoder = joblib.load(self.models_dir / "label_encoder.pkl")
            self.imputer = joblib.load(self.models_dir / "imputer.pkl")
            self.feature_columns = joblib.load(self.models_dir / "feature_columns.pkl")

            logger.info("Model components loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"Model components not found: {e}")
            logger.error("Please train the model first by running train_models.py")

    def preprocess_input(self, data):
        """Preprocess input data for prediction."""
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        else:
            # Batch prediction
            df = pd.DataFrame(data)

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Select and order features
        X = df[self.feature_columns]

        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X), columns=X.columns, index=X.index
        )

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index,
        )

        return X_scaled

    def predict(self, data):
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")

        # Preprocess input
        X_processed = self.preprocess_input(data)

        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)

        # Convert predictions back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        # Prepare results
        results = []
        for i, (pred_label, pred_idx) in enumerate(zip(predicted_labels, predictions)):
            prob_dict = {
                class_name: float(prob)
                for class_name, prob in zip(
                    self.label_encoder.classes_, probabilities[i]
                )
            }

            results.append(
                {
                    "prediction": pred_label,
                    "confidence": float(probabilities[i][pred_idx]),
                    "probabilities": prob_dict,
                }
            )

        return results[0] if isinstance(data, dict) else results


# Initialize prediction service
prediction_service = PredictionService()


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": prediction_service.model is not None,
        }
    )


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Single prediction endpoint."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = [
            "orbital_period",
            "transit_duration",
            "transit_depth",
            "planet_radius",
        ]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return (
                jsonify(
                    {
                        "error": f"Missing required fields: {missing_fields}",
                        "required_fields": required_fields,
                    }
                ),
                400,
            )

        # Make prediction
        result = prediction_service.predict(data)

        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        result["model_version"] = "1.0"

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()

        if not data or "samples" not in data:
            return jsonify({"error": "No samples provided"}), 400

        samples = data["samples"]
        if not isinstance(samples, list):
            return jsonify({"error": "Samples must be a list"}), 400

        # Make batch predictions
        results = prediction_service.predict(samples)

        return jsonify(
            {
                "predictions": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat(),
                "model_version": "1.0",
            }
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/upload", methods=["POST"])
def upload_data():
    """Upload CSV data for batch prediction."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "File must be CSV format"}), 400

        # Read CSV data
        df = pd.read_csv(file)

        # Convert to list of dictionaries
        samples = df.to_dict("records")

        # Make predictions
        results = prediction_service.predict(samples)

        # Add original data to results
        for i, result in enumerate(results):
            result["input_data"] = samples[i]

        return jsonify(
            {
                "predictions": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat(),
                "filename": file.filename,
            }
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/model/info", methods=["GET"])
def model_info():
    """Get model information."""
    try:
        metadata_path = prediction_service.models_dir / "model_metadata.pkl"

        info = {
            "model_loaded": prediction_service.model is not None,
            "feature_columns": prediction_service.feature_columns,
            "classes": (
                list(prediction_service.label_encoder.classes_)
                if prediction_service.label_encoder
                else None
            ),
            "model_version": "1.0",
        }

        # Load additional metadata if available
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            info.update(
                {
                    "best_model_name": metadata.get("best_model_name"),
                    "model_scores": metadata.get("model_scores"),
                }
            )

        return jsonify(info)

    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/features/template", methods=["GET"])
def feature_template():
    """Get template for input features."""
    try:
        if not prediction_service.feature_columns:
            return jsonify({"error": "Model not loaded"}), 500

        # Create template with example values
        template = {}
        feature_descriptions = {
            "orbital_period": {
                "example": 365.25,
                "unit": "days",
                "description": "Orbital period of the planet",
            },
            "transit_duration": {
                "example": 6.5,
                "unit": "hours",
                "description": "Duration of transit event",
            },
            "transit_depth": {
                "example": 1000,
                "unit": "ppm",
                "description": "Depth of transit in parts per million",
            },
            "impact_parameter": {
                "example": 0.5,
                "unit": "dimensionless",
                "description": "Impact parameter (0-1)",
            },
            "planet_radius": {
                "example": 1.0,
                "unit": "Earth radii",
                "description": "Radius of the planet",
            },
            "equilibrium_temp": {
                "example": 288,
                "unit": "Kelvin",
                "description": "Equilibrium temperature",
            },
            "stellar_temp": {
                "example": 5778,
                "unit": "Kelvin",
                "description": "Stellar effective temperature",
            },
            "stellar_radius": {
                "example": 1.0,
                "unit": "Solar radii",
                "description": "Radius of the host star",
            },
            "stellar_magnitude": {
                "example": 12.5,
                "unit": "magnitude",
                "description": "Apparent magnitude of star",
            },
            "false_positive_flag": {
                "example": 0,
                "unit": "count",
                "description": "Number of false positive flags",
            },
            "koi_score": {
                "example": 0.8,
                "unit": "dimensionless",
                "description": "KOI disposition score (0-1)",
            },
            "radius_ratio": {
                "example": 0.01,
                "unit": "dimensionless",
                "description": "Planet to star radius ratio",
            },
            "signal_strength": {
                "example": 0.001,
                "unit": "fraction",
                "description": "Transit signal strength",
            },
            "in_habitable_zone": {
                "example": 1,
                "unit": "boolean",
                "description": "Whether planet is in habitable zone (0 or 1)",
            },
        }

        for feature in prediction_service.feature_columns:
            if feature in feature_descriptions:
                template[feature] = feature_descriptions[feature]
            else:
                template[feature] = {
                    "example": 0.0,
                    "unit": "unknown",
                    "description": "No description available",
                }

        return jsonify(
            {
                "template": template,
                "required_features": prediction_service.feature_columns,
                "example_input": {
                    feature: info["example"] for feature, info in template.items()
                },
            }
        )

    except Exception as e:
        logger.error(f"Template error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/stats", methods=["GET"])
def get_stats():
    """Get dataset and model statistics."""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "model_status": (
                "loaded" if prediction_service.model is not None else "not_loaded"
            ),
        }

        # Try to load dataset stats from available data files
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"

        # Check for processed data first, then individual datasets
        processed_path = data_dir / "processed_data.csv"
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            # Convert pandas data types to native Python types for JSON serialization
            class_dist = {}
            if "disposition" in df.columns:
                class_dist = {
                    str(k): int(v)
                    for k, v in df["disposition"].value_counts().to_dict().items()
                }

            stats.update(
                {
                    "dataset_size": int(len(df)),
                    "feature_count": int(len(df.columns) - 1),  # Exclude target
                    "class_distribution": class_dist,
                    "missing_values": int(df.isnull().sum().sum()),
                }
            )
        else:
            # Check individual dataset files
            dataset_files = ["kepler_koi.csv", "tess_toi.csv", "k2_planets.csv"]
            total_records = 0
            datasets_found = []

            for file in dataset_files:
                file_path = data_dir / file
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        total_records += len(df)
                        datasets_found.append(
                            {
                                "name": file.replace(".csv", ""),
                                "records": int(len(df)),  # Convert to native Python int
                                "columns": int(
                                    len(df.columns)
                                ),  # Convert to native Python int
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error reading {file}: {e}")

            if datasets_found:
                stats.update(
                    {
                        "dataset_size": int(
                            total_records
                        ),  # Convert to native Python int
                        "datasets": datasets_found,
                        "status": "raw_data_available",
                    }
                )

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500
