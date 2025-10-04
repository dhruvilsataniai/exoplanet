"""
Main Flask application for the Exoplanet Detection System.
Serves the web interface and API endpoints.
"""

import logging
import os
from pathlib import Path

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configuration
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "dev-secret-key-change-in-production"
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

    # Enable CORS for API endpoints
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    from src.api.routes import api_bp

    app.register_blueprint(api_bp)

    # Web routes
    @app.route("/")
    def index():
        """Main dashboard page."""
        return render_template("index.html")

    @app.route("/predict")
    def predict_page():
        """Prediction interface page."""
        return render_template("predict.html")

    @app.route("/upload")
    def upload_page():
        """Data upload page."""
        return render_template("upload.html")

    @app.route("/explore")
    def explore_page():
        """Data exploration page."""
        return render_template("explore.html")

    @app.route("/about")
    def about_page():
        """About page."""
        return render_template("about.html")

    # Static file serving
    @app.route("/static/<path:filename>")
    def static_files(filename):
        """Serve static files."""
        return send_from_directory("static", filename)

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template("500.html"), 500

    @app.errorhandler(413)
    def too_large(error):
        return "File too large", 413

    return app


# Create the app
app = create_app()

if __name__ == "__main__":
    # Ensure required directories exist
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)

    # Run the application
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") == "development"

    logger.info(f"Starting Exoplanet Detection System on port {port}")
    logger.info(f"Debug mode: {debug}")

    app.run(host="0.0.0.0", port=port, debug=debug)
