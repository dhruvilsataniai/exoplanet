# Exoplanet Detection AI System

An advanced machine learning system for detecting and classifying exoplanets using NASA's Kepler, K2, and TESS mission data. This project combines state-of-the-art ML algorithms with an interactive web interface for researchers and scientists.

## Features

- **Multi-Mission Data Integration**: Processes data from Kepler, K2, and TESS missions
- **Advanced ML Models**: Implements multiple algorithms including Random Forest, XGBoost, and Neural Networks
- **Interactive Web Interface**: User-friendly dashboard for data upload, analysis, and predictions
- **Real-time Predictions**: Classify new observations as confirmed exoplanets, candidates, or false positives
- **Model Retraining**: Continuously improve models with new user-provided data
- **Comprehensive Visualizations**: Interactive plots and dashboards for data exploration

## Project Structure

```
exoplanet/
├── data/                   # Raw and processed datasets
├── models/                 # Trained ML models
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Source code
│   ├── data_processing/   # Data loading and preprocessing
│   ├── training_scripts/           # ML model implementations
│   └── api/              # Flask API endpoints
├── static/               # CSS, JS, and other static files
├── templates/            # HTML templates
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd exoplanet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing
```bash
python3 src/data_processing/fetch_data.py
python3 src/data_processing/preprocess.py
```

### Model Training
```bash
python3 src/training_scripts/train_models.py
python3 src/training_scripts/enhanced_training.py
```

### Web Application
```bash
python3 app.py
```

Visit `http://localhost:5001` to access the web interface.

## Data Sources

This project uses publicly available datasets from:
- **NASA Exoplanet Archive**: Kepler Objects of Interest (KOI) and TESS Objects of Interest (TOI)
- **Kepler Mission**: Light curves and stellar parameters
- **K2 Mission**: Extended mission data
- **TESS Mission**: Transiting Exoplanet Survey Satellite data

## Acknowledgments

- NASA Exoplanet Archive for providing open-source datasets
- Kepler, K2, and TESS mission teams
- The astronomical community for their contributions to exoplanet research
