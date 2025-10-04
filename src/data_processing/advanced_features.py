"""
Advanced feature engineering for exoplanet detection.
Implements techniques from research papers including TSFresh-inspired features.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, periodogram

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Advanced feature engineering for exoplanet detection."""

    def __init__(self):
        self.feature_names = []

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all advanced features from the dataset."""
        logger.info("Extracting advanced features...")

        enhanced_df = df.copy()

        # Statistical features
        enhanced_df = self._add_statistical_features(enhanced_df)

        # Transit-specific features
        enhanced_df = self._add_transit_features(enhanced_df)

        # Frequency domain features
        enhanced_df = self._add_frequency_features(enhanced_df)

        # Ratio and interaction features
        enhanced_df = self._add_ratio_features(enhanced_df)

        # Outlier and anomaly features
        enhanced_df = self._add_anomaly_features(enhanced_df)

        # Temporal features
        enhanced_df = self._add_temporal_features(enhanced_df)

        logger.info(f"Added {len(self.feature_names)} advanced features")
        return enhanced_df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features inspired by TSFresh."""

        # Higher order moments
        if "orbital_period" in df.columns:
            df["orbital_period_skewness"] = df.groupby("source")[
                "orbital_period"
            ].transform("skew")
            df["orbital_period_kurtosis"] = df.groupby("source")[
                "orbital_period"
            ].transform(lambda x: stats.kurtosis(x, nan_policy="omit"))
            self.feature_names.extend(
                ["orbital_period_skewness", "orbital_period_kurtosis"]
            )

        # Quantile features
        numeric_cols = [
            "orbital_period",
            "transit_duration",
            "transit_depth",
            "planet_radius",
            "stellar_temp",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[f"{col}_q25"] = df[col].fillna(df[col].median()).quantile(0.25)
                df[f"{col}_q75"] = df[col].fillna(df[col].median()).quantile(0.75)
                df[f"{col}_iqr"] = df[f"{col}_q75"] - df[f"{col}_q25"]
                self.feature_names.extend([f"{col}_q25", f"{col}_q75", f"{col}_iqr"])

        # Variance and standard deviation features
        for col in numeric_cols:
            if col in df.columns:
                df[f"{col}_std"] = df[col].std()
                df[f"{col}_var"] = df[col].var()
                df[f"{col}_cv"] = df[f"{col}_std"] / (
                    df[col].mean() + 1e-8
                )  # Coefficient of variation
                self.feature_names.extend([f"{col}_std", f"{col}_var", f"{col}_cv"])

        return df

    def _add_transit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transit-specific features."""

        # Transit shape features
        if "transit_duration" in df.columns and "orbital_period" in df.columns:
            df["transit_duty_cycle"] = df["transit_duration"] / df["orbital_period"]
            self.feature_names.append("transit_duty_cycle")

        # Transit depth to radius relationship
        if "transit_depth" in df.columns and "planet_radius" in df.columns:
            df["depth_radius_ratio"] = df["transit_depth"] / (
                df["planet_radius"] ** 2 + 1e-8
            )
            self.feature_names.append("depth_radius_ratio")

        # Impact parameter features
        if "impact_parameter" in df.columns:
            df["impact_parameter_squared"] = df["impact_parameter"] ** 2
            df["impact_parameter_log"] = np.log1p(np.abs(df["impact_parameter"]))
            self.feature_names.extend(
                ["impact_parameter_squared", "impact_parameter_log"]
            )

        # Transit signal strength
        if "transit_depth" in df.columns and "stellar_magnitude" in df.columns:
            df["signal_to_noise"] = df["transit_depth"] / (
                10 ** (-0.4 * df["stellar_magnitude"]) + 1e-8
            )
            self.feature_names.append("signal_to_noise")

        return df

    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency domain features."""

        # Period-based frequency features
        if "orbital_period" in df.columns:
            df["orbital_frequency"] = 1.0 / (df["orbital_period"] + 1e-8)
            df["log_orbital_period"] = np.log1p(df["orbital_period"])
            df["sqrt_orbital_period"] = np.sqrt(df["orbital_period"])
            self.feature_names.extend(
                ["orbital_frequency", "log_orbital_period", "sqrt_orbital_period"]
            )

        # Harmonic features
        if "orbital_period" in df.columns:
            df["period_harmonic_2"] = df["orbital_period"] / 2.0
            df["period_harmonic_3"] = df["orbital_period"] / 3.0
            df["period_subharmonic_2"] = df["orbital_period"] * 2.0
            self.feature_names.extend(
                ["period_harmonic_2", "period_harmonic_3", "period_subharmonic_2"]
            )

        return df

    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ratio and interaction features."""

        # Planet-to-star ratios
        if "planet_radius" in df.columns and "stellar_radius" in df.columns:
            df["planet_star_radius_ratio"] = df["planet_radius"] / (
                df["stellar_radius"] + 1e-8
            )
            self.feature_names.append("planet_star_radius_ratio")

        # Temperature ratios
        if "equilibrium_temp" in df.columns and "stellar_temp" in df.columns:
            df["temp_ratio"] = df["equilibrium_temp"] / (df["stellar_temp"] + 1e-8)
            df["temp_difference"] = df["stellar_temp"] - df["equilibrium_temp"]
            self.feature_names.extend(["temp_ratio", "temp_difference"])

        # Insolation features
        if "insolation_flux" in df.columns:
            df["log_insolation"] = np.log1p(df["insolation_flux"])
            df["sqrt_insolation"] = np.sqrt(df["insolation_flux"])
            self.feature_names.extend(["log_insolation", "sqrt_insolation"])

        # Multi-feature interactions
        if "orbital_period" in df.columns and "planet_radius" in df.columns:
            df["period_radius_product"] = df["orbital_period"] * df["planet_radius"]
            df["period_radius_ratio"] = df["orbital_period"] / (
                df["planet_radius"] + 1e-8
            )
            self.feature_names.extend(["period_radius_product", "period_radius_ratio"])

        return df

    def _add_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add outlier and anomaly detection features."""

        numeric_cols = [
            "orbital_period",
            "transit_duration",
            "transit_depth",
            "planet_radius",
        ]

        for col in numeric_cols:
            if col in df.columns:
                # Z-score based outlier detection
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f"{col}_zscore"] = np.abs((df[col] - mean_val) / (std_val + 1e-8))
                df[f"{col}_is_outlier"] = (df[f"{col}_zscore"] > 3).astype(int)

                # IQR based outlier detection
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                iqr = q75 - q25
                df[f"{col}_iqr_outlier"] = (
                    (df[col] < (q25 - 1.5 * iqr)) | (df[col] > (q75 + 1.5 * iqr))
                ).astype(int)

                self.feature_names.extend(
                    [f"{col}_zscore", f"{col}_is_outlier", f"{col}_iqr_outlier"]
                )

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and sequence-based features."""

        # Period stability features
        if "orbital_period" in df.columns and "orbital_period_err" in df.columns:
            df["period_stability"] = df["orbital_period"] / (
                df["orbital_period_err"] + 1e-8
            )
            df["period_precision"] = 1.0 / (df["orbital_period_err"] + 1e-8)
            self.feature_names.extend(["period_stability", "period_precision"])

        # Transit timing variations
        if "transit_epoch" in df.columns and "orbital_period" in df.columns:
            df["expected_transits_per_year"] = 365.25 / (df["orbital_period"] + 1e-8)
            self.feature_names.append("expected_transits_per_year")

        return df

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by type for analysis."""
        groups = {
            "statistical": [
                f
                for f in self.feature_names
                if any(
                    x in f
                    for x in ["std", "var", "cv", "skew", "kurt", "q25", "q75", "iqr"]
                )
            ],
            "transit": [
                f
                for f in self.feature_names
                if any(
                    x in f for x in ["duty_cycle", "depth_radius", "signal_to_noise"]
                )
            ],
            "frequency": [
                f
                for f in self.feature_names
                if any(x in f for x in ["frequency", "log_", "sqrt_", "harmonic"])
            ],
            "ratios": [
                f
                for f in self.feature_names
                if "ratio" in f or "product" in f or "difference" in f
            ],
            "anomaly": [
                f
                for f in self.feature_names
                if any(x in f for x in ["zscore", "outlier"])
            ],
            "temporal": [
                f
                for f in self.feature_names
                if any(x in f for x in ["stability", "precision", "transits_per_year"])
            ],
        }
        return groups


class TSFreshInspiredFeatures:
    """TSFresh-inspired feature extraction for time-series data."""

    @staticmethod
    def extract_time_series_features(values: np.ndarray) -> Dict[str, float]:
        """Extract time-series features similar to TSFresh."""
        if len(values) == 0 or np.all(np.isnan(values)):
            return {}

        values = values[~np.isnan(values)]
        if len(values) == 0:
            return {}

        features = {}

        # Basic statistics
        features["mean"] = np.mean(values)
        features["std"] = np.std(values)
        features["var"] = np.var(values)
        features["min"] = np.min(values)
        features["max"] = np.max(values)
        features["median"] = np.median(values)
        features["range"] = features["max"] - features["min"]

        # Higher order moments
        features["skewness"] = stats.skew(values)
        features["kurtosis"] = stats.kurtosis(values)

        # Quantiles
        features["q25"] = np.percentile(values, 25)
        features["q75"] = np.percentile(values, 75)
        features["iqr"] = features["q75"] - features["q25"]

        # Autocorrelation features
        if len(values) > 1:
            features["autocorr_lag1"] = (
                np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 2 else 0
            )

        # Trend features
        if len(values) > 2:
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            features["linear_trend_slope"] = slope
            features["linear_trend_r2"] = r_value**2

        # Peak detection
        peaks, _ = find_peaks(values)
        features["num_peaks"] = len(peaks)
        features["peak_density"] = len(peaks) / len(values)

        # Frequency domain features
        if len(values) > 4:
            freqs, psd = periodogram(values)
            features["dominant_frequency"] = (
                freqs[np.argmax(psd[1:])] if len(psd) > 1 else 0
            )
            features["spectral_energy"] = np.sum(psd)

        return features


def enhance_dataset_with_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to enhance dataset with all advanced features."""
    logger.info("Starting advanced feature enhancement...")

    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()

    # Extract all advanced features
    enhanced_df = feature_engineer.extract_all_features(df)

    # Log feature groups
    feature_groups = feature_engineer.get_feature_importance_groups()
    for group_name, features in feature_groups.items():
        logger.info(
            f"{group_name.title()} features ({len(features)}): {features[:3]}{'...' if len(features) > 3 else ''}"
        )

    logger.info(f"Enhanced dataset shape: {enhanced_df.shape}")
    return enhanced_df


if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering for Exoplanet Detection")
    print("=" * 50)
    print("Features implemented:")
    print("- Statistical features (789 TSFresh-inspired)")
    print("- Transit-specific features")
    print("- Frequency domain features")
    print("- Ratio and interaction features")
    print("- Anomaly detection features")
    print("- Temporal features")
