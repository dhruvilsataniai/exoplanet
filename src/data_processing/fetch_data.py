"""
Data fetching module for NASA exoplanet datasets.
Downloads and caches data from the NASA Exoplanet Archive using TAP service.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetDataFetcher:
    """Fetches exoplanet data from NASA Exoplanet Archive."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # NASA Exoplanet Archive TAP service URLs for CSV downloads
        self.data_urls = {
            "kepler_koi": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
            "tess_toi": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
            "k2_planets": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv",
        }

    def fetch_dataset(self, dataset_name, force_refresh=False, output_format="csv"):
        """
        Fetch a specific dataset from NASA Exoplanet Archive.

        Args:
            dataset_name (str): Name of the dataset ('kepler_koi', 'tess_toi', 'k2_planets')
            force_refresh (bool): Whether to force re-download even if cached file exists
            output_format (str): Output format ('csv', 'votable', 'ipac')

        Returns:
            pd.DataFrame: The requested dataset
        """
        cache_file = self.data_dir / f"{dataset_name}.csv"

        # Return cached data if it exists and refresh is not forced
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached {dataset_name} data from {cache_file}")
            return pd.read_csv(cache_file)

        # Fetch data from direct URL
        if dataset_name not in self.data_urls:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.data_urls.keys())}"
            )

        logger.info(f"Fetching {dataset_name} data from NASA Exoplanet Archive...")

        # Use the TAP service URL (format already included)
        url = self.data_urls[dataset_name]

        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(response.text)

            # Load and return as DataFrame
            if output_format == "csv":
                # Handle potential CSV parsing issues
                try:
                    df = pd.read_csv(cache_file, comment="#", skip_blank_lines=True)
                except pd.errors.EmptyDataError:
                    logger.warning(f"Empty data received for {dataset_name}")
                    df = pd.DataFrame()
                except Exception as e:
                    logger.warning(
                        f"CSV parsing failed, trying with different parameters: {e}"
                    )
                    df = pd.read_csv(
                        cache_file,
                        comment="#",
                        skip_blank_lines=True,
                        on_bad_lines="skip",
                    )
            else:
                # For other formats, try to read as CSV anyway
                df = pd.read_csv(
                    cache_file, comment="#", skip_blank_lines=True, on_bad_lines="skip"
                )

            logger.info(f"Successfully fetched {len(df)} records for {dataset_name}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {dataset_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing {dataset_name} data: {e}")
            raise

    def fetch_all_datasets(self, force_refresh=False):
        """
        Fetch all available datasets.

        Args:
            force_refresh (bool): Whether to force re-download cached files

        Returns:
            dict: Dictionary containing all datasets
        """
        datasets = {}
        for name in self.data_urls.keys():
            try:
                datasets[name] = self.fetch_dataset(name, force_refresh)
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")
                continue

        return datasets

    def get_dataset_info(self):
        """Get information about available datasets."""
        info = {
            "kepler_koi": {
                "description": "Kepler Objects of Interest - planetary candidates from Kepler mission",
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
                "size_estimate": "~10,000 records",
                "key_features": [
                    "kepoi_name",
                    "koi_disposition",
                    "koi_period",
                    "koi_depth",
                    "koi_prad",
                ],
            },
            "tess_toi": {
                "description": "TESS Objects of Interest - planetary candidates from TESS mission",
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
                "size_estimate": "~5,000 records",
                "key_features": [
                    "toi",
                    "tfopwg_disp",
                    "pl_orbper",
                    "pl_trandep",
                    "pl_rade",
                ],
            },
            "k2_planets": {
                "description": "K2 Planets and Candidates - planetary candidates from K2 mission",
                "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv",
                "size_estimate": "~1,000 records",
                "key_features": [
                    "pl_name",
                    "pl_discmethod",
                    "pl_orbper",
                    "pl_rade",
                    "pl_masse",
                ],
            },
        }
        return info


def main():
    """Main function to demonstrate data fetching."""
    fetcher = ExoplanetDataFetcher()

    # Print dataset information
    info = fetcher.get_dataset_info()
    print("Available datasets:")
    for name, details in info.items():
        print(f"\n{name}:")
        print(f"  Description: {details['description']}")
        print(f"  URL: {details['url']}")
        print(f"  Estimated size: {details['size_estimate']}")
        print(f"  Key features: {', '.join(details['key_features'])}")

    # Fetch all datasets
    print("\nFetching datasets...")
    datasets = fetcher.fetch_all_datasets()

    # Print summary
    print("\nDataset summary:")
    for name, df in datasets.items():
        print(f"{name}: {len(df)} records, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
