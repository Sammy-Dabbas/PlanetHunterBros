"""
Real-time TESS Data Fetcher
Fetches recent TESS Object of Interest (TOI) discoveries from NASA's Exoplanet Archive
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import warnings


class TESSRealtimeFetcher:
    """
    Fetches and processes real-time TESS Object of Interest (TOI) data
    from NASA's Exoplanet Archive API.
    """

    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    def __init__(self):
        """Initialize the TESS real-time fetcher."""
        self.toi_data = None
        self.last_fetch = None

    def fetch_all_tois(self, timeout: int = 30) -> pd.DataFrame:
        """
        Fetch all TOIs from NASA Exoplanet Archive.

        Parameters:
        -----------
        timeout : int
            Request timeout in seconds

        Returns:
        --------
        df : pd.DataFrame
            DataFrame containing all TOI data
        """
        print("Fetching all TESS Objects of Interest from NASA Exoplanet Archive...")

        query = """
        SELECT top 10000 * FROM toi
        """

        params = {
            'query': query,
            'format': 'csv'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()

            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            self.toi_data = df
            self.last_fetch = datetime.now()

            print(f"Successfully fetched {len(df)} TOI entries")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching TOI data: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error parsing TOI data: {e}")
            return pd.DataFrame()

    def get_recent_tois(self, days: int = 30) -> pd.DataFrame:
        """
        Get TOIs discovered or updated in the last N days.

        Parameters:
        -----------
        days : int
            Number of days to look back

        Returns:
        --------
        df : pd.DataFrame
            DataFrame containing recent TOI data
        """
        if self.toi_data is None:
            self.fetch_all_tois()

        if self.toi_data is None or len(self.toi_data) == 0:
            print("No TOI data available")
            return pd.DataFrame()

        print(f"\nFiltering TOIs from the last {days} days...")

        # Try to use the date column if available
        date_columns = ['toi_created', 'rowupdate', 'toi_disposition_date']
        date_col = None

        for col in date_columns:
            if col in self.toi_data.columns:
                date_col = col
                break

        if date_col is None:
            print("Warning: No date column found, returning all TOIs")
            return self.toi_data

        try:
            # Parse dates
            self.toi_data[date_col] = pd.to_datetime(self.toi_data[date_col], errors='coerce')

            # Filter recent entries
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_mask = self.toi_data[date_col] >= cutoff_date
            recent_tois = self.toi_data[recent_mask].copy()

            print(f"Found {len(recent_tois)} TOIs from the last {days} days")
            return recent_tois

        except Exception as e:
            print(f"Error filtering by date: {e}")
            return self.toi_data

    def harmonize_toi_to_kepler_format(self, toi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert TOI data to match Kepler exoplanet format for model compatibility.

        Parameters:
        -----------
        toi_df : pd.DataFrame
            TOI data from NASA archive

        Returns:
        --------
        df : pd.DataFrame
            Harmonized DataFrame compatible with Kepler-trained models
        """
        print("\nHarmonizing TOI data to Kepler format...")

        if len(toi_df) == 0:
            return pd.DataFrame()

        # Mapping from TOI columns to Kepler columns
        column_mapping = {
            'toi': 'kepid',  # Use TOI ID as identifier
            'pl_orbper': 'koi_period',  # Orbital period
            'pl_rade': 'koi_prad',  # Planet radius
            'pl_insol': 'koi_insol',  # Insolation flux
            'st_rad': 'koi_srad',  # Stellar radius
            'st_mass': 'koi_smass',  # Stellar mass
            'st_teff': 'koi_steff',  # Stellar effective temperature
            'tfopwg_disp': 'koi_disposition'  # Disposition (confirmed/false positive)
        }

        harmonized = pd.DataFrame()

        for toi_col, kepler_col in column_mapping.items():
            if toi_col in toi_df.columns:
                harmonized[kepler_col] = toi_df[toi_col]
            else:
                # Fill with NaN if column doesn't exist
                harmonized[kepler_col] = np.nan

        # Reset index to avoid alignment issues
        harmonized = harmonized.reset_index(drop=True)

        # Add commonly used Kepler columns with estimated/default values
        if 'koi_period' in harmonized.columns:
            # Estimate transit duration based on period and radius (simplified formula)
            harmonized.loc[:, 'koi_duration'] = harmonized['koi_period'].fillna(10).values * 0.1
        else:
            harmonized['koi_duration'] = 1.0

        if 'koi_prad' not in harmonized.columns or harmonized['koi_prad'].isna().all():
            harmonized.loc[:, 'koi_prad'] = 2.0  # Default Earth-sized planet

        # Add other common columns with reasonable defaults
        harmonized.loc[:, 'koi_depth'] = 100  # Transit depth in ppm
        harmonized.loc[:, 'koi_teq'] = 300  # Equilibrium temperature in K
        harmonized.loc[:, 'koi_model_snr'] = 10  # Signal-to-noise ratio

        # Convert disposition to numeric (1 for planet, 0 for false positive)
        if 'koi_disposition' in harmonized.columns:
            harmonized['koi_disposition'] = harmonized['koi_disposition'].map({
                'PC': 1,  # Planet Candidate
                'CP': 1,  # Confirmed Planet
                'FP': 0,  # False Positive
                'KP': 1,  # Known Planet
                'APC': 1  # Astrophysical False Positive
            })

        # Add label column (2 = candidate, 1 = confirmed, 0 = false positive)
        if 'tfopwg_disp' in toi_df.columns:
            harmonized['label'] = toi_df['tfopwg_disp'].map({
                'PC': 2,  # Planet Candidate
                'CP': 1,  # Confirmed Planet
                'FP': 0,  # False Positive
                'KP': 1,  # Known Planet
                'APC': 0  # Astrophysical False Positive
            })

        print(f"Harmonized {len(harmonized)} TOI entries to Kepler format")
        print(f"Columns: {list(harmonized.columns)}")

        # Print class distribution
        if 'koi_disposition' in harmonized.columns:
            class_dist = harmonized['koi_disposition'].value_counts()
            print(f"\nClass distribution:")
            print(f"  Planets (1): {class_dist.get(1, 0)}")
            print(f"  False Positives (0): {class_dist.get(0, 0)}")
            print(f"  Unknown/NaN: {harmonized['koi_disposition'].isna().sum()}")

            # Warning if only one class
            n_classes = harmonized['koi_disposition'].nunique()
            if n_classes < 2:
                print("\nWARNING: Only one class found in data!")
                print("TESS data is typically for PREDICTION, not training.")
                print("Train your model on Kepler data first, then predict on TESS data.")

        return harmonized

    def get_toi_metadata(self, toi_number: float) -> Dict:
        """
        Get detailed metadata for a specific TOI.

        Parameters:
        -----------
        toi_number : float
            TOI number (e.g., 1234.01)

        Returns:
        --------
        metadata : dict
            Dictionary containing TOI metadata
        """
        if self.toi_data is None:
            self.fetch_all_tois()

        if self.toi_data is None or len(self.toi_data) == 0:
            return {}

        # Find the TOI
        if 'toi' in self.toi_data.columns:
            toi_row = self.toi_data[self.toi_data['toi'] == toi_number]

            if len(toi_row) == 0:
                print(f"TOI {toi_number} not found")
                return {}

            # Convert to dictionary, handling NaN values
            metadata = toi_row.iloc[0].to_dict()

            # Clean up NaN values
            metadata = {k: (v if pd.notna(v) else None) for k, v in metadata.items()}

            return metadata

        return {}

    def get_confirmed_planets(self) -> pd.DataFrame:
        """
        Get only confirmed planets from TOI data.

        Returns:
        --------
        df : pd.DataFrame
            DataFrame containing confirmed planets
        """
        if self.toi_data is None:
            self.fetch_all_tois()

        if self.toi_data is None or len(self.toi_data) == 0:
            return pd.DataFrame()

        if 'tfopwg_disp' in self.toi_data.columns:
            confirmed = self.toi_data[
                self.toi_data['tfopwg_disp'].isin(['CP', 'KP'])
            ].copy()

            print(f"Found {len(confirmed)} confirmed planets")
            return confirmed

        return pd.DataFrame()


def demo_recent_discoveries():
    """
    Demonstrate fetching and processing recent TESS discoveries.
    """
    print("=" * 60)
    print("TESS Real-time Data Fetcher Demo")
    print("=" * 60)

    fetcher = TESSRealtimeFetcher()

    # Fetch all TOIs
    print("\n[1] Fetching all TOIs...")
    all_tois = fetcher.fetch_all_tois()

    if len(all_tois) > 0:
        print(f"\nTotal TOIs fetched: {len(all_tois)}")
        print(f"Columns available: {len(all_tois.columns)}")

        # Show sample columns
        key_columns = ['toi', 'tfopwg_disp', 'pl_orbper', 'pl_rade', 'st_teff']
        available_key_cols = [col for col in key_columns if col in all_tois.columns]

        if available_key_cols:
            print(f"\nSample TOI data (first 5 rows):")
            print(all_tois[available_key_cols].head())

        # Get disposition statistics
        if 'tfopwg_disp' in all_tois.columns:
            print(f"\nTOI Disposition breakdown:")
            print(all_tois['tfopwg_disp'].value_counts())

    # Get recent TOIs
    print("\n" + "-" * 60)
    print("[2] Getting recent TOIs (last 90 days)...")
    recent_tois = fetcher.get_recent_tois(days=90)

    if len(recent_tois) > 0:
        print(f"\nRecent TOIs: {len(recent_tois)}")

    # Get confirmed planets
    print("\n" + "-" * 60)
    print("[3] Getting confirmed planets...")
    confirmed = fetcher.get_confirmed_planets()

    if len(confirmed) > 0:
        print(f"\nConfirmed planets: {len(confirmed)}")

    # Harmonize to Kepler format
    print("\n" + "-" * 60)
    print("[4] Harmonizing to Kepler format...")
    harmonized = fetcher.harmonize_toi_to_kepler_format(all_tois.head(100))

    if len(harmonized) > 0:
        print(f"\nHarmonized data shape: {harmonized.shape}")
        print(f"Sample harmonized data:")
        print(harmonized.head())

    # Get specific TOI metadata
    if len(all_tois) > 0 and 'toi' in all_tois.columns:
        print("\n" + "-" * 60)
        print("[5] Getting metadata for specific TOI...")

        # Get first TOI number
        first_toi = all_tois['toi'].iloc[0]
        metadata = fetcher.get_toi_metadata(first_toi)

        if metadata:
            print(f"\nMetadata for TOI {first_toi}:")
            for key, value in list(metadata.items())[:10]:  # Show first 10 fields
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_recent_discoveries()
