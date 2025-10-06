"""
Adversarial Testing Module
Creates synthetic test cases to evaluate model robustness on edge cases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings


class AdversarialTestGenerator:
    """
    Generates adversarial test cases to challenge exoplanet detection models.
    Creates synthetic light curves that mimic false positives.
    """

    def __init__(self, time_steps: int = 3000, seed: Optional[int] = None):
        """
        Initialize the adversarial test generator.

        Parameters:
        -----------
        time_steps : int
            Number of time steps in generated light curves
        seed : int, optional
            Random seed for reproducibility
        """
        self.time_steps = time_steps
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.test_cases = []

    def create_binary_star_eclipse(
        self,
        n_samples: int = 100,
        eclipse_depth: float = 0.02,
        eclipse_duration: float = 0.15
    ) -> None:
        """
        Create binary star eclipse events that can mimic planet transits.

        Parameters:
        -----------
        n_samples : int
            Number of test cases to generate
        eclipse_depth : float
            Depth of the eclipse (fraction of flux)
        eclipse_duration : float
            Duration of eclipse as fraction of time series
        """
        print(f"Generating {n_samples} binary star eclipse test cases...")

        for i in range(n_samples):
            # Create baseline flux with stellar variability
            flux = np.ones(self.time_steps) + np.random.randn(self.time_steps) * 0.001

            # Add periodic eclipses
            period = np.random.uniform(10, 50)  # Period in time steps
            phase = np.random.uniform(0, period)

            for t in range(self.time_steps):
                phase_position = (t + phase) % period
                if phase_position < eclipse_duration * period:
                    # V-shaped or U-shaped eclipse (more gradual than planet transit)
                    progress = phase_position / (eclipse_duration * period)
                    if progress < 0.5:
                        depth = eclipse_depth * (2 * progress)
                    else:
                        depth = eclipse_depth * (2 * (1 - progress))
                    flux[t] -= depth

            # Add random variations
            flux += np.random.randn(self.time_steps) * 0.0005

            self.test_cases.append({
                'flux': flux,
                'label': 2,  # Not a planet
                'type': 'binary_star_eclipse',
                'description': f'Binary star eclipse {i+1}'
            })

    def create_stellar_variability(
        self,
        n_samples: int = 100,
        variability_scale: float = 0.01
    ) -> None:
        """
        Create stellar variability patterns (starspots, rotation, pulsation).

        Parameters:
        -----------
        n_samples : int
            Number of test cases to generate
        variability_scale : float
            Scale of variability
        """
        print(f"Generating {n_samples} stellar variability test cases...")

        for i in range(n_samples):
            flux = np.ones(self.time_steps)

            # Add multiple periodicities (rotation, pulsation)
            n_components = np.random.randint(2, 5)
            for _ in range(n_components):
                period = np.random.uniform(5, 100)
                amplitude = np.random.uniform(0.001, variability_scale)
                phase = np.random.uniform(0, 2 * np.pi)

                t = np.arange(self.time_steps)
                flux += amplitude * np.sin(2 * np.pi * t / period + phase)

            # Add random noise
            flux += np.random.randn(self.time_steps) * 0.0005

            self.test_cases.append({
                'flux': flux,
                'label': 2,  # Not a planet
                'type': 'stellar_variability',
                'description': f'Stellar variability {i+1}'
            })

    def create_instrumental_noise(
        self,
        n_samples: int = 100,
        noise_level: float = 0.005
    ) -> None:
        """
        Create instrumental artifacts and systematic noise patterns.

        Parameters:
        -----------
        n_samples : int
            Number of test cases to generate
        noise_level : float
            Level of instrumental noise
        """
        print(f"Generating {n_samples} instrumental noise test cases...")

        for i in range(n_samples):
            flux = np.ones(self.time_steps)

            # Add drift
            drift_type = np.random.choice(['linear', 'exponential', 'step'])
            t = np.arange(self.time_steps)

            if drift_type == 'linear':
                drift = np.linspace(0, noise_level, self.time_steps)
            elif drift_type == 'exponential':
                drift = noise_level * (1 - np.exp(-t / (self.time_steps / 3)))
            else:  # step
                step_position = np.random.randint(self.time_steps // 4, 3 * self.time_steps // 4)
                drift = np.zeros(self.time_steps)
                drift[step_position:] = noise_level

            flux += drift

            # Add high-frequency noise
            flux += np.random.randn(self.time_steps) * noise_level * 0.5

            # Add occasional outliers
            n_outliers = np.random.randint(0, 10)
            outlier_positions = np.random.choice(self.time_steps, n_outliers, replace=False)
            flux[outlier_positions] += np.random.randn(n_outliers) * noise_level * 3

            self.test_cases.append({
                'flux': flux,
                'label': 2,  # Not a planet
                'type': 'instrumental_noise',
                'description': f'Instrumental noise {i+1}'
            })

    def create_grazing_transits(
        self,
        n_samples: int = 100,
        transit_depth: float = 0.005
    ) -> None:
        """
        Create grazing transit events (planet barely crossing stellar disk).
        These are challenging edge cases with shallow, unusual transit shapes.

        Parameters:
        -----------
        n_samples : int
            Number of test cases to generate
        transit_depth : float
            Depth of grazing transit
        """
        print(f"Generating {n_samples} grazing transit test cases...")

        for i in range(n_samples):
            flux = np.ones(self.time_steps) + np.random.randn(self.time_steps) * 0.0005

            # Create grazing transit with unusual shape
            period = np.random.uniform(20, 100)
            phase = np.random.uniform(0, period)
            duration = np.random.uniform(0.05, 0.1) * period

            for t in range(self.time_steps):
                phase_position = (t + phase) % period
                if phase_position < duration:
                    # Asymmetric, shallow transit shape
                    progress = phase_position / duration
                    # Use a non-standard transit shape
                    if progress < 0.3:
                        depth = transit_depth * (progress / 0.3) ** 2
                    elif progress < 0.7:
                        depth = transit_depth
                    else:
                        depth = transit_depth * ((1 - progress) / 0.3) ** 2

                    flux[t] -= depth

            self.test_cases.append({
                'flux': flux,
                'label': 2,  # Ambiguous - could be planet or false positive
                'type': 'grazing_transit',
                'description': f'Grazing transit {i+1}'
            })

    def generate_all_adversarial_cases(self, n_samples: int = 20):
        """
        Generate all types of adversarial test cases.

        Parameters:
        -----------
        n_samples : int
            Number of samples per adversarial type (default 20)
        """
        print(f"\nGenerating {n_samples} adversarial cases of each type...")

        # Clear existing test cases
        self.test_cases = []

        # Generate all types
        self.create_binary_star_eclipse(n_samples)
        self.create_stellar_variability(n_samples)
        self.create_instrumental_noise(n_samples)
        self.create_grazing_transits(n_samples)

        print(f"Generated {len(self.test_cases)} total adversarial test cases")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert test cases to a pandas DataFrame with flux values as columns.

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with flux values and metadata
        """
        if not self.test_cases:
            warnings.warn("No test cases generated yet")
            return pd.DataFrame()

        data = []
        for case in self.test_cases:
            row = {'label': case['label'], 'type': case['type']}
            # Add flux values as FLUX.1, FLUX.2, etc.
            for i, flux_val in enumerate(case['flux'], 1):
                row[f'FLUX.{i}'] = flux_val
            data.append(row)

        df = pd.DataFrame(data)
        print(f"\nCreated DataFrame with {len(df)} test cases and {len(df.columns)} columns")
        return df

    def get_summary(self) -> Dict[str, int]:
        """
        Get summary statistics of generated test cases.

        Returns:
        --------
        summary : dict
            Dictionary with counts by type
        """
        summary = {}
        for case in self.test_cases:
            case_type = case['type']
            summary[case_type] = summary.get(case_type, 0) + 1
        return summary


def test_adversarial_robustness(model, X_adversarial: pd.DataFrame, y_adversarial: pd.Series):
    """
    Test a model's robustness on adversarial test cases.

    Parameters:
    -----------
    model : object
        Trained model with predict() method
    X_adversarial : pd.DataFrame
        Adversarial test features
    y_adversarial : pd.Series
        True labels for adversarial tests

    Returns:
    --------
    results : dict
        Dictionary with performance metrics
    """
    print("\n" + "=" * 60)
    print("Testing Model Robustness on Adversarial Cases")
    print("=" * 60)

    # Get predictions
    y_pred = model.predict(X_adversarial)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Handle multi-class case
    avg_method = 'weighted' if len(np.unique(y_adversarial)) > 2 else 'binary'

    accuracy = accuracy_score(y_adversarial, y_pred)
    precision = precision_score(y_adversarial, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_adversarial, y_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_adversarial, y_pred, average=avg_method, zero_division=0)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_samples': len(y_adversarial)
    }

    print(f"\nResults on {results['n_samples']} adversarial test cases:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")

    # Analyze by type if available
    if 'type' in X_adversarial.columns:
        print("\nPerformance by adversarial type:")
        print("-" * 60)

        for adv_type in X_adversarial['type'].unique():
            mask = X_adversarial['type'] == adv_type
            type_accuracy = accuracy_score(y_adversarial[mask], y_pred[mask])
            print(f"  {adv_type}: {type_accuracy:.4f} ({mask.sum()} samples)")

    print("=" * 60)

    return results


if __name__ == "__main__":
    # Demo the adversarial test generator
    print("Adversarial Test Generator Demo")
    print("=" * 60)

    generator = AdversarialTestGenerator(time_steps=3000, seed=42)

    # Generate different types of adversarial cases
    generator.create_binary_star_eclipse(n_samples=50)
    generator.create_stellar_variability(n_samples=50)
    generator.create_instrumental_noise(n_samples=50)
    generator.create_grazing_transits(n_samples=50)

    # Get summary
    summary = generator.get_summary()
    print("\nGenerated test cases:")
    for case_type, count in summary.items():
        print(f"  {case_type}: {count}")

    # Convert to DataFrame
    df = generator.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
