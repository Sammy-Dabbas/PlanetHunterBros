"""
Uncertainty Quantification Module
Implements conformal prediction for uncertainty intervals in exoplanet detection
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class UncertaintyQuantifier:
    """
    Implements conformal prediction to provide calibrated uncertainty intervals
    for exoplanet detection predictions.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize the uncertainty quantifier.

        Parameters:
        -----------
        alpha : float
            Desired miscoverage rate (default 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.quantile = None
        self.is_calibrated = False

    def calibrate(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
        """
        Calibrate the uncertainty quantifier using a trained model and data.

        Parameters:
        -----------
        model : sklearn model
            Trained model with predict_proba method
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
        cv : int
            Number of cross-validation folds (default 5)
        """
        from sklearn.model_selection import cross_val_predict

        # Get out-of-fold predictions
        if hasattr(model, 'predict_proba'):
            y_pred = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
            # Use probability scores for positive class
            if len(y_pred.shape) > 1:
                y_pred = y_pred[:, 1]
        else:
            y_pred = cross_val_predict(model, X, y, cv=cv)

        # Compute nonconformity scores
        cal_scores = self.compute_nonconformity_scores(y, y_pred)

        n = len(cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(cal_scores, q_level)
        self.is_calibrated = True
        print(f"Calibrated uncertainty quantifier with quantile: {self.quantile:.4f}")
        print(f"Used {cv}-fold cross-validation on {n} samples")

    def predict_with_intervals(
        self,
        predictions: np.ndarray,
        base_uncertainty: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty intervals.

        Parameters:
        -----------
        predictions : np.ndarray
            Point predictions from the model
        base_uncertainty : np.ndarray, optional
            Base uncertainty estimates (if None, uses quantile)

        Returns:
        --------
        predictions : np.ndarray
            Point predictions
        lower_bounds : np.ndarray
            Lower bounds of prediction intervals
        upper_bounds : np.ndarray
            Upper bounds of prediction intervals
        """
        if not self.is_calibrated:
            raise ValueError("Quantifier must be calibrated before making predictions")

        if base_uncertainty is None:
            # Use constant quantile for all predictions
            interval_width = self.quantile
            lower_bounds = predictions - interval_width
            upper_bounds = predictions + interval_width
        else:
            # Scale base uncertainty by calibrated quantile
            lower_bounds = predictions - base_uncertainty * self.quantile
            upper_bounds = predictions + base_uncertainty * self.quantile

        return predictions, lower_bounds, upper_bounds

    def compute_nonconformity_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity scores as absolute errors.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns:
        --------
        scores : np.ndarray
            Nonconformity scores
        """
        return np.abs(y_true - y_pred)

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> float:
        """
        Evaluate empirical coverage of prediction intervals.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        lower_bounds : np.ndarray
            Lower bounds of prediction intervals
        upper_bounds : np.ndarray
            Upper bounds of prediction intervals

        Returns:
        --------
        coverage : float
            Fraction of true values within prediction intervals
        """
        covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        coverage = np.mean(covered)
        return coverage


def demo_uncertainty_quantification():
    """
    Demonstrate uncertainty quantification with conformal prediction.
    """
    print("=" * 60)
    print("Uncertainty Quantification Demo")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Simulate predictions and true values
    y_true = np.random.randn(n_samples)
    y_pred = y_true + np.random.randn(n_samples) * 0.3  # Add prediction error

    # Split into calibration and test sets
    indices = np.arange(n_samples)
    cal_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=42)

    y_true_cal = y_true[cal_idx]
    y_pred_cal = y_pred[cal_idx]
    y_true_test = y_true[test_idx]
    y_pred_test = y_pred[test_idx]

    print(f"\nDataset sizes:")
    print(f"  Calibration: {len(cal_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Initialize and calibrate uncertainty quantifier
    print(f"\nCalibrating with target coverage: {90}%")
    uq = UncertaintyQuantifier(alpha=0.1)

    # Compute nonconformity scores on calibration set
    cal_scores = uq.compute_nonconformity_scores(y_true_cal, y_pred_cal)
    uq.calibrate(cal_scores)

    # Generate predictions with intervals on test set
    print("\nGenerating prediction intervals...")
    predictions, lower, upper = uq.predict_with_intervals(y_pred_test)

    # Evaluate coverage
    coverage = uq.evaluate_coverage(y_true_test, lower, upper)
    print(f"\nEmpirical coverage on test set: {coverage * 100:.2f}%")
    print(f"Target coverage: {(1 - uq.alpha) * 100:.2f}%")

    # Show example predictions
    print("\nExample predictions with uncertainty intervals:")
    print("-" * 60)
    print(f"{'Index':<8} {'True':<10} {'Pred':<10} {'Lower':<10} {'Upper':<10} {'Covered':<8}")
    print("-" * 60)

    for i in range(min(10, len(predictions))):
        true_val = y_true_test[i]
        pred_val = predictions[i]
        lower_val = lower[i]
        upper_val = upper[i]
        covered = "Yes" if lower_val <= true_val <= upper_val else "No"

        print(f"{i:<8} {true_val:<10.4f} {pred_val:<10.4f} {lower_val:<10.4f} {upper_val:<10.4f} {covered:<8}")

    # Compute interval statistics
    interval_widths = upper - lower
    print(f"\nInterval statistics:")
    print(f"  Mean width: {np.mean(interval_widths):.4f}")
    print(f"  Std width: {np.std(interval_widths):.4f}")
    print(f"  Min width: {np.min(interval_widths):.4f}")
    print(f"  Max width: {np.max(interval_widths):.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_uncertainty_quantification()
