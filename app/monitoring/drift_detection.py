"""Model drift detection using Evidently AI."""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


class DriftDetector:
    """Detect model drift using Evidently AI."""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (training/baseline data)
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently AI not installed. Install with: pip install evidently")
        
        self.reference_data = reference_data
    
    def analyze_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Analyze drift between reference and current data.
        
        Args:
            current_data: Recent predictions data
            reference_data: Reference dataset (uses default if not provided)
            
        Returns:
            Dict with drift analysis results
        """
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None:
            return {
                "error": "No reference data provided",
                "drift_detected": False
            }
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name="confidence_spam"),
            ColumnDriftMetric(column_name="text_length"),
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Extract results
        results = report.as_dict()
        
        # Parse drift information
        dataset_drift = results["metrics"][1]["result"]
        
        return {
            "drift_detected": dataset_drift.get("dataset_drift", False),
            "number_of_drifted_columns": dataset_drift.get("number_of_drifted_columns", 0),
            "share_of_drifted_columns": dataset_drift.get("share_of_drifted_columns", 0),
            "drift_by_columns": dataset_drift.get("drift_by_columns", {}),
            "timestamp": datetime.utcnow().isoformat(),
            "n_reference": len(reference_data),
            "n_current": len(current_data)
        }
    
    def generate_html_report(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        output_path: str = "data/drift_report.html"
    ):
        """Generate HTML drift report.
        
        Args:
            current_data: Recent predictions data
            reference_data: Reference dataset
            output_path: Path to save HTML report
        """
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None:
            raise ValueError("No reference data provided")
        
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        report.save_html(output_path)
        
        return output_path


def prepare_data_for_drift_analysis(predictions: List[Dict]) -> pd.DataFrame:
    """Convert prediction logs to DataFrame for drift analysis.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        DataFrame with features for drift analysis
    """
    df = pd.DataFrame(predictions)
    
    # Keep relevant columns
    if not df.empty:
        df = df[[
            "text_length",
            "prediction",
            "confidence_spam",
            "confidence_ham",
            "timestamp"
        ]]
        
        # Convert prediction to numeric
        df["prediction_numeric"] = (df["prediction"] == "spam").astype(int)
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df
