"""Metrics collection for model monitoring."""

import time
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict, deque


class MetricsCollector:
    """Collects and aggregates runtime metrics."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector.
        
        Args:
            window_size: Number of recent predictions to keep in memory
        """
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        
        # Counters
        self.total_predictions = 0
        self.total_spam = 0
        self.total_ham = 0
        self.total_errors = 0
        
        # Baseline for drift detection (set after initial observations)
        self.baseline_spam_ratio: float = None
        self.baseline_confidence: float = None
        self.drift_threshold: float = 0.15  # 15% change triggers alert
    
    def record_prediction(
        self,
        prediction: str,
        confidence_spam: float,
        response_time_ms: float,
        is_error: bool = False
    ):
        """Record a single prediction."""
        self.total_predictions += 1
        
        if is_error:
            self.total_errors += 1
            self.errors.append(datetime.utcnow())
            return
        
        # Update counters
        if prediction == "spam":
            self.total_spam += 1
        else:
            self.total_ham += 1
        
        # Update rolling windows
        self.response_times.append(response_time_ms)
        self.predictions.append(prediction)
        self.confidences.append(confidence_spam)
        
        # Set baseline after 100 predictions
        if self.total_predictions == 100:
            self._set_baseline()
    
    def _set_baseline(self):
        """Set baseline metrics for drift detection."""
        if len(self.predictions) > 0:
            spam_count = sum(1 for p in self.predictions if p == "spam")
            self.baseline_spam_ratio = spam_count / len(self.predictions)
            self.baseline_confidence = sum(self.confidences) / len(self.confidences)
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        current_spam_ratio = 0.0
        current_confidence = 0.0
        
        if len(self.predictions) > 0:
            spam_count = sum(1 for p in self.predictions if p == "spam")
            current_spam_ratio = spam_count / len(self.predictions)
            current_confidence = sum(self.confidences) / len(self.confidences)
        
        # Calculate percentiles for response time
        sorted_times = sorted(self.response_times) if self.response_times else [0]
        p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else 0
        p99 = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else 0
        
        return {
            "total_predictions": self.total_predictions,
            "total_spam": self.total_spam,
            "total_ham": self.total_ham,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_predictions, 1),
            "current_spam_ratio": current_spam_ratio,
            "current_avg_confidence": current_confidence,
            "baseline_spam_ratio": self.baseline_spam_ratio,
            "baseline_confidence": self.baseline_confidence,
            "response_time_p50_ms": p50,
            "response_time_p95_ms": p95,
            "response_time_p99_ms": p99,
            "window_size": len(self.predictions)
        }
    
    def detect_drift(self) -> Dict:
        """Detect if model performance has drifted."""
        if not self.baseline_spam_ratio or len(self.predictions) < 50:
            return {
                "drift_detected": False,
                "reason": "Insufficient data for drift detection",
                "baseline_set": self.baseline_spam_ratio is not None
            }
        
        metrics = self.get_current_metrics()
        current_spam_ratio = metrics["current_spam_ratio"]
        current_confidence = metrics["current_avg_confidence"]
        
        # Check spam ratio drift
        spam_ratio_change = abs(current_spam_ratio - self.baseline_spam_ratio)
        spam_drift = spam_ratio_change > self.drift_threshold
        
        # Check confidence drift (should not drop significantly)
        confidence_change = abs(current_confidence - self.baseline_confidence)
        confidence_drift = confidence_change > 0.10  # 10% confidence drop
        
        drift_detected = spam_drift or confidence_drift
        
        return {
            "drift_detected": drift_detected,
            "spam_ratio_drift": spam_drift,
            "confidence_drift": confidence_drift,
            "spam_ratio_change": spam_ratio_change,
            "confidence_change": confidence_change,
            "current_spam_ratio": current_spam_ratio,
            "baseline_spam_ratio": self.baseline_spam_ratio,
            "current_confidence": current_confidence,
            "baseline_confidence": self.baseline_confidence,
            "threshold": self.drift_threshold
        }
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_current_metrics()
        
        lines = [
            f"# HELP ham_spam_predictions_total Total number of predictions",
            f"# TYPE ham_spam_predictions_total counter",
            f"ham_spam_predictions_total {metrics['total_predictions']}",
            f"",
            f"# HELP ham_spam_predictions_by_class Predictions by class",
            f"# TYPE ham_spam_predictions_by_class counter",
            f'ham_spam_predictions_by_class{{class="spam"}} {metrics["total_spam"]}',
            f'ham_spam_predictions_by_class{{class="ham"}} {metrics["total_ham"]}',
            f"",
            f"# HELP ham_spam_errors_total Total number of errors",
            f"# TYPE ham_spam_errors_total counter",
            f"ham_spam_errors_total {metrics['total_errors']}",
            f"",
            f"# HELP ham_spam_response_time_ms Response time percentiles in milliseconds",
            f"# TYPE ham_spam_response_time_ms gauge",
            f'ham_spam_response_time_ms{{quantile="0.5"}} {metrics["response_time_p50_ms"]}',
            f'ham_spam_response_time_ms{{quantile="0.95"}} {metrics["response_time_p95_ms"]}',
            f'ham_spam_response_time_ms{{quantile="0.99"}} {metrics["response_time_p99_ms"]}',
            f"",
            f"# HELP ham_spam_confidence Average prediction confidence",
            f"# TYPE ham_spam_confidence gauge",
            f"ham_spam_confidence {metrics['current_avg_confidence']}",
        ]
        
        return "\n".join(lines)
