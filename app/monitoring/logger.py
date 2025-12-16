"""Enhanced logging setup for monitoring."""

import logging
import json
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_monitoring_logger(name: str = "monitoring", level: int = logging.INFO) -> logging.Logger:
    """Setup structured logger for monitoring.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_prediction(
    logger: logging.Logger,
    text_length: int,
    prediction: str,
    confidence: float,
    response_time_ms: float
):
    """Log prediction with structured data."""
    extra_fields = {
        "event": "prediction",
        "text_length": text_length,
        "prediction": prediction,
        "confidence": confidence,
        "response_time_ms": response_time_ms
    }
    
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"Prediction: {prediction} (confidence: {confidence:.3f})",
        (),
        None
    )
    record.extra_fields = extra_fields
    logger.handle(record)
