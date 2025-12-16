"""Database storage for predictions and metrics."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

Base = declarative_base()


class PredictionLog(Base):
    """Model for storing prediction logs."""
    
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    text_hash = Column(String(64), index=True)  # SHA256 hash for privacy
    text_length = Column(Integer)
    prediction = Column(String(10))  # 'spam' or 'ham'
    confidence_spam = Column(Float)
    confidence_ham = Column(Float)
    response_time_ms = Column(Float)
    error = Column(Text, nullable=True)
    

class MonitoringDatabase:
    """Database interface for monitoring data."""
    
    def __init__(self, db_path: str = "data/monitoring.db"):
        """Initialize database connection."""
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f"sqlite:///{db_file}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def log_prediction(
        self,
        text: str,
        prediction: str,
        confidence_spam: float,
        confidence_ham: float,
        response_time_ms: float,
        error: Optional[str] = None
    ):
        """Log a prediction to database."""
        session = self.get_session()
        try:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            log = PredictionLog(
                text_hash=text_hash,
                text_length=len(text),
                prediction=prediction,
                confidence_spam=confidence_spam,
                confidence_ham=confidence_ham,
                response_time_ms=response_time_ms,
                error=error
            )
            session.add(log)
            session.commit()
        finally:
            session.close()
    
    def get_predictions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Retrieve predictions within time range."""
        session = self.get_session()
        try:
            query = session.query(PredictionLog)
            
            if start_time:
                query = query.filter(PredictionLog.timestamp >= start_time)
            if end_time:
                query = query.filter(PredictionLog.timestamp <= end_time)
            
            query = query.order_by(PredictionLog.timestamp.desc()).limit(limit)
            
            results = []
            for log in query.all():
                results.append({
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "text_length": log.text_length,
                    "prediction": log.prediction,
                    "confidence_spam": log.confidence_spam,
                    "confidence_ham": log.confidence_ham,
                    "response_time_ms": log.response_time_ms,
                    "error": log.error
                })
            
            return results
        finally:
            session.close()
    
    def get_statistics(self, days: int = 30) -> Dict:
        """Get prediction statistics for last N days."""
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Total predictions
            total = session.query(func.count(PredictionLog.id)).filter(
                PredictionLog.timestamp >= cutoff
            ).scalar()
            
            # Spam/Ham counts
            spam_count = session.query(func.count(PredictionLog.id)).filter(
                PredictionLog.timestamp >= cutoff,
                PredictionLog.prediction == "spam"
            ).scalar()
            
            ham_count = session.query(func.count(PredictionLog.id)).filter(
                PredictionLog.timestamp >= cutoff,
                PredictionLog.prediction == "ham"
            ).scalar()
            
            # Average confidence
            avg_confidence_spam = session.query(func.avg(PredictionLog.confidence_spam)).filter(
                PredictionLog.timestamp >= cutoff
            ).scalar() or 0.0
            
            # Average response time
            avg_response_time = session.query(func.avg(PredictionLog.response_time_ms)).filter(
                PredictionLog.timestamp >= cutoff
            ).scalar() or 0.0
            
            # Error count
            error_count = session.query(func.count(PredictionLog.id)).filter(
                PredictionLog.timestamp >= cutoff,
                PredictionLog.error.isnot(None)
            ).scalar()
            
            return {
                "period_days": days,
                "total_predictions": total or 0,
                "spam_count": spam_count or 0,
                "ham_count": ham_count or 0,
                "spam_ratio": (spam_count / total) if total > 0 else 0.0,
                "avg_confidence_spam": float(avg_confidence_spam),
                "avg_response_time_ms": float(avg_response_time),
                "error_count": error_count or 0,
                "error_rate": (error_count / total) if total > 0 else 0.0
            }
        finally:
            session.close()
    
    def cleanup_old_records(self, retention_days: int = 30):
        """Delete records older than retention period."""
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=retention_days)
            deleted = session.query(PredictionLog).filter(
                PredictionLog.timestamp < cutoff
            ).delete()
            session.commit()
            return deleted
        finally:
            session.close()
