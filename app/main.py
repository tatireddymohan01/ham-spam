import time
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.responses import PlainTextResponse
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler

from app.schemas import TextRequest, PredictionResponse
from app.model import SpamModel
from app.config import APP_NAME, VERSION
from app.monitoring import (
    MetricsCollector,
    setup_monitoring_logger,
    MonitoringDatabase,
    MonitoringMiddleware
)
from app.monitoring.drift_detection import DriftDetector, prepare_data_for_drift_analysis

# Initialize monitoring components
monitoring_db = MonitoringDatabase()
metrics_collector = MetricsCollector()
monitoring_logger = setup_monitoring_logger()

app = FastAPI(title=APP_NAME, version=VERSION)

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware, metrics_collector=metrics_collector)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SpamModel()

# Mount static files
web_dir = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

# Initialize scheduler for cleanup tasks
scheduler = BackgroundScheduler()

def cleanup_old_records():
    """Background task to clean up old records."""
    deleted = monitoring_db.cleanup_old_records(retention_days=30)
    monitoring_logger.info(f"Cleaned up {deleted} old records")

@app.on_event("startup")
def startup_event():
    """Initialize background tasks on startup."""
    # Schedule daily cleanup at midnight
    scheduler.add_job(cleanup_old_records, "cron", hour=0, minute=0)
    scheduler.start()
    monitoring_logger.info("Monitoring system initialized")

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    scheduler.shutdown()
    monitoring_logger.info("Monitoring system shutdown")

@app.get("/")
def home():
    return {
        "message": "Ham-Spam Classifier API is running!",
        "ui": "/web",
        "monitoring": "/monitoring",
        "api_docs": "/docs"
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/web")
def web_ui():
    """Serve the web UI"""
    return FileResponse(str(web_dir / "index.html"))

@app.get("/monitoring")
def monitoring_ui():
    """Serve the monitoring dashboard UI"""
    return FileResponse(str(web_dir / "monitoring.html"))

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: TextRequest, background_tasks: BackgroundTasks):
    """Predict spam/ham with monitoring."""
    start_time = time.time()
    error = None
    
    try:
        label, prob_spam, prob_ham = model.predict(req.text)
        response_time_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        metrics_collector.record_prediction(
            prediction=label,
            confidence_spam=prob_spam or 0.0,
            response_time_ms=response_time_ms,
            is_error=False
        )
        
        # Log to database (in background)
        background_tasks.add_task(
            monitoring_db.log_prediction,
            text=req.text,
            prediction=label,
            confidence_spam=prob_spam or 0.0,
            confidence_ham=prob_ham or 0.0,
            response_time_ms=response_time_ms,
            error=None
        )
        
        return PredictionResponse(
            label=label,
            probability_spam=prob_spam,
            probability_ham=prob_ham
        )
    
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        error = str(e)
        
        # Record error
        metrics_collector.record_prediction(
            prediction="error",
            confidence_spam=0.0,
            response_time_ms=response_time_ms,
            is_error=True
        )
        
        monitoring_logger.error(f"Prediction error: {error}")
        raise

@app.get("/metrics")
def get_metrics():
    """Get current metrics (Prometheus format)."""
    return Response(
        content=metrics_collector.get_prometheus_metrics(),
        media_type="text/plain"
    )

@app.get("/metrics/json")
def get_metrics_json():
    """Get current metrics in JSON format."""
    return metrics_collector.get_current_metrics()

@app.get("/metrics/drift")
def get_drift_status():
    """Check for model drift."""
    return metrics_collector.detect_drift()

@app.get("/admin/dashboard")
def get_dashboard():
    """Get monitoring dashboard data."""
    # Get statistics for different periods
    stats_24h = monitoring_db.get_statistics(days=1)
    stats_7d = monitoring_db.get_statistics(days=7)
    stats_30d = monitoring_db.get_statistics(days=30)
    
    # Get current metrics
    current_metrics = metrics_collector.get_current_metrics()
    
    # Check drift
    drift_status = metrics_collector.detect_drift()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health": "healthy",
        "statistics": {
            "last_24_hours": stats_24h,
            "last_7_days": stats_7d,
            "last_30_days": stats_30d
        },
        "current_metrics": current_metrics,
        "drift_detection": drift_status
    }

@app.get("/admin/predictions")
def get_recent_predictions(hours: int = 24, limit: int = 100):
    """Get recent predictions."""
    start_time = datetime.utcnow() - timedelta(hours=hours)
    predictions = monitoring_db.get_predictions(
        start_time=start_time,
        limit=limit
    )
    return {
        "count": len(predictions),
        "predictions": predictions
    }

@app.post("/admin/drift-report")
async def generate_drift_report(days: int = 7):
    """Generate Evidently AI drift report."""
    try:
        # Get recent predictions
        start_time = datetime.utcnow() - timedelta(days=days)
        predictions = monitoring_db.get_predictions(start_time=start_time, limit=5000)
        
        if len(predictions) < 50:
            return JSONResponse(
                status_code=400,
                content={"error": "Insufficient data for drift analysis (minimum 50 predictions required)"}
            )
        
        # Prepare data
        current_df = prepare_data_for_drift_analysis(predictions)
        
        # For reference data, use older predictions
        ref_start = datetime.utcnow() - timedelta(days=days*2)
        ref_end = datetime.utcnow() - timedelta(days=days)
        reference_predictions = monitoring_db.get_predictions(
            start_time=ref_start,
            end_time=ref_end,
            limit=5000
        )
        
        if len(reference_predictions) < 50:
            return JSONResponse(
                status_code=400,
                content={"error": "Insufficient reference data"}
            )
        
        reference_df = prepare_data_for_drift_analysis(reference_predictions)
        
        # Analyze drift
        detector = DriftDetector(reference_data=reference_df)
        drift_analysis = detector.analyze_drift(current_df)
        
        # Generate HTML report
        report_path = detector.generate_html_report(
            current_data=current_df,
            reference_data=reference_df,
            output_path="data/drift_report.html"
        )
        
        return {
            "drift_analysis": drift_analysis,
            "report_path": report_path,
            "report_url": "/admin/drift-report-view"
        }
    
    except Exception as e:
        monitoring_logger.error(f"Drift report generation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/admin/drift-report-view")
def view_drift_report():
    """View the generated drift report."""
    report_path = Path("data/drift_report.html")
    if report_path.exists():
        return FileResponse(str(report_path))
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "No drift report found. Generate one first at POST /admin/drift-report"}
        )
