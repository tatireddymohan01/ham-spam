# Model Monitoring & Performance Tracking

This module provides comprehensive monitoring for the ham-spam classifier, including prediction logging, metrics collection, drift detection, and performance tracking.

## Features

### 1. **Prediction Logging**
- Every prediction is logged to SQLite database
- Stores: timestamp, text hash (for privacy), prediction, confidence scores, response time
- 30-day retention policy (automatic cleanup)

### 2. **Real-time Metrics**
- Total predictions counter
- Spam/Ham distribution
- Average confidence scores
- Response time percentiles (p50, p95, p99)
- Error rate tracking

### 3. **Model Drift Detection**
- Baseline establishment after 100 predictions
- Alerts when spam ratio shifts >15% from baseline
- Confidence degradation monitoring
- Evidently AI integration for detailed drift analysis

### 4. **Performance Monitoring**
- Request/response timing
- Throughput tracking
- Error logging
- Prometheus-compatible metrics export

## API Endpoints

### Metrics Endpoints

```bash
# Prometheus format metrics
GET /metrics

# JSON format metrics
GET /metrics/json

# Drift detection status
GET /metrics/drift
```

### Admin Dashboard

```bash
# Main dashboard (statistics + drift + health)
GET /admin/dashboard

# Recent predictions
GET /admin/predictions?hours=24&limit=100

# Generate Evidently AI drift report
POST /admin/drift-report?days=7

# View generated drift report
GET /admin/drift-report-view
```

## Example Responses

### Dashboard (`GET /admin/dashboard`)

```json
{
  "timestamp": "2025-12-16T10:30:00",
  "health": "healthy",
  "statistics": {
    "last_24_hours": {
      "total_predictions": 1250,
      "spam_count": 425,
      "ham_count": 825,
      "spam_ratio": 0.34,
      "avg_confidence_spam": 0.87,
      "avg_response_time_ms": 12.5,
      "error_count": 3,
      "error_rate": 0.0024
    },
    "last_7_days": {...},
    "last_30_days": {...}
  },
  "current_metrics": {
    "total_predictions": 15420,
    "response_time_p95_ms": 25.3,
    "current_spam_ratio": 0.35
  },
  "drift_detection": {
    "drift_detected": false,
    "spam_ratio_drift": false,
    "confidence_drift": false,
    "current_spam_ratio": 0.35,
    "baseline_spam_ratio": 0.33
  }
}
```

### Drift Report (`POST /admin/drift-report`)

```json
{
  "drift_analysis": {
    "drift_detected": true,
    "number_of_drifted_columns": 1,
    "share_of_drifted_columns": 0.25,
    "drift_by_columns": {
      "confidence_spam": {
        "drift_detected": true,
        "drift_score": 0.18
      }
    }
  },
  "report_path": "data/drift_report.html",
  "report_url": "/admin/drift-report-view"
}
```

## Grafana Integration

### Setup

1. **Install Prometheus:**
   ```bash
   docker run -d -p 9090:9090 \
     -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
     prom/prometheus
   ```

2. **Configure Prometheus** (`prometheus.yml`):
   ```yaml
   scrape_configs:
     - job_name: 'ham-spam-api'
       scrape_interval: 15s
       static_configs:
         - targets: ['your-api-url.azurewebsites.net']
       metrics_path: '/metrics'
   ```

3. **Install Grafana:**
   ```bash
   docker run -d -p 3000:3000 grafana/grafana
   ```

4. **Add Prometheus Data Source in Grafana:**
   - URL: `http://localhost:9090`
   - Access: Server (default)

5. **Import Dashboard:**
   - Use the included `grafana_dashboard.json` template
   - Or create custom panels with these metrics:
     - `ham_spam_predictions_total`
     - `ham_spam_predictions_by_class`
     - `ham_spam_response_time_ms`
     - `ham_spam_confidence`

## Evidently AI Reports

### Generate Report via API

```bash
curl -X POST "https://your-api.azurewebsites.net/admin/drift-report?days=7"
```

### View Report

Open the report in browser:
```bash
https://your-api.azurewebsites.net/admin/drift-report-view
```

### Manual Report Generation

```python
from app.monitoring.drift_detection import DriftDetector, prepare_data_for_drift_analysis
from app.monitoring.storage import MonitoringDatabase

# Get data
db = MonitoringDatabase()
predictions = db.get_predictions(limit=5000)

# Prepare data
current_df = prepare_data_for_drift_analysis(predictions)

# Generate report
detector = DriftDetector()
detector.generate_html_report(
    current_data=current_df,
    reference_data=reference_df,
    output_path="drift_report.html"
)
```

## Database Schema

### PredictionLog Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | Prediction timestamp (indexed) |
| text_hash | String(64) | SHA256 hash of input text |
| text_length | Integer | Length of input text |
| prediction | String(10) | 'spam' or 'ham' |
| confidence_spam | Float | Spam probability (0-1) |
| confidence_ham | Float | Ham probability (0-1) |
| response_time_ms | Float | Response time in milliseconds |
| error | Text | Error message (if any) |

## Alerting

### Drift Alerts

The system automatically detects:
- **Spam Ratio Drift**: When spam ratio changes >15% from baseline
- **Confidence Drift**: When average confidence drops >10%

Check drift status:
```bash
curl https://your-api.azurewebsites.net/metrics/drift
```

Response:
```json
{
  "drift_detected": true,
  "spam_ratio_drift": true,
  "spam_ratio_change": 0.18,
  "threshold": 0.15,
  "current_spam_ratio": 0.48,
  "baseline_spam_ratio": 0.30
}
```

## Data Retention

- **Automatic cleanup**: Runs daily at midnight
- **Retention period**: 30 days (configurable)
- **Manual cleanup**:
  ```python
  from app.monitoring.storage import MonitoringDatabase
  db = MonitoringDatabase()
  deleted = db.cleanup_old_records(retention_days=30)
  ```

## Performance Impact

- **Prediction overhead**: ~2-5ms per request
- **Storage**: ~200 bytes per prediction
- **30-day storage estimate**: 200 bytes × 10,000 predictions/day × 30 days = ~60 MB

## Troubleshooting

### Database locked error
If you see "database is locked", ensure only one worker process is running:
```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker app.main:app
```

### Missing Evidently AI
Install with:
```bash
pip install evidently
```

### Prometheus scraping fails
Ensure `/metrics` endpoint is accessible and returns text/plain format.
