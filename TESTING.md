# 🧪 Local Testing Guide

## Quick Start

### 1. Start the Server

```bash
# Using Python virtual environment
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# Or if not using venv
python -m uvicorn app.main:app --reload --port 8000
```

Server will be available at: **http://127.0.0.1:8000**

### 2. Open API Documentation

Open your browser: **http://127.0.0.1:8000/docs**

This interactive page lets you test all endpoints directly!

---

## Manual Testing with Browser

### Test Endpoints in Browser:

1. **Home**: http://127.0.0.1:8000/
2. **Health Check**: http://127.0.0.1:8000/health
3. **Metrics (JSON)**: http://127.0.0.1:8000/metrics/json
4. **Drift Status**: http://127.0.0.1:8000/metrics/drift
5. **Dashboard**: http://127.0.0.1:8000/admin/dashboard
6. **Prometheus Metrics**: http://127.0.0.1:8000/metrics

---

## Testing with PowerShell/Terminal

### Make a prediction:

```powershell
$body = @{ text = "Congratulations! You've won a free iPhone!" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

### Make multiple predictions:

```powershell
$messages = @(
    "Win free cash now!",
    "Hey, how are you?",
    "URGENT: Claim your prize",
    "See you at the meeting",
    "Click here for FREE gift",
    "Thanks for your help"
)

foreach ($msg in $messages) {
    $body = @{ text = $msg } | ConvertTo-Json
    $result = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json"
    Write-Host "$msg -> $($result.label) (spam: $($result.probability_spam))"
}
```

### Check metrics:

```powershell
# JSON format
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metrics/json"

# Dashboard
Invoke-RestMethod -Uri "http://127.0.0.1:8000/admin/dashboard"

# Drift detection
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metrics/drift"
```

### View recent predictions:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/admin/predictions?hours=1&limit=10"
```

---

## Using Python Test Script

Run the automated test script:

```bash
python test_monitoring.py
```

This will:
- Make 10 test predictions
- Check all monitoring endpoints
- Display metrics and statistics
- Verify drift detection

---

## Using cURL (Alternative)

### Make prediction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Win free iPhone now!"}'
```

### Get metrics:
```bash
curl http://127.0.0.1:8000/metrics/json
```

### Get dashboard:
```bash
curl http://127.0.0.1:8000/admin/dashboard
```

---

## Checking the Database

The SQLite database is created at: **`data/monitoring.db`**

### View database with Python:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/monitoring.db")
df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 10", conn)
print(df)
conn.close()
```

### View database with SQLite CLI:

```bash
sqlite3 data/monitoring.db
```

```sql
SELECT COUNT(*) FROM prediction_logs;
SELECT prediction, COUNT(*) FROM prediction_logs GROUP BY prediction;
SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 5;
```

---

## Expected Results

After making 10+ predictions, you should see:

### `/metrics/json` Response:
```json
{
  "total_predictions": 10,
  "total_spam": 5,
  "total_ham": 5,
  "error_rate": 0.0,
  "current_spam_ratio": 0.5,
  "response_time_p95_ms": 15.2
}
```

### `/admin/dashboard` Response:
```json
{
  "health": "healthy",
  "statistics": {
    "last_24_hours": {
      "total_predictions": 10,
      "spam_count": 5,
      "ham_count": 5,
      "spam_ratio": 0.5,
      "avg_response_time_ms": 12.5
    }
  },
  "drift_detection": {
    "drift_detected": false,
    "reason": "Insufficient data for drift detection"
  }
}
```

---

## Troubleshooting

### Port already in use:
```bash
# Use different port
python -m uvicorn app.main:app --reload --port 8001
```

### Module not found errors:
```bash
# Install dependencies
pip install -r requirements.txt
```

### Database locked error:
```bash
# Use single worker (not multiple processes)
python -m uvicorn app.main:app --workers 1
```

---

## What to Test

✅ **Basic functionality:**
- [ ] Server starts without errors
- [ ] `/health` returns healthy status
- [ ] `/predict` endpoint works
- [ ] Predictions are logged to database

✅ **Monitoring:**
- [ ] `/metrics/json` shows prediction counts
- [ ] `/admin/dashboard` displays statistics
- [ ] Response times are tracked
- [ ] Error rate is 0%

✅ **Drift detection:**
- [ ] After 100 predictions, baseline is set
- [ ] `/metrics/drift` shows drift status
- [ ] Can generate Evidently report

✅ **Performance:**
- [ ] Response time < 50ms per prediction
- [ ] Database writes don't block requests
- [ ] Memory usage stays stable

---

## Next Steps

Once local testing is complete:
1. Commit and push changes
2. GitHub Actions will deploy to Azure
3. Test same endpoints on production URL
4. Setup Grafana for visualization (optional)
