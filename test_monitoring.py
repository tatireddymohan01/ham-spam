"""Test script for monitoring endpoints."""

import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint."""
    print("\n🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predictions():
    """Make some test predictions."""
    print("📝 Making test predictions...")
    
    test_messages = [
        "Congratulations! You've won a free iPhone! Click here now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account will be closed unless you verify now!",
        "Can you send me the meeting notes from yesterday?",
        "Win $1000 cash prize! Text WIN to 12345",
        "Thanks for your help with the project!",
        "FREE GIFT! Claim your prize now before it expires!",
        "Let me know when you're free to chat",
        "CONGRATULATIONS!!! You're our lucky winner!",
        "What time does the meeting start?"
    ]
    
    for i, text in enumerate(test_messages, 1):
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        result = response.json()
        print(f"{i}. {text[:50]}...")
        print(f"   → {result['label'].upper()} (spam: {result['probability_spam']:.2%})")
        time.sleep(0.1)  # Small delay
    
    print("\n✅ Predictions completed!\n")

def test_metrics():
    """Test metrics endpoints."""
    print("📊 Testing /metrics/json endpoint...")
    response = requests.get(f"{BASE_URL}/metrics/json")
    metrics = response.json()
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Spam: {metrics['total_spam']}, Ham: {metrics['total_ham']}")
    print(f"Error rate: {metrics['error_rate']:.2%}")
    print(f"Response time (p95): {metrics['response_time_p95_ms']:.2f}ms\n")

def test_drift():
    """Test drift detection."""
    print("🔔 Testing /metrics/drift endpoint...")
    response = requests.get(f"{BASE_URL}/metrics/drift")
    drift = response.json()
    print(f"Drift detected: {drift['drift_detected']}")
    print(f"Reason: {drift.get('reason', 'N/A')}")
    if drift.get('baseline_spam_ratio'):
        print(f"Baseline spam ratio: {drift['baseline_spam_ratio']:.2%}")
        print(f"Current spam ratio: {drift['current_spam_ratio']:.2%}")
    print()

def test_dashboard():
    """Test admin dashboard."""
    print("📈 Testing /admin/dashboard endpoint...")
    response = requests.get(f"{BASE_URL}/admin/dashboard")
    data = response.json()
    
    print(f"Health: {data['health']}")
    stats_24h = data['statistics']['last_24_hours']
    print(f"\n📅 Last 24 hours:")
    print(f"  Total predictions: {stats_24h['total_predictions']}")
    print(f"  Spam ratio: {stats_24h['spam_ratio']:.2%}")
    print(f"  Avg confidence: {stats_24h['avg_confidence_spam']:.2%}")
    print(f"  Avg response time: {stats_24h['avg_response_time_ms']:.2f}ms")
    print(f"  Error rate: {stats_24h['error_rate']:.2%}\n")

def test_prometheus_metrics():
    """Test Prometheus format metrics."""
    print("🎯 Testing /metrics (Prometheus format)...")
    response = requests.get(f"{BASE_URL}/metrics")
    lines = response.text.split('\n')[:10]  # First 10 lines
    for line in lines:
        if line and not line.startswith('#'):
            print(f"  {line}")
    print(f"  ... ({len(response.text.split(chr(10)))} total lines)\n")

def test_recent_predictions():
    """Test recent predictions endpoint."""
    print("📋 Testing /admin/predictions endpoint...")
    response = requests.get(f"{BASE_URL}/admin/predictions?hours=1&limit=5")
    data = response.json()
    print(f"Found {data['count']} predictions")
    if data['predictions']:
        print("Latest predictions:")
        for pred in data['predictions'][:3]:
            print(f"  - {pred['prediction']} (confidence: {pred['confidence_spam']:.2%}) at {pred['timestamp']}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Ham-Spam Monitoring Test Suite")
    print("=" * 60)
    
    try:
        # Test basic health
        test_health()
        
        # Make predictions to generate data
        test_predictions()
        
        # Wait a moment for async logging
        time.sleep(1)
        
        # Test metrics endpoints
        test_metrics()
        test_drift()
        test_prometheus_metrics()
        
        # Test dashboard
        test_dashboard()
        test_recent_predictions()
        
        print("=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        print("\n📖 Next steps:")
        print("1. Open http://127.0.0.1:8000/docs for API documentation")
        print("2. View dashboard: http://127.0.0.1:8000/admin/dashboard")
        print("3. Check metrics: http://127.0.0.1:8000/metrics")
        print("4. Generate drift report: POST http://127.0.0.1:8000/admin/drift-report")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server.")
        print("Make sure the server is running: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
