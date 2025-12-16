"""FastAPI middleware for automatic request/response monitoring."""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to track request/response metrics."""
    
    def __init__(self, app, metrics_collector=None):
        """Initialize middleware.
        
        Args:
            app: FastAPI application
            metrics_collector: Optional MetricsCollector instance
        """
        super().__init__(app)
        self.metrics_collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics."""
        # Skip monitoring for certain endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/metrics", "/health"]:
            return await call_next(request)
        
        # Track request timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Add response time header
        response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        
        return response
