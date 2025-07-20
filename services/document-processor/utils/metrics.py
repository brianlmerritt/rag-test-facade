from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import time
import structlog

logger = structlog.get_logger()

# Metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

document_processing_count = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['format', 'strategy']
)

chunk_creation_count = Counter(
    'chunks_created_total',
    'Total chunks created',
    ['strategy']
)

processing_duration = Histogram(
    'document_processing_duration_seconds',
    'Document processing duration',
    ['format', 'strategy']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

def setup_metrics(app: FastAPI):
    """Setup metrics middleware and endpoint."""
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        active_requests.inc()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
        finally:
            active_requests.dec()
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint():
        return generate_latest()
    
    return {
        'request_count': request_count,
        'request_duration': request_duration,
        'document_processing_count': document_processing_count,
        'chunk_creation_count': chunk_creation_count,
        'processing_duration': processing_duration,
        'active_requests': active_requests
    }