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

embedding_requests_total = Counter(
    'embedding_requests_total',
    'Total embedding requests',
    ['model', 'provider']
)

embedding_tokens_total = Counter(
    'embedding_tokens_total',
    'Total tokens processed for embeddings',
    ['model']
)

embedding_duration = Histogram(
    'embedding_duration_seconds',
    'Embedding generation duration',
    ['model', 'batch_size']
)

cache_operations = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']
)

model_load_duration = Histogram(
    'model_load_duration_seconds',
    'Model loading duration',
    ['model']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

loaded_models = Gauge(
    'loaded_models',
    'Number of loaded models'
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
        'embedding_requests_total': embedding_requests_total,
        'embedding_tokens_total': embedding_tokens_total,
        'embedding_duration': embedding_duration,
        'cache_operations': cache_operations,
        'model_load_duration': model_load_duration,
        'active_requests': active_requests,
        'loaded_models': loaded_models
    }