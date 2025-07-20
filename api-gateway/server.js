const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const { createProxyMiddleware } = require('http-proxy-middleware');
const winston = require('winston');
const swaggerJsdoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');
const client = require('prom-client');
require('dotenv').config();

const app = express();
const PORT = process.env.API_GATEWAY_PORT || 8000;

// Prometheus metrics
const register = new client.Registry();
client.collectDefaultMetrics({ register });

const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.5, 1, 2, 5]
});

const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

register.registerMetric(httpRequestDuration);
register.registerMetric(httpRequestsTotal);

// Logger configuration
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use(limiter);

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Metrics middleware
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route?.path || req.path;
    
    httpRequestDuration
      .labels(req.method, route, res.statusCode)
      .observe(duration);
    
    httpRequestsTotal
      .labels(req.method, route, res.statusCode)
      .inc();
  });
  
  next();
});

// Swagger configuration
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'RAG Test Facade API',
      version: '1.0.0',
      description: 'A modular, library-agnostic RAG system with OpenAI-compatible APIs'
    },
    servers: [
      {
        url: `http://localhost:${PORT}`,
        description: 'Development server'
      }
    ]
  },
  apis: ['./routes/*.js', './server.js']
};

const specs = swaggerJsdoc(swaggerOptions);
app.use('/docs', swaggerUi.serve, swaggerUi.setup(specs));

// Service routes and proxy configuration
const serviceProxies = {
  '/v1/documents': {
    target: process.env.DOCUMENT_PROCESSOR_URL || 'http://document-processor:8001',
    pathRewrite: { '^/v1/documents': '/v1/documents' }
  },
  '/v1/embeddings': {
    target: process.env.EMBEDDING_SERVICE_URL || 'http://embedding-service:8002',
    pathRewrite: { '^/v1/embeddings': '/v1/embeddings' }
  },
  '/v1/search/sparse': {
    target: process.env.SPARSE_RETRIEVAL_URL || 'http://sparse-retrieval:8003',
    pathRewrite: { '^/v1/search/sparse': '/v1/search/sparse' }
  },
  '/v1/search/dense': {
    target: process.env.DENSE_RETRIEVAL_URL || 'http://dense-retrieval:8004',
    pathRewrite: { '^/v1/search/dense': '/v1/search/dense' }
  },
  '/v1/search/graph': {
    target: process.env.GRAPH_RETRIEVAL_URL || 'http://graph-retrieval:8005',
    pathRewrite: { '^/v1/search/graph': '/v1/search/graph' }
  },
  '/v1/search/hybrid': {
    target: process.env.HYBRID_SEARCH_URL || 'http://hybrid-search:8006',
    pathRewrite: { '^/v1/search/hybrid': '/v1/search/hybrid' }
  },
  '/v1/rerank': {
    target: process.env.RERANKER_SERVICE_URL || 'http://reranker:8007',
    pathRewrite: { '^/v1/rerank': '/v1/rerank' }
  },
  '/v1/query/transform': {
    target: process.env.QUERY_TRANSFORM_URL || 'http://query-transform:8008',
    pathRewrite: { '^/v1/query/transform': '/v1/query/transform' }
  },
  '/v1/chat/completions': {
    target: process.env.RAG_GENERATION_URL || 'http://rag-generation:8009',
    pathRewrite: { '^/v1/chat/completions': '/v1/chat/completions' }
  },
  '/v1/evaluate': {
    target: process.env.EVALUATION_SERVICE_URL || 'http://evaluation:8010',
    pathRewrite: { '^/v1/evaluate': '/v1/evaluate' }
  }
};

// Create proxy middleware for each service
Object.entries(serviceProxies).forEach(([path, config]) => {
  const proxyMiddleware = createProxyMiddleware({
    target: config.target,
    changeOrigin: true,
    pathRewrite: config.pathRewrite,
    onProxyReq: (proxyReq, req) => {
      logger.info(`Proxying ${req.method} ${req.path} to ${config.target}`);
    },
    onError: (err, req, res) => {
      logger.error(`Proxy error for ${req.path}:`, err);
      res.status(503).json({
        error: 'Service Unavailable',
        message: 'The requested service is currently unavailable'
      });
    }
  });
  
  app.use(path, proxyMiddleware);
});

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Health check endpoint
 *     responses:
 *       200:
 *         description: Service is healthy
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: Object.keys(serviceProxies)
  });
});

/**
 * @swagger
 * /metrics:
 *   get:
 *     summary: Prometheus metrics endpoint
 *     responses:
 *       200:
 *         description: Metrics in Prometheus format
 */
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  const metrics = await register.metrics();
  res.end(metrics);
});

/**
 * @swagger
 * /:
 *   get:
 *     summary: API Gateway info
 *     responses:
 *       200:
 *         description: Gateway information
 */
app.get('/', (req, res) => {
  res.json({
    name: 'RAG Test Facade API Gateway',
    version: '1.0.0',
    description: 'A modular, library-agnostic RAG system',
    endpoints: {
      documentation: '/docs',
      health: '/health',
      metrics: '/metrics'
    },
    services: Object.keys(serviceProxies)
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.originalUrl} not found`
  });
});

app.listen(PORT, '0.0.0.0', () => {
  logger.info(`API Gateway started on port ${PORT}`);
  logger.info(`Documentation available at http://localhost:${PORT}/docs`);
});

module.exports = app;