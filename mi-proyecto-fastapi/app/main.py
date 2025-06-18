from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import structlog
from datetime import datetime

from app.core.config import settings
from app.api.api import api_router
from app.schemas.movement import HealthCheckResponse

# Configurar logging estructurado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Crear instancia de FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API para análisis de movimientos de surf consciente",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para tracking de uptime
start_time = datetime.now()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de requests"""
    start_time_request = time.time()
    
    # Log del request entrante
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    # Procesar request
    response = await call_next(request)
    
    # Calcular tiempo de procesamiento
    process_time = time.time() - start_time_request
    
    # Log del response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=round(process_time, 4)
    )
    
    # Agregar header con tiempo de procesamiento
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para excepciones no manejadas"""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        method=request.method,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# Incluir routers de la API
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Endpoint raíz con health check"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        uptime_seconds=uptime
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Endpoint de health check detallado"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        uptime_seconds=uptime
    )


@app.on_event("startup")
async def startup_event():
    """Eventos al iniciar la aplicación"""
    logger.info(
        "Application startup",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Eventos al cerrar la aplicación"""
    logger.info("Application shutdown")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )