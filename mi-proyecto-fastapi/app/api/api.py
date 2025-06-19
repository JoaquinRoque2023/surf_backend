from fastapi import APIRouter

from app.api.endpoints import movements, analysis, analysis_video

# Router principal de la API
api_router = APIRouter()

# Incluir todos los endpoints
api_router.include_router(
    movements.router,
    prefix="/movements",
    tags=["movements"]
)

api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["analysis"]
)

# Separar el análisis de video en su propio prefijo para evitar conflictos
api_router.include_router(
    analysis_video.router,
    prefix="/video-analysis",  # Cambiar prefijo para evitar conflictos
    tags=["video-analysis"]
)