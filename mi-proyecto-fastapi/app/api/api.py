from fastapi import APIRouter

from app.api.endpoints import movements, analysis

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