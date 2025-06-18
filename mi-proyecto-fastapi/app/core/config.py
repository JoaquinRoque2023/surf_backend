from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Configuración de la aplicación
    APP_NAME: str = "Surf Consciente API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    
    # Configuración del servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Base de datos
    DATABASE_URL: str = "sqlite:///./surf_consciente.db"
    
    # Seguridad
    SECRET_KEY: str = "tu-clave-secreta-muy-segura-aqui"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Configuración de análisis de movimientos
    MAX_ANALYSIS_TIME: int = 30  # segundos
    MIN_CONFIDENCE_SCORE: float = 0.7
    SUPPORTED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov"]
    
    class Config:
        env_file = ".env"


# Instancia global de configuración
settings = Settings()