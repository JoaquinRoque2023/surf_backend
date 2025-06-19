# core/config.py - Configuración actualizada para MySQL
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Configuración de la aplicación
    APP_NAME: str = "Surf Consciente API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    
    # Configuración del servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Base de datos - MySQL como principal
    DATABASE_TYPE: str = "mysql"
    
    # MySQL Configuration
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "tu_password_aqui"
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "surf_consciente"
    
    # MySQL Connection Pool Settings
    MYSQL_POOL_SIZE: int = 10
    MYSQL_MAX_OVERFLOW: int = 20
    MYSQL_POOL_RECYCLE: int = 3600
    MYSQL_POOL_TIMEOUT: int = 30
    
    # MySQL Connection Arguments
    MYSQL_CHARSET: str = "utf8mb4"
    MYSQL_COLLATION: str = "utf8mb4_unicode_ci"
    MYSQL_AUTOCOMMIT: bool = False
    MYSQL_SSL_DISABLED: bool = True  # Para desarrollo local
    
    # Seguridad
    SECRET_KEY: str = "tu-clave-secreta-muy-segura-cambiar-en-produccion"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://localhost:4200"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Configuración de análisis de movimientos
    MAX_ANALYSIS_TIME: int = 30
    MIN_CONFIDENCE_SCORE: float = 0.7
    SUPPORTED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "webm"]
    
    # Configuración de archivos
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    @property
    def database_url(self) -> str:
        """Construye la URL de MySQL con todos los parámetros"""
        base_url = (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        )
        
        # Agregar parámetros de conexión
        params = []
        params.append(f"charset={self.MYSQL_CHARSET}")
        params.append(f"collation={self.MYSQL_COLLATION}")
        
        if self.MYSQL_SSL_DISABLED:
            params.append("ssl_disabled=true")
        
        if params:
            base_url += "?" + "&".join(params)
        
        return base_url
    
    @property
    def mysql_connect_args(self) -> dict:
        """Argumentos adicionales para la conexión MySQL"""
        return {
            "charset": self.MYSQL_CHARSET,
            "collation": self.MYSQL_COLLATION,
            "autocommit": self.MYSQL_AUTOCOMMIT,
            "connect_timeout": self.MYSQL_POOL_TIMEOUT,
            "read_timeout": 60,
            "write_timeout": 60,
            "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore" 

# Instancia global de configuración
settings = Settings()