#!/usr/bin/env python3
"""
Script para ejecutar el servidor de desarrollo de Surf Consciente
"""

import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print(f"🏄‍♂️ Iniciando {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"🌊 Servidor disponible en: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 Documentación API: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔧 Debug mode: {settings.DEBUG}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )