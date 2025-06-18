from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import structlog
from datetime import datetime

from app.schemas.movement import (
    MovementAnalysisRequest,
    MovementAnalysisResponse,
    MovementSessionCreate,
    MovementSessionResponse,
    MovementType,
    SurfStance
)
from app.services.movement_analyzer import MovementAnalyzerService

logger = structlog.get_logger()
router = APIRouter()

# Instancia del servicio de análisis
analyzer_service = MovementAnalyzerService()


@router.post("/analyze", response_model=MovementAnalysisResponse)
async def analyze_movement(
    analysis_request: MovementAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analizar un movimiento de surf basado en datos biomecánicos
    """
    try:
        logger.info(
            "Starting movement analysis",
            movement_type=analysis_request.movement_type,
            stance=analysis_request.stance,
            frames_count=len(analysis_request.biomechanical_data)
        )
        
        # Validaciones básicas
        if len(analysis_request.biomechanical_data) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Se requieren datos biomecánicos para el análisis"
            )
        
        if analysis_request.duration_seconds <= 0:
            raise HTTPException(
                status_code=400,
                detail="La duración debe ser mayor a 0 segundos"
            )
        
        # Realizar el análisis
        result = await analyzer_service.analyze_movement(analysis_request)
        
        # Agregar tarea en background para logging/analytics
        background_tasks.add_task(
            log_analysis_completion,
            analysis_request.movement_type,
            result.overall_score,
            result.confidence_level
        )
        
        logger.info(
            "Movement analysis completed",
            movement_type=analysis_request.movement_type,
            overall_score=result.overall_score,
            confidence=result.confidence_level
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error during movement analysis",
            error=str(e),
            movement_type=analysis_request.movement_type if analysis_request else "unknown"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el análisis: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=MovementSessionResponse)
async def get_analysis_session(session_id: int):
    """
    Obtener detalles de una sesión de análisis específica
    """
    try:
        # Simulación de obtener sesión por ID
        # En implementación real, consultaría la base de datos
        session = MovementSessionResponse(
            id=session_id,
            surfer_name="Surfista Demo",
            location="Playa Demo",
            wave_conditions="Olas 1-2m, viento offshore",
            notes="Sesión de práctica",
            created_at=datetime.now(),
            total_analyses=5,
            average_score=7.5
        )
        
        logger.info("Retrieved analysis session", session_id=session_id)
        return session
        
    except Exception as e:
        logger.error(
            "Error retrieving analysis session",
            error=str(e),
            session_id=session_id
        )
        raise HTTPException(
            status_code=404,
            detail="Sesión de análisis no encontrada"
        )


@router.get("/sessions")
async def get_analysis_sessions(
    skip: int = 0,
    limit: int = 100,
    surfer_name: Optional[str] = None
):
    """
    Obtener lista de sesiones de análisis con paginación
    """
    try:
        # Simulación de lista de sesiones
        # En implementación real, consultaría la base de datos con filtros
        sessions = [
            MovementSessionResponse(
                id=i,
                surfer_name=f"Surfista {i}",
                location="Playa Demo",
                wave_conditions="Olas 1-2m",
                notes=f"Sesión {i}",
                created_at=datetime.now(),
                total_analyses=3,
                average_score=7.0 + (i * 0.5)
            )
            for i in range(1, min(limit + 1, 6))  # Máximo 5 sesiones demo
        ]
        
        # Aplicar filtro por nombre si se proporciona
        if surfer_name:
            sessions = [s for s in sessions if surfer_name.lower() in s.surfer_name.lower()]
        
        # Aplicar paginación
        paginated_sessions = sessions[skip:skip + limit]
        
        logger.info(
            "Retrieved analysis sessions",
            total=len(sessions),
            returned=len(paginated_sessions),
            skip=skip,
            limit=limit
        )
        
        return {
            "sessions": paginated_sessions,
            "total": len(sessions),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error("Error retrieving analysis sessions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Error al obtener sesiones de análisis"
        )


@router.get("/movements/{movement_type}/criteria")
async def get_movement_criteria(movement_type: MovementType, stance: Optional[SurfStance] = None):
    """
    Obtener criterios de evaluación específicos para un tipo de movimiento
    """
    try:
        criteria = analyzer_service.get_movement_criteria(movement_type, stance)
        
        logger.info(
            "Retrieved movement criteria",
            movement_type=movement_type,
            stance=stance,
            criteria_count=len(criteria)
        )
        
        return {
            "movement_type": movement_type,
            "stance": stance,
            "criteria": criteria,
            "total_criteria": len(criteria)
        }
        
    except Exception as e:
        logger.error(
            "Error retrieving movement criteria",
            error=str(e),
            movement_type=movement_type
        )
        raise HTTPException(
            status_code=500,
            detail="Error al obtener criterios de movimiento"
        )


@router.post("/sessions", response_model=MovementSessionResponse)
async def create_analysis_session(session_data: MovementSessionCreate):
    """
    Crear una nueva sesión de análisis de movimientos
    """
    try:
        # Validaciones básicas
        if not session_data.surfer_name or not session_data.surfer_name.strip():
            raise HTTPException(
                status_code=400,
                detail="El nombre del surfista es requerido"
            )
        
        if not session_data.location or not session_data.location.strip():
            raise HTTPException(
                status_code=400,
                detail="La ubicación es requerida"
            )
        
        # Por ahora simulamos la creación de una sesión
        # En una implementación real, esto se guardaría en base de datos
        session = MovementSessionResponse(
            id=1,  # Esto vendría de la base de datos
            surfer_name=session_data.surfer_name,
            location=session_data.location,
            wave_conditions=session_data.wave_conditions,
            notes=session_data.notes,
            created_at=datetime.now(),
            total_analyses=0,
            average_score=None
        )
        
        logger.info(
            "Analysis session created",
            session_id=session.id,
            surfer_name=session.surfer_name
        )
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error creating analysis session",
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Error al crear sesión de análisis"
        )


@router.get("/movements/types")
async def get_movement_types():
    """
    Obtener lista de tipos de movimientos disponibles para análisis
    """
    try:
        movement_types = [
            {
                "type": MovementType.TAKE_OFF,
                "name": "Take Off",
                "description": "Movimiento de ponerse de pie en la tabla"
            },
            {
                "type": MovementType.BOTTOM_TURN,
                "name": "Bottom Turn",
                "description": "Giro en la parte baja de la ola"
            },
            {
                "type": MovementType.TOP_TURN,
                "name": "Top Turn",
                "description": "Giro en la parte alta de la ola"
            }
        ]
        
        logger.info("Retrieved movement types", count=len(movement_types))
        
        return {
            "movement_types": movement_types,
            "total": len(movement_types)
        }
        
    except Exception as e:
        logger.error("Error retrieving movement types", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Error al obtener tipos de movimientos"
        )


@router.get("/stances")
async def get_surf_stances():
    """
    Obtener lista de posturas/stances disponibles
    """
    try:
        stances = [
            {
                "stance": SurfStance.REGULAR,
                "name": "Regular",
                "description": "Pie izquierdo adelante"
            },
            {
                "stance": SurfStance.GOOFY,
                "name": "Goofy",
                "description": "Pie derecho adelante"
            }
        ]
        
        logger.info("Retrieved surf stances", count=len(stances))
        
        return {
            "stances": stances,
            "total": len(stances)
        }
        
    except Exception as e:
        logger.error("Error retrieving surf stances", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Error al obtener posturas de surf"
        )


@router.delete("/sessions/{session_id}")
async def delete_analysis_session(session_id: int):
    """
    Eliminar una sesión de análisis específica
    """
    try:
        # En implementación real, esto eliminaría de la base de datos
        # Por ahora simulamos la eliminación
        logger.info("Analysis session deleted", session_id=session_id)
        
        return {
            "message": f"Sesión {session_id} eliminada exitosamente",
            "deleted_session_id": session_id
        }
        
    except Exception as e:
        logger.error(
            "Error deleting analysis session",
            error=str(e),
            session_id=session_id
        )
        raise HTTPException(
            status_code=500,
            detail="Error al eliminar sesión de análisis"
        )


@router.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud del servicio
    """
    try:
        # Verificar que el servicio analizador está funcionando
        test_criteria = analyzer_service.get_movement_criteria(MovementType.TAKE_OFF)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "movement_analyzer",
            "version": "1.0.0",
            "analyzer_available": len(test_criteria) > 0
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible"
        )


async def log_analysis_completion(
    movement_type: MovementType,
    score: float,
    confidence: float
):
    """
    Función de background task para logging de análisis completados
    """
    logger.info(
        "Analysis logged for analytics",
        movement_type=movement_type,
        score=score,
        confidence=confidence,
        timestamp=datetime.now().isoformat()
    )