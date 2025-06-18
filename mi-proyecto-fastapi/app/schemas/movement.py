from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class MovementType(str, Enum):
    """Tipos de movimientos de surf según el documento"""
    TAKE_OFF = "take_off"
    BOTTOM_TURN = "bottom_turn"
    TOP_TURN = "top_turn"
    CUTBACK = "cutback"
    FLOATER = "floater"
    REENTRY = "reentry"
    SNAP = "snap"
    CARVING = "carving"
    ROUNDHOUSE_CUTBACK = "roundhouse_cutback"
    FOAM_CLIMB = "foam_climb"
    CLOSEOUT_REENTRY = "closeout_reentry"
    TAIL_SLIDE = "tail_slide"
    RAIL_TO_RAIL = "rail_to_rail"
    HORIZONTAL_SPEED = "horizontal_speed"


class SurfStance(str, Enum):
    """Posición del surfista respecto a la ola"""
    FRONTSIDE = "frontside"
    BACKSIDE = "backside"


class BodyPartPosition(BaseModel):
    """Posición de una parte del cuerpo en coordenadas 3D"""
    x: float = Field(..., description="Coordenada X")
    y: float = Field(..., description="Coordenada Y")
    z: float = Field(..., description="Coordenada Z")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la detección")


class BiomechanicalData(BaseModel):
    """Datos biomecánicos del surfista"""
    head_position: BodyPartPosition
    torso_position: BodyPartPosition
    hip_position: BodyPartPosition
    left_shoulder: BodyPartPosition
    right_shoulder: BodyPartPosition
    left_elbow: BodyPartPosition
    right_elbow: BodyPartPosition
    left_wrist: BodyPartPosition
    right_wrist: BodyPartPosition
    left_knee: BodyPartPosition
    right_knee: BodyPartPosition
    left_ankle: BodyPartPosition
    right_ankle: BodyPartPosition
    
    # Ángulos calculados
    torso_angle: Optional[float] = Field(None, description="Ángulo del torso respecto a la tabla")
    knee_flexion_left: Optional[float] = Field(None, description="Flexión rodilla izquierda")
    knee_flexion_right: Optional[float] = Field(None, description="Flexión rodilla derecha")
    hip_angle: Optional[float] = Field(None, description="Ángulo de la cadera")
    

class MovementCriteria(BaseModel):
    """Criterios específicos evaluados para cada movimiento"""
    criteria_name: str = Field(..., description="Nombre del criterio evaluado")
    expected: str = Field(..., description="Lo que se esperaba")
    actual: str = Field(..., description="Lo que se detectó")
    passed: bool = Field(..., description="Si cumple el criterio")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza en la evaluación")


class MovementAnalysisRequest(BaseModel):
    """Request para análisis de movimiento"""
    movement_type: MovementType = Field(..., description="Tipo de movimiento a analizar")
    stance: SurfStance = Field(..., description="Posición respecto a la ola")
    biomechanical_data: List[BiomechanicalData] = Field(..., description="Datos biomecánicos por frame")
    video_fps: int = Field(30, description="FPS del video")
    duration_seconds: float = Field(..., description="Duración del movimiento en segundos")


class MovementAnalysisResponse(BaseModel):
    """Response del análisis de movimiento"""
    movement_type: MovementType
    stance: SurfStance
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Puntuación general (0-10)")
    execution_time: float = Field(..., description="Tiempo de ejecución del movimiento")
    criteria_results: List[MovementCriteria] = Field(..., description="Resultados por criterio")
    
    # Métricas específicas
    balance_score: float = Field(..., ge=0.0, le=10.0, description="Puntuación de equilibrio")
    technique_score: float = Field(..., ge=0.0, le=10.0, description="Puntuación de técnica")
    timing_score: float = Field(..., ge=0.0, le=10.0, description="Puntuación de timing")
    
    # Feedback
    strengths: List[str] = Field(default_factory=list, description="Fortalezas detectadas")
    improvements: List[str] = Field(default_factory=list, description="Áreas de mejora")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones específicas")
    
    # Metadatos
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confianza general del análisis")


class MovementSessionCreate(BaseModel):
    """Crear una sesión de análisis"""
    surfer_name: Optional[str] = Field(None, description="Nombre del surfista")
    location: Optional[str] = Field(None, description="Ubicación de la sesión")
    wave_conditions: Optional[str] = Field(None, description="Condiciones de las olas")
    notes: Optional[str] = Field(None, description="Notas adicionales")


class MovementSessionResponse(BaseModel):
    """Response de sesión de análisis"""
    id: int
    surfer_name: Optional[str]
    location: Optional[str]
    wave_conditions: Optional[str]
    notes: Optional[str]
    created_at: datetime
    total_analyses: int = Field(0, description="Total de análisis en la sesión")
    average_score: Optional[float] = Field(None, description="Puntuación promedio")


class HealthCheckResponse(BaseModel):
    """Response del health check"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    uptime_seconds: float