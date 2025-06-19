from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class HealthCheckResponse(BaseModel):
    """Response del health check"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    uptime_seconds: float
    
class BodyPartPosition(BaseModel):
    """Posición de una parte del cuerpo en coordenadas 3D"""
    x: float = Field(..., description="Coordenada X")
    y: float = Field(..., description="Coordenada Y")
    z: float = Field(..., description="Coordenada Z")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la detección")

class MovementCriteria(BaseModel):
    """Criterios específicos evaluados para cada movimiento"""
    criteria_name: str = Field(..., description="Nombre del criterio evaluado")
    expected: str = Field(..., description="Lo que se esperaba")
    actual: str = Field(..., description="Lo que se detectó")
    passed: bool = Field(..., description="Si cumple el criterio")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza en la evaluación")

class SurfStance(str, Enum):
    """Posición del surfista respecto a la ola"""
    FRONTSIDE = "frontside"
    BACKSIDE = "backside"


class MovementType(str, Enum):
    """Tipos específicos de movimientos de surf"""
    TAKE_OFF = "take_off"
    BOTTOM_TURN = "bottom_turn"
    TOP_TURN = "top_turn"
    CUTBACK = "cutback"
    FLOATER = "floater"
    SNAP = "snap"
    TUBE_RIDE = "tube_ride"
    AERIAL = "aerial"
    DUCK_DIVE = "duck_dive"
    PADDLE_OUT = "paddle_out"

class MovementDifficulty(str, Enum):
    """Niveles de dificultad de movimientos"""
    PRINCIPIANTE = "principiante"
    INTERMEDIO = "intermedio"
    AVANZADO = "avanzado"
    EXPERTO = "experto"

class MovementCategory(str, Enum):
    """Categorías de movimientos de surf"""
    BASICO = "basico"
    MANIOBRA = "maniobra"
    AEREO = "aereo"
    TUBO = "tubo"

class MovementBase(BaseModel):
    """Base para movimientos de surf"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    difficulty: MovementDifficulty
    category: MovementCategory
    prerequisites: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)

class MovementCreate(MovementBase):
    """Schema para crear un nuevo movimiento"""
    pass

class MovementUpdate(BaseModel):
    """Schema para actualizar un movimiento"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, min_length=1, max_length=500)
    difficulty: Optional[MovementDifficulty] = None
    category: Optional[MovementCategory] = None
    prerequisites: Optional[List[str]] = None
    key_points: Optional[List[str]] = None

class Movement(MovementBase):
    """Schema completo de movimiento"""
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None
        
    class Config:
        from_attributes = True

class MovementList(BaseModel):
    """Lista de movimientos con metadatos"""
    movements: List[Movement]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool

class MovementAnalysisResult(BaseModel):
    """Resultado del análisis de un movimiento"""
    movement_id: int
    movement_name: str
    execution_score: float = Field(..., ge=0, le=100)
    technique_feedback: Dict[str, str]
    improvement_suggestions: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]
    
class MovementProgress(BaseModel):
    """Progreso del usuario en un movimiento"""
    movement_id: int
    user_id: int
    current_level: MovementDifficulty
    attempts: int
    best_score: Optional[float] = None
    last_attempt: Optional[datetime] = None
    mastery_percentage: float = Field(..., ge=0, le=100)

# Schemas para análisis de movimientos

class BiomechanicalData(BaseModel):
    """Datos biomecánicos de un frame"""
    frame_number: int
    timestamp: float
    body_keypoints: Dict[str, Any]  # Puntos clave del cuerpo
    angles: Dict[str, float]  # Ángulos articulares
    velocities: Dict[str, float]  # Velocidades
    accelerations: Dict[str, float]  # Aceleraciones

class MovementAnalysisRequest(BaseModel):
    """Request para análisis de movimiento"""
    movement_type: MovementType
    stance: SurfStance
    biomechanical_data: List[BiomechanicalData]
    duration_seconds: float
    video_metadata: Optional[Dict[str, Any]] = None
    surfer_info: Optional[Dict[str, str]] = None

class AnalysisFeedback(BaseModel):
    """Feedback específico de análisis"""
    aspect: str
    score: float = Field(..., ge=0, le=10)
    feedback: str
    improvement_tip: str

class MovementAnalysisResponse(BaseModel):
    """Response del análisis de movimiento"""
    movement_type: MovementType
    stance: SurfStance
    overall_score: float = Field(..., ge=0, le=10)
    confidence_level: float = Field(..., ge=0, le=1)
    execution_time_seconds: float
    detailed_feedback: List[AnalysisFeedback]
    key_metrics: Dict[str, float]
    recommendations: List[str]
    analysis_timestamp: datetime
    session_id: Optional[int] = None

class MovementSessionCreate(BaseModel):
    """Schema para crear sesión de análisis"""
    surfer_name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=200)
    wave_conditions: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = Field(None, max_length=1000)

class MovementSessionResponse(BaseModel):
    """Response de sesión de análisis"""
    id: int
    surfer_name: str
    location: str
    wave_conditions: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    total_analyses: int
    average_score: Optional[float] = None

  