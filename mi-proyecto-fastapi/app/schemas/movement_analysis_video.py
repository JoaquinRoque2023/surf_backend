from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Any, Optional

class MovementType(str, Enum):
    """Tipos de movimientos de surf disponibles para análisis"""
    TAKE_OFF = "take_off"
    BOTTOM_TURN = "bottom_turn"
    TOP_TURN = "top_turn"
    CUTBACK = "cutback"
    FLOATER = "floater"
    SNAP = "snap"
    AERIAL = "aerial"
    TUBE_RIDING = "tube_riding"

class MovementVideoAnalysisResponse(BaseModel):
    """Respuesta del análisis de video de movimiento"""
    movement_type: MovementType
    analysis_result: Dict[str, Any]
    biomechanical_feedback: Dict[str, str]
    video_url: str
    
    class Config:
        json_encoders = {
            MovementType: lambda v: v.value
        }

class VideoAnalysisRequest(BaseModel):
    """Request para análisis de video"""
    movement_type: MovementType

class AnalysisIssue(BaseModel):
    """Problema detectado en el análisis"""
    severity: str  # "low", "medium", "high"
    description: str
    recommendation: str
    timestamp: Optional[float] = None

class DetailedAnalysisResult(BaseModel):
    """Resultado detallado del análisis"""
    status: str
    message: str
    frames_processed: int
    total_frames: int
    video_duration: float
    fps: float
    movement_detected: str
    issues_detected: List[str]
    confidence_score: Optional[float] = None