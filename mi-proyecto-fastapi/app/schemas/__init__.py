# app/schemas/__init__.py
"""
Schemas for the Surf Consciente API
"""

from .movement import (
    Movement,
    MovementCreate,
    MovementUpdate,
    MovementList,
    MovementDifficulty,
    MovementCategory,
    MovementAnalysisResult,
    MovementProgress
)

from .movement_analysis_video import (
    MovementType,
    MovementVideoAnalysisResponse,
    VideoAnalysisRequest,
    AnalysisIssue,
    DetailedAnalysisResult
)

__all__ = [
    # Movement schemas
    "Movement",
    "MovementCreate", 
    "MovementUpdate",
    "MovementList",
    "MovementDifficulty",
    "MovementCategory",
    "MovementAnalysisResult",
    "MovementProgress",
    
    # Video analysis schemas
    "MovementType",
    "MovementVideoAnalysisResponse",
    "VideoAnalysisRequest",
    "AnalysisIssue",
    "DetailedAnalysisResult",
]