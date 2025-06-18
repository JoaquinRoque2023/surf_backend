from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from ..core.database import Base


class MovementType(str, enum.Enum):
    TAKEOFF = "takeoff"
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


class MovementDirection(str, enum.Enum):
    FRONTSIDE = "frontside"
    BACKSIDE = "backside"


class MovementDefinition(Base):
    """
    Tabla que define las características de cada movimiento
    """
    __tablename__ = "movement_definitions"
    
    id = Column(Integer, primary_key=True, index=True)
    movement_type = Column(Enum(MovementType), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    difficulty_level = Column(String(20))  # beginner, intermediate, advanced
    
    # Biomechanical parameters
    max_duration = Column(Float)  # Maximum duration in seconds
    key_checkpoints = Column(JSON)  # Key body positions to check
    biomechanical_profile = Column(JSON)  # Detailed biomechanical requirements
    
    # Thresholds and tolerances
    angle_tolerances = Column(JSON)  # Acceptable angle ranges
    timing_requirements = Column(JSON)  # Timing constraints
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    analyses = relationship("MovementAnalysis", back_populates="movement_definition")


class MovementAnalysis(Base):
    """
    Tabla que almacena los análisis de movimientos realizados
    """
    __tablename__ = "movement_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    movement_definition_id = Column(Integer, nullable=False)
    movement_direction = Column(Enum(MovementDirection), nullable=False)
    
    # Analysis results
    overall_score = Column(Float)  # 0-100 score
    is_correct = Column(Boolean, default=False)
    duration = Column(Float)  # Actual duration in seconds
    
    # Detailed analysis results
    body_alignment_score = Column(Float)
    timing_score = Column(Float)
    stability_score = Column(Float)
    technique_score = Column(Float)
    
    # Specific checkpoints results
    checkpoints_results = Column(JSON)  # Detailed results for each checkpoint
    
    # Keypoints data
    body_keypoints = Column(JSON)  # Array of body keypoints over time
    movement_sequence = Column(JSON)  # Sequence of movement phases
    
    # Feedback and recommendations
    feedback_message = Column(Text)
    recommendations = Column(JSON)  # Array of improvement suggestions
    
    # Metadata
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_time = Column(Float)  # Time taken to analyze in seconds
    
    # Relationships
    movement_definition = relationship("MovementDefinition", back_populates="analyses")