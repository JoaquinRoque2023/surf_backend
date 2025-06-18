from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from enum import Enum

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MovementAnalysis(Base):
    __tablename__ = "movement_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("surf_sessions.id"), nullable=True)
    movement_id = Column(Integer, ForeignKey("movements.id"), nullable=False)
    
    # Analysis metadata
    video_filename = Column(String(255), nullable=False)
    video_path = Column(String(500), nullable=False)
    video_duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    
    # Analysis status
    status = Column(String(20), default=AnalysisStatus.PENDING)
    error_message = Column(Text, nullable=True)
    
    # Processing metrics
    processing_start = Column(DateTime(timezone=True), nullable=True)
    processing_end = Column(DateTime(timezone=True), nullable=True)
    processing_duration = Column(Float, nullable=True)
    
    # Analysis results
    detected_movement_type = Column(String(50), nullable=True)
    confidence_level = Column(Float, nullable=True)
    
    # Detailed analysis data
    frame_by_frame_data = Column(JSON, nullable=True)
    pose_landmarks = Column(JSON, nullable=True)
    biomechanical_metrics = Column(JSON, nullable=True)
    
    # Scores breakdown
    technical_score = Column(Float, nullable=True)
    balance_score = Column(Float, nullable=True)
    timing_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # Specific feedback
    strengths = Column(JSON, nullable=True)
    weaknesses = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Visual feedback data
    annotated_frames = Column(JSON, nullable=True)  # Key frame annotations
    skeleton_overlay = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("SurfSession", back_populates="analyses")
    movement = relationship("Movement")
    
    def __repr__(self):
        return f"<MovementAnalysis(id={self.id}, status={self.status}, score={self.overall_score})>"