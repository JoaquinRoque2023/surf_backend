from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text
from sqlalchemy.sql import func
from ..core.database import Base


class SurfSession(Base):
    """
    Tabla que almacena las sesiones de surf y análisis
    """
    __tablename__ = "surf_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)  # Could be UUID or username
    session_name = Column(String(200))
    
    # Session metadata
    location = Column(String(200))
    wave_conditions = Column(JSON)  # height, period, direction, etc.
    weather_conditions = Column(JSON)  # wind, temperature, etc.
    
    # Video/data information
    video_filename = Column(String(500))
    video_duration = Column(Float)  # in seconds
    fps = Column(Float)  # frames per second
    resolution = Column(String(20))  # e.g., "1920x1080"
    
    # Session statistics
    total_movements_analyzed = Column(Integer, default=0)
    average_score = Column(Float)
    movements_summary = Column(JSON)  # Summary of all movements detected
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Notes and observations
    notes = Column(Text)
    tags = Column(JSON)  # Array of tags for categorization