# app/services/__init__.py
"""
Services for the Surf Consciente API
"""

from .video_analysis_service import analyze_video

__all__ = [
    "analyze_video"
]