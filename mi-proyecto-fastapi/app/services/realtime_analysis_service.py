# app/services/realtime_analysis_service.py

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Any
import time
import asyncio
from dataclasses import dataclass
from enum import Enum
import math

# Importar tipos necesarios (asumiendo que están definidos en otro módulo)
try:
    from app.models.movement_types import MovementType
    from app.services.biomechanical_analysis import calculate_angle, calculate_distance
except ImportError:
    # Fallback en caso de que no estén disponibles
    class MovementType(Enum):
        POPUP = "popup"
        CUTBACK = "cutback"
        BOTTOM_TURN = "bottom_turn"
        TOP_TURN = "top_turn"
        CARVING = "carving"


@dataclass
class FrameAnalysisResult:
    """Resultado del análisis de un frame individual."""
    poses: List[Dict]
    measurements: Dict[str, float]
    quality_score: float
    issues: List[str]
    frame_analysis: Dict[str, Any]
    timestamp: float


class RealtimeAnalyzer:
    """
    Analizador en tiempo real para análisis biomecánico de movimientos de surf.
    Procesa frames individuales y mantiene estado para análisis temporal.
    """
    
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar el modelo de pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Estado interno para análisis temporal
        self.frame_history: List[Dict] = []
        self.pose_history: List[Optional[Dict]] = []
        self.quality_history: List[float] = []
        self.max_history_length = 30  # Mantener últimos 30 frames para análisis temporal
        
        # Métricas de rendimiento
        self.processing_times: List[float] = []
        self.frame_count = 0
        
        # Configuración de análisis por tipo de movimiento
        self.movement_configs = {
            MovementType.POPUP: {
                "key_angles": ["knee_left", "knee_right", "hip_left", "hip_right", "shoulder_left", "shoulder_right"],
                "critical_phases": ["lying", "pushing_up", "knee_drive", "standing"],
                "quality_thresholds": {"min_visibility": 0.6, "min_pose_confidence": 0.7}
            },
            MovementType.CUTBACK: {
                "key_angles": ["shoulder_rotation", "hip_rotation", "knee_left", "knee_right"],
                "critical_phases": ["setup", "initiation", "carve", "completion"],
                "quality_thresholds": {"min_visibility": 0.5, "min_pose_confidence": 0.6}
            },
            MovementType.BOTTOM_TURN: {
                "key_angles": ["ankle_left", "ankle_right", "knee_left", "knee_right", "hip_rotation"],
                "critical_phases": ["approach", "compression", "turn", "extension"],
                "quality_thresholds": {"min_visibility": 0.6, "min_pose_confidence": 0.7}
            },
            MovementType.TOP_TURN: {
                "key_angles": ["shoulder_left", "shoulder_right", "hip_left", "hip_right"],
                "critical_phases": ["setup", "compression", "pivot", "completion"],
                "quality_thresholds": {"min_visibility": 0.5, "min_pose_confidence": 0.6}
            },
            MovementType.CARVING: {
                "key_angles": ["ankle_left", "ankle_right", "knee_left", "knee_right", "hip_rotation"],
                "critical_phases": ["rail_engagement", "carve", "transition"],
                "quality_thresholds": {"min_visibility": 0.6, "min_pose_confidence": 0.7}
            }
        }

    def analyze_frame(self, frame: np.ndarray, movement_type: MovementType) -> Dict[str, Any]:
        """
        Analiza un frame individual y devuelve el resultado del análisis.
        
        Args:
            frame: Frame de video como array numpy (BGR)
            movement_type: Tipo de movimiento a analizar
            
        Returns:
            Diccionario con resultado del análisis del frame
        """
        start_time = time.time()
        
        try:
            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Extraer poses y landmarks
            poses_data = self._extract_poses(results)
            
            # Calcular métricas biomecánicas
            measurements = self._calculate_measurements(poses_data, movement_type)
            
            # Evaluar calidad del frame
            quality_score = self._evaluate_frame_quality(poses_data, results)
            
            # Detectar issues biomecánicos
            issues = self._detect_biomechanical_issues(measurements, movement_type)
            
            # Análisis específico del frame
            frame_analysis = self._analyze_frame_specifics(poses_data, movement_type)
            
            # Actualizar historial
            self._update_history(poses_data, quality_score, measurements)
            
            # Calcular tiempo de procesamiento
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # Mantener solo los últimos 100 tiempos para estadísticas
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            
            return {
                "poses": poses_data,
                "measurements": measurements,
                "quality_score": quality_score,
                "issues": issues,
                "frame_analysis": frame_analysis,
                "performance_stats": {
                    "processing_time": processing_time,
                    "avg_processing_time": np.mean(self.processing_times),
                    "fps_estimate": 1.0 / processing_time if processing_time > 0 else 0,
                    "frame_count": self.frame_count
                }
            }
            
        except Exception as e:
            return {
                "poses": [],
                "measurements": {},
                "quality_score": 0.0,
                "issues": [f"Error en análisis: {str(e)}"],
                "frame_analysis": {},
                "performance_stats": {
                    "processing_time": time.time() - start_time,
                    "error": str(e)
                }
            }

    def _extract_poses(self, results) -> List[Dict]:
        """Extrae y formatea los datos de pose de MediaPipe."""
        poses = []
        
        if results.pose_landmarks:
            landmarks = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks.append({
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            
            poses.append({
                "landmarks": landmarks,
                "confidence": self._calculate_pose_confidence(landmarks)
            })
        
        return poses

    def _calculate_pose_confidence(self, landmarks: List[Dict]) -> float:
        """Calcula la confianza promedio de la pose."""
        if not landmarks:
            return 0.0
        
        visibilities = [lm["visibility"] for lm in landmarks]
        return np.mean(visibilities)

    def _calculate_measurements(self, poses_data: List[Dict], movement_type: MovementType) -> Dict[str, float]:
        """Calcula métricas biomecánicas específicas del movimiento."""
        measurements = {}
        
        if not poses_data or not poses_data[0]["landmarks"]:
            return measurements
        
        landmarks = poses_data[0]["landmarks"]
        config = self.movement_configs.get(movement_type, {})
        key_angles = config.get("key_angles", [])
        
        # Calcular ángulos clave según el tipo de movimiento
        for angle_name in key_angles:
            angle_value = self._calculate_specific_angle(landmarks, angle_name)
            if angle_value is not None:
                measurements[angle_name] = angle_value
        
        # Métricas adicionales
        measurements.update({
            "center_of_mass_x": self._calculate_center_of_mass_x(landmarks),
            "center_of_mass_y": self._calculate_center_of_mass_y(landmarks),
            "body_symmetry": self._calculate_body_symmetry(landmarks),
            "stability_index": self._calculate_stability_index(landmarks)
        })
        
        return measurements

    def _calculate_specific_angle(self, landmarks: List[Dict], angle_name: str) -> Optional[float]:
        """Calcula ángulos específicos basados en el nombre."""
        try:
            if angle_name == "knee_left":
                return self._calculate_joint_angle(landmarks, 23, 25, 27)  # Cadera-Rodilla-Tobillo izq
            elif angle_name == "knee_right":
                return self._calculate_joint_angle(landmarks, 24, 26, 28)  # Cadera-Rodilla-Tobillo der
            elif angle_name == "hip_left":
                return self._calculate_joint_angle(landmarks, 11, 23, 25)  # Hombro-Cadera-Rodilla izq
            elif angle_name == "hip_right":
                return self._calculate_joint_angle(landmarks, 12, 24, 26)  # Hombro-Cadera-Rodilla der
            elif angle_name == "shoulder_left":
                return self._calculate_joint_angle(landmarks, 13, 11, 23)  # Codo-Hombro-Cadera izq
            elif angle_name == "shoulder_right":
                return self._calculate_joint_angle(landmarks, 14, 12, 24)  # Codo-Hombro-Cadera der
            elif angle_name == "ankle_left":
                return self._calculate_joint_angle(landmarks, 25, 27, 31)  # Rodilla-Tobillo-Pie izq
            elif angle_name == "ankle_right":
                return self._calculate_joint_angle(landmarks, 26, 28, 32)  # Rodilla-Tobillo-Pie der
            elif angle_name == "shoulder_rotation":
                return self._calculate_shoulder_rotation(landmarks)
            elif angle_name == "hip_rotation":
                return self._calculate_hip_rotation(landmarks)
            else:
                return None
        except (IndexError, KeyError, TypeError):
            return None

    def _calculate_joint_angle(self, landmarks: List[Dict], p1_idx: int, p2_idx: int, p3_idx: int) -> Optional[float]:
        """Calcula el ángulo entre tres puntos."""
        try:
            if (p1_idx >= len(landmarks) or p2_idx >= len(landmarks) or p3_idx >= len(landmarks)):
                return None
            
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            p3 = landmarks[p3_idx]
            
            if not all([p1, p2, p3]):
                return None
            
            # Verificar visibilidad mínima
            if any(p.get("visibility", 0) < 0.5 for p in [p1, p2, p3]):
                return None
            
            # Calcular vectores
            v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
            v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
            
            # Calcular ángulo
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            return float(angle)
            
        except (ValueError, ZeroDivisionError):
            return None

    def _calculate_shoulder_rotation(self, landmarks: List[Dict]) -> Optional[float]:
        """Calcula la rotación de hombros."""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            if not all([left_shoulder, right_shoulder]):
                return None
            
            dx = right_shoulder["x"] - left_shoulder["x"]
            dy = right_shoulder["y"] - left_shoulder["y"]
            
            angle = math.atan2(dy, dx) * 180 / math.pi
            return float(angle)
            
        except (KeyError, IndexError, TypeError):
            return None

    def _calculate_hip_rotation(self, landmarks: List[Dict]) -> Optional[float]:
        """Calcula la rotación de cadera."""
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            if not all([left_hip, right_hip]):
                return None
            
            dx = right_hip["x"] - left_hip["x"]
            dy = right_hip["y"] - left_hip["y"]
            
            angle = math.atan2(dy, dx) * 180 / math.pi
            return float(angle)
            
        except (KeyError, IndexError, TypeError):
            return None

    def _calculate_center_of_mass_x(self, landmarks: List[Dict]) -> float:
        """Calcula el centro de masa horizontal."""
        try:
            # Usar puntos clave del torso
            key_points = [11, 12, 23, 24]  # Hombros y caderas
            valid_points = [landmarks[i] for i in key_points if i < len(landmarks) and landmarks[i]]
            
            if not valid_points:
                return 0.5
            
            x_coords = [p["x"] for p in valid_points if p.get("visibility", 0) > 0.5]
            return float(np.mean(x_coords)) if x_coords else 0.5
            
        except (KeyError, IndexError, TypeError):
            return 0.5

    def _calculate_center_of_mass_y(self, landmarks: List[Dict]) -> float:
        """Calcula el centro de masa vertical."""
        try:
            # Usar puntos clave del torso
            key_points = [11, 12, 23, 24]  # Hombros y caderas
            valid_points = [landmarks[i] for i in key_points if i < len(landmarks) and landmarks[i]]
            
            if not valid_points:
                return 0.5
            
            y_coords = [p["y"] for p in valid_points if p.get("visibility", 0) > 0.5]
            return float(np.mean(y_coords)) if y_coords else 0.5
            
        except (KeyError, IndexError, TypeError):
            return 0.5

    def _calculate_body_symmetry(self, landmarks: List[Dict]) -> float:
        """Calcula un índice de simetría corporal."""
        try:
            # Comparar lados izquierdo y derecho
            symmetry_pairs = [
                (11, 12),  # Hombros
                (13, 14),  # Codos
                (15, 16),  # Muñecas
                (23, 24),  # Caderas
                (25, 26),  # Rodillas
                (27, 28)   # Tobillos
            ]
            
            symmetry_scores = []
            
            for left_idx, right_idx in symmetry_pairs:
                if (left_idx < len(landmarks) and right_idx < len(landmarks) and
                    landmarks[left_idx] and landmarks[right_idx]):
                    
                    left = landmarks[left_idx]
                    right = landmarks[right_idx]
                    
                    if (left.get("visibility", 0) > 0.5 and right.get("visibility", 0) > 0.5):
                        # Calcular diferencia en altura (y coordinate)
                        height_diff = abs(left["y"] - right["y"])
                        symmetry_score = max(0, 1 - height_diff * 2)  # Normalizar
                        symmetry_scores.append(symmetry_score)
            
            return float(np.mean(symmetry_scores)) if symmetry_scores else 0.5
            
        except (KeyError, IndexError, TypeError):
            return 0.5

    def _calculate_stability_index(self, landmarks: List[Dict]) -> float:
        """Calcula un índice de estabilidad basado en la base de soporte."""
        try:
            # Usar pies para calcular base de soporte
            left_foot = landmarks[27] if 27 < len(landmarks) else None
            right_foot = landmarks[28] if 28 < len(landmarks) else None
            
            if not (left_foot and right_foot):
                return 0.5
            
            if (left_foot.get("visibility", 0) < 0.5 or right_foot.get("visibility", 0) < 0.5):
                return 0.5
            
            # Calcular distancia entre pies
            foot_distance = abs(left_foot["x"] - right_foot["x"])
            
            # Normalizar (0.1 = muy estrecho, 0.3+ = muy amplio)
            stability = min(1.0, foot_distance / 0.3)
            
            return float(stability)
            
        except (KeyError, IndexError, TypeError):
            return 0.5

    def _evaluate_frame_quality(self, poses_data: List[Dict], results) -> float:
        """Evalúa la calidad del análisis del frame."""
        if not poses_data:
            return 0.0
        
        try:
            pose = poses_data[0]
            landmarks = pose["landmarks"]
            
            # Factores de calidad
            visibility_score = np.mean([lm.get("visibility", 0) for lm in landmarks])
            pose_confidence = pose.get("confidence", 0)
            
            # Detectar oclusiones o poses incompletas
            key_points = [11, 12, 23, 24, 25, 26]  # Puntos críticos del torso y piernas
            key_visibility = np.mean([landmarks[i].get("visibility", 0) 
                                    for i in key_points if i < len(landmarks)])
            
            # Combinar métricas
            quality_score = (visibility_score * 0.4 + 
                           pose_confidence * 0.4 + 
                           key_visibility * 0.2)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except (KeyError, IndexError, TypeError):
            return 0.0

    def _detect_biomechanical_issues(self, measurements: Dict[str, float], movement_type: MovementType) -> List[str]:
        """Detecta problemas biomecánicos en tiempo real."""
        issues = []
        
        try:
            config = self.movement_configs.get(movement_type, {})
            
            # Análisis específico por tipo de movimiento
            if movement_type == MovementType.POPUP:
                issues.extend(self._analyze_popup_issues(measurements))
            elif movement_type == MovementType.CUTBACK:
                issues.extend(self._analyze_cutback_issues(measurements))
            elif movement_type == MovementType.BOTTOM_TURN:
                issues.extend(self._analyze_bottom_turn_issues(measurements))
            elif movement_type == MovementType.TOP_TURN:
                issues.extend(self._analyze_top_turn_issues(measurements))
            elif movement_type == MovementType.CARVING:
                issues.extend(self._analyze_carving_issues(measurements))
            
            # Análisis general
            issues.extend(self._analyze_general_issues(measurements))
            
        except Exception as e:
            issues.append(f"Error en detección de issues: {str(e)}")
        
        return issues

    def _analyze_popup_issues(self, measurements: Dict[str, float]) -> List[str]:
        """Analiza issues específicos del popup."""
        issues = []
        
        # Verificar ángulos de rodilla
        knee_left = measurements.get("knee_left")
        knee_right = measurements.get("knee_right")
        
        if knee_left and knee_left < 90:
            issues.append("Rodilla izquierda muy flexionada durante popup")
        if knee_right and knee_right < 90:
            issues.append("Rodilla derecha muy flexionada durante popup")
        
        # Verificar simetría
        body_symmetry = measurements.get("body_symmetry", 1.0)
        if body_symmetry < 0.7:
            issues.append("Falta de simetría en el popup")
        
        return issues

    def _analyze_cutback_issues(self, measurements: Dict[str, float]) -> List[str]:
        """Analiza issues específicos del cutback."""
        issues = []
        
        shoulder_rotation = measurements.get("shoulder_rotation")
        hip_rotation = measurements.get("hip_rotation")
        
        if shoulder_rotation and hip_rotation:
            rotation_diff = abs(shoulder_rotation - hip_rotation)
            if rotation_diff > 30:
                issues.append("Descoordinación entre rotación de hombros y cadera")
        
        return issues

    def _analyze_bottom_turn_issues(self, measurements: Dict[str, float]) -> List[str]:
        """Analiza issues específicos del bottom turn."""
        issues = []
        
        knee_left = measurements.get("knee_left")
        knee_right = measurements.get("knee_right")
        
        if knee_left and knee_left > 160:
            issues.append("Rodilla izquierda muy extendida en bottom turn")
        if knee_right and knee_right > 160:
            issues.append("Rodilla derecha muy extendida en bottom turn")
        
        return issues

    def _analyze_top_turn_issues(self, measurements: Dict[str, float]) -> List[str]:
        """Analiza issues específicos del top turn."""
        issues = []
        
        stability = measurements.get("stability_index", 0.5)
        if stability < 0.3:
            issues.append("Base de soporte muy estrecha para top turn")
        
        return issues

    def _analyze_carving_issues(self, measurements: Dict[str, float]) -> List[str]:
        """Analiza issues específicos del carving."""
        issues = []
        
        ankle_left = measurements.get("ankle_left")
        ankle_right = measurements.get("ankle_right")
        
        if ankle_left and ankle_left < 70:
            issues.append("Tobillo izquierdo muy flexionado para carving")
        if ankle_right and ankle_right < 70:
            issues.append("Tobillo derecho muy flexionado para carving")
        
        return issues

    def _analyze_general_issues(self, measurements: Dict[str, float]) -> List[str]:
        """Analiza issues generales de postura."""
        issues = []
        
        # Verificar centro de masa
        com_x = measurements.get("center_of_mass_x", 0.5)
        if com_x < 0.3 or com_x > 0.7:
            issues.append("Centro de masa desplazado lateralmente")
        
        return issues

    def _analyze_frame_specifics(self, poses_data: List[Dict], movement_type: MovementType) -> Dict[str, Any]:
        """Análisis específico del frame actual."""
        analysis = {
            "poses_detected": len(poses_data),
            "movement_phase": "unknown",
            "key_points_visible": 0,
            "temporal_analysis": {}
        }
        
        if poses_data:
            landmarks = poses_data[0]["landmarks"]
            
            # Contar puntos clave visibles
            key_points = [11, 12, 23, 24, 25, 26, 27, 28]  # Puntos críticos
            visible_points = sum(1 for i in key_points 
                               if i < len(landmarks) and landmarks[i] and 
                               landmarks[i].get("visibility", 0) > 0.5)
            analysis["key_points_visible"] = visible_points
            
            # Análisis temporal si hay historial
            if len(self.pose_history) > 5:
                analysis["temporal_analysis"] = self._analyze_temporal_patterns()
        
        return analysis

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analiza patrones temporales en el historial de poses."""
        temporal_analysis = {
            "movement_speed": 0.0,
            "movement_consistency": 0.0,
            "trend_analysis": {}
        }
        
        try:
            if len(self.pose_history) < 3:
                return temporal_analysis
            
            # Calcular velocidad de movimiento
            recent_poses = self.pose_history[-5:]
            movements = []
            
            for i in range(1, len(recent_poses)):
                if recent_poses[i] and recent_poses[i-1]:
                    # Calcular desplazamiento del centro de masa
                    curr_landmarks = recent_poses[i]["landmarks"]
                    prev_landmarks = recent_poses[i-1]["landmarks"]
                    
                    if len(curr_landmarks) > 24 and len(prev_landmarks) > 24:
                        # Usar hombros como referencia
                        curr_center = np.mean([(curr_landmarks[11]["x"] + curr_landmarks[12]["x"]) / 2,
                                             (curr_landmarks[11]["y"] + curr_landmarks[12]["y"]) / 2])
                        prev_center = np.mean([(prev_landmarks[11]["x"] + prev_landmarks[12]["x"]) / 2,
                                             (prev_landmarks[11]["y"] + prev_landmarks[12]["y"]) / 2])
                        
                        movement = abs(curr_center - prev_center)
                        movements.append(movement)
            
            if movements:
                temporal_analysis["movement_speed"] = float(np.mean(movements))
                temporal_analysis["movement_consistency"] = float(1.0 - np.std(movements))
            
        except Exception as e:
            temporal_analysis["error"] = str(e)
        
        return temporal_analysis

    def _update_history(self, poses_data: List[Dict], quality_score: float, measurements: Dict[str, float]):
        """Actualiza el historial de análisis."""
        # Agregar datos actuales
        current_pose = poses_data[0] if poses_data else None
        self.pose_history.append(current_pose)
        self.quality_history.append(quality_score)
        
        # Mantener solo los últimos N frames
        if len(self.pose_history) > self.max_history_length:
            self.pose_history = self.pose_history[-self.max_history_length:]
        if len(self.quality_history) > self.max_history_length:
            self.quality_history = self.quality_history[-self.max_history_length:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento del analizador."""
        return {
            "frames_processed": self.frame_count,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "avg_fps": 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
            "avg_quality": np.mean(self.quality_history) if self.quality_history else 0,
            "history_length": len(self.pose_history)
        }

    def reset(self):
        """Reinicia el estado del analizador."""
        self.pose_history.clear()
        self.quality_history.clear()
        self.frame_history.clear()
        self.processing_times.clear()
        self.frame_count = 0

    def __del__(self):
        """Limpieza al destruir el objeto."""
        if hasattr(self, 'pose'):
            self.pose.close()